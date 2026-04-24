import os
import re
import ast
import json
import logging

from utils import file_to_string
from models import get_response_method, vllm_model_setup, get_answer, get_token_log


REQUIRED_SCORE_KEYS = [
    "history_taking",
    "ddx_reasoning",
    "clinical_communication",
    "safety_risk_management",
    "diagnosis_management_plan",
]


class ReviewerAgent:
    """Evaluates a completed doctor-patient consultation using the 5-category rubric.

    Returns a structured record consumable by MedicalMemoryStore:
        {
            "scores": { "history_taking": 1-5, ..., "diagnosis_management_plan": 1-5 },
            "summary": str,
            "questioning_strategy": str,
            "identified_errors": [str, ...],
            "key_lessons": [str, ...],
        }
    """

    def __init__(
        self,
        backend_str: str = "gemini-2.5-flash",
        backend_api_type: str = "genai",
        prompt_dir: str = "./prompts/review",
        rubric_prompt_file: str = "rubric_system",
        distill_prompt_file: str = "distillation",
        client_params: dict = None,
        verbose: bool = False,
    ):
        self.prompt_dir = prompt_dir
        self.backend = backend_str
        self.backend_api_type = backend_api_type
        self.client_params = client_params if client_params is not None else {"temperature": 0, "seed": 42}
        self.verbose = verbose

        self.client = get_response_method(self.backend_api_type)
        self.model = vllm_model_setup(self.backend) if self.backend_api_type == "vllm" else self.backend

        self.rubric_system_prompt = file_to_string(os.path.join(self.prompt_dir, rubric_prompt_file + ".txt"))
        self.distill_prompt_template = file_to_string(os.path.join(self.prompt_dir, distill_prompt_file + ".txt"))

        self.token_log = {"review": [], "distill": []}

    # ---------- helpers ----------

    @staticmethod
    def _extract_json(text: str) -> dict:
        """Pull the first JSON object out of a model response."""
        if text is None:
            raise ValueError("Empty response from reviewer model.")
        cleaned = text.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
            cleaned = re.sub(r"\s*```$", "", cleaned)
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if not match:
            raise ValueError(f"No JSON object found in reviewer output:\n{text[:500]}")
        blob = match.group()
        try:
            return json.loads(blob)
        except json.JSONDecodeError:
            return ast.literal_eval(blob)

    @staticmethod
    def _format_dialog(dialog_history: list) -> str:
        lines = []
        for utter in dialog_history:
            role = utter.get("role", "?")
            content = utter.get("content", "").strip()
            lines.append(f"{role}: {content}")
        return "\n".join(lines)

    @staticmethod
    def _format_patient_context(patient_profile: dict) -> str:
        return (
            f"{patient_profile.get('age','?')}{patient_profile.get('gender','?')}, "
            f"chief complaint: {patient_profile.get('chiefcomplaint','?')}, "
            f"persona: cefr={patient_profile.get('cefr_option') or patient_profile.get('cefr','?')}, "
            f"personality={patient_profile.get('personality_option') or patient_profile.get('personality','?')}, "
            f"recall={patient_profile.get('recall_level_option') or patient_profile.get('recall_level','?')}, "
            f"dazed={patient_profile.get('dazed_level_option') or patient_profile.get('dazed_level','?')}"
        )

    # ---------- main ----------

    def review(self, dialog_history: list, patient_profile: dict, max_retries: int = 3) -> dict:
        """Score a completed consultation and return a structured memory record."""
        dialog_text = self._format_dialog(dialog_history)
        patient_context = self._format_patient_context(patient_profile)
        gt_diagnosis = patient_profile.get("diagnosis", "?")

        user_prompt = (
            f"PATIENT CONTEXT\n  {patient_context}\n"
            f"GROUND-TRUTH DIAGNOSIS (for your reference only, not shown to doctor)\n  {gt_diagnosis}\n\n"
            f"CONSULTATION TRANSCRIPT\n{dialog_text}\n\n"
            "Evaluate the doctor strictly per the rubric and return the JSON object now."
        )

        messages = [
            {"role": "system", "content": self.rubric_system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        last_err = None
        for attempt in range(max_retries):
            try:
                response = self.client(messages, model=self.model, **self.client_params)
                answer_text = get_answer(response)
                parsed = self._extract_json(answer_text)
                self._validate_review(parsed)
                self.token_log["review"].append(get_token_log(response))
                record = self._build_record(parsed, patient_context, gt_diagnosis, patient_profile)
                return record
            except Exception as e:
                last_err = e
                logging.warning(f"Reviewer attempt {attempt+1}/{max_retries} failed: {e}")

        raise RuntimeError(f"Reviewer failed after {max_retries} attempts: {last_err}")

    @staticmethod
    def _validate_review(parsed: dict) -> None:
        if "scores" not in parsed:
            raise ValueError("Review missing 'scores' key.")
        for k in REQUIRED_SCORE_KEYS:
            if k not in parsed["scores"]:
                raise ValueError(f"Review missing score key '{k}'. Got: {list(parsed['scores'].keys())}")
            v = parsed["scores"][k]
            if not isinstance(v, int) or not (1 <= v <= 5):
                raise ValueError(f"Score '{k}' must be int 1-5, got {v!r}")

    @staticmethod
    def _build_record(parsed: dict, patient_context: str, gt_diagnosis: str, patient_profile: dict) -> dict:
        scores = parsed["scores"]
        total = sum(scores.values())
        return {
            "hadm_id": patient_profile.get("hadm_id"),
            "patient_context": patient_context,
            "gt_diagnosis": gt_diagnosis,
            "scores": scores,
            "total_score": total,
            "total_over_25": f"{total}/25 ({int(round(100*total/25))}%)",
            "summary": parsed.get("summary", ""),
            "questioning_strategy": parsed.get("questioning_strategy", ""),
            "identified_errors": parsed.get("identified_errors", []),
            "key_lessons": parsed.get("key_lessons", []),
        }

    def distill(self, current_distilled: str, archived_record: dict, max_retries: int = 2) -> str:
        """Fold an archived record into the distilled wisdom notebook."""
        prompt = self.distill_prompt_template.format(
            current_distilled=current_distilled if current_distilled else "(empty - this is the first entry)",
            patient_context=archived_record.get("patient_context", "?"),
            gt_diagnosis=archived_record.get("gt_diagnosis", "?"),
            scores=json.dumps(archived_record.get("scores", {})),
            questioning_strategy=archived_record.get("questioning_strategy", ""),
            identified_errors="; ".join(archived_record.get("identified_errors", [])) or "none",
            key_lessons="; ".join(archived_record.get("key_lessons", [])) or "none",
            summary=archived_record.get("summary", ""),
        )
        messages = [{"role": "user", "content": prompt}]

        last_err = None
        for attempt in range(max_retries):
            try:
                response = self.client(messages, model=self.model, **self.client_params)
                text = get_answer(response).strip()
                if text.startswith("```"):
                    text = re.sub(r"^```(?:markdown|md)?\s*", "", text)
                    text = re.sub(r"\s*```$", "", text)
                if not text:
                    raise ValueError("empty distillation output")
                self.token_log["distill"].append(get_token_log(response))
                return text
            except Exception as e:
                last_err = e
                logging.warning(f"Distill attempt {attempt+1}/{max_retries} failed: {e}")

        raise RuntimeError(f"Distillation failed after {max_retries} attempts: {last_err}")
