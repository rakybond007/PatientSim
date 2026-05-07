"""Reviewer that scores the doctor's INTERPERSONAL behavior toward the patient.

Distinct from agent.reviewer_agent.ReviewerAgent (which scores clinical quality).
This one is the data source that drives Patient persona evolution.
"""
import os
import re
import ast
import json
import logging

from utils import file_to_string
from models import get_response_method, vllm_model_setup, get_answer, get_token_log


REQUIRED_SCORE_KEYS = [
    "listening",
    "validation",
    "empathy",
    "respect",
    "information_sharing",
]
ALLOWED_LABELS = {"dismissive", "rushed", "competent_but_cold", "warm_and_thorough", "mixed"}


class PatientExperienceReviewer:
    def __init__(
        self,
        backend_str: str = "gemini-2.5-flash",
        backend_api_type: str = "genai",
        prompt_dir: str = "./prompts/persona_evolution",
        rubric_file: str = "patient_experience_rubric",
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
        self.system_prompt = file_to_string(os.path.join(self.prompt_dir, rubric_file + ".txt"))
        self.token_log = []

    @staticmethod
    def _extract_json(text: str) -> dict:
        if text is None:
            raise ValueError("Empty response.")
        cleaned = text.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
            cleaned = re.sub(r"\s*```$", "", cleaned)
        m = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if not m:
            raise ValueError(f"No JSON in patient-experience reviewer output:\n{text[:400]}")
        blob = m.group()
        try:
            return json.loads(blob)
        except json.JSONDecodeError:
            return ast.literal_eval(blob)

    @staticmethod
    def _format_dialog(dialog_history: list) -> str:
        return "\n".join(f"{m.get('role','?')}: {m.get('content','').strip()}" for m in dialog_history)

    def review(self, dialog_history: list, base_persona_summary: str = "", max_retries: int = 3) -> dict:
        dialog_text = self._format_dialog(dialog_history)
        user_prompt = (
            f"PATIENT BASE PERSONA\n  {base_persona_summary}\n\n"
            f"CONSULTATION DIALOG\n{dialog_text}\n\n"
            "Score the doctor's interpersonal behavior from the patient's perspective. Return JSON only."
        )
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        last_err = None
        for attempt in range(max_retries):
            try:
                response = self.client(messages, model=self.model, **self.client_params)
                text = get_answer(response)
                parsed = self._extract_json(text)
                self._validate(parsed)
                self.token_log.append(get_token_log(response))
                return parsed
            except Exception as e:
                last_err = e
                logging.warning(f"PatientExperienceReviewer attempt {attempt+1} failed: {e}")
        raise RuntimeError(f"PatientExperienceReviewer failed after {max_retries} attempts: {last_err}")

    @staticmethod
    def _validate(parsed: dict) -> None:
        if "scores" not in parsed:
            raise ValueError("Missing 'scores'.")
        for k in REQUIRED_SCORE_KEYS:
            if k not in parsed["scores"]:
                raise ValueError(f"Missing score key '{k}'. Got {list(parsed['scores'].keys())}")
            v = parsed["scores"][k]
            if not isinstance(v, int) or not (1 <= v <= 5):
                raise ValueError(f"Score '{k}' must be int 1-5, got {v!r}")
        label = parsed.get("doctor_behavior_label", "")
        if label not in ALLOWED_LABELS:
            # Tolerate unexpected labels by coercing to 'mixed'
            parsed["doctor_behavior_label"] = "mixed"
