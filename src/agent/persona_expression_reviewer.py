"""Quantifies how strongly each persona trait is expressed in a patient's dialog.

This is the *dependent variable* for the persona-evolution experiments —
across multiple visits, we want to see whether (e.g.) the `distrust` trait
intensifies under a dismissive doctor, or wanes under an empathetic one.
"""
import os
import re
import ast
import json
import logging

from utils import file_to_string
from models import get_response_method, vllm_model_setup, get_answer, get_token_log


TRAITS = [
    "distrust", "pleasing", "impatience", "overanxiousness",
    "verbosity", "guardedness", "resignation", "directness",
]


class PersonaExpressionReviewer:
    def __init__(
        self,
        backend_str: str = "gemini-2.5-flash",
        backend_api_type: str = "genai",
        prompt_dir: str = "./prompts/persona_evolution",
        rubric_file: str = "persona_expression_rubric",
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
    def _format_dialog_patient_only(dialog_history: list) -> str:
        # Show full dialog but keep doctor turns short for context only.
        out = []
        for m in dialog_history:
            role = m.get("role", "?")
            content = m.get("content", "").strip()
            if role.lower() == "doctor":
                out.append(f"Doctor: {content[:200]}")
            else:
                out.append(f"Patient: {content}")
        return "\n".join(out)

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
            raise ValueError(f"No JSON in expression-reviewer output:\n{text[:400]}")
        try:
            return json.loads(m.group())
        except json.JSONDecodeError:
            return ast.literal_eval(m.group())

    def review(self, dialog_history: list, max_retries: int = 3) -> dict:
        dialog_text = self._format_dialog_patient_only(dialog_history)
        user = (
            f"DIALOG\n{dialog_text}\n\n"
            "Score each trait 0-10 based only on the patient's utterances. Return JSON only."
        )
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user},
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
                logging.warning(f"PersonaExpressionReviewer attempt {attempt+1} failed: {e}")
        raise RuntimeError(f"PersonaExpressionReviewer failed after {max_retries} attempts: {last_err}")

    @staticmethod
    def _validate(parsed: dict) -> None:
        if "scores" not in parsed:
            raise ValueError("Missing 'scores'.")
        for t in TRAITS:
            if t not in parsed["scores"]:
                raise ValueError(f"Missing trait '{t}'.")
            v = parsed["scores"][t]
            if not isinstance(v, int) or not (0 <= v <= 10):
                raise ValueError(f"Score '{t}' must be int 0-10, got {v!r}")
