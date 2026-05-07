"""Persona Evolution Updater.

Given a base persona, prior evolved state, and the most recent visit (dialog +
patient-experience review), produce a 2-4 sentence update that the patient's
NEXT visit prompt will read.
"""
import os
import logging

from utils import file_to_string
from models import get_response_method, vllm_model_setup, get_answer, get_token_log


class PersonaEvolutionUpdater:
    def __init__(
        self,
        backend_str: str = "gemini-2.5-flash",
        backend_api_type: str = "genai",
        prompt_dir: str = "./prompts/persona_evolution",
        prompt_file: str = "persona_update",
        client_params: dict = None,
        verbose: bool = False,
    ):
        self.prompt_dir = prompt_dir
        self.backend = backend_str
        self.backend_api_type = backend_api_type
        self.client_params = client_params if client_params is not None else {"temperature": 0.4, "seed": 42}
        self.verbose = verbose

        self.client = get_response_method(self.backend_api_type)
        self.model = vllm_model_setup(self.backend) if self.backend_api_type == "vllm" else self.backend
        self.template = file_to_string(os.path.join(self.prompt_dir, prompt_file + ".txt"))
        self.token_log = []

    @staticmethod
    def _format_dialog(dialog_history: list, max_chars: int = 4000) -> str:
        text = "\n".join(f"{m.get('role','?')}: {m.get('content','').strip()}" for m in dialog_history)
        if len(text) > max_chars:
            text = text[: max_chars // 2] + "\n[... transcript truncated ...]\n" + text[-max_chars // 2 :]
        return text

    @staticmethod
    def _format_review(review: dict) -> str:
        scores = review.get("scores", {})
        summary = review.get("summary", "")
        label = review.get("doctor_behavior_label", "?")
        takeaway = review.get("patient_emotional_takeaway", "")
        score_str = ", ".join(f"{k}={v}" for k, v in scores.items())
        return (
            f"Doctor behavior label: {label}\n"
            f"Interpersonal scores (1-5): {score_str}\n"
            f"Reviewer summary (patient POV): {summary}\n"
            f"Likely emotional takeaway: {takeaway}"
        )

    def update(
        self,
        base_persona_summary: str,
        prior_evolution: str,
        dialog_history: list,
        review: dict,
        visit_outcome: str = "",
        max_retries: int = 2,
    ) -> str:
        prompt = self.template.format(
            base_persona=base_persona_summary,
            prior_evolution=prior_evolution if prior_evolution else "(none — this is the first visit)",
            doctor_behavior_summary=self._format_review(review),
            dialog_transcript=self._format_dialog(dialog_history),
            visit_outcome=visit_outcome or "(no outcome summary provided)",
        )
        messages = [{"role": "user", "content": prompt}]
        last_err = None
        for attempt in range(max_retries):
            try:
                response = self.client(messages, model=self.model, **self.client_params)
                text = get_answer(response).strip()
                if not text:
                    raise ValueError("empty update output")
                self.token_log.append(get_token_log(response))
                return text
            except Exception as e:
                last_err = e
                logging.warning(f"PersonaEvolutionUpdater attempt {attempt+1} failed: {e}")
        raise RuntimeError(f"PersonaEvolutionUpdater failed after {max_retries} attempts: {last_err}")
