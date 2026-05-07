"""LLM-as-judge: how far has this dialog drifted from a 'fresh' first-visit expression of the base persona?

Returns:
  {
    "deviation_score": 0-10,
    "direction": "negative" | "positive" | "mixed" | "none",
    "evidence": [str, str, str],
    "rationale": str
  }

Designed to be run on dialogs from existing experiment runs (no API call during the
consultation itself). Supports an ensemble mode: call review_ensemble(N) to average
across multiple seeds for stability.
"""
import os
import re
import ast
import json
import logging
import statistics

from utils import file_to_string
from models import get_response_method, vllm_model_setup, get_answer, get_token_log


class PersonaDeviationReviewer:
    def __init__(
        self,
        backend_str: str = "gemini-2.5-flash",
        backend_api_type: str = "genai",
        prompt_dir: str = "./prompts/persona_evolution",
        rubric_file: str = "persona_deviation_rubric",
        client_params: dict = None,
        verbose: bool = False,
    ):
        self.prompt_dir = prompt_dir
        self.backend = backend_str
        self.backend_api_type = backend_api_type
        # NOTE: leave temperature 0 for the single-call path; ensemble uses different seeds.
        self.client_params = client_params if client_params is not None else {"temperature": 0, "seed": 42}
        self.verbose = verbose

        self.client = get_response_method(self.backend_api_type)
        self.model = vllm_model_setup(self.backend) if self.backend_api_type == "vllm" else self.backend
        self.system_prompt = file_to_string(os.path.join(self.prompt_dir, rubric_file + ".txt"))
        self.token_log = []

    @staticmethod
    def _format_dialog(dialog_history: list) -> str:
        return "\n".join(f"{m.get('role','?')}: {m.get('content','').strip()}" for m in dialog_history)

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
            raise ValueError(f"No JSON in deviation reviewer output:\n{text[:400]}")
        try:
            return json.loads(m.group())
        except json.JSONDecodeError:
            return ast.literal_eval(m.group())

    def review(self, dialog_history: list, base_persona_summary: str, max_retries: int = 3, seed_override: int = None) -> dict:
        params = dict(self.client_params)
        if seed_override is not None:
            params["seed"] = seed_override
        user = (
            f"BASE PERSONA\n  {base_persona_summary}\n\n"
            f"DIALOG\n{self._format_dialog(dialog_history)}\n\n"
            "Return JSON only."
        )
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user},
        ]
        last_err = None
        for attempt in range(max_retries):
            try:
                response = self.client(messages, model=self.model, **params)
                text = get_answer(response)
                parsed = self._extract_json(text)
                self._validate(parsed)
                self.token_log.append(get_token_log(response))
                return parsed
            except Exception as e:
                last_err = e
                logging.warning(f"PersonaDeviationReviewer attempt {attempt+1} failed: {e}")
        raise RuntimeError(f"PersonaDeviationReviewer failed after {max_retries} attempts: {last_err}")

    def review_ensemble(self, dialog_history, base_persona_summary, n: int = 3, seeds=None):
        """Run the reviewer N times with different seeds, return individual + aggregate."""
        if seeds is None:
            seeds = [42, 7, 99][:n]
        runs = []
        for s in seeds:
            try:
                runs.append(self.review(dialog_history, base_persona_summary, seed_override=s))
            except Exception as e:
                logging.warning(f"ensemble seed {s} failed: {e}")
        if not runs:
            raise RuntimeError("All ensemble runs failed.")
        scores = [r["deviation_score"] for r in runs]
        directions = [r["direction"] for r in runs]
        from collections import Counter
        agg_dir = Counter(directions).most_common(1)[0][0]
        return {
            "n_runs": len(runs),
            "score_mean": round(statistics.mean(scores), 2),
            "score_std": round(statistics.pstdev(scores), 2) if len(scores) > 1 else 0.0,
            "score_min": min(scores),
            "score_max": max(scores),
            "direction_mode": agg_dir,
            "individual": runs,
        }

    @staticmethod
    def _validate(parsed: dict) -> None:
        if "deviation_score" not in parsed:
            raise ValueError("Missing 'deviation_score'.")
        v = parsed["deviation_score"]
        if not isinstance(v, int) or not (0 <= v <= 10):
            raise ValueError(f"deviation_score must be int 0-10, got {v!r}")
        d = parsed.get("direction", "")
        if d not in ("negative", "positive", "mixed", "none"):
            parsed["direction"] = "mixed"
