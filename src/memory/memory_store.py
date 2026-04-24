import os
import json
import logging
from datetime import datetime
from typing import Callable, Optional


class MedicalMemoryStore:
    """Sliding window + rolling summary memory for a Doctor Agent.

    Layout:
        recent:    list of up to N structured consultation records (intact)
        distilled: accumulated wisdom text (rolling summary) absorbed from records
                   that fell out of the recent window.

    When a new record is added and the window is full, the oldest record is
    popped and absorbed into `distilled` via a caller-supplied distill function.
    """

    def __init__(self, path: str, window_size: int = 3):
        self.path = path
        self.N = window_size
        self.data = {
            "meta": {
                "window_size": window_size,
                "num_consultations": 0,
                "created_at": datetime.utcnow().isoformat(),
            },
            "distilled": "",
            "recent": [],
        }
        self._load()

    def _load(self) -> None:
        if self.path and os.path.isfile(self.path):
            with open(self.path, "r", encoding="utf-8") as f:
                loaded = json.load(f)
            self.data = loaded
            self.N = loaded.get("meta", {}).get("window_size", self.N)

    def _save(self) -> None:
        if not self.path:
            return
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self.data, f, ensure_ascii=False, indent=2)

    def num_consultations(self) -> int:
        return self.data["meta"]["num_consultations"]

    def has_memory(self) -> bool:
        return bool(self.data["distilled"]) or len(self.data["recent"]) > 0

    def add(self, record: dict, distill_fn: Optional[Callable[[str, dict], str]] = None) -> None:
        """Add a new consultation record. If window is full, pop the oldest
        and distill it into `distilled` using `distill_fn(current_distilled, archived_record)`.
        """
        if len(self.data["recent"]) >= self.N:
            oldest = self.data["recent"].pop(0)
            if distill_fn is not None:
                try:
                    new_distilled = distill_fn(self.data["distilled"], oldest)
                    if isinstance(new_distilled, str) and new_distilled.strip():
                        self.data["distilled"] = new_distilled.strip()
                    else:
                        logging.warning("Distillation returned empty text; keeping previous distilled.")
                except Exception as e:
                    logging.warning(f"Distillation failed: {e}; keeping previous distilled.")
        self.data["recent"].append(record)
        self.data["meta"]["num_consultations"] += 1
        self.data["meta"]["updated_at"] = datetime.utcnow().isoformat()
        self._save()

    def render_for_prompt(self) -> str:
        """Format memory as a text block for injection into the Doctor system prompt.
        Returns empty string when there is no memory to inject.
        """
        if not self.has_memory():
            return ""

        parts = []
        if self.data["distilled"]:
            parts.append("[Accumulated Clinical Wisdom from Past Consultations]\n" + self.data["distilled"])

        if self.data["recent"]:
            recent_lines = ["[Recent Detailed Case Records]"]
            for i, rec in enumerate(self.data["recent"], start=1):
                scores = rec.get("scores", {})
                score_str = ", ".join(f"{k}={v}" for k, v in scores.items()) if scores else "n/a"
                errors = rec.get("identified_errors", [])
                errors_str = "; ".join(errors) if errors else "none"
                recent_lines.append(
                    f"  Case {i}: {rec.get('patient_context','?')}\n"
                    f"    GT diagnosis: {rec.get('gt_diagnosis','?')}\n"
                    f"    Scores: {score_str}\n"
                    f"    Questioning strategy: {rec.get('questioning_strategy','?')}\n"
                    f"    Errors: {errors_str}\n"
                    f"    Summary: {rec.get('summary','')}"
                )
            parts.append("\n".join(recent_lines))

        return "\n\n".join(parts)
