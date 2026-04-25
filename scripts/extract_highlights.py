"""Scan completed memory experiment runs for qualitatively interesting consultations.

Criteria for being a "highlight":
  - top_score: total_score >= 23 (excellent consultation)
  - poor_score: total_score <= 14 (notable failure)
  - score_jump: same-mode jump of >= 5 points between consecutive consultations
  - score_drop: same-mode drop of >= 4 points (often = distillation event or hard scenario)
  - perfect: total_score == 25
  - error_free: identified_errors == [] (rare)
  - notable_persona: personality in {distrust, pleasing} or recall == low — proposal hot spots

For each highlighted consultation, attach:
  - mode, scenario_idx, hadm_id, persona, gt_diagnosis, scores, summary
  - 2-4 representative dialog turn pairs (chosen heuristically)
  - the lessons that the reviewer extracted
  - which "highlight" tags it earned

Outputs:
  docs/highlights.md          - human-readable list, grouped by tag
  figures/highlights.json     - machine-readable for slide builders
"""
import os
import sys
import json
import glob
from collections import defaultdict


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_ROOT = os.path.join(REPO_ROOT, "src", "results", "memory")
DOCS_DIR = os.path.join(REPO_ROOT, "docs")
FIGURES_DIR = os.path.join(REPO_ROOT, "figures")


def load_jsonl(p):
    if not os.path.isfile(p):
        return []
    with open(p, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def find_dirs():
    return sorted(d for d in glob.glob(os.path.join(RESULTS_ROOT, "*"))
                  if os.path.isdir(d) and not d.endswith(".partial"))


def short_persona(ctx_or_dialog_meta):
    """Extract a short persona label like 'distrust+lowrecall+CEFR_B'."""
    if isinstance(ctx_or_dialog_meta, str):
        return ctx_or_dialog_meta
    p = ctx_or_dialog_meta.get("personality_type") or ""
    r = ctx_or_dialog_meta.get("recall_level_type") or ""
    c = ctx_or_dialog_meta.get("cefr_type") or ""
    parts = [p]
    if r == "low":
        parts.append("lowrecall")
    if c:
        parts.append(f"CEFR_{c}")
    return "+".join(parts) if parts else "?"


def pick_representative_turns(dialog_history, max_pairs=3):
    """Select interesting Doctor↔Patient pairs.

    Heuristic:
      - Skip the greeting pair (Doctor #1 = "Hello", Patient #1 typically chief complaint).
      - First pair (first real exchange).
      - Middle pair (where memory is most likely visible).
      - Last pair before [DDX] (where the doctor synthesizes).
    """
    pairs = []
    cursor = 0
    while cursor + 1 < len(dialog_history):
        d = dialog_history[cursor]
        p = dialog_history[cursor + 1] if cursor + 1 < len(dialog_history) else None
        if d and p and d.get("role") == "Doctor" and p.get("role") == "Patient":
            pairs.append((d, p))
        cursor += 2
    # also include patient-then-doctor pair
    pairs2 = []
    cursor = 1
    while cursor + 1 < len(dialog_history):
        p = dialog_history[cursor]
        d = dialog_history[cursor + 1] if cursor + 1 < len(dialog_history) else None
        if p and d and p.get("role") == "Patient" and d.get("role") == "Doctor":
            pairs2.append((p, d))
        cursor += 2

    # Use pairs2 (Patient -> Doctor) since that's where the doctor's reaction shows
    if not pairs2:
        return []
    chosen = []
    n = len(pairs2)
    if n <= max_pairs:
        chosen = pairs2
    else:
        idxs = sorted({0, n // 2, n - 1})[:max_pairs]
        chosen = [pairs2[i] for i in idxs]
    out = []
    for p, d in chosen:
        out.append({
            "patient": (p.get("content") or "").strip()[:600],
            "doctor": (d.get("content") or "").strip()[:600],
        })
    return out


def derive_persona_from_dialog_entry(dialog_entry):
    """Pull persona out of dialog_entry's nested patient_token_log info or meta."""
    # The dialog_entry stores hadm_id; we cannot reliably know the persona without
    # cross-referencing the patient_profile dataset. Carry whatever metadata is
    # in the memory record.
    return None


def index_reviews_by_mode():
    out = {}
    for d in find_dirs():
        mode = os.path.basename(d)
        reviews = load_jsonl(os.path.join(d, "reviews.jsonl"))
        dialogs = load_jsonl(os.path.join(d, "dialogues.jsonl"))
        # Index dialogs by scenario_idx (1-based)
        dialog_by_idx = {}
        for entry in dialogs:
            idx = entry.get("scenario_idx") or entry.get("consultation_idx")
            if idx is not None:
                dialog_by_idx[idx] = entry
        out[mode] = {"reviews": reviews, "dialogs": dialog_by_idx, "dir": d}
    return out


def tag_highlights(reviews):
    tags = []  # list of (idx, set_of_tags)
    prev = None
    for r in reviews:
        scen = []
        ts = r.get("total_score", 0)
        if ts >= 25:
            scen.append("perfect")
        elif ts >= 23:
            scen.append("top_score")
        if ts <= 14:
            scen.append("poor_score")
        if not r.get("identified_errors"):
            scen.append("error_free")
        if prev is not None:
            delta = ts - prev
            if delta >= 5:
                scen.append("score_jump")
            elif delta <= -4:
                scen.append("score_drop")
        ctx = (r.get("patient_context") or "").lower()
        if "distrust" in ctx or "pleasing" in ctx:
            scen.append("notable_persona")
        if "recall=low" in ctx or "lowrecall" in ctx:
            scen.append("low_recall")
        tags.append(scen)
        prev = ts
    return tags


def render_md(highlights_by_mode, output_path):
    lines = ["# Qualitative Highlights — interesting consultations from memory experiments\n"]
    lines.append("Auto-generated by `scripts/extract_highlights.py`. Each entry is sourced from `src/results/memory/<mode>/reviews.jsonl` and `dialogues.jsonl` and tagged by simple criteria.\n")
    lines.append("Tags: **perfect** (25/25) · **top_score** (≥23) · **poor_score** (≤14) · **score_jump** (+5 vs prev) · **score_drop** (-4 vs prev) · **error_free** · **notable_persona** (distrust/pleasing) · **low_recall**\n")

    # Group by tag for narrative
    by_tag = defaultdict(list)
    for mode, items in highlights_by_mode.items():
        for it in items:
            for t in it["tags"]:
                by_tag[t].append((mode, it))

    # Tag-ordered narrative section
    tag_order = ["perfect", "top_score", "score_jump", "error_free", "score_drop", "poor_score", "notable_persona", "low_recall"]
    for tag in tag_order:
        bucket = by_tag.get(tag, [])
        if not bucket:
            continue
        lines.append(f"\n## {tag} ({len(bucket)} entries)")
        for mode, it in bucket[:6]:  # cap per tag
            lines.append(f"\n### {mode} · scenario {it['idx']} · {it['patient_context']}")
            lines.append(f"- gt diagnosis: **{it['gt_diagnosis']}**")
            lines.append(f"- score: **{it['total_score']}/25** ({it['scores']})")
            lines.append(f"- tags: {', '.join(it['tags'])}")
            if it.get("summary"):
                lines.append(f"- reviewer summary: *{it['summary']}*")
            if it.get("identified_errors"):
                errors = "; ".join(it["identified_errors"])
                lines.append(f"- identified errors: {errors}")
            if it.get("key_lessons"):
                lessons = "; ".join(it["key_lessons"])
                lines.append(f"- key lessons: {lessons}")
            if it.get("turns"):
                lines.append("- representative turns:")
                for pair in it["turns"]:
                    lines.append("  ")
                    lines.append(f"  > **Patient**: {pair['patient']}")
                    lines.append(f"  > **Doctor**: {pair['doctor']}")

    # Mode-by-mode appendix
    lines.append("\n---\n## Appendix: All highlights by mode")
    for mode, items in highlights_by_mode.items():
        lines.append(f"\n### {mode}")
        if not items:
            lines.append("(no highlighted consultations)")
            continue
        for it in items:
            tags = ", ".join(it["tags"]) or "—"
            lines.append(f"- s{it['idx']} | {it['patient_context']} → **{it['gt_diagnosis']}** | score **{it['total_score']}/25** | tags: {tags}")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"saved {output_path}")


def main():
    os.makedirs(DOCS_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    indexed = index_reviews_by_mode()
    highlights_by_mode = {}
    flat = []

    for mode, data in indexed.items():
        reviews = data["reviews"]
        dialog_by_idx = data["dialogs"]
        if not reviews:
            highlights_by_mode[mode] = []
            continue
        tags_per = tag_highlights(reviews)
        items = []
        for r, tags in zip(reviews, tags_per):
            if not tags:
                continue  # not a highlight
            idx = r.get("scenario_idx") or r.get("consultation_idx") or 0
            dialog_entry = dialog_by_idx.get(idx)
            turns = pick_representative_turns(dialog_entry["dialog_history"]) if dialog_entry else []
            entry = {
                "mode": mode,
                "idx": idx,
                "hadm_id": r.get("hadm_id"),
                "patient_context": r.get("patient_context", ""),
                "gt_diagnosis": r.get("gt_diagnosis"),
                "scores": r.get("scores", {}),
                "total_score": r.get("total_score", 0),
                "summary": r.get("summary", ""),
                "questioning_strategy": r.get("questioning_strategy", ""),
                "identified_errors": r.get("identified_errors", []),
                "key_lessons": r.get("key_lessons", []),
                "tags": tags,
                "turns": turns,
            }
            items.append(entry)
            flat.append(entry)
        highlights_by_mode[mode] = items

    # Outputs
    md_path = os.path.join(DOCS_DIR, "highlights.md")
    render_md(highlights_by_mode, md_path)

    json_path = os.path.join(FIGURES_DIR, "highlights.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"by_mode": {m: [{k: v for k, v in it.items()} for it in items] for m, items in highlights_by_mode.items()}}, f, ensure_ascii=False, indent=2)
    print(f"saved {json_path}")

    print(f"\nSummary: {len(flat)} highlighted consultations across {len(indexed)} modes")


if __name__ == "__main__":
    main()
