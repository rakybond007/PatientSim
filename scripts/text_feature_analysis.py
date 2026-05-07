"""Deterministic text-feature analysis of patient utterances per visit.

Extracts behavioral metrics that do NOT require an LLM judge — useful as a
robust, reproducible counterpart to PersonaExpressionReviewer.

Per-visit metrics (per patient utterance, then aggregated):
  - utterance_count, total_words
  - mean_words_per_utterance, std_words_per_utterance
  - frac_short_utterances        (≤5 words : measure of withholding)
  - hedge_count                  (markers like "maybe", "I think", "I don't know")
  - past_care_reference_count    ("other hospital", "before", "last time", "they said", "they gave")
  - sarcasm_marker_count         (scare-quotes around doctor's words; exclamation+question combos)
  - trauma_marker_count          ("blood", "awful", "twice", "again", "still")
  - emotional_adjective_count    ("frustrated", "scared", "anxious", "tired", "fine")
  - patient_questions_count      (utterances ending in '?' from patient)
  - first_person_count           ("I", "me", "my")

Outputs:
  figures/text_features_<exp>.svg     - per-experiment line trajectories
  figures/text_features_summary.svg   - cross-experiment overlay (key metrics)
  figures/text_features_table.md      - raw numbers per visit per experiment

Usage:
  python scripts/text_feature_analysis.py
"""
import os
import re
import json
import glob


REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(REPO, "src", "results", "persona")
OUT = os.path.join(REPO, "figures")
DOCS_OUT = os.path.join(REPO, "docs")
os.makedirs(OUT, exist_ok=True)


HEDGE = [r"\bmaybe\b", r"\bi think\b", r"\bi don'?t know\b", r"\bi guess\b", r"\bsort of\b", r"\bkind of\b", r"\bi'?m not sure\b"]
PAST_CARE = [
    r"\bother hospital\b", r"\bother place\b", r"\bbefore\b", r"\blast time\b",
    r"\bthey said\b", r"\bthey gave\b", r"\bthey tried\b", r"\bthey told\b",
    r"\blast (?:time|visit|hospital|place|appointment)\b", r"\bprevious\b",
    r"\bagain\b", r"\bonce\b",
]
TRAUMA = [
    r"\bblood\b", r"\bawful\b", r"\btwice\b", r"\bstill\b",
    r"\bworse\b", r"\bhurt\b", r"\bterrible\b",
    r"\bnot fully\b", r"\bnot really\b",
]
SARCASM_QUOTES = re.compile(r"(['\"])([^'\"]{1,40})\1\s*\??", re.IGNORECASE)
EMOTIONAL = [
    r"\bfrustrated\b", r"\bscared\b", r"\banxious\b", r"\btired\b",
    r"\bangry\b", r"\bworried\b", r"\bunheard\b", r"\bdismissed\b",
    r"\bconfused\b", r"\bfine\b", r"\bnothing\b", r"\bjust\b",
]
FIRST_PERSON = re.compile(r"\b(?:i|me|my|mine)\b", re.IGNORECASE)


def count_pattern_hits(text: str, patterns) -> int:
    n = 0
    for p in patterns:
        n += len(re.findall(p, text, flags=re.IGNORECASE))
    return n


def patient_utterances(dialog):
    return [m.get("content", "").strip() for m in dialog if m.get("role", "").lower() == "patient"]


def per_visit_features(dialog):
    utts = patient_utterances(dialog)
    text = " ".join(utts)
    word_counts = [len(u.split()) for u in utts]
    n_utts = len(utts)
    if n_utts == 0:
        return None
    short = sum(1 for w in word_counts if w <= 5)
    sarcasm = 0
    for m in SARCASM_QUOTES.findall(text):
        # Each scare-quote-around-short-phrase counts once.
        sarcasm += 1
    return {
        "utterance_count": n_utts,
        "total_words": sum(word_counts),
        "mean_words_per_utterance": round(sum(word_counts) / n_utts, 2),
        "std_words_per_utterance": round(_std(word_counts), 2),
        "frac_short_utterances": round(short / n_utts, 3),
        "hedge_count": count_pattern_hits(text, HEDGE),
        "past_care_reference_count": count_pattern_hits(text, PAST_CARE),
        "trauma_marker_count": count_pattern_hits(text, TRAUMA),
        "sarcasm_marker_count": sarcasm,
        "emotional_adjective_count": count_pattern_hits(text, EMOTIONAL),
        "patient_questions_count": sum(1 for u in utts if u.rstrip().endswith("?")),
        "first_person_count": len(FIRST_PERSON.findall(text)),
    }


def _std(xs):
    if not xs:
        return 0.0
    m = sum(xs) / len(xs)
    return (sum((x - m) ** 2 for x in xs) / len(xs)) ** 0.5


def load_visits(exp):
    p = os.path.join(RESULTS, exp, "visits.jsonl")
    if not os.path.isfile(p):
        return []
    with open(p, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def list_experiments():
    return sorted(os.path.basename(d) for d in glob.glob(os.path.join(RESULTS, "*"))
                  if os.path.isfile(os.path.join(d, "visits.jsonl")))


# -------- charting (SVG) --------

EXP_COLORS = {
    "P1_distrust_dismissive": "#d62728",
    "P2_distrust_empathetic": "#2ca02c",
    "P3_prefix_then_repair": "#ff7f0e",
    "P5_memory_off_distrust_dismissive": "#888888",
}
EXP_LABEL = {
    "P1_distrust_dismissive": "P1: dismissive ×5 (mem ON)",
    "P2_distrust_empathetic": "P2: empathetic ×5 (mem ON)",
    "P3_prefix_then_repair": "P3: bad×2→good×3 (mem ON)",
    "P5_memory_off_distrust_dismissive": "P5: dismissive ×5 (mem OFF, control)",
}


def svg_open(w, h, title, subtitle=""):
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}" '
        'font-family="Helvetica, Arial, sans-serif">'
        '<style>'
        '.title{font-size:15px;font-weight:700;text-anchor:middle}'
        '.sub{font-size:11px;text-anchor:middle;fill:#444}'
        '.axis{stroke:#333;stroke-width:1}'
        '.grid{stroke:#bbb;stroke-width:0.4;stroke-dasharray:3,3}'
        '.lblr{font-size:10px;text-anchor:end}'
        '.lblc{font-size:10px;text-anchor:middle}'
        '.legend{font-size:10px}'
        '.metric{font-size:11px;font-weight:600}'
        '</style>'
        f'<text x="{w//2}" y="20" class="title">{title}</text>'
        f'<text x="{w//2}" y="36" class="sub">{subtitle}</text>'
    )


def panel_chart(g_x, g_y, g_w, g_h, title, series, x_labels, y_max=None):
    """Returns SVG string for a single small-multiples panel.
    series is dict[exp_name -> list-of-floats]."""
    parts = [f'<g transform="translate({g_x},{g_y})">']
    parts.append(f'<text x="{g_w/2}" y="-6" class="metric" text-anchor="middle">{title}</text>')
    plot_w = g_w - 50
    plot_h = g_h - 30
    parts.append(f'<line class="axis" x1="40" y1="0" x2="40" y2="{plot_h}"/>')
    parts.append(f'<line class="axis" x1="40" y1="{plot_h}" x2="{40+plot_w}" y2="{plot_h}"/>')
    if y_max is None:
        all_vals = [v for s in series.values() for v in s]
        y_max = max(all_vals) if all_vals else 1
    if y_max <= 0:
        y_max = 1
    # gridlines & y labels (4 ticks)
    for i in range(5):
        v = y_max * i / 4
        y = plot_h - (v / y_max) * plot_h
        parts.append(f'<line class="grid" x1="40" y1="{y}" x2="{40+plot_w}" y2="{y}"/>')
        parts.append(f'<text x="36" y="{y+3}" class="lblr">{v:.1f}</text>')
    # x labels
    n = len(x_labels)
    for i, lbl in enumerate(x_labels):
        x = 40 + (i / max(n - 1, 1)) * plot_w if n > 1 else 40 + plot_w / 2
        parts.append(f'<text x="{x}" y="{plot_h+12}" class="lblc">{lbl}</text>')
    # plot lines
    for exp, ys in series.items():
        color = EXP_COLORS.get(exp, "#444")
        pts = []
        for i, v in enumerate(ys):
            x = 40 + (i / max(n - 1, 1)) * plot_w if n > 1 else 40 + plot_w / 2
            y = plot_h - (v / y_max) * plot_h
            pts.append((x, y))
        d = "M " + " L ".join(f"{x:.1f},{y:.1f}" for x, y in pts)
        parts.append(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="2" opacity="0.85"/>')
        for x, y in pts:
            parts.append(f'<circle cx="{x}" cy="{y}" r="2.6" fill="{color}"/>')
    parts.append('</g>')
    return "".join(parts)


def grid_summary_chart(features_by_exp, save_path):
    """6-metric small multiples grid, each panel showing all experiments."""
    metrics = [
        ("mean_words_per_utterance", "Mean utterance words"),
        ("frac_short_utterances", "Fraction short utterances (≤5 words)"),
        ("past_care_reference_count", "References to prior care (count)"),
        ("trauma_marker_count", "Trauma markers (count)"),
        ("sarcasm_marker_count", "Sarcasm markers (count)"),
        ("emotional_adjective_count", "Emotional adjectives (count)"),
    ]
    rows, cols = 2, 3
    panel_w, panel_h = 320, 200
    pad = 24
    w = pad + cols * (panel_w + pad)
    h = 80 + rows * (panel_h + pad) + 60

    out = [svg_open(w, h, "Deterministic text-feature trajectories",
                    "Per-visit metrics extracted from patient utterances. No LLM judgment used.")]

    # determine x labels and series
    exp_order = ["P1_distrust_dismissive", "P5_memory_off_distrust_dismissive",
                 "P2_distrust_empathetic", "P3_prefix_then_repair"]
    available = [e for e in exp_order if e in features_by_exp]
    if not available:
        return False
    n_visits = max(len(v) for v in features_by_exp.values())
    x_labels = [f"V{i+1}" for i in range(n_visits)]

    # legend at top
    legend_x = pad
    for i, exp in enumerate(available):
        color = EXP_COLORS.get(exp, "#444")
        out.append(f'<rect x="{legend_x}" y="48" width="14" height="10" fill="{color}"/>')
        out.append(f'<text x="{legend_x+18}" y="57" class="legend">{EXP_LABEL.get(exp, exp)}</text>')
        legend_x += 240

    for idx, (key, title) in enumerate(metrics):
        r, c = divmod(idx, cols)
        gx = pad + c * (panel_w + pad)
        gy = 80 + r * (panel_h + pad)
        series = {exp: [features_by_exp[exp][i].get(key, 0) for i in range(len(features_by_exp[exp]))] for exp in available}
        out.append(panel_chart(gx, gy, panel_w, panel_h, title, series, x_labels))

    # footnote
    out.append(f'<text x="{w/2}" y="{h-30}" class="sub" text-anchor="middle">'
               'P5 (memory OFF) should stay near a flat baseline across visits; P1 (memory ON, dismissive) shows trauma-marker, sarcasm, and past-care reference accumulation.'
               '</text>')
    out.append('</svg>')
    with open(save_path, "w", encoding="utf-8") as f:
        f.write("".join(out))
    print(f"saved {save_path}")
    return True


def write_table_md(features_by_exp, save_path):
    metrics = [
        "mean_words_per_utterance",
        "frac_short_utterances",
        "past_care_reference_count",
        "trauma_marker_count",
        "sarcasm_marker_count",
        "emotional_adjective_count",
        "hedge_count",
        "patient_questions_count",
    ]
    lines = ["# Text-feature analysis — per visit per experiment", ""]
    for exp, feats in features_by_exp.items():
        lines.append(f"\n## {EXP_LABEL.get(exp, exp)}")
        head = "| Visit | " + " | ".join(metrics) + " |"
        sep = "|---|" + "|".join("---" for _ in metrics) + "|"
        lines.append(head)
        lines.append(sep)
        for i, f in enumerate(feats):
            row = [f"V{i+1}"] + [str(f.get(k, 0)) for k in metrics]
            lines.append("| " + " | ".join(row) + " |")
    with open(save_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    print(f"saved {save_path}")


def main():
    features_by_exp = {}
    for exp in list_experiments():
        if exp.startswith("smoke"):
            continue
        visits = load_visits(exp)
        if not visits:
            continue
        per_visit = []
        for v in visits:
            feats = per_visit_features(v.get("dialog_history", []))
            if feats is None:
                continue
            per_visit.append(feats)
        if per_visit:
            features_by_exp[exp] = per_visit
            print(f"{exp}: {len(per_visit)} visits")

    if not features_by_exp:
        print("No experiment data found.")
        return

    grid_summary_chart(features_by_exp, os.path.join(OUT, "text_features_summary.svg"))
    write_table_md(features_by_exp, os.path.join(DOCS_OUT, "text_features_table.md"))


if __name__ == "__main__":
    main()
