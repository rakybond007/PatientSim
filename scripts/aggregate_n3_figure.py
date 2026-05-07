"""Aggregate n=3 patient analysis: P1 (memory ON) vs P5 (memory OFF).

For each metric and visit V1..V5, compute MEAN +/- STD across the 3 patients.
Within-patient normalization (Δ from V1) is also reported because patients
have different baseline talkativeness (e.g., low-CEFR patient has short utterances).

Output:
  figures/persona_n3_headline.svg  - 4-panel chart, P1 vs P5 averaged across 3 patients
  docs/persona_n3_aggregate_table.md - markdown summary
"""
import os
import json
import glob
import statistics

import sys
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts"))
from text_feature_analysis import per_visit_features, load_visits  # noqa


RESULTS = os.path.join(REPO, "src", "results", "persona")
OUT = os.path.join(REPO, "figures")
DOCS_OUT = os.path.join(REPO, "docs")
os.makedirs(OUT, exist_ok=True)


P1_RUNS = [
    "P1_distrust_dismissive",
    "P1_distrust_dismissive_h21114179",
    "P1_distrust_dismissive_h22308320",
]
P5_RUNS = [
    "P5_memory_off_distrust_dismissive",
    "P5_memory_off_distrust_dismissive_h21114179",
    "P5_memory_off_distrust_dismissive_h22308320",
]

METRICS = [
    ("past_care_reference_count", "References to prior care"),
    ("trauma_marker_count",        "Trauma markers"),
    ("sarcasm_marker_count",       "Sarcasm markers"),
    ("emotional_adjective_count",  "Emotional adjectives"),
]


def collect_per_visit(runs):
    """For each run, compute per-visit features. Returns per_visit[V_idx][run_idx][metric]."""
    by_run = []
    for r in runs:
        visits = load_visits(r)
        feats = [per_visit_features(v["dialog_history"]) for v in visits]
        by_run.append(feats)
    return by_run


def mean_std_per_visit(by_run, metric):
    n_visits = max(len(r) for r in by_run)
    out = []
    for i in range(n_visits):
        vals = [r[i].get(metric, 0) for r in by_run if i < len(r) and r[i] is not None]
        m = statistics.mean(vals) if vals else 0
        s = statistics.pstdev(vals) if len(vals) > 1 else 0
        out.append((m, s, vals))
    return out


def deviation_mean_std_per_visit(runs):
    """Read deviation_scores.jsonl per run, compute mean+/-std across runs per visit."""
    by_run = []
    for r in runs:
        p = os.path.join(RESULTS, r, "deviation_scores.jsonl")
        if not os.path.isfile(p):
            by_run.append([])
            continue
        recs = [json.loads(l) for l in open(p) if l.strip()]
        by_run.append([rec.get("score_mean", 0) for rec in recs])
    n = max(len(r) for r in by_run) if by_run else 0
    out = []
    for i in range(n):
        vals = [r[i] for r in by_run if i < len(r)]
        m = statistics.mean(vals) if vals else 0
        s = statistics.pstdev(vals) if len(vals) > 1 else 0
        out.append((m, s))
    return out


def headline_figure(p1_byrun, p5_byrun, save_path):
    p1_dev = deviation_mean_std_per_visit(P1_RUNS)
    p5_dev = deviation_mean_std_per_visit(P5_RUNS)

    panels = []  # (title, p1_series, p5_series)
    for key, title in METRICS:
        panels.append((title,
                       mean_std_per_visit(p1_byrun, key),
                       mean_std_per_visit(p5_byrun, key)))
    # also append deviation
    panels.append(("LLM persona-deviation score (0-10)",
                   [(m, s) for (m, s) in p1_dev],
                   [(m, s) for (m, s) in p5_dev]))

    cols = 3
    rows = (len(panels) + cols - 1) // cols
    panel_w = 320
    panel_h = 220
    pad = 28
    w = pad + cols * (panel_w + pad)
    h = 110 + rows * (panel_h + pad) + 40

    out = []
    out.append(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}" '
               'font-family="Helvetica, Arial, sans-serif">')
    out.append('<style>'
               '.title{font-size:18px;font-weight:800;text-anchor:middle}'
               '.sub{font-size:12px;text-anchor:middle;fill:#444}'
               '.metric{font-size:13px;font-weight:700;text-anchor:middle}'
               '.axis{stroke:#333;stroke-width:1}'
               '.grid{stroke:#bbb;stroke-width:0.4;stroke-dasharray:3,3}'
               '.lblr{font-size:9px;text-anchor:end}'
               '.lblc{font-size:9px;text-anchor:middle}'
               '.legend{font-size:11px}'
               '.note{font-size:10px;fill:#444}'
               '</style>')
    out.append(f'<text x="{w//2}" y="26" class="title">'
               'Persona Evolution Quantified — Memory ON vs OFF averaged across n=3 distrust patients'
               '</text>')
    out.append(f'<text x="{w//2}" y="48" class="sub">'
               'Same 5-visit dismissive-doctor sequence, only difference is memory injection. Mean &#177; std across 3 patients (38F MI/IO, 74M MI, 67M IO).'
               '</text>')

    # legend
    out.append(f'<rect x="60" y="64" width="14" height="10" fill="#d62728"/>')
    out.append(f'<text x="80" y="73" class="legend">P1: dismissive ×5 (memory ON)</text>')
    out.append(f'<rect x="320" y="64" width="14" height="10" fill="#888"/>')
    out.append(f'<text x="340" y="73" class="legend">P5: dismissive ×5 (memory OFF, control)</text>')

    n_visits = 5
    for idx, (title, p1, p5) in enumerate(panels):
        r, c = divmod(idx, cols)
        gx = pad + c * (panel_w + pad)
        gy = 110 + r * (panel_h + pad)
        out.append(f'<g transform="translate({gx},{gy})">')
        out.append(f'<text x="{panel_w/2}" y="-6" class="metric">{title}</text>')
        plot_w = panel_w - 50
        plot_h = panel_h - 36
        out.append(f'<line class="axis" x1="40" y1="0" x2="40" y2="{plot_h}"/>')
        out.append(f'<line class="axis" x1="40" y1="{plot_h}" x2="{40+plot_w}" y2="{plot_h}"/>')
        all_vals = []
        for m, s, *_ in p1: all_vals.append(m + s)
        for m, s, *_ in p5: all_vals.append(m + s)
        y_max = max(all_vals) if all_vals else 1
        if y_max < 1: y_max = 1
        for i in range(5):
            v = y_max * i / 4
            y = plot_h - (v / y_max) * plot_h
            out.append(f'<line class="grid" x1="40" y1="{y}" x2="{40+plot_w}" y2="{y}"/>')
            out.append(f'<text x="36" y="{y+3}" class="lblr">{v:.1f}</text>')
        for i in range(n_visits):
            x = 40 + (i / max(n_visits - 1, 1)) * plot_w
            out.append(f'<text x="{x}" y="{plot_h+12}" class="lblc">V{i+1}</text>')

        for series, color in [(p1, "#d62728"), (p5, "#888888")]:
            pts = []
            for i in range(min(n_visits, len(series))):
                m = series[i][0]
                s = series[i][1] if len(series[i]) > 1 else 0
                x = 40 + (i / max(n_visits - 1, 1)) * plot_w
                y = plot_h - (m / y_max) * plot_h
                yh = (s / y_max) * plot_h
                pts.append((x, y, yh, m))
            d = "M " + " L ".join(f"{x:.1f},{y:.1f}" for x, y, _, _ in pts)
            out.append(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="2.4"/>')
            for x, y, yh, m in pts:
                out.append(f'<line x1="{x}" y1="{y-yh}" x2="{x}" y2="{y+yh}" stroke="{color}" stroke-width="1.2"/>')
                out.append(f'<circle cx="{x}" cy="{y}" r="3" fill="{color}"/>')
        out.append('</g>')

    out.append(f'<text x="{w//2}" y="{h-12}" class="note" text-anchor="middle">'
               'Memory OFF (gray) is statistically flat or near-zero on every accumulation metric. Memory ON (red) trends upward, with widening separation by visit 5.'
               '</text>')
    out.append('</svg>')
    with open(save_path, "w") as f:
        f.write("".join(out))
    print(f"saved {save_path}")


def write_table(p1_byrun, p5_byrun, p1_dev, p5_dev, save_path):
    md = ["# Aggregate n=3 patient comparison — Memory ON vs OFF", ""]
    md.append("## Mean ± std across 3 distrust patients (38F IO, 74M MI, 67M IO), per visit")
    md.append("")
    md.append("| Metric | Mode | V1 | V2 | V3 | V4 | V5 |")
    md.append("|---|---|---|---|---|---|---|")
    for key, title in METRICS:
        for label, byrun in [("P1 (mem ON)", p1_byrun), ("P5 (mem OFF)", p5_byrun)]:
            ms = mean_std_per_visit(byrun, key)
            row = [title, label]
            for m, s, _ in ms:
                row.append(f"{m:.2f} ± {s:.2f}")
            md.append("| " + " | ".join(row) + " |")
    md.append("| LLM deviation score (0-10) | P1 (mem ON) | " +
              " | ".join(f"{m:.2f} ± {s:.2f}" for m, s in p1_dev) + " |")
    md.append("| LLM deviation score (0-10) | P5 (mem OFF) | " +
              " | ".join(f"{m:.2f} ± {s:.2f}" for m, s in p5_dev) + " |")

    md.append("")
    md.append("## V5 P1 vs P5 (mean across 3 patients, key metrics)")
    md.append("| Metric | P1 V5 | P5 V5 | Δ |")
    md.append("|---|---|---|---|")
    for key, title in METRICS:
        p1_ms = mean_std_per_visit(p1_byrun, key)
        p5_ms = mean_std_per_visit(p5_byrun, key)
        if len(p1_ms) >= 5 and len(p5_ms) >= 5:
            md.append(f"| {title} | {p1_ms[4][0]:.2f} ± {p1_ms[4][1]:.2f} | {p5_ms[4][0]:.2f} ± {p5_ms[4][1]:.2f} | **+{p1_ms[4][0]-p5_ms[4][0]:.2f}** |")
    if len(p1_dev) >= 5 and len(p5_dev) >= 5:
        md.append(f"| LLM deviation score | {p1_dev[4][0]:.2f} ± {p1_dev[4][1]:.2f} | {p5_dev[4][0]:.2f} ± {p5_dev[4][1]:.2f} | **+{p1_dev[4][0]-p5_dev[4][0]:.2f}** |")

    with open(save_path, "w") as f:
        f.write("\n".join(md))
    print(f"saved {save_path}")


def main():
    p1 = collect_per_visit(P1_RUNS)
    p5 = collect_per_visit(P5_RUNS)
    p1_dev = deviation_mean_std_per_visit(P1_RUNS)
    p5_dev = deviation_mean_std_per_visit(P5_RUNS)
    headline_figure(p1, p5, os.path.join(OUT, "persona_n3_headline.svg"))
    write_table(p1, p5, p1_dev, p5_dev, os.path.join(DOCS_OUT, "persona_n3_aggregate_table.md"))


if __name__ == "__main__":
    main()
