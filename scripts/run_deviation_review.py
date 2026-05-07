"""Apply PersonaDeviationReviewer (with ensemble) to all completed visits across
  results/persona/* and emit a per-experiment trajectory.

Outputs:
  results/persona/<exp>/deviation_scores.jsonl  - per visit, per ensemble seed
  figures/persona_deviation_trajectories.svg     - line chart per experiment
  docs/persona_deviation_table.md                - markdown table (mean +- std)
"""
import os
import sys
import json
import glob
import statistics


REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src"))

from agent.persona_deviation_reviewer import PersonaDeviationReviewer  # noqa: E402

RESULTS = os.path.join(REPO, "src", "results", "persona")
OUT = os.path.join(REPO, "figures")
DOCS_OUT = os.path.join(REPO, "docs")
os.makedirs(OUT, exist_ok=True)


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


def list_experiments():
    return sorted(os.path.basename(d) for d in glob.glob(os.path.join(RESULTS, "*"))
                  if os.path.isfile(os.path.join(d, "visits.jsonl")) and not os.path.basename(d).startswith("smoke"))


def load_visits(exp):
    p = os.path.join(RESULTS, exp, "visits.jsonl")
    with open(p, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def base_persona_summary_from_summary(exp):
    p = os.path.join(RESULTS, exp, "summary.json")
    if not os.path.isfile(p):
        return ""
    s = json.load(open(p, encoding="utf-8"))
    pat = s.get("patient", {})
    return (
        f"personality={pat.get('personality')}, cefr={pat.get('cefr')}, "
        f"recall={pat.get('recall_level')}, dazed={pat.get('dazed_level')}"
    )


def main():
    reviewer = PersonaDeviationReviewer()
    series = {}  # exp -> list of {"visit", "score_mean", "score_std", "direction"}

    for exp in list_experiments():
        base = base_persona_summary_from_summary(exp)
        out_path = os.path.join(RESULTS, exp, "deviation_scores.jsonl")
        # if already computed, reuse
        if os.path.isfile(out_path) and os.path.getsize(out_path) > 0:
            print(f"reusing existing {out_path}")
            recs = [json.loads(l) for l in open(out_path) if l.strip()]
            series[exp] = [
                {"visit": r["visit"],
                 "score_mean": r["score_mean"],
                 "score_std": r["score_std"],
                 "direction": r["direction_mode"]}
                for r in recs
            ]
            continue

        visits = load_visits(exp)
        recs = []
        for v in visits:
            try:
                ens = reviewer.review_ensemble(v["dialog_history"], base, n=3)
                rec = {"visit": v["visit"], **ens, "doctor_style": v.get("doctor_style")}
                recs.append(rec)
                with open(out_path, "a") as f:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                print(f"  {exp} V{v['visit']} score={ens['score_mean']:.1f}±{ens['score_std']} dir={ens['direction_mode']}")
            except Exception as e:
                print(f"  {exp} V{v['visit']} ERROR {e}")
        series[exp] = [
            {"visit": r["visit"],
             "score_mean": r["score_mean"],
             "score_std": r["score_std"],
             "direction": r["direction_mode"]}
            for r in recs
        ]

    # ---- Generate figure ----
    if not series:
        print("No data.")
        return
    n_visits = max(len(v) for v in series.values())
    w, h = 800, 460
    pad_l, pad_r, pad_t, pad_b = 70, 240, 60, 60
    plot_w = w - pad_l - pad_r
    plot_h = h - pad_t - pad_b
    y_max = 10
    out = []
    out.append(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}" '
               'font-family="Helvetica, Arial, sans-serif">')
    out.append('<style>'
               '.title{font-size:16px;font-weight:700;text-anchor:middle}'
               '.sub{font-size:11px;text-anchor:middle;fill:#444}'
               '.axis{stroke:#333;stroke-width:1.2}'
               '.grid{stroke:#bbb;stroke-width:0.5;stroke-dasharray:3,3}'
               '.lblr{font-size:11px;text-anchor:end}'
               '.lblc{font-size:11px;text-anchor:middle}'
               '.legend{font-size:11px}'
               '.val{font-size:10px;font-weight:600;text-anchor:middle}'
               '</style>')
    out.append(f'<text x="{w//2}" y="22" class="title">Persona deviation score per visit (LLM ensemble, n=3 seeds)</text>')
    out.append(f'<text x="{w//2}" y="40" class="sub">0 = first-visit-fresh; 10 = heavy accumulation. P5 (memory OFF) should stay near 0.</text>')
    out.append(f'<line class="axis" x1="{pad_l}" y1="{pad_t}" x2="{pad_l}" y2="{pad_t+plot_h}"/>')
    out.append(f'<line class="axis" x1="{pad_l}" y1="{pad_t+plot_h}" x2="{w-pad_r}" y2="{pad_t+plot_h}"/>')
    for v in range(0, y_max + 1, 2):
        y = pad_t + plot_h - (v / y_max) * plot_h
        out.append(f'<line class="grid" x1="{pad_l}" y1="{y}" x2="{w-pad_r}" y2="{y}"/>')
        out.append(f'<text class="lblr" x="{pad_l-8}" y="{y+4}">{v}</text>')
    for i in range(n_visits):
        x = pad_l + (i / max(n_visits - 1, 1)) * plot_w if n_visits > 1 else pad_l + plot_w / 2
        out.append(f'<text class="lblc" x="{x}" y="{pad_t+plot_h+18}">V{i+1}</text>')

    legend_y = pad_t
    for exp, points in series.items():
        color = EXP_COLORS.get(exp, "#444")
        pts = []
        for i, pt in enumerate(points):
            x = pad_l + (i / max(n_visits - 1, 1)) * plot_w if n_visits > 1 else pad_l + plot_w / 2
            y = pad_t + plot_h - (pt["score_mean"] / y_max) * plot_h
            pts.append((x, y, pt["score_std"], pt["score_mean"]))
        d = "M " + " L ".join(f"{x:.1f},{y:.1f}" for x, y, _, _ in pts)
        out.append(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="2.4"/>')
        for (x, y, std, mean) in pts:
            # std bar
            half = (std / y_max) * plot_h
            out.append(f'<line x1="{x}" y1="{y-half}" x2="{x}" y2="{y+half}" stroke="{color}" stroke-width="1.2"/>')
            out.append(f'<circle cx="{x}" cy="{y}" r="3.5" fill="{color}"/>')
            out.append(f'<text class="val" x="{x}" y="{y-9}">{mean:.1f}</text>')
        out.append(f'<line x1="{w-pad_r+10}" y1="{legend_y}" x2="{w-pad_r+30}" y2="{legend_y}" stroke="{color}" stroke-width="2.5"/>')
        out.append(f'<text class="legend" x="{w-pad_r+36}" y="{legend_y+4}">{EXP_LABEL.get(exp, exp)}</text>')
        legend_y += 22
    out.append(f'<text class="lblc" x="35" y="{pad_t+plot_h/2}" transform="rotate(-90 35 {pad_t+plot_h/2})">Deviation from base persona (0-10)</text>')
    out.append('</svg>')
    save_path = os.path.join(OUT, "persona_deviation_trajectories.svg")
    with open(save_path, "w", encoding="utf-8") as f:
        f.write("\n".join(out))
    print(f"saved {save_path}")

    # ---- Markdown table ----
    md = ["# Persona deviation scores (LLM ensemble, n=3 seeds, 0-10)", ""]
    md.append("| Exp | V1 | V2 | V3 | V4 | V5 |")
    md.append("|---|---|---|---|---|---|")
    for exp, points in series.items():
        cells = []
        for i in range(5):
            if i < len(points):
                p = points[i]
                cells.append(f"{p['score_mean']:.1f}±{p['score_std']:.1f}")
            else:
                cells.append("—")
        md.append(f"| {EXP_LABEL.get(exp, exp)} | " + " | ".join(cells) + " |")
    md_path = os.path.join(DOCS_OUT, "persona_deviation_table.md")
    with open(md_path, "w") as f:
        f.write("\n".join(md))
    print(f"saved {md_path}")


if __name__ == "__main__":
    main()
