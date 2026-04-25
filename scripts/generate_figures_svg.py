"""SVG-only fallback figure generator (no matplotlib dependency).

Reads reviews.jsonl + memory.json from each mode directory and emits SVG
charts directly via Python string templates. Useful when the matplotlib
toolchain or its bash invocation is unavailable.

Usage:
    python scripts/generate_figures_svg.py
"""
import os
import json
import glob
import sys

CATEGORIES = [
    "history_taking",
    "ddx_reasoning",
    "clinical_communication",
    "safety_risk_management",
    "diagnosis_management_plan",
]
CAT_LABEL = ["Hist", "DDx", "Comm", "Safety", "Plan"]
MODE_COLOR = {
    "expA_baseline": "#888888",
    "expB_same_type": "#1f77b4",
    "expC_cross_type": "#d62728",
}
MODE_DISPLAY = {
    "expA_baseline": "A: No Memory",
    "expB_same_type": "B: Same-type Memory",
    "expC_cross_type": "C: Cross-type Memory",
}


def find_dir(root, name):
    direct = os.path.join(root, name)
    if os.path.isdir(direct):
        return direct
    cands = sorted(glob.glob(os.path.join(root, f"*{name}*")))
    return cands[-1] if cands else None


def load_reviews(d):
    p = os.path.join(d, "reviews.jsonl")
    if not os.path.isfile(p):
        return []
    with open(p, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def load_memory(d):
    p = os.path.join(d, "memory.json")
    return json.load(open(p, encoding="utf-8")) if os.path.isfile(p) else None


def svg_open(w, h, title):
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}" '
        'font-family="Helvetica, Arial, sans-serif">\n'
        '  <style>\n'
        '    .title { font-size:16px; font-weight:700; text-anchor:middle; }\n'
        '    .axis  { stroke:#333; stroke-width:1.2; }\n'
        '    .gridl { stroke:#bbb; stroke-width:0.5; stroke-dasharray:3,3; }\n'
        '    .lbl   { font-size:11px; }\n'
        '    .lblc  { font-size:11px; text-anchor:middle; }\n'
        '    .lblr  { font-size:11px; text-anchor:end; }\n'
        '    .val   { font-size:11px; text-anchor:middle; font-weight:600; }\n'
        '    .legend{ font-size:11px; }\n'
        '  </style>\n'
        f'  <text x="{w//2}" y="22" class="title">{title}</text>\n'
    )


def svg_close():
    return "</svg>\n"


def bar_chart_scores_by_mode(data, save_path):
    w, h = 700, 400
    pad_l, pad_r, pad_t, pad_b = 70, 30, 50, 80
    plot_w = w - pad_l - pad_r
    plot_h = h - pad_t - pad_b
    max_y = 25
    n_modes = len(data)
    if n_modes == 0:
        return
    bar_w = plot_w / (n_modes * 2 + 1)

    out = [svg_open(w, h, "Total score by mode (out of 25)")]
    # axes
    out.append(f'  <line class="axis" x1="{pad_l}" y1="{pad_t}" x2="{pad_l}" y2="{pad_t+plot_h}"/>\n')
    out.append(f'  <line class="axis" x1="{pad_l}" y1="{pad_t+plot_h}" x2="{w-pad_r}" y2="{pad_t+plot_h}"/>\n')
    # gridlines + y labels
    for v in range(0, 26, 5):
        y = pad_t + plot_h - (v / max_y) * plot_h
        out.append(f'  <line class="gridl" x1="{pad_l}" y1="{y}" x2="{w-pad_r}" y2="{y}"/>\n')
        out.append(f'  <text class="lblr" x="{pad_l-8}" y="{y+4}">{v}</text>\n')
    # bars
    for i, (mode, recs) in enumerate(data.items()):
        if not recs:
            continue
        scores = [r["total_score"] for r in recs]
        mean = sum(scores) / len(scores)
        x = pad_l + bar_w * (i * 2 + 1)
        bar_height = (mean / max_y) * plot_h
        y_top = pad_t + plot_h - bar_height
        color = MODE_COLOR.get(mode, "#999")
        out.append(f'  <rect x="{x}" y="{y_top}" width="{bar_w}" height="{bar_height}" fill="{color}" stroke="#222" stroke-width="0.8"/>\n')
        # value label on top
        out.append(f'  <text class="val" x="{x+bar_w/2}" y="{y_top-6}">{mean:.1f}</text>\n')
        # x label
        cx = x + bar_w / 2
        out.append(f'  <text class="lblc" x="{cx}" y="{pad_t+plot_h+18}">{MODE_DISPLAY.get(mode, mode)}</text>\n')
        # show scores in small text below mode label
        score_str = ", ".join(str(s) for s in scores)
        out.append(f'  <text class="lblc" x="{cx}" y="{pad_t+plot_h+34}" font-size="9" fill="#555">[{score_str}]</text>\n')
        # n
        out.append(f'  <text class="lblc" x="{cx}" y="{pad_t+plot_h+50}" font-size="9" fill="#555">n={len(scores)}</text>\n')

    out.append(f'  <text class="lblc" x="{pad_l-50}" y="{pad_t+plot_h/2}" transform="rotate(-90 {pad_l-50} {pad_t+plot_h/2})">Total score (out of 25)</text>\n')
    out.append(svg_close())
    with open(save_path, "w", encoding="utf-8") as f:
        f.write("".join(out))
    print(f"saved {save_path}")


def line_chart_progression(data, save_path):
    w, h = 800, 420
    pad_l, pad_r, pad_t, pad_b = 70, 240, 50, 60
    plot_w = w - pad_l - pad_r
    plot_h = h - pad_t - pad_b
    max_x = max((len(recs) for recs in data.values()), default=1)
    max_x = max(max_x, 5)
    max_y = 25

    out = [svg_open(w, h, "Per-consultation score progression")]
    out.append(f'  <line class="axis" x1="{pad_l}" y1="{pad_t}" x2="{pad_l}" y2="{pad_t+plot_h}"/>\n')
    out.append(f'  <line class="axis" x1="{pad_l}" y1="{pad_t+plot_h}" x2="{w-pad_r}" y2="{pad_t+plot_h}"/>\n')
    for v in range(0, 26, 5):
        y = pad_t + plot_h - (v / max_y) * plot_h
        out.append(f'  <line class="gridl" x1="{pad_l}" y1="{y}" x2="{w-pad_r}" y2="{y}"/>\n')
        out.append(f'  <text class="lblr" x="{pad_l-8}" y="{y+4}">{v}</text>\n')
    # x axis labels
    for v in range(1, max_x + 1):
        x = pad_l + ((v - 1) / max(max_x - 1, 1)) * plot_w if max_x > 1 else pad_l + plot_w / 2
        out.append(f'  <text class="lblc" x="{x}" y="{pad_t+plot_h+18}">{v}</text>\n')
    out.append(f'  <text class="lblc" x="{pad_l+plot_w/2}" y="{pad_t+plot_h+40}">Consultation index</text>\n')
    out.append(f'  <text class="lblc" x="{pad_l-50}" y="{pad_t+plot_h/2}" transform="rotate(-90 {pad_l-50} {pad_t+plot_h/2})">Total score (out of 25)</text>\n')

    # legend
    leg_x = w - pad_r + 10
    leg_y = pad_t + 10
    for i, (mode, recs) in enumerate(data.items()):
        if not recs:
            continue
        color = MODE_COLOR.get(mode, "#999")
        ly = leg_y + i * 22
        out.append(f'  <line x1="{leg_x}" y1="{ly}" x2="{leg_x+18}" y2="{ly}" stroke="{color}" stroke-width="2.5"/>\n')
        out.append(f'  <circle cx="{leg_x+9}" cy="{ly}" r="3" fill="{color}"/>\n')
        out.append(f'  <text class="legend" x="{leg_x+24}" y="{ly+4}">{MODE_DISPLAY.get(mode, mode)}</text>\n')

    # lines
    for mode, recs in data.items():
        if not recs:
            continue
        color = MODE_COLOR.get(mode, "#999")
        pts = []
        for i, r in enumerate(recs):
            x = pad_l + (i / max(max_x - 1, 1)) * plot_w if max_x > 1 else pad_l + plot_w / 2
            y = pad_t + plot_h - (r["total_score"] / max_y) * plot_h
            pts.append((x, y, r["total_score"]))
        # path
        d = "M " + " L ".join(f"{x:.1f},{y:.1f}" for x, y, _ in pts)
        out.append(f'  <path d="{d}" fill="none" stroke="{color}" stroke-width="2"/>\n')
        for x, y, sc in pts:
            out.append(f'  <circle cx="{x}" cy="{y}" r="4" fill="{color}" stroke="#222" stroke-width="0.7"/>\n')
            out.append(f'  <text class="val" x="{x}" y="{y-9}" fill="{color}">{sc}</text>\n')

    out.append(svg_close())
    with open(save_path, "w", encoding="utf-8") as f:
        f.write("".join(out))
    print(f"saved {save_path}")


def grouped_bar_categories(data, save_path):
    w, h = 800, 420
    pad_l, pad_r, pad_t, pad_b = 70, 200, 50, 50
    plot_w = w - pad_l - pad_r
    plot_h = h - pad_t - pad_b
    n_cats = len(CATEGORIES)
    n_modes = len(data)
    if n_modes == 0:
        return
    group_w = plot_w / n_cats
    bar_w = group_w / (n_modes + 1)
    max_y = 5

    out = [svg_open(w, h, "Category-level score breakdown (avg, out of 5)")]
    out.append(f'  <line class="axis" x1="{pad_l}" y1="{pad_t}" x2="{pad_l}" y2="{pad_t+plot_h}"/>\n')
    out.append(f'  <line class="axis" x1="{pad_l}" y1="{pad_t+plot_h}" x2="{w-pad_r}" y2="{pad_t+plot_h}"/>\n')
    for v in range(0, 6):
        y = pad_t + plot_h - (v / max_y) * plot_h
        out.append(f'  <line class="gridl" x1="{pad_l}" y1="{y}" x2="{w-pad_r}" y2="{y}"/>\n')
        out.append(f'  <text class="lblr" x="{pad_l-8}" y="{y+4}">{v}</text>\n')

    # legend
    leg_x = w - pad_r + 10
    for i, (mode, _) in enumerate(data.items()):
        ly = pad_t + 10 + i * 22
        color = MODE_COLOR.get(mode, "#999")
        out.append(f'  <rect x="{leg_x}" y="{ly-9}" width="14" height="14" fill="{color}" stroke="#222"/>\n')
        out.append(f'  <text class="legend" x="{leg_x+22}" y="{ly+2}">{MODE_DISPLAY.get(mode, mode)}</text>\n')

    # bars
    for ci, cat in enumerate(CATEGORIES):
        gx = pad_l + ci * group_w
        # category label
        out.append(f'  <text class="lblc" x="{gx + group_w/2}" y="{pad_t+plot_h+18}">{CAT_LABEL[ci]}</text>\n')
        for mi, (mode, recs) in enumerate(data.items()):
            if not recs:
                continue
            mean = sum(r["scores"][cat] for r in recs) / len(recs)
            x = gx + (mi + 0.5) * bar_w + group_w * 0.05
            bh = (mean / max_y) * plot_h
            yt = pad_t + plot_h - bh
            color = MODE_COLOR.get(mode, "#999")
            out.append(f'  <rect x="{x}" y="{yt}" width="{bar_w*0.9}" height="{bh}" fill="{color}" stroke="#222" stroke-width="0.6"/>\n')
            out.append(f'  <text class="val" x="{x+bar_w*0.45}" y="{yt-3}" font-size="9">{mean:.1f}</text>\n')

    out.append(svg_close())
    with open(save_path, "w", encoding="utf-8") as f:
        f.write("".join(out))
    print(f"saved {save_path}")


def write_summary_md(data, memories, save_path):
    lines = ["# Auto-generated Result Summary", ""]
    lines.append("## Overall scores")
    lines.append("| Mode | n | Mean | Per-scenario |")
    lines.append("|---|---|---|---|")
    for mode, recs in data.items():
        if not recs:
            lines.append(f"| {MODE_DISPLAY.get(mode, mode)} | 0 | — | (no data) |")
            continue
        scores = [r["total_score"] for r in recs]
        lines.append(f"| {MODE_DISPLAY.get(mode, mode)} | {len(scores)} | {sum(scores)/len(scores):.2f} | {scores} |")

    lines.append("")
    lines.append("## Category averages (out of 5)")
    header = "| Mode |" + "|".join(f" {c} " for c in CATEGORIES) + "|"
    sep = "|---|" + "|".join("---" for _ in CATEGORIES) + "|"
    lines.append(header)
    lines.append(sep)
    for mode, recs in data.items():
        if not recs:
            row = [MODE_DISPLAY.get(mode, mode)] + ["—"] * len(CATEGORIES)
            lines.append("| " + " | ".join(row) + " |")
            continue
        row = [MODE_DISPLAY.get(mode, mode)]
        for c in CATEGORIES:
            avg = sum(r["scores"][c] for r in recs) / len(recs)
            row.append(f"{avg:.2f}")
        lines.append("| " + " | ".join(row) + " |")

    lines.append("")
    lines.append("## Memory state (final)")
    lines.append("| Mode | num_consultations | distilled chars | recent count |")
    lines.append("|---|---|---|---|")
    for mode, mem in memories.items():
        if not mem:
            lines.append(f"| {MODE_DISPLAY.get(mode, mode)} | — | — | — |")
            continue
        n = mem["meta"]["num_consultations"]
        dc = len(mem.get("distilled", "") or "")
        rc = len(mem.get("recent", []))
        lines.append(f"| {MODE_DISPLAY.get(mode, mode)} | {n} | {dc} | {rc} |")

    with open(save_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"saved {save_path}")


def main():
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    results_root = os.path.join(repo_root, "src", "results", "memory")
    out_dir = os.path.join(repo_root, "figures")
    os.makedirs(out_dir, exist_ok=True)

    modes = ["expA_baseline", "expB_same_type", "expC_cross_type"]
    data = {}
    memories = {}
    for m in modes:
        d = find_dir(results_root, m)
        if not d:
            print(f"[WARN] no dir for {m}", file=sys.stderr)
            data[m] = []
            memories[m] = None
            continue
        recs = load_reviews(d)
        mem = load_memory(d)
        data[m] = recs
        memories[m] = mem
        print(f"loaded {m}: {len(recs)} reviews, memory={'yes' if mem else 'no'}")

    bar_chart_scores_by_mode(data, os.path.join(out_dir, "scores_by_mode.svg"))
    line_chart_progression(data, os.path.join(out_dir, "score_progression.svg"))
    grouped_bar_categories(data, os.path.join(out_dir, "category_scores.svg"))
    write_summary_md(data, memories, os.path.join(out_dir, "score_table.md"))


if __name__ == "__main__":
    main()
