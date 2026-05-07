"""Build persona-evolution trajectory figures from results/persona/* runs.

Outputs (SVG so it stays toolchain-independent):
  figures/persona_trait_trajectories.svg     — line chart of 8 traits across visits per experiment
  figures/persona_doctor_experience.svg      — listening/validation/empathy/etc per visit per experiment
  figures/persona_evolution_text_growth.svg  — chars-of-evolution-text per visit per experiment
  figures/persona_p1_vs_p5.svg               — direct memory-on vs memory-off contrast (P1 vs P5)

Usage:
  python scripts/persona_trajectory_figures.py
"""
import os
import sys
import json
import glob


REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(REPO, "src", "results", "persona")
OUT = os.path.join(REPO, "figures")
os.makedirs(OUT, exist_ok=True)


TRAITS = ["distrust","pleasing","impatience","overanxiousness","verbosity","guardedness","resignation","directness"]
EXP_AXES = ["listening","validation","empathy","respect","information_sharing"]

EXP_COLORS = {
    "P1_distrust_dismissive": "#d62728",
    "P2_distrust_empathetic": "#2ca02c",
    "P3_prefix_then_repair":  "#ff7f0e",
    "P5_memory_off_distrust_dismissive": "#888888",
    "smoke_distrust_dismissive": "#aaaaaa",
}

EXP_LABEL = {
    "P1_distrust_dismissive": "P1: distrust × dismissive (mem ON)",
    "P2_distrust_empathetic": "P2: distrust × empathetic (mem ON)",
    "P3_prefix_then_repair":  "P3: distrust × bad→good (mem ON)",
    "P5_memory_off_distrust_dismissive": "P5: distrust × dismissive (mem OFF, control)",
}


def load_summary(exp_name):
    p = os.path.join(RESULTS, exp_name, "summary.json")
    if not os.path.isfile(p):
        return None
    return json.load(open(p, encoding="utf-8"))


def list_experiments():
    out = []
    for d in sorted(glob.glob(os.path.join(RESULTS, "*"))):
        if not os.path.isdir(d):
            continue
        if os.path.isfile(os.path.join(d, "summary.json")):
            out.append(os.path.basename(d))
    return out


def svg_open(w, h, title, subtitle=""):
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}" '
        'font-family="Helvetica, Arial, sans-serif">\n'
        '<style>'
        '.title{font-size:16px;font-weight:700;text-anchor:middle}'
        '.sub{font-size:11px;text-anchor:middle;fill:#444}'
        '.axis{stroke:#333;stroke-width:1.2}'
        '.grid{stroke:#bbb;stroke-width:0.5;stroke-dasharray:3,3}'
        '.lblr{font-size:11px;text-anchor:end}'
        '.lblc{font-size:11px;text-anchor:middle}'
        '.legend{font-size:11px}'
        '.val{font-size:10px;font-weight:600;text-anchor:middle}'
        '</style>\n'
        f'<text x="{w//2}" y="22" class="title">{title}</text>\n'
        f'<text x="{w//2}" y="40" class="sub">{subtitle}</text>\n'
    )


def svg_close():
    return "</svg>\n"


def trajectory_chart(exp_name, summary, save_path, traits=TRAITS, y_max=10):
    visits = [v["visit"] for v in summary["visits"]]
    n = len(visits)
    if n == 0:
        return False
    series = {t: [] for t in traits}
    for v in summary["visits"]:
        sc = v.get("expression_scores") or {}
        for t in traits:
            series[t].append(sc.get(t, 0) if isinstance(sc.get(t), int) else 0)

    w, h = 880, 420
    pad_l, pad_r, pad_t, pad_b = 70, 240, 60, 70
    plot_w = w - pad_l - pad_r
    plot_h = h - pad_t - pad_b

    out = [svg_open(w, h, f"Persona trait trajectories — {EXP_LABEL.get(exp_name, exp_name)}",
                    f"Patient: {summary['patient']['age']}{summary['patient']['gender']} "
                    f"({summary['patient']['personality']}) · {n} visits · doctor styles per visit")]
    out.append(f'<line class="axis" x1="{pad_l}" y1="{pad_t}" x2="{pad_l}" y2="{pad_t+plot_h}"/>')
    out.append(f'<line class="axis" x1="{pad_l}" y1="{pad_t+plot_h}" x2="{w-pad_r}" y2="{pad_t+plot_h}"/>')
    for v in range(0, y_max + 1, 2):
        y = pad_t + plot_h - (v / y_max) * plot_h
        out.append(f'<line class="grid" x1="{pad_l}" y1="{y}" x2="{w-pad_r}" y2="{y}"/>')
        out.append(f'<text class="lblr" x="{pad_l-8}" y="{y+4}">{v}</text>')

    for i, vn in enumerate(visits):
        x = pad_l + (i / max(n - 1, 1)) * plot_w if n > 1 else pad_l + plot_w / 2
        out.append(f'<text class="lblc" x="{x}" y="{pad_t+plot_h+18}">V{vn}</text>')
        # doctor style
        ds = summary["visits"][i].get("doctor_style", "?")
        out.append(f'<text class="lblc" x="{x}" y="{pad_t+plot_h+34}" font-size="9" fill="#555">{ds}</text>')

    out.append(f'<text class="lblc" x="{pad_l-50}" y="{pad_t+plot_h/2}" transform="rotate(-90 {pad_l-50} {pad_t+plot_h/2})">Trait expression (0-10)</text>')

    palette = ["#d62728","#1f77b4","#ff7f0e","#9467bd","#2ca02c","#bcbd22","#8c564b","#17becf"]
    for ti, t in enumerate(traits):
        color = palette[ti % len(palette)]
        pts = []
        for i, v in enumerate(series[t]):
            x = pad_l + (i / max(n - 1, 1)) * plot_w if n > 1 else pad_l + plot_w / 2
            y = pad_t + plot_h - (v / y_max) * plot_h
            pts.append((x, y))
        d = "M " + " L ".join(f"{x:.1f},{y:.1f}" for x, y in pts)
        out.append(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="2" opacity="0.9"/>')
        for (x, y) in pts:
            out.append(f'<circle cx="{x}" cy="{y}" r="3" fill="{color}"/>')
        # legend
        ly = pad_t + 10 + ti * 18
        out.append(f'<line x1="{w-pad_r+10}" y1="{ly}" x2="{w-pad_r+30}" y2="{ly}" stroke="{color}" stroke-width="2.5"/>')
        out.append(f'<circle cx="{w-pad_r+20}" cy="{ly}" r="3" fill="{color}"/>')
        out.append(f'<text class="legend" x="{w-pad_r+36}" y="{ly+4}">{t}</text>')

    out.append(svg_close())
    with open(save_path, "w", encoding="utf-8") as f:
        f.write("".join(out))
    print(f"saved {save_path}")
    return True


def evolution_text_growth_chart(experiments, save_path):
    """For each experiment, plot len(persona_evolution_after) per visit."""
    valid = [e for e in experiments if e[1]]
    if not valid:
        return False
    w, h = 800, 420
    pad_l, pad_r, pad_t, pad_b = 70, 220, 60, 60
    plot_w = w - pad_l - pad_r
    plot_h = h - pad_t - pad_b

    # find max chars across experiments
    all_lens = []
    series_by_exp = {}
    max_visits = 0
    for exp_name, summary in valid:
        path = os.path.join(RESULTS, exp_name, "persona_state.jsonl")
        if not os.path.isfile(path):
            continue
        chars = []
        with open(path) as f:
            for line in f:
                if not line.strip():
                    continue
                rec = json.loads(line)
                chars.append(len(rec.get("persona_evolution_after", "") or ""))
        series_by_exp[exp_name] = chars
        all_lens += chars
        max_visits = max(max_visits, len(chars))
    if not all_lens:
        return False
    y_max = max(max(all_lens), 100)
    y_max = ((y_max // 200) + 1) * 200

    out = [svg_open(w, h, "Evolution text growth across visits",
                    "Per-visit length (chars) of accumulated persona-evolution text. memory_off control should stay flat at 0.")]
    out.append(f'<line class="axis" x1="{pad_l}" y1="{pad_t}" x2="{pad_l}" y2="{pad_t+plot_h}"/>')
    out.append(f'<line class="axis" x1="{pad_l}" y1="{pad_t+plot_h}" x2="{w-pad_r}" y2="{pad_t+plot_h}"/>')
    for step in range(0, y_max + 1, max(y_max // 5, 1)):
        y = pad_t + plot_h - (step / y_max) * plot_h
        out.append(f'<line class="grid" x1="{pad_l}" y1="{y}" x2="{w-pad_r}" y2="{y}"/>')
        out.append(f'<text class="lblr" x="{pad_l-8}" y="{y+4}">{step}</text>')
    for v in range(1, max_visits + 1):
        x = pad_l + ((v - 1) / max(max_visits - 1, 1)) * plot_w if max_visits > 1 else pad_l + plot_w / 2
        out.append(f'<text class="lblc" x="{x}" y="{pad_t+plot_h+18}">V{v}</text>')
    out.append(f'<text class="lblc" x="{pad_l-50}" y="{pad_t+plot_h/2}" transform="rotate(-90 {pad_l-50} {pad_t+plot_h/2})">evolution text length (chars)</text>')

    legend_y = pad_t
    for exp_name, chars in series_by_exp.items():
        color = EXP_COLORS.get(exp_name, "#555")
        pts = []
        for i, c in enumerate(chars):
            x = pad_l + (i / max(max_visits - 1, 1)) * plot_w if max_visits > 1 else pad_l + plot_w / 2
            y = pad_t + plot_h - (c / y_max) * plot_h
            pts.append((x, y))
        d = "M " + " L ".join(f"{x:.1f},{y:.1f}" for x, y in pts)
        out.append(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="2.5"/>')
        for (x, y) in pts:
            out.append(f'<circle cx="{x}" cy="{y}" r="4" fill="{color}"/>')
        out.append(f'<line x1="{w-pad_r+10}" y1="{legend_y}" x2="{w-pad_r+30}" y2="{legend_y}" stroke="{color}" stroke-width="2.5"/>')
        out.append(f'<text class="legend" x="{w-pad_r+36}" y="{legend_y+4}">{EXP_LABEL.get(exp_name, exp_name)}</text>')
        legend_y += 20

    out.append(svg_close())
    with open(save_path, "w", encoding="utf-8") as f:
        f.write("".join(out))
    print(f"saved {save_path}")
    return True


def main():
    experiments = []
    for name in list_experiments():
        s = load_summary(name)
        if s:
            experiments.append((name, s))

    print(f"Found {len(experiments)} experiments: {[e[0] for e in experiments]}")

    # Per-experiment trait trajectory
    for name, summary in experiments:
        trajectory_chart(name, summary, os.path.join(OUT, f"persona_trajectory_{name}.svg"))

    # Evolution-text growth comparison
    evolution_text_growth_chart(experiments, os.path.join(OUT, "persona_evolution_text_growth.svg"))


if __name__ == "__main__":
    main()
