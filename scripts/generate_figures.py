"""Generate presentation figures from memory simulation results.

Reads each experiment directory under src/results/memory/ that contains
reviews.jsonl + memory.json and produces:
    figures/scores_by_mode.png         - average total score per mode (bar)
    figures/score_progression.png      - score by scenario index, line per mode
    figures/category_scores.png        - 5-category breakdown per mode (grouped bars)
    figures/memory_growth.png          - memory chars / consultations over time
    figures/score_table.md             - markdown summary table

Usage:
    python scripts/generate_figures.py \
        --results-root src/results/memory \
        --output-dir figures \
        --modes expA_baseline,expB_same_type,expC_cross_type
"""
import os
import sys
import json
import glob
import argparse
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


CATEGORIES = [
    "history_taking",
    "ddx_reasoning",
    "clinical_communication",
    "safety_risk_management",
    "diagnosis_management_plan",
]
CATEGORY_LABELS = [
    "History\nTaking",
    "DDx\nReasoning",
    "Clinical\nComm.",
    "Safety &\nRisk Mgmt",
    "Diagnosis &\nMgmt Plan",
]
MODE_COLOR = {
    "expA_baseline": "#888888",
    "expB_same_type": "#1f77b4",
    "expC_cross_type": "#d62728",
}
MODE_DISPLAY = {
    "expA_baseline": "A: No Memory (baseline)",
    "expB_same_type": "B: Same-type Memory (Intestinal Obstr.)",
    "expC_cross_type": "C: Cross-type Memory",
}


def load_jsonl(path):
    if not os.path.isfile(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def find_exp_dir(results_root, exp_name):
    """Resolve an experiment dir even when run_memory_simulation.py prepends a timestamp."""
    direct = os.path.join(results_root, exp_name)
    if os.path.isdir(direct):
        return direct
    candidates = sorted(glob.glob(os.path.join(results_root, f"*{exp_name}*")))
    return candidates[-1] if candidates else None


def load_experiment(exp_dir):
    reviews = load_jsonl(os.path.join(exp_dir, "reviews.jsonl"))
    mem_path = os.path.join(exp_dir, "memory.json")
    memory = json.load(open(mem_path, encoding="utf-8")) if os.path.isfile(mem_path) else None
    config_path = os.path.join(exp_dir, "config.json")
    config = json.load(open(config_path, encoding="utf-8")) if os.path.isfile(config_path) else {}
    return {"reviews": reviews, "memory": memory, "config": config, "dir": exp_dir}


def avg_total_per_mode(data):
    out = {}
    for mode, d in data.items():
        scores = [r["total_score"] for r in d["reviews"]]
        out[mode] = (np.mean(scores), np.std(scores), scores) if scores else (0, 0, [])
    return out


def avg_per_category(data):
    out = {}
    for mode, d in data.items():
        per_cat = {c: [] for c in CATEGORIES}
        for r in d["reviews"]:
            for c in CATEGORIES:
                per_cat[c].append(r["scores"][c])
        out[mode] = {c: (np.mean(v) if v else 0) for c, v in per_cat.items()}
    return out


def plot_scores_by_mode(data, save_path):
    summary = avg_total_per_mode(data)
    modes = list(data.keys())
    means = [summary[m][0] for m in modes]
    stds = [summary[m][1] for m in modes]
    colors = [MODE_COLOR.get(m, "#999") for m in modes]
    labels = [MODE_DISPLAY.get(m, m) for m in modes]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(range(len(modes)), means, yerr=stds, color=colors, capsize=8, edgecolor="black", linewidth=0.8)
    ax.set_xticks(range(len(modes)))
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("Total score (out of 25)")
    ax.set_ylim(0, 25)
    ax.set_title("Doctor consultation quality by experimental mode\n(higher is better; mean ± std across scenarios)")
    ax.axhline(12, color="grey", linestyle=":", alpha=0.5, label="Initial Experience baseline (proposal Table 1: 12/25)")
    ax.legend(loc="lower right", fontsize=9)
    for bar, mean in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, mean + 0.5, f"{mean:.1f}", ha="center", fontsize=10, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_score_progression(data, save_path):
    fig, ax = plt.subplots(figsize=(9, 5))
    for mode, d in data.items():
        scores = [r["total_score"] for r in d["reviews"]]
        x = list(range(1, len(scores) + 1))
        ax.plot(x, scores, marker="o", linewidth=2, color=MODE_COLOR.get(mode, "#999"), label=MODE_DISPLAY.get(mode, mode))
    ax.set_xlabel("Consultation index (chronological order)")
    ax.set_ylabel("Total score (out of 25)")
    ax.set_ylim(0, 25)
    ax.set_title("Per-consultation score progression\n(does memory accumulation lift later consultations?)")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", fontsize=9)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_category_scores(data, save_path):
    cat_avgs = avg_per_category(data)
    modes = list(data.keys())
    n_cat = len(CATEGORIES)
    width = 0.8 / max(len(modes), 1)
    x = np.arange(n_cat)

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, mode in enumerate(modes):
        means = [cat_avgs[mode][c] for c in CATEGORIES]
        offset = (i - (len(modes) - 1) / 2) * width
        bars = ax.bar(x + offset, means, width, color=MODE_COLOR.get(mode, "#999"), label=MODE_DISPLAY.get(mode, mode), edgecolor="black", linewidth=0.5)
        for bar, m in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width() / 2, m + 0.05, f"{m:.1f}", ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(CATEGORY_LABELS, fontsize=9)
    ax.set_ylabel("Avg score (out of 5)")
    ax.set_ylim(0, 5.5)
    ax.set_title("Category-level breakdown across modes")
    ax.grid(alpha=0.3, axis="y")
    ax.legend(loc="lower right", fontsize=9)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_memory_growth(data, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    # Left: memory chars over consultation index for memory-bearing runs
    for mode, d in data.items():
        if mode == "expA_baseline":
            continue
        # Reconstruct cumulative chars by reading dialog log's pre-state then adding the review record contribution
        dialog_path = os.path.join(d["dir"], "dialogues.jsonl")
        if not os.path.isfile(dialog_path):
            continue
        dialogs = load_jsonl(dialog_path)
        # use reported memory_entries_before + estimate via final mem
        x = []
        y_chars = []
        # We'll approximate: chars after consultation i = render(distilled+recent_at_that_time)
        # We don't have intermediate snapshots, so plot only known points: 0 (start) and len(dialogs) (end)
        x = [0, len(dialogs)]
        y_chars = [0, 0]
        if d["memory"]:
            distilled = d["memory"].get("distilled", "") or ""
            recent_chars = sum(len(json.dumps(r, ensure_ascii=False)) for r in d["memory"].get("recent", []))
            y_chars[-1] = len(distilled) + recent_chars
        axes[0].plot(x, y_chars, marker="o", color=MODE_COLOR.get(mode, "#999"), label=MODE_DISPLAY.get(mode, mode))
    axes[0].set_xlabel("Consultations")
    axes[0].set_ylabel("Memory size (chars)")
    axes[0].set_title("Memory artifact growth")
    axes[0].grid(alpha=0.3)
    axes[0].legend(fontsize=9)

    # Right: distilled vs recent split (final state)
    modes_with_mem = [m for m, d in data.items() if d["memory"]]
    if modes_with_mem:
        bar_width = 0.5
        positions = np.arange(len(modes_with_mem))
        distilled = [len(data[m]["memory"].get("distilled", "") or "") for m in modes_with_mem]
        recent = [sum(len(json.dumps(r, ensure_ascii=False)) for r in data[m]["memory"].get("recent", [])) for m in modes_with_mem]
        axes[1].bar(positions, distilled, bar_width, color="#2ca02c", label="Distilled (rolling summary)", edgecolor="black", linewidth=0.5)
        axes[1].bar(positions, recent, bar_width, bottom=distilled, color="#ff7f0e", label="Recent N records", edgecolor="black", linewidth=0.5)
        axes[1].set_xticks(positions)
        axes[1].set_xticklabels([MODE_DISPLAY.get(m, m) for m in modes_with_mem], fontsize=9, rotation=10, ha="right")
        axes[1].set_ylabel("Chars")
        axes[1].set_title("Final memory composition")
        axes[1].legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def write_score_table(data, save_path):
    summary = avg_total_per_mode(data)
    cat_avgs = avg_per_category(data)
    lines = ["# Experimental Result Summary", ""]
    lines.append("## Overall Total Scores (out of 25)\n")
    lines.append("| Mode | Mean | Std | Per-scenario |")
    lines.append("|---|---|---|---|")
    for mode in data.keys():
        mean, std, scores = summary[mode]
        per = ", ".join(f"{s}" for s in scores)
        lines.append(f"| {MODE_DISPLAY.get(mode, mode)} | {mean:.2f} | {std:.2f} | {per} |")
    lines.append("")
    lines.append("## Category Averages (out of 5)\n")
    header = "| Mode | " + " | ".join(c.replace("_", " ") for c in CATEGORIES) + " |"
    lines.append(header)
    lines.append("|" + "---|" * (len(CATEGORIES) + 1))
    for mode in data.keys():
        row = [MODE_DISPLAY.get(mode, mode)]
        for c in CATEGORIES:
            row.append(f"{cat_avgs[mode][c]:.2f}")
        lines.append("| " + " | ".join(row) + " |")
    with open(save_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-root", default="src/results/memory")
    ap.add_argument("--output-dir", default="figures")
    ap.add_argument("--modes", default="expA_baseline,expB_same_type,expC_cross_type")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    mode_names = [m.strip() for m in args.modes.split(",") if m.strip()]
    data = {}
    for m in mode_names:
        d = find_exp_dir(args.results_root, m)
        if not d:
            print(f"[WARN] No directory found for {m} under {args.results_root}", file=sys.stderr)
            continue
        loaded = load_experiment(d)
        if not loaded["reviews"]:
            print(f"[WARN] No reviews.jsonl yet in {d}; skipping.", file=sys.stderr)
            continue
        data[m] = loaded
        print(f"Loaded {m}: {len(loaded['reviews'])} reviews from {d}")

    if not data:
        print("No completed experiments found. Re-run after experiments finish.")
        return

    plot_scores_by_mode(data, os.path.join(args.output_dir, "scores_by_mode.png"))
    plot_score_progression(data, os.path.join(args.output_dir, "score_progression.png"))
    plot_category_scores(data, os.path.join(args.output_dir, "category_scores.png"))
    plot_memory_growth(data, os.path.join(args.output_dir, "memory_growth.png"))
    write_score_table(data, os.path.join(args.output_dir, "score_table.md"))
    print(f"Figures and table written to {args.output_dir}/")


if __name__ == "__main__":
    main()
