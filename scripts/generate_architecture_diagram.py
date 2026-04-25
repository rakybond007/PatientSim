"""Generate the Medical Memory architecture diagram for the slide deck.

Produces:
    figures/architecture.png        - 3-component loop overview
    figures/memory_mechanism.png    - sliding window + rolling summary detail
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch


OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")
os.makedirs(OUT_DIR, exist_ok=True)


# ---------- 1. Top-level architecture ----------

def draw_arch():
    fig, ax = plt.subplots(figsize=(11, 6.2))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 6.2)
    ax.axis("off")

    def box(x, y, w, h, label, color, fontsize=11):
        rect = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08", linewidth=1.5,
                              edgecolor="#222", facecolor=color, alpha=0.9)
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center",
                fontsize=fontsize, fontweight="bold", wrap=True)

    def arrow(x1, y1, x2, y2, label=None, color="#222", style="-|>", offset_y=0.2, lw=1.5):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle=style, color=color, lw=lw,
                                    shrinkA=4, shrinkB=4))
        if label:
            mx, my = (x1 + x2) / 2, (y1 + y2) / 2 + offset_y
            ax.text(mx, my, label, ha="center", va="bottom", fontsize=9,
                    color=color, style="italic")

    # Patient (left)
    box(0.4, 4.0, 2.6, 1.3, "Patient Simulator\n(PatientAgent)", "#cde4f9")
    # Doctor (center)
    box(4.0, 4.0, 3.0, 1.3, "Doctor Agent\n(reads memory)", "#fde2c4")
    # Reviewer (right)
    box(8.0, 4.0, 2.6, 1.3, "Reviewer Agent\n(5-rubric judge)", "#f9d3d4")
    # Memory store (bottom)
    box(4.0, 1.0, 3.0, 1.3, "Medical Memory\n(distilled + recent N)", "#d6f0d4")

    # Conversation arrows (Patient ↔ Doctor)
    arrow(3.0, 4.9, 4.0, 4.9, label="patient utterance", color="#1f77b4", offset_y=0.18)
    arrow(4.0, 4.4, 3.0, 4.4, label="doctor utterance", color="#1f77b4", offset_y=-0.42)

    # Doctor → Reviewer (completed dialogue)
    arrow(7.0, 4.65, 8.0, 4.65, label="completed dialogue + GT", color="#222", offset_y=0.18)
    # Reviewer → Memory
    arrow(9.3, 4.0, 6.0, 2.3, label="review record\n(scores, lessons,\nerrors)", color="#d62728", offset_y=-0.45, lw=1.7)
    # Memory → Doctor
    arrow(5.0, 2.3, 5.0, 4.0, label="injected into\nsystem prompt", color="#2ca02c", offset_y=-0.05, lw=1.7)

    # Side annotation: patient profile feeds into both Patient & Reviewer
    box(0.4, 1.2, 2.8, 0.9, "MIMIC-ED/IV\nPatient Profile + Persona", "#e7e7e7", fontsize=10)
    arrow(1.8, 2.1, 1.8, 4.0, color="#888", style="-|>")
    arrow(3.2, 1.65, 8.4, 4.0, color="#888", style="-|>")

    # Loop label
    ax.text(5.5, 5.85, "Iterative Doctor-Patient-Reviewer Loop with Medical Memory",
            ha="center", fontsize=14, fontweight="bold")
    ax.text(5.5, 0.45,
            "Each consultation produces a memory record. Older records are distilled into a rolling 'wisdom notebook'.\n"
            "Subsequent doctors enter the room already 'experienced' via the memory injection.",
            ha="center", fontsize=9, style="italic", color="#444")

    plt.tight_layout()
    out = os.path.join(OUT_DIR, "architecture.png")
    plt.savefig(out, dpi=170)
    plt.close()
    print(f"saved {out}")


# ---------- 2. Memory mechanism detail ----------

def draw_memory_mechanism():
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 5.5)
    ax.axis("off")

    def box(x, y, w, h, label, color, fontsize=10):
        rect = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.06", linewidth=1.2,
                              edgecolor="#333", facecolor=color, alpha=0.95)
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center",
                fontsize=fontsize, wrap=True)

    # Title
    ax.text(5.5, 5.15, "Sliding Window + Rolling Summary", ha="center",
            fontsize=14, fontweight="bold")
    ax.text(5.5, 4.75, "Recent N records stay intact; older ones are absorbed into a distilled wisdom notebook.",
            ha="center", fontsize=10, style="italic", color="#555")

    # State BEFORE
    ax.text(2.0, 4.2, "State before adding r$_{t+1}$ (window full, N = 3)",
            ha="center", fontsize=10, fontweight="bold")
    box(0.2, 3.2, 1.1, 0.8, "r$_{t-2}$\n(oldest)", "#fde2c4")
    box(1.4, 3.2, 1.1, 0.8, "r$_{t-1}$", "#fde2c4")
    box(2.6, 3.2, 1.1, 0.8, "r$_t$\n(newest)", "#fde2c4")
    ax.text(0.2, 3.05, "recent (intact, FIFO)", fontsize=8, color="#555")
    box(0.2, 1.6, 3.5, 1.0, "distilled notebook (text)\n• History Taking …\n• DDx …\n• Safety …", "#d6f0d4", fontsize=9)

    # Pop arrow
    arrow1 = FancyArrowPatch((1.3, 3.6), (4.3, 3.6), arrowstyle="-|>", lw=1.8, color="#d62728",
                             connectionstyle="arc3,rad=0.0")
    ax.add_patch(arrow1)
    ax.text(2.8, 3.78, "pop oldest", color="#d62728", fontsize=9, fontweight="bold")

    # Distill arrow (pop → distill)
    arrow2 = FancyArrowPatch((4.3, 3.4), (4.3, 2.2), arrowstyle="-|>", lw=1.8, color="#d62728",
                             connectionstyle="arc3,rad=0.0")
    ax.add_patch(arrow2)
    ax.text(4.45, 2.8, "LLM distill\n(notebook + r$_{t-2}$\n→ updated notebook)",
            color="#d62728", fontsize=9, fontweight="bold")

    # Append arrow
    arrow3 = FancyArrowPatch((9.5, 4.6), (8.7, 3.7), arrowstyle="-|>", lw=1.8, color="#2ca02c",
                             connectionstyle="arc3,rad=-0.2")
    ax.add_patch(arrow3)
    ax.text(9.6, 4.3, "append r$_{t+1}$", color="#2ca02c", fontsize=9, fontweight="bold")

    # State AFTER
    ax.text(8.6, 4.2, "State after",
            ha="center", fontsize=10, fontweight="bold")
    box(5.8, 3.2, 1.1, 0.8, "r$_{t-1}$\n(now oldest)", "#fde2c4")
    box(7.0, 3.2, 1.1, 0.8, "r$_t$", "#fde2c4")
    box(8.2, 3.2, 1.1, 0.8, "r$_{t+1}$\n(newest)", "#fde2c4")
    ax.text(5.8, 3.05, "recent (size still N)", fontsize=8, color="#555")
    box(5.8, 1.6, 3.5, 1.0, "distilled notebook (UPDATED)\n• absorbed lessons from r$_{t-2}$\n• merged with prior wisdom",
        "#a8d99c", fontsize=9)

    # Side note
    ax.text(5.5, 0.6,
            "Doctor system prompt receives BOTH:  [Recent Detailed Cases]  +  [Accumulated Clinical Wisdom]",
            ha="center", fontsize=10, fontweight="bold", color="#222",
            bbox=dict(facecolor="#fff4c2", edgecolor="#aa7", boxstyle="round,pad=0.4"))

    plt.tight_layout()
    out = os.path.join(OUT_DIR, "memory_mechanism.png")
    plt.savefig(out, dpi=170)
    plt.close()
    print(f"saved {out}")


if __name__ == "__main__":
    draw_arch()
    draw_memory_mechanism()
