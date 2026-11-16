import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

BLUE = "#0072B2"
VERMILLION = "#D55E00"

plt.rcParams.update({
    "font.family": "DejaVu Serif",
    "font.size": 7.0,
    "figure.dpi": 200,
})


def box(ax, x, y, width, text, color):
    patch = FancyBboxPatch(
        (x, y), width, 0.30,
        boxstyle="round,pad=0.008,rounding_size=0.015",
        facecolor=color, edgecolor="white", linewidth=0.8)
    ax.add_patch(patch)
    ax.text(x + width / 2, y + 0.15, text, ha="center", va="center",
            color="black", linespacing=1.25)


def arrow(ax, left, right, y=0.51):
    ax.annotate("", xy=(right, y), xytext=(left, y),
                arrowprops={"arrowstyle": "->", "color": "0.35", "lw": 0.9})


def panel(ax, title, labels, color, note, note_color):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(0.01, 0.91, title, ha="left", va="center",
            fontsize=8.2, fontweight="bold", color="black")
    gap = 0.035
    width = (0.98 - gap * (len(labels) - 1)) / len(labels)
    for index, label in enumerate(labels):
        x = 0.01 + index * (width + gap)
        box(ax, x, 0.36, width, label, color)
        if index + 1 < len(labels):
            arrow(ax, x + width + 0.005, x + width + gap - 0.005)
    ax.text(0.50, 0.13, note, ha="center", va="center",
            fontsize=7.2, color=note_color, fontweight="bold")


def main():
    fig, axes = plt.subplots(2, 1, figsize=(5.5, 2.35), layout="constrained")
    panel(
        axes[0], "A  Continuous gate: original diagnostics",
        ["Token states\n(all valid rows)", "Run FFN\non every row", "Multiply output\nby fractional g"],
        "#F7E4D4", "Every row is computed: no executed FFN savings", VERMILLION)
    panel(
        axes[1], "B  Binary exact-budget mask: new experiments",
        ["Score current\npre-FFN states", "Select exactly\nB valid rows", "Gather rows\nand run FFN", "Scatter outputs\nto full sequence"],
        "#DCECF5", "Actual row skipping; attention remains dense", BLUE)
    fig.savefig("paper/execution_overview.pdf", bbox_inches="tight", pad_inches=0.02)
    print("wrote paper/execution_overview.pdf")


if __name__ == "__main__":
    main()
