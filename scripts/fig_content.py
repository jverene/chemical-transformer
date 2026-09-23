"""Figure 2 (punchline): supervision content is irrelevant.

Message-first design (neuromodulation-house style): verdict colors,
paired-seed dots, tie bands, annotated deltas, and an inset carrying the
scale replication.

(a) 11M Stage-2 arms, identical inference budget (~19 MFLOPs/tok incl.
    oracle-pass billing), paired by seed (thin lines; same Stage-1 body
    per seed): constant-gate-from-scratch vs shuffled (noise targets) vs
    gate-grad (measured value) vs tag (true labels). The three
    varying-target arms sit inside a green tie band (max paired delta
    0.3 pts); the constant arm sits ~2.4 pts below (vermillion).
    Inset (lower-right, empty region): the 1.4B web-text replication —
    gate-grad 2.7077 vs shuffled 2.7085 held loss (|delta| = 0.0008 vs
    the 1.0-pt pre-registered threshold), dense Stage-1 reference line.
(b) Realized E/M/H gate spread per arm (per-seed dots): only the tag arm
    builds difficulty-ordered gates (r = 0.95) — structure without
    accuracy benefit, which is the point.

Sources:
  results-p0/tagged/{predictor-supervised,gategrad,shuffled}_seed*.json
  results/tagged/fixed-schedule_seed*.json        (constant-gate arm)
  results-nl/webtext/{ours-gategrad_seed1,shuffled_seed1}.json + dense

Writes paper/content.pdf. Run from repo root:
  .venv/bin/python scripts/fig_content.py
"""
import glob
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Okabe-Ito
C_BLUE = "#0072B2"
C_VERM = "#D55E00"
C_GREEN = "#009E73"
C_GRAY = "#555555"
plt.rcParams.update({
    "font.size": 6.8, "axes.titlesize": 7.4, "axes.labelsize": 6.6,
    "legend.fontsize": 6.0, "xtick.labelsize": 6.0, "ytick.labelsize": 6.2,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 200})


def load_accs():
    def accs(pat):
        return [100 * json.load(open(p))["final"]["acc"]
                for p in sorted(glob.glob(pat))]
    return {
        "const": accs("results/tagged/fixed-schedule_seed*.json"),
        "shuffled": accs("results-p0/tagged/shuffled_seed*.json"),
        "gategrad": accs("results-p0/tagged/gategrad_seed*.json"),
        "tag": accs("results-p0/tagged/predictor-supervised_seed*.json"),
    }


def load_spreads():
    def spreads(pat):
        out = []
        for p in sorted(glob.glob(pat)):
            gb = json.load(open(p))["final"].get("gate_by_bin") or []
            out.append(gb[-1] - gb[0] if len(gb) >= 2 else 0.0)
        return out
    return {
        "const": [0.0] * 3,
        "shuffled": spreads("results-p0/tagged/shuffled_seed*.json"),
        "gategrad": spreads("results-p0/tagged/gategrad_seed*.json"),
        "tag": spreads("results-p0/tagged/predictor-supervised_seed*.json"),
    }


def nl_losses():
    gg = json.load(open("results-nl/webtext/ours-gategrad_seed1.json"))["final"]
    sh = json.load(open("results-nl/webtext/shuffled_seed1.json"))["final"]
    de = json.load(open("results-nl/webtext/baseline_seed0.json"))["final"]
    return de["held_loss"], gg["held_loss"], sh["held_loss"]


def panel_a(ax, accs):
    order = ["const", "shuffled", "gategrad", "tag"]
    labels = ["Constant\n$g{=}0.5$", "Shuffled\n(noise)", "Gate-grad\n(measured)",
              "Tag\n(labels)"]
    xs = np.arange(4)
    # paired seed lines first (under the dots)
    for s in range(3):
        ax.plot(xs, [accs[a][s] for a in order], "-", color="0.78",
                lw=0.7, zorder=1)
    # green tie band across the three varying-target arms
    means = {a: float(np.mean(accs[a])) for a in order}
    band_lo = min(means[a] for a in order[1:]) - 0.45
    band_hi = max(means[a] for a in order[1:]) + 0.45
    ax.axhspan(band_lo, band_hi, xmin=0.14, xmax=0.99, color=C_GREEN,
               alpha=0.13, lw=0)
    ax.annotate("content-irrelevant:\nmax paired $|\\Delta| = 0.3$ pts",
                (2.0, band_hi + 0.25), fontsize=5.6, color=C_GREEN,
                ha="center")
    # constant-arm verdict
    ax.annotate(f"$-{means['shuffled'] - means['const']:.1f}$ pts (paired)",
                (0.0, means["const"] - 0.75), fontsize=5.6, color=C_VERM,
                ha="center")
    # seed dots + mean markers
    for i, a in enumerate(order):
        col = C_VERM if a == "const" else C_BLUE
        ax.scatter([i] * 3, accs[a], s=9, color=col, zorder=3, alpha=0.75)
        ax.plot([i - 0.16, i + 0.16], [means[a]] * 2, color="black", lw=1.5,
                zorder=4)
    ax.set_xticks(xs, labels)
    ax.tick_params(axis="x", labelsize=5.8)
    ax.set_ylabel("held-out accuracy (%)")
    ax.set_ylim(69.2, 77.6)
    ax.set_xlim(-0.45, 3.45)
    ax.set_title("(a) Same accuracy, whatever the targets", loc="left")

    # inset: 1.4B replication, lower-right (tag/shuffled dots live >= 73.4)
    axi = ax.inset_axes([0.56, 0.06, 0.41, 0.44])
    dense, gg, sh = nl_losses()
    axi.axhline(dense, color=C_GREEN, ls="--", lw=0.9)
    axi.text(1.62, dense + 0.004, "dense", fontsize=5.0, color=C_GREEN,
             ha="left", va="bottom")
    axi.scatter([0.85], [gg], s=13, color=C_BLUE, zorder=3)
    axi.scatter([1.15], [sh], s=13, color=C_GRAY, zorder=3)
    axi.plot([0.85, 1.15], [gg, sh], color="0.6", lw=0.7)
    axi.annotate("$|\\Delta|=0.0008$\n(threshold: 1.0 pt)", (1.0, 2.732),
                 fontsize=5.0, ha="center")
    axi.set_xticks([0.85, 1.15], ["gate-grad", "shuf."])
    axi.set_xlim(0.5, 1.5)
    axi.tick_params(axis="x", labelsize=5.0)
    axi.tick_params(axis="y", labelsize=5.0)
    axi.set_ylim(2.50, 2.80)
    axi.set_title("1.4B web text (loss)", fontsize=5.6, pad=2)
    for sp in axi.spines.values():
        sp.set_visible(True); sp.set_linewidth(0.7); sp.set_color("0.3")
    axi.patch.set_facecolor("white"); axi.patch.set_alpha(1.0)


def panel_b(ax, spreads):
    order = ["const", "shuffled", "gategrad", "tag"]
    labels = ["Constant", "Shuffled", "Gate-grad", "Tag"]
    xs = np.arange(4)
    means = [float(np.mean(spreads[a])) for a in order]
    bars = ax.bar(xs, means, width=0.55,
                  color=["0.75", "0.75", "0.75", C_BLUE], zorder=2)
    for i, a in enumerate(order):
        ax.scatter([i] * len(spreads[a]), spreads[a], s=8, color="black",
                   zorder=3, alpha=0.7)
    ax.annotate("only tag orders gates\n(bin--difficulty $r=0.95$)",
                (0.05, 0.232), fontsize=5.6, color=C_BLUE, ha="left",
                va="center",
                arrowprops=dict(arrowstyle="->", lw=0.7, color=C_BLUE))
    ax.annotate("no structure, same accuracy (a)", (1.0, 0.092), fontsize=5.6,
                color=C_GRAY, ha="center")
    ax.set_xticks(xs, ["Constant", "Shuffled", "Gate-\ngrad", "Tag"])
    ax.set_ylabel("realized gate spread (H $-$ E)")
    ax.set_ylim(0, 0.30)
    ax.set_title("(b) Structure without benefit", loc="left")


def main():
    accs, spreads = load_accs(), load_spreads()
    fig, axes = plt.subplots(1, 2, figsize=(5.5, 1.9),
                             gridspec_kw={"width_ratios": [1.5, 1]})
    panel_a(axes[0], accs)
    panel_b(axes[1], spreads)
    fig.tight_layout(w_pad=1.6)
    fig.savefig("paper/content.pdf")
    print("wrote paper/content.pdf")


if __name__ == "__main__":
    main()
