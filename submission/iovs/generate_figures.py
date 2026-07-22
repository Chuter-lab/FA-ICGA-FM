#!/usr/bin/env python3
"""Generate publication figures for FA-ICGA-FM IOVS submission.

All results are from synthetic smoke runs (is_synthetic=true).
Labels on every figure say so explicitly -- no real-data claims.
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from pathlib import Path

FIG_DIR = Path(__file__).parent / "figures"
FIG_DIR.mkdir(exist_ok=True)

SYNTHETIC_NOTE = "NOTE: All results from synthetic smoke-test data only. Real-data performance pending APTOS 2023 staging."

# ── colour palette (ARVO accessible) ─────────────────────────────────────────
C_BLUE   = "#1f77b4"
C_ORANGE = "#ff7f0e"
C_GREEN  = "#2ca02c"
C_RED    = "#d62728"
C_GREY   = "#7f7f7f"
C_PURPLE = "#9467bd"
C_BROWN  = "#8c564b"
C_PINK   = "#e377c2"
C_OLIVE  = "#bcbd22"
C_CYAN   = "#17becf"
C_TEAL   = "#006064"
C_DARK   = "#333333"

# ─────────────────────────────────────────────────────────────────────────────
# Figure 1 — Pipeline Architecture Overview (schematic)
# ─────────────────────────────────────────────────────────────────────────────
def fig1_pipeline_overview():
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)
    ax.axis("off")

    def box(cx, cy, w, h, text, color="#dce8f5", fontsize=9, textcolor="black"):
        rect = mpatches.FancyBboxPatch(
            (cx - w/2, cy - h/2), w, h,
            boxstyle="round,pad=0.1", linewidth=1.2,
            edgecolor="#2c5f8a", facecolor=color, zorder=3
        )
        ax.add_patch(rect)
        ax.text(cx, cy, text, ha="center", va="center",
                fontsize=fontsize, fontweight="bold", color=textcolor,
                wrap=True, zorder=4)

    def arrow(x1, y1, x2, y2):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", color="#555", lw=1.5),
                    zorder=2)

    # Data input
    box(1.5, 6.5, 2.6, 1.0, "APTOS 2023\nFA + ICGA\n(58 K images, 24 classes)",
        color="#e8f5e9", fontsize=8)

    # Preprocessing
    box(4.5, 6.5, 2.4, 1.0, "Pre-processing\nAugmentation\n(CutMix, MixUp, Fourier,\nPhase-swap)",
        color="#fff3e0", fontsize=7.5)
    arrow(2.8, 6.5, 3.3, 6.5)

    # Pre-training block
    box(4.5, 4.8, 2.4, 1.6, "Pre-training\n─────────────\nMAE (A1)\nCross-modal Contrastive (C1)\nMTL (A3)",
        color="#e3f2fd", fontsize=7.5)
    arrow(4.5, 5.9, 4.5, 5.65)

    # Backbones
    box(8.2, 6.5, 3.2, 1.0,
        "Backbones\nViT-B · Swin-B · ConvNeXt-B\nRETFound · BiomedCLIP · DINOv2\nEffNetV2-S · DeiT-III · MaxViT",
        color="#f3e5f5", fontsize=7.5)
    arrow(5.7, 6.5, 6.6, 6.5)

    # Fine-tuning
    box(8.2, 4.8, 3.2, 1.6,
        "Fine-tuning\n────────────\nLabel-smooth + Temp-scale (H5)\nFocal / PolyLoss / ASL (H12,N9,M16)\nCurriculum (H14) · SWA (H17) · EMA\nSAM (M2) · LLRD (M17)",
        color="#fce4ec", fontsize=7.5)
    arrow(8.2, 5.9, 8.2, 5.65)

    # Evaluation block
    box(11.5, 4.8, 2.6, 2.4,
        "Evaluation\n──────────\nLinear probe · KNN · SVM-RBF\nFew-shot (5/10/25)\nCORAL / CORN ordinal\nConformal sets · ECE\nGradCAM++ · Attention\nVessel seg (Dice/CAL)\nTop-k · QWK · Per-class PR-AUC",
        color="#e8eaf6", fontsize=7.0)
    arrow(9.8, 4.8, 10.2, 4.8)

    # Ensemble
    box(8.2, 2.8, 3.2, 1.2,
        "Ensemble\n──────────\nGreedy Forward Selection (F1)\nDeep Ensemble (N4)\nFedAvg Simulation (F4)",
        color="#fff9c4", fontsize=7.5)
    arrow(8.2, 4.0, 8.2, 3.4)
    arrow(10.2, 4.0, 8.8, 3.4)

    # Interpretability
    box(4.5, 2.8, 2.4, 1.2,
        "Interpretability\n──────────────\nGradCAM++ (H15)\nAttention Rollout (E5)\nt-SNE (M30) · ViT-FPN (H22)",
        color="#e0f7fa", fontsize=7.5)
    arrow(8.2, 4.0, 4.5, 3.4)

    # Output
    box(1.5, 2.8, 2.0, 1.2,
        "Outputs\n────────\nCheckpoints\nMetrics JSON\nFigures",
        color="#f5f5f5", fontsize=8)
    arrow(3.3, 2.8, 2.5, 2.8)

    # Phase-conditioned ViT
    box(4.5, 1.2, 2.4, 0.9, "Phase-conditioned ViT (D1)\n+ IQA Head (E6)",
        color="#ffe0b2", fontsize=7.5)
    arrow(4.5, 2.2, 4.5, 1.65)

    # Cross-modal
    box(8.2, 1.2, 3.2, 0.9,
        "Cross-modal Alignment\nCross-Attention FA/ICGA (H18)\nPhase-aware IQA (G3)",
        color="#e8f5e9", fontsize=7.5)
    arrow(8.2, 2.2, 8.2, 1.65)

    # Title
    ax.text(7, 7.7, "FA-ICGA-FM: Pipeline Architecture",
            ha="center", va="center", fontsize=13, fontweight="bold", color="#1a1a2e")
    ax.text(7, 7.3, SYNTHETIC_NOTE,
            ha="center", va="center", fontsize=7, color="red", style="italic")

    plt.tight_layout()
    out = FIG_DIR / "fig1_pipeline_overview.tiff"
    fig.savefig(out, dpi=300, bbox_inches="tight", format="tiff")
    plt.close(fig)
    print(f"Saved {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 2 — Backbone Comparison (synthetic AUC)
# ─────────────────────────────────────────────────────────────────────────────
def fig2_backbone_comparison():
    backbones = {
        "ViT-B/16":        0.7840,
        "DINOv2-FT":       0.7818,
        "ConvNeXt-B":      0.7296,
        "Phase-ViT":       0.7285,
        "RETFound":        0.7206,
        "BiomedCLIP":      0.6951,
        "DINOv2-LP":       0.6794,
        "DeiT-III-B":      0.6618,
        "EfficientNetV2-S":0.6584,
        "MaxViT-Tiny":     0.5550,
        "CLIP (zero-shot)":0.5351,
        "Swin-B":          0.4592,
    }
    names  = list(backbones.keys())
    aucs   = list(backbones.values())
    colors = [C_BLUE if a > 0.75 else C_ORANGE if a > 0.65 else C_GREY for a in aucs]

    fig, ax = plt.subplots(figsize=(10, 7))
    bars = ax.barh(names, aucs, color=colors, edgecolor="white", linewidth=0.5)
    ax.axvline(0.5, color="black", linestyle="--", linewidth=0.8, label="Random (0.5)")
    for bar, auc in zip(bars, aucs):
        ax.text(auc + 0.005, bar.get_y() + bar.get_height()/2,
                f"{auc:.3f}", va="center", ha="left", fontsize=8)
    ax.set_xlabel("One-vs-Rest Macro AUC", fontsize=11)
    ax.set_title("Backbone Linear-Probe AUC — Synthetic Smoke-Test Data",
                 fontsize=11, fontweight="bold")
    ax.set_xlim(0.0, 0.95)
    ax.invert_yaxis()
    legend_patches = [
        mpatches.Patch(color=C_BLUE,   label="AUC > 0.75"),
        mpatches.Patch(color=C_ORANGE, label="AUC 0.65–0.75"),
        mpatches.Patch(color=C_GREY,   label="AUC < 0.65"),
    ]
    ax.legend(handles=legend_patches, loc="lower right", fontsize=8)
    plt.tight_layout()
    fig.text(0.5, 0.01,
             "NOTE: values obtained on synthetic data only — not indicative of real-world performance.",
             ha="center", fontsize=7, color="red", style="italic")
    out = FIG_DIR / "fig2_backbone_comparison.tiff"
    fig.savefig(out, dpi=300, bbox_inches="tight", format="tiff")
    plt.close(fig)
    print(f"Saved {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 3 — Few-shot & Calibration Panel
# ─────────────────────────────────────────────────────────────────────────────
def fig3_fewshot_calibration():
    fig = plt.figure(figsize=(12, 5))
    gs = GridSpec(1, 2, figure=fig, wspace=0.35)

    # Panel A: few-shot
    ax1 = fig.add_subplot(gs[0])
    shots    = [5, 10, 25]
    means    = [0.0774, 0.0952, 0.0536]
    stds     = [0.0607, 0.0168, 0.0000]
    ax1.errorbar(shots, means, yerr=stds, marker="o", color=C_BLUE,
                 linewidth=2, capsize=5, label="ViT-B")
    ax1.axhline(1/24, color=C_GREY, linestyle="--", linewidth=1, label="Random (1/24)")
    ax1.set_xlabel("Shots per Class", fontsize=11)
    ax1.set_ylabel("Accuracy", fontsize=11)
    ax1.set_title("(A) Few-Shot Classification\n(synthetic data, ViT-B features)", fontsize=10)
    ax1.set_xticks(shots)
    ax1.legend(fontsize=8)
    ax1.set_ylim(0, 0.25)
    ax1.text(0.5, -0.18, SYNTHETIC_NOTE,
             transform=ax1.transAxes, ha="center", fontsize=7, color="red", style="italic")

    # Panel B: ECE reliability diagram
    ax2 = fig.add_subplot(gs[1])
    bin_confs = [0.0, 0.1190, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    bin_accs  = [0.0, 0.0714, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    bin_counts= [0,   56,     0,   0,   0,   0,   0,   0,   0,   0]
    bin_edges = np.linspace(0, 1, 11)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    widths = np.diff(bin_edges)

    # bar per bin (only plot non-zero counts)
    ax2.bar(bin_centers, bin_accs, width=widths * 0.8, color=C_BLUE,
            alpha=0.7, label="Accuracy", align="center")
    ax2.plot([0, 1], [0, 1], "k--", linewidth=1, label="Perfect calibration")
    ax2.set_xlabel("Mean Confidence", fontsize=11)
    ax2.set_ylabel("Fraction Correct", fontsize=11)
    ax2.set_title(f"(B) Reliability Diagram\n(ECE = 0.048, synthetic data)", fontsize=10)
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.legend(fontsize=8)
    ax2.text(0.5, -0.18, SYNTHETIC_NOTE,
             transform=ax2.transAxes, ha="center", fontsize=7, color="red", style="italic")

    fig.suptitle("Few-Shot Evaluation and Calibration Assessment", fontsize=12, fontweight="bold", y=1.02)
    plt.tight_layout()
    out = FIG_DIR / "fig3_fewshot_calibration.tiff"
    fig.savefig(out, dpi=300, bbox_inches="tight", format="tiff")
    plt.close(fig)
    print(f"Saved {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 4 — Methods Coverage Wheel
# ─────────────────────────────────────────────────────────────────────────────
def fig4_methods_coverage():
    categories = {
        "Pre-training\n(A1,A3,C1)":      3,
        "Backbones\n(B1–B7,N1–N3)":      10,
        "Augmentation\n(H4,H10,H24)":    3,
        "Loss Functions\n(H5,H12,M15\nM16,N9,N11)": 6,
        "Evaluation\n(E1–E8,N8,M19\nM21,M31,M32,M38)": 11,
        "Calibration &\nUncertainty\n(H6,H16,H23,M22)": 4,
        "Interpretability\n(H15,E5,M30,N21)": 4,
        "Ordinal\nRegression\n(E3,N10)":  2,
        "Ensemble\n(F1,F2,N4,F4)":        4,
        "Training\nSchedule\n(H14,H17,M2,M17)": 4,
        "Phase/Cross-\nmodal (D1,H18,G3)": 3,
        "CLIP Zero-shot\n(H2)":            1,
        "Architecture\n(H18,H22)":         2,
        "Segmentation\n(E2,E8)":           2,
        "Other (A4–A6\nG1,G2)":            1,
    }

    labels = list(categories.keys())
    sizes  = list(categories.values())
    colors = plt.cm.tab20(np.linspace(0, 1, len(sizes)))

    fig, ax = plt.subplots(figsize=(10, 9))
    wedges, texts, autotexts = ax.pie(
        sizes, labels=None, colors=colors,
        autopct=lambda pct: f"{pct:.0f}%" if pct > 3 else "",
        startangle=90, pctdistance=0.8,
        wedgeprops=dict(linewidth=0.5, edgecolor="white")
    )
    for at in autotexts:
        at.set_fontsize(7)

    ax.legend(wedges, [f"{l} ({s})" for l, s in zip(labels, sizes)],
              loc="center left", bbox_to_anchor=(1.0, 0.5),
              fontsize=7.5, frameon=False)
    ax.set_title(f"FA-ICGA-FM: {sum(sizes)} Implemented Approaches\nAcross 15 Methodological Categories",
                 fontsize=12, fontweight="bold", pad=20)
    plt.tight_layout()
    out = FIG_DIR / "fig4_methods_coverage.tiff"
    fig.savefig(out, dpi=300, bbox_inches="tight", format="tiff")
    plt.close(fig)
    print(f"Saved {out}")


if __name__ == "__main__":
    fig1_pipeline_overview()
    fig2_backbone_comparison()
    fig3_fewshot_calibration()
    fig4_methods_coverage()
    print("All figures generated.")
