import os
import subprocess
import sys

# Ensure required packages are installed
required_packages = ["matplotlib", "numpy"]
for pkg in required_packages:
    try:
        __import__(pkg)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import textwrap

OUTPUT_FOLDER = "PEUM"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)


def save_figure(fig, file_name):
    path = os.path.join(OUTPUT_FOLDER, file_name)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)


def draw_flowchart():
    fig, ax = plt.subplots(figsize=(10, 14))
    fig.patch.set_facecolor("#1E2761")
    ax.set_facecolor("#1E2761")
    ax.axis("off")

    title = "PU-EM Architecture & Pipeline"
    ax.text(0.5, 0.97, title, ha="center", va="center", fontsize=24, color="white", weight="bold")

    box_texts = [
        "Step 1: Positive Set P — Confirmed LTBI from 2014–15 Survey",
        "Step 2: Initial Classifier — Train on P=1, Random Sample U=0",
        "Step 3: E-Step — Estimate P(Latent TB | x) for Each Unlabeled Record",
        "Step 4: M-Step — Re-fit Classifier on Soft-Labeled Full Dataset",
        "Step 5: Convergence — Log-Likelihood Plateau after 18 Iterations",
        "OUTPUT: Per-District LTBI Prevalence % + Ranked Priority List → MoH Dashboard",
    ]
    box_colors = ["#CADCFC"] * 5 + ["#F9E795"]
    y_positions = np.linspace(0.82, 0.1, len(box_texts))

    for idx, (text, color, y) in enumerate(zip(box_texts, box_colors, y_positions)):
        wrapped = textwrap.fill(text, width=30)
        box = patches.FancyBboxPatch(
            (0.1, y - 0.05),
            0.8,
            0.12,
            boxstyle="round,pad=0.03,rounding_size=0.04",
            linewidth=1.5,
            edgecolor="white",
            facecolor=color,
            mutation_aspect=1.8,
        )
        ax.add_patch(box)
        ax.text(0.5, y + 0.01, wrapped, ha="center", va="center", fontsize=12, color="#1E2761" if color == "#CADCFC" else "#1E2761")

        if idx < len(box_texts) - 1:
            arrow_y = y - 0.02
            ax.annotate(
                "",
                xy=(0.5, arrow_y - 0.02),
                xytext=(0.5, y - 0.03),
                arrowprops=dict(arrowstyle="->", color="white", lw=2),
            )

    save_figure(fig, "slide3_pipeline_flowchart.png")


def draw_radar_chart():
    labels = [
        "Combating LTBI",
        "Resource Optimization",
        "Ending Active TB",
        "Uganda Context",
        "ML Implementation",
        "Govt Actionability",
        "Evidence Quality",
    ]
    num_vars = len(labels)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]

    scores = {
        "PUEM": [9, 9, 10, 9, 9, 10, 9],
        "NMF": [6, 7, 5, 8, 8, 6, 7],
        "SEM": [1, 6, 0, 7, 7, 4, 5],
    }

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    for name, values, color, alpha in [
        ("PUEM", scores["PUEM"], "#028090", 0.25),
        ("NMF", scores["NMF"], "#F96167", 0.2),
        ("SEM", scores["SEM"], "#6D2E46", 0.2),
    ]:
        values = values + values[:1]
        ax.plot(angles, values, color=color, linewidth=2, label=name)
        ax.fill(angles, values, color=color, alpha=alpha)

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_thetagrids(np.degrees(angles[:-1]), labels)
    ax.set_ylim(0, 10)
    ax.set_rlabel_position(180 / num_vars)
    ax.tick_params(colors="#333333")
    ax.grid(color="#999999", linestyle="--", linewidth=0.7)
    ax.spines["polar"].set_color("#333333")

    ax.set_title("Model Alignment Scorecard: PUEM vs NMF vs SEM", va="bottom", fontsize=18, weight="bold")
    ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.1))

    save_figure(fig, "slide6_radar_chart.png")


def draw_resource_bar_chart():
    categories = ["Uniform Allocation", "NMF Targeted", "PUEM Targeted"]
    values = [400.5, 331.6, 256.8]
    colors = ["#B85042", "#F9E795", "#028090"]
    savings = [0, 25.63, 35.89]

    fig, ax = plt.subplots(figsize=(12, 8))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    bars = ax.bar(categories, values, color=colors, edgecolor="black", linewidth=0.8)
    ax.set_ylabel("Cost (Billion UGX)", fontsize=14)
    ax.set_title("Resource Optimization: Targeted vs Uniform Allocation", fontsize=18, weight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    baseline = 400.5
    ax.axhline(baseline, color="#333333", linestyle="--", linewidth=1.25)
    ax.text(0.98, baseline + 8, "Uniform Baseline", ha="right", va="bottom", fontsize=12, color="#333333")

    for bar, value, save in zip(bars, values, savings):
        height = bar.get_height()
        label = f"{value:.1f}B UGX"
        if save > 0:
            label += f"\n{save:.2f}% saved"
        ax.text(bar.get_x() + bar.get_width() / 2, height + 8, label, ha="center", va="bottom", fontsize=11, color="#333333")

    puem_bar = bars[2]
    x = puem_bar.get_x() + puem_bar.get_width() / 2
    y = puem_bar.get_height()
    ax.annotate(
        "143.7B UGX Saved = USD 38.3M",
        xy=(x, y),
        xytext=(x + 0.5, y + 70),
        arrowprops=dict(arrowstyle="->", color="#028090", lw=2),
        fontsize=12,
        color="#028090",
    )

    save_figure(fig, "slide5_resource_bar_chart.png")


def draw_roadmap():
    fig, ax = plt.subplots(figsize=(16, 10))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.axis("off")

    top_y = 0.75
    left_x = 0.05
    width = 0.16
    height = 0.12
    gap = 0.03
    items = [
        "PU-EM Prevalence",
        "BN Risk Rank",
        "SEM System Diagnosis",
        "Dashboard",
        "MoH Allocation",
    ]

    for idx, text in enumerate(items):
        x = left_x + idx * (width + gap)
        box = patches.FancyBboxPatch(
            (x, top_y),
            width,
            height,
            boxstyle="round,pad=0.03,rounding_size=0.03",
            linewidth=1.5,
            edgecolor="#028090",
            facecolor="#1E2761",
        )
        ax.add_patch(box)
        ax.text(x + width / 2, top_y + height / 2, text, ha="center", va="center", fontsize=12, color="white", wrap=True)
        if idx < len(items) - 1:
            ax.annotate(
                "",
                xy=(x + width, top_y + height / 2),
                xytext=(x + width + gap, top_y + height / 2),
                arrowprops=dict(arrowstyle="->", color="#028090", lw=2),
            )

    bottom_y = 0.25
    phase_texts = [
        "Phase 2A — Expand PU-EM to all 135 Uganda Districts",
        "Phase 2B — Replace Simulated SEM Data with Real DHIS2 Data",
        "Phase 2C — PU-EM Labels Replace Proxy Labels in BN Training",
    ]
    phase_width = 0.26
    phase_gap = 0.03
    for idx, text in enumerate(phase_texts):
        x = left_x + idx * (phase_width + phase_gap)
        box = patches.FancyBboxPatch(
            (x, bottom_y),
            phase_width,
            height,
            boxstyle="round,pad=0.03,rounding_size=0.03",
            linewidth=1.5,
            edgecolor="#333333",
            facecolor="#F9E795",
        )
        ax.add_patch(box)
        ax.text(x + phase_width / 2, bottom_y + height / 2, text, ha="center", va="center", fontsize=12, color="#1E2761", wrap=True)

    ax.plot([0.05, 0.95], [0.5, 0.5], color="#333333", linewidth=2)
    ax.text(0.05, 0.52, "CURRENT PIPELINE", ha="left", va="bottom", fontsize=14, weight="bold", color="#1E2761")
    ax.text(0.95, 0.52, "PHASE 2 ROADMAP", ha="right", va="bottom", fontsize=14, weight="bold", color="#1E2761")

    save_figure(fig, "slide7_roadmap.png")


if __name__ == "__main__":
    draw_flowchart()
    draw_radar_chart()
    draw_resource_bar_chart()
    draw_roadmap()
    print("All 4 visuals saved to PEUM folder successfully.")
