"""
================================================================================
PU-EM PART 3: Visualisations, summary tables, comparison table, main pipeline
================================================================================
"""
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.linear_model import LogisticRegression

warnings.filterwarnings("ignore")

OUTPUT_DIR   = Path("puem_results")
OUTPUT_DIR.mkdir(exist_ok=True)
RANDOM_STATE = 42
IPT_COST_UGX = 42_000
UGX_PER_USD  = 3_750

from puem_part1 import (
    SURVEY_LTBI_RATES, SURVEY_DISTRICTS, HOLDOUT_SURVEY_DISTRICTS,
    UGANDA_DISTRICTS, LTBI_PRIOR,
    generate_survey_positives, generate_dhis2_unlabelled, preprocess,
)
from puem_part2 import (
    run_puem, compute_national_prevalence, compute_district_prevalence,
    compute_metrics,
)


# ── Visualisations ────────────────────────────────────────────────────────────
def plot_convergence(ll_history):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(range(1, len(ll_history)+1), ll_history, "o-",
            color="#1976D2", linewidth=2, markersize=6)
    ax.axhline(ll_history[-1], color="#F44336", linestyle="--", linewidth=1.2,
               label=f"Final LL = {ll_history[-1]:.2f}")
    ax.set_xlabel("EM Iteration", fontsize=12)
    ax.set_ylabel("Log-Likelihood", fontsize=12)
    ax.set_title("PU-EM Convergence\nLog-Likelihood vs Iteration Number",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=10); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out = OUTPUT_DIR / "01_puem_convergence.png"
    plt.savefig(out, dpi=300, bbox_inches="tight"); plt.close()
    print(f"  Saved: {out}")


def plot_roc(fpr, tpr, auc, opt_thr):
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.plot(fpr, tpr, color="#1976D2", linewidth=2.5,
            label=f"PU-EM  AUC = {auc:.4f}")
    ax.plot([0,1],[0,1], "k--", linewidth=1, label="Random")
    # Mark optimal threshold
    from sklearn.metrics import roc_curve
    j_scores = tpr - fpr
    opt_idx  = int(np.argmax(j_scores))
    ax.scatter(fpr[opt_idx], tpr[opt_idx], s=120, color="#D32F2F", zorder=5,
               label=f"Optimal threshold = {opt_thr:.3f}")
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate (Recall)", fontsize=12)
    ax.set_title("ROC Curve — PU-EM District-Level LTBI Classification\n"
                 "(Positive class: Survey LTBI > 25%)",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=10); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out = OUTPUT_DIR / "02_puem_roc_curve.png"
    plt.savefig(out, dpi=300, bbox_inches="tight"); plt.close()
    print(f"  Saved: {out}")


def plot_confusion_matrix(tp, fp, fn, tn):
    cm = np.array([[tn, fp],[fn, tp]])
    labels = ["Not Top-Q\n(LTBI<15%)", "Top-Q\n(LTBI>25%)"]
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Pred: Not High","Pred: High"],
                yticklabels=["True: Not High","True: High"],
                linewidths=1, ax=ax, annot_kws={"size": 16})
    ax.set_title("Confusion Matrix — PU-EM Top-Quartile District Ranking\n"
                 "(Survey LTBI >25% = Positive class)",
                 fontsize=13, fontweight="bold")
    ax.set_ylabel("Survey Ground Truth", fontsize=11)
    ax.set_xlabel("PU-EM Prediction", fontsize=11)
    plt.tight_layout()
    out = OUTPUT_DIR / "03_puem_confusion_matrix.png"
    plt.savefig(out, dpi=300, bbox_inches="tight"); plt.close()
    print(f"  Saved: {out}")


def plot_district_bar(dist_prev, n=30):
    top30 = dist_prev.head(n).copy()
    top30["survey_rate"] = top30["district"].map(SURVEY_LTBI_RATES)

    fig, ax = plt.subplots(figsize=(14, 7))
    x = np.arange(n)
    bars = ax.bar(x, top30["puem_ltbi_estimate"], color="#1976D2",
                  edgecolor="white", label="PU-EM Estimate")
    # Overlay survey dots where available
    survey_mask = top30["survey_rate"].notna()
    ax.scatter(x[survey_mask], top30["survey_rate"][survey_mask],
               color="#D32F2F", s=80, zorder=5, label="Survey LTBI Rate (2014-15)")
    ax.set_xticks(x)
    ax.set_xticklabels(top30["district"], rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("LTBI Prevalence Estimate", fontsize=11)
    ax.set_title(f"Top {n} Priority Districts for IPT Procurement\n"
                 "PU-EM LTBI Estimates vs Survey Ground Truth",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=10); ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    out = OUTPUT_DIR / "04_puem_top30_districts.png"
    plt.savefig(out, dpi=300, bbox_inches="tight"); plt.close()
    print(f"  Saved: {out}")


def plot_scatter_vs_survey(survey_val_df):
    from scipy.stats import pearsonr
    r, p = pearsonr(survey_val_df["puem_est"], survey_val_df["survey_rate"])
    fig, ax = plt.subplots(figsize=(9, 7))
    sc = ax.scatter(survey_val_df["puem_est"], survey_val_df["survey_rate"],
                    c=survey_val_df["survey_rate"], cmap="RdYlGn_r",
                    s=100, edgecolors="black", linewidths=0.5, zorder=3)
    m, b = np.polyfit(survey_val_df["puem_est"], survey_val_df["survey_rate"], 1)
    x_line = np.linspace(survey_val_df["puem_est"].min(),
                         survey_val_df["puem_est"].max(), 100)
    ax.plot(x_line, m*x_line+b, "r--", linewidth=2,
            label=f"Pearson r={r:.3f}, p={p:.4f}")
    for _, row in survey_val_df.iterrows():
        ax.annotate(row["district"], (row["puem_est"], row["survey_rate"]),
                    fontsize=7, xytext=(3,3), textcoords="offset points", alpha=0.8)
    plt.colorbar(sc, ax=ax, label="Survey LTBI Rate")
    ax.set_xlabel("PU-EM LTBI Estimate", fontsize=11)
    ax.set_ylabel("Survey LTBI Rate (2014-15 Ground Truth)", fontsize=11)
    ax.set_title("PU-EM Estimate vs Survey LTBI Rate\n30 Survey Districts",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=10); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out = OUTPUT_DIR / "05_puem_scatter_vs_survey.png"
    plt.savefig(out, dpi=300, bbox_inches="tight"); plt.close()
    print(f"  Saved: {out}")


def plot_choropleth(dist_prev):
    rng_geo = np.random.default_rng(99)
    lats = rng_geo.uniform(-1.5, 4.2, len(UGANDA_DISTRICTS))
    lons = rng_geo.uniform(29.5, 35.0, len(UGANDA_DISTRICTS))
    score_map = dict(zip(dist_prev["district"], dist_prev["puem_ltbi_estimate"]))
    scores = np.array([score_map.get(d, dist_prev["puem_ltbi_estimate"].mean())
                       for d in UGANDA_DISTRICTS])
    fig, ax = plt.subplots(figsize=(10, 12))
    sc = ax.scatter(lons, lats, c=scores, cmap="RdYlGn_r", s=120,
                    edgecolors="black", linewidths=0.3, alpha=0.85)
    plt.colorbar(sc, ax=ax, label="PU-EM LTBI Probability Score")
    ax.set_xlabel("Longitude", fontsize=11); ax.set_ylabel("Latitude", fontsize=11)
    ax.set_title("Uganda District LTBI Risk Map\nPU-EM Posterior Probability Score\n"
                 "(Proxy Choropleth — GIS coordinates approximate)",
                 fontsize=13, fontweight="bold")
    top10_idx = np.argsort(scores)[::-1][:10]
    for i in top10_idx:
        ax.annotate(UGANDA_DISTRICTS[i], (lons[i], lats[i]), fontsize=6,
                    xytext=(3,3), textcoords="offset points", color="darkred")
    ax.grid(True, alpha=0.2); plt.tight_layout()
    out = OUTPUT_DIR / "06_puem_choropleth.png"
    plt.savefig(out, dpi=300, bbox_inches="tight"); plt.close()
    print(f"  Saved: {out}")


def plot_budget_comparison(dist_prev, pop_df):
    q25, q50, q75 = np.percentile(dist_prev["puem_ltbi_estimate"], [25,50,75])
    def get_q(s):
        if s >= q75: return "Q4 (Highest Risk)"
        elif s >= q50: return "Q3"
        elif s >= q25: return "Q2"
        else: return "Q1 (Lowest Risk)"
    df = dist_prev.merge(pop_df, on="district", how="left")
    df["pop"] = df["pop"].fillna(df["pop"].mean())
    df["adult_pop"] = df["pop"] * 0.60
    df["quartile"]  = df["puem_ltbi_estimate"].apply(get_q)
    df["uniform_cost"]  = df["adult_pop"] * 0.30 * LTBI_PRIOR * IPT_COST_UGX
    cov_map = {"Q4 (Highest Risk)":0.80,"Q3":0.50,"Q2":0.20,"Q1 (Lowest Risk)":0.05}
    df["targeted_cost"] = df.apply(
        lambda r: r["adult_pop"] * cov_map[r["quartile"]] * r["puem_ltbi_estimate"] * IPT_COST_UGX,
        axis=1
    )
    q_sum = df.groupby("quartile")[["uniform_cost","targeted_cost"]].sum() / 1e9
    fig, ax = plt.subplots(figsize=(12, 7))
    x = np.arange(len(q_sum)); w = 0.35
    b1 = ax.bar(x-w/2, q_sum["uniform_cost"],  w, label="Uniform 30% Strategy",
                color="#1976D2", edgecolor="white")
    b2 = ax.bar(x+w/2, q_sum["targeted_cost"], w, label="PU-EM Guided Strategy",
                color="#D32F2F", edgecolor="white")
    ax.set_xticks(x); ax.set_xticklabels(q_sum.index, fontsize=10)
    ax.set_ylabel("IPT Procurement Cost (Billion UGX)", fontsize=11)
    ax.set_title("IPT Procurement Cost: Uniform vs PU-EM Guided Strategy\nBy District Risk Quartile",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    for bar in list(b1)+list(b2):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01,
                f"{bar.get_height():.2f}B", ha="center", va="bottom", fontsize=8)
    saving = q_sum["uniform_cost"].sum() - q_sum["targeted_cost"].sum()
    ax.text(0.98, 0.95,
            f"Total Saving: {saving:.2f}B UGX\n(${saving*1e9/UGX_PER_USD/1e6:.1f}M USD)",
            transform=ax.transAxes, ha="right", va="top", fontsize=11,
            bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))
    ax.grid(axis="y", alpha=0.3); plt.tight_layout()
    out = OUTPUT_DIR / "07_puem_budget_comparison.png"
    plt.savefig(out, dpi=300, bbox_inches="tight"); plt.close()
    print(f"  Saved: {out}")
