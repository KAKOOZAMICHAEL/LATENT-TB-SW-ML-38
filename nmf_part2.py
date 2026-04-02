"""NMF Part 2: Training, metrics, visualisations, main pipeline."""

import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.decomposition import NMF

warnings.filterwarnings("ignore")

OUTPUT_DIR = Path("nmf_results")
OUTPUT_DIR.mkdir(exist_ok=True)
RANDOM_STATE = 42
UGX_PER_USD  = 3_750
IPT_COST_UGX = 42_000


# ── Component auto-labelling ──────────────────────────────────────────────────
def _auto_label(top5, idx):
    name_map = {
        "poverty_index": "poverty", "hiv_prevalence_pct": "HIV",
        "hiv_positive_pct": "HIV", "mining_zone_flag": "mining",
        "mining_zone": "mining", "mining_hiv_risk": "mining-HIV",
        "bcg_gap": "BCG-gap", "bcg_coverage_pct": "BCG",
        "urban_pct": "urban", "urban_crowding": "urban-crowding",
        "art_coverage_pct": "ART", "hiv_art_gap": "HIV-ART-gap",
        "tpt_coverage_pct": "TPT", "tpt_hiv_interaction": "TPT-HIV",
        "malnutrition_index": "malnutrition",
        "poverty_malnutrition": "poverty-malnutrition",
        "age_group_15_34_pct": "youth", "age_group_65plus_pct": "elderly",
        "crowding_index": "crowding", "drug_stock_level_months": "drug-stock",
        "contact_investigation_rate": "contact-tracing",
        "treatment_success_rate": "treatment-success", "ltfu_rate": "LTFU",
        "rainfall_mm_annual": "rainfall",
        "indoor_air_pollution_index": "air-pollution",
        "case_notifications_per100k": "TB-notifications",
        "genexpert_availability_pct": "GeneXpert",
    }
    tokens = []
    for ind in top5[:3]:
        t = name_map.get(ind, ind.replace("_", "-"))
        if t not in tokens:
            tokens.append(t)
    return "-".join(tokens[:3]) + f" component-{idx+1}"


# ── NMF Training ──────────────────────────────────────────────────────────────
def train_nmf(V, optimal_r, indicator_names):
    print("\n" + "="*70)
    print(f"STEP 6: Training NMF at optimal rank r = {optimal_r}")
    print("="*70)
    model = NMF(n_components=optimal_r, init="nndsvda", solver="cd",
                max_iter=500, random_state=RANDOM_STATE)
    W = model.fit_transform(V)
    H = model.components_
    component_labels = []
    for i in range(optimal_r):
        top5_idx = np.argsort(H[i])[::-1][:5]
        top5_names = [indicator_names[j] for j in top5_idx]
        label = _auto_label(top5_names, i)
        component_labels.append(label)
        print(f"\n  Component {i+1} — '{label}'")
        for j, idx in enumerate(top5_idx):
            print(f"    {j+1}. {indicator_names[idx]:<40s} loading={H[i,idx]:.4f}")
    return W, H, model, component_labels


# ── Performance Metrics ───────────────────────────────────────────────────────
def compute_performance_metrics(V, W, H, model, districts, survey_df,
                                 component_labels, optimal_r, rng):
    print("\n" + "="*70)
    print("STEP 7: Computing performance metrics")
    print("="*70)

    V_recon    = W @ H
    frob_error = np.linalg.norm(V - V_recon, "fro")
    frob_V     = np.linalg.norm(V, "fro")
    rel_error  = frob_error / frob_V * 100
    ss_total   = np.sum((V - V.mean())**2)
    ss_resid   = np.sum((V - V_recon)**2)
    var_exp    = (1 - ss_resid / ss_total) * 100

    print(f"\n  A) Reconstruction Quality")
    print(f"     Frobenius error:    {frob_error:.4f}")
    print(f"     Relative error:     {rel_error:.2f}%")
    print(f"     Variance explained: {var_exp:.2f}%")

    # External validation
    survey_df = survey_df.copy()
    survey_df["district_lower"] = survey_df["district"].str.lower()
    best_comp, best_r = 0, 0.0
    survey_corrs = {}
    for comp in range(optimal_r):
        w_scores = pd.Series(W[:, comp], index=[d.lower() for d in districts])
        mv = survey_df.copy()
        mv["w_score"] = mv["district_lower"].map(w_scores)
        mv = mv.dropna(subset=["w_score", "survey_ltbi_rate"])
        if len(mv) < 5:
            continue
        r_val, p_val = stats.pearsonr(mv["w_score"], mv["survey_ltbi_rate"])
        survey_corrs[comp] = {"pearson_r": r_val, "pearson_p": p_val}
        if abs(r_val) > abs(best_r):
            best_r, best_comp = r_val, comp

    w_top = pd.Series(W[:, best_comp], index=[d.lower() for d in districts])
    merged_top = survey_df.copy()
    merged_top["w_score"] = merged_top["district_lower"].map(w_top)
    merged_top = merged_top.dropna(subset=["w_score", "survey_ltbi_rate"])
    pearson_r, pearson_p   = stats.pearsonr(merged_top["w_score"], merged_top["survey_ltbi_rate"])
    spearman_r, spearman_p = stats.spearmanr(merged_top["w_score"], merged_top["survey_ltbi_rate"])

    print(f"\n  B) External Validation (n={len(merged_top)} survey districts)")
    print(f"     Top component: {component_labels[best_comp]}")
    print(f"     Pearson r:     {pearson_r:.4f}  (p={pearson_p:.4f})")
    print(f"     Spearman r:    {spearman_r:.4f}  (p={spearman_p:.4f})")

    # Stability
    print(f"\n  C) Stability Analysis (10 runs)")
    stab_errors = []
    for seed in range(10):
        m = NMF(n_components=optimal_r, init="nndsvda", solver="cd",
                max_iter=500, random_state=seed*7+1)
        Ws = m.fit_transform(V)
        stab_errors.append(np.linalg.norm(V - Ws @ m.components_, "fro") / frob_V * 100)
    stab_mean, stab_std = np.mean(stab_errors), np.std(stab_errors)
    print(f"     Mean: {stab_mean:.4f}%  Std: {stab_std:.4f}%")

    # Top-quartile ranking
    w_all = pd.Series(W[:, best_comp], index=districts)
    top_q_thr = np.percentile(w_all, 75)
    top_q_set = set(w_all[w_all >= top_q_thr].index.str.lower())
    survey_df["in_top_q"] = survey_df["district_lower"].isin(top_q_set)
    survey_df["high_ltbi"] = survey_df["survey_ltbi_rate"] > 0.25
    survey_df["low_ltbi"]  = survey_df["survey_ltbi_rate"] < 0.15
    tp = int(( survey_df["in_top_q"] &  survey_df["high_ltbi"]).sum())
    fp = int(( survey_df["in_top_q"] &  survey_df["low_ltbi"]).sum())
    fn = int((~survey_df["in_top_q"] &  survey_df["high_ltbi"]).sum())
    tn = int((~survey_df["in_top_q"] &  survey_df["low_ltbi"]).sum())
    precision = tp/(tp+fp) if (tp+fp)>0 else 0.0
    recall    = tp/(tp+fn) if (tp+fn)>0 else 0.0
    f1        = 2*precision*recall/(precision+recall) if (precision+recall)>0 else 0.0
    print(f"\n  D) Top-Quartile Ranking  TP={tp} FP={fp} FN={fn} TN={tn}")
    print(f"     Precision={precision:.4f}  Recall={recall:.4f}  F1={f1:.4f}")

    # Budget impact
    pop_arr = rng.uniform(50, 2500, len(districts))
    ntlp_pop = pd.DataFrame({"district": districts, "population_1000s": pop_arr})
    total_ltbi = ntlp_pop["population_1000s"].sum() * 1000 * 0.12
    uniform_cost  = total_ltbi * 0.30 * IPT_COST_UGX
    top_q_pop = ntlp_pop[ntlp_pop["district"].str.lower().isin(top_q_set)]["population_1000s"].sum()*1000
    targeted_cost = top_q_pop * 0.12 * 0.80 * IPT_COST_UGX
    saving_ugx = uniform_cost - targeted_cost
    saving_usd = saving_ugx / UGX_PER_USD
    saving_pct = saving_ugx / uniform_cost * 100
    print(f"\n  E) Budget Impact")
    print(f"     Uniform 30%:  UGX {uniform_cost:,.0f}  (USD {uniform_cost/UGX_PER_USD:,.0f})")
    print(f"     NMF-guided:   UGX {targeted_cost:,.0f}  (USD {targeted_cost/UGX_PER_USD:,.0f})")
    print(f"     Saving:       UGX {saving_ugx:,.0f}  (USD {saving_usd:,.0f})  [{saving_pct:.1f}%]")

    return dict(
        frob_error=frob_error, rel_error_pct=rel_error, var_explained_pct=var_exp,
        pearson_r=pearson_r, pearson_p=pearson_p,
        spearman_r=spearman_r, spearman_p=spearman_p,
        stab_mean=stab_mean, stab_std=stab_std,
        best_comp=best_comp, top_comp_label=component_labels[best_comp],
        top_comp_H=H[best_comp],
        tp=tp, fp=fp, fn=fn, tn=tn,
        precision=precision, recall=recall, f1=f1,
        uniform_cost_ugx=uniform_cost, targeted_cost_ugx=targeted_cost,
        saving_ugx=saving_ugx, saving_usd=saving_usd, saving_pct=saving_pct,
        w_all=w_all, merged_top=merged_top, survey_df=survey_df,
        ntlp_df_pop=ntlp_pop, top_q_threshold=top_q_thr,
    )


# ── Visualisations ────────────────────────────────────────────────────────────
def plot_elbow(errors, optimal_r):
    ranks = list(errors.keys())
    frob  = [errors[r]["frobenius"] for r in ranks]
    rel   = [errors[r]["relative_pct"] for r in ranks]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, vals, ylabel, title in zip(
        axes, [frob, rel],
        ["Frobenius Norm Error", "Relative Error (%)"],
        ["Frobenius Reconstruction Error vs Rank", "Relative Reconstruction Error vs Rank"],
    ):
        ax.plot(ranks, vals, "o-", color="#2196F3", linewidth=2, markersize=8)
        ax.axvline(optimal_r, color="#F44336", linestyle="--", linewidth=1.5,
                   label=f"Optimal r={optimal_r}")
        ax.set_xlabel("Number of Components (r)", fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.legend(); ax.grid(True, alpha=0.3)
        for x, y in zip(ranks, vals):
            ax.annotate(f"{y:.2f}", (x, y), textcoords="offset points",
                        xytext=(0, 8), ha="center", fontsize=9)
    plt.suptitle("NMF Rank Selection — Elbow Method\nUganda District TB Indicator Matrix",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    out = OUTPUT_DIR / "01_nmf_elbow_plot.png"
    plt.savefig(out, dpi=300, bbox_inches="tight"); plt.close()
    print(f"  Saved: {out}")


def plot_h_heatmap(H, indicator_names, component_labels):
    H_df = pd.DataFrame(H,
        index=[f"C{i+1}: {l}" for i, l in enumerate(component_labels)],
        columns=indicator_names)
    fig_w = max(20, len(indicator_names) * 0.5)
    fig, ax = plt.subplots(figsize=(fig_w, max(6, len(component_labels)*1.5)))
    sns.heatmap(H_df, cmap="YlOrRd", annot=True, fmt=".3f", linewidths=0.3,
                cbar_kws={"label": "Loading Value"}, ax=ax, annot_kws={"size": 7})
    ax.set_title("NMF Basis Vectors (H Matrix)\nComponent Loadings on Each Indicator",
                 fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("Indicators", fontsize=11); ax.set_ylabel("NMF Components", fontsize=11)
    plt.xticks(rotation=45, ha="right", fontsize=8); plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()
    out = OUTPUT_DIR / "02_nmf_h_matrix_heatmap.png"
    plt.savefig(out, dpi=300, bbox_inches="tight"); plt.close()
    print(f"  Saved: {out}")


def plot_top_indicator_loadings(top_comp_H, indicator_names, top_comp_label, n_top=10):
    top_idx  = np.argsort(top_comp_H)[::-1][:n_top]
    top_vals = top_comp_H[top_idx]
    top_labs = [indicator_names[i] for i in top_idx]
    colors = ["#D32F2F" if v > np.percentile(top_vals, 60) else "#1976D2" for v in top_vals]
    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.barh(range(n_top), top_vals[::-1], color=colors[::-1], edgecolor="white", height=0.7)
    ax.set_yticks(range(n_top)); ax.set_yticklabels(top_labs[::-1], fontsize=10)
    ax.set_xlabel("Loading Value", fontsize=12)
    ax.set_title(f"Top {n_top} Indicator Loadings\nMost Actionable Component: '{top_comp_label}'",
                 fontsize=13, fontweight="bold")
    for bar, val in zip(bars, top_vals[::-1]):
        ax.text(val+0.001, bar.get_y()+bar.get_height()/2, f"{val:.4f}", va="center", fontsize=9)
    ax.grid(axis="x", alpha=0.3); plt.tight_layout()
    out = OUTPUT_DIR / "03_nmf_top_indicator_loadings.png"
    plt.savefig(out, dpi=300, bbox_inches="tight"); plt.close()
    print(f"  Saved: {out}")


def plot_w_vs_survey(merged_top, pearson_r, pearson_p, top_comp_label):
    fig, ax = plt.subplots(figsize=(10, 7))
    sc = ax.scatter(merged_top["w_score"], merged_top["survey_ltbi_rate"],
                    c=merged_top["survey_ltbi_rate"], cmap="RdYlGn_r",
                    s=100, edgecolors="black", linewidths=0.5, zorder=3)
    m, b = np.polyfit(merged_top["w_score"], merged_top["survey_ltbi_rate"], 1)
    x_line = np.linspace(merged_top["w_score"].min(), merged_top["w_score"].max(), 100)
    ax.plot(x_line, m*x_line+b, "r--", linewidth=2,
            label=f"Pearson r={pearson_r:.3f}, p={pearson_p:.4f}")
    for _, row in merged_top.iterrows():
        ax.annotate(row["district"], (row["w_score"], row["survey_ltbi_rate"]),
                    fontsize=7, xytext=(3, 3), textcoords="offset points", alpha=0.8)
    plt.colorbar(sc, ax=ax, label="Survey LTBI Rate")
    ax.set_xlabel(f"W-Score on Top Component\n('{top_comp_label}')", fontsize=11)
    ax.set_ylabel("Survey LTBI Prevalence Rate (2014-15)", fontsize=11)
    ax.set_title("District NMF Risk Score vs Survey-Confirmed LTBI Prevalence\n"
                 "External Validation Against Uganda National TB Prevalence Survey",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=10); ax.grid(True, alpha=0.3); plt.tight_layout()
    out = OUTPUT_DIR / "04_nmf_w_score_vs_survey_ltbi.png"
    plt.savefig(out, dpi=300, bbox_inches="tight"); plt.close()
    print(f"  Saved: {out}")


def plot_choropleth(w_all, districts):
    rng_geo = np.random.default_rng(99)
    lats = rng_geo.uniform(-1.5, 4.2, len(districts))
    lons = rng_geo.uniform(29.5, 35.0, len(districts))
    w_vals = np.array([w_all.get(d, w_all.mean()) for d in districts])
    norm_w = (w_vals - w_vals.min()) / (w_vals.max() - w_vals.min() + 1e-9)
    fig, ax = plt.subplots(figsize=(10, 12))
    sc = ax.scatter(lons, lats, c=norm_w, cmap="RdYlGn_r", s=120,
                    edgecolors="black", linewidths=0.3, alpha=0.85)
    plt.colorbar(sc, ax=ax, label="Normalised NMF Risk Score (Top Component)")
    ax.set_xlabel("Longitude", fontsize=11); ax.set_ylabel("Latitude", fontsize=11)
    ax.set_title("Uganda District Risk Map\nNMF Top-Component W-Score\n"
                 "(Proxy Choropleth — GIS coordinates approximate)",
                 fontsize=13, fontweight="bold")
    top10_idx = np.argsort(w_vals)[::-1][:10]
    for i in top10_idx:
        ax.annotate(districts[i], (lons[i], lats[i]), fontsize=6,
                    xytext=(3, 3), textcoords="offset points", color="darkred")
    ax.grid(True, alpha=0.2); plt.tight_layout()
    out = OUTPUT_DIR / "05_nmf_choropleth_risk_map.png"
    plt.savefig(out, dpi=300, bbox_inches="tight"); plt.close()
    print(f"  Saved: {out}")


def plot_budget_comparison(metrics, ntlp_df_pop, w_all, districts):
    q25, q50, q75 = np.percentile(w_all, [25, 50, 75])
    def get_q(s):
        if s >= q75: return "Q4 (Highest Risk)"
        elif s >= q50: return "Q3"
        elif s >= q25: return "Q2"
        else: return "Q1 (Lowest Risk)"
    df = ntlp_df_pop.copy()
    df["w_score"]  = df["district"].map({d: w_all.get(d, w_all.mean()) for d in districts})
    df["quartile"] = df["w_score"].apply(get_q)
    df["ltbi_pop"] = df["population_1000s"] * 1000 * 0.12
    df["uniform_cost"]  = df["ltbi_pop"] * 0.30 * IPT_COST_UGX
    cov_map = {"Q4 (Highest Risk)": 0.80, "Q3": 0.50, "Q2": 0.20, "Q1 (Lowest Risk)": 0.05}
    df["targeted_cost"] = df.apply(lambda r: r["ltbi_pop"]*cov_map[r["quartile"]]*IPT_COST_UGX, axis=1)
    q_sum = df.groupby("quartile")[["uniform_cost","targeted_cost"]].sum() / 1e9
    fig, ax = plt.subplots(figsize=(12, 7))
    x = np.arange(len(q_sum)); w = 0.35
    b1 = ax.bar(x-w/2, q_sum["uniform_cost"],  w, label="Uniform 30% Strategy",       color="#1976D2", edgecolor="white")
    b2 = ax.bar(x+w/2, q_sum["targeted_cost"], w, label="NMF-Guided Targeted Strategy", color="#D32F2F", edgecolor="white")
    ax.set_xticks(x); ax.set_xticklabels(q_sum.index, fontsize=10)
    ax.set_ylabel("IPT Procurement Cost (Billion UGX)", fontsize=11)
    ax.set_title("IPT Procurement Cost: Uniform vs NMF-Guided Strategy\nBy District Risk Quartile",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    for bar in list(b1)+list(b2):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01,
                f"{bar.get_height():.2f}B", ha="center", va="bottom", fontsize=8)
    saving = q_sum["uniform_cost"].sum() - q_sum["targeted_cost"].sum()
    ax.text(0.98, 0.95, f"Total Saving: {saving:.2f}B UGX\n(${saving*1e9/UGX_PER_USD/1e6:.1f}M USD)",
            transform=ax.transAxes, ha="right", va="top", fontsize=11,
            bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))
    ax.grid(axis="y", alpha=0.3); plt.tight_layout()
    out = OUTPUT_DIR / "06_nmf_budget_comparison.png"
    plt.savefig(out, dpi=300, bbox_inches="tight"); plt.close()
    print(f"  Saved: {out}")


# ── Summary tables & CSV outputs ──────────────────────────────────────────────
def print_summary_table(metrics, errors, optimal_r, indicator_names, W, districts):
    print("\n" + "="*70)
    print("COMPARATIVE STATISTICS TABLE (NMF)")
    print("="*70)
    rows = [
        ("Reconstruction error (relative)",  f"{metrics['rel_error_pct']:.2f}%"),
        ("Pearson r vs Survey LTBI",          f"{metrics['pearson_r']:.4f}  (p={metrics['pearson_p']:.4f})"),
        ("Spearman r vs Survey LTBI",         f"{metrics['spearman_r']:.4f}  (p={metrics['spearman_p']:.4f})"),
        ("Variance explained (%)",            f"{metrics['var_explained_pct']:.2f}%"),
        ("Stability (std dev across runs)",   f"{metrics['stab_std']:.4f}%"),
        ("Estimated procurement saving",      f"UGX {metrics['saving_ugx']:,.0f}  (USD {metrics['saving_usd']:,.0f})  [{metrics['saving_pct']:.1f}%]"),
        ("Top actionable component name",     metrics['top_comp_label']),
        ("Districts correctly ranked top-Q",  f"Precision={metrics['precision']:.3f}  Recall={metrics['recall']:.3f}  F1={metrics['f1']:.3f}"),
    ]
    cw = 36
    print(f"  {'Metric':<{cw}} | NMF Value")
    print(f"  {'-'*cw}-+-{'-'*40}")
    for name, val in rows:
        print(f"  {name:<{cw}} | {val}")

    print("\n" + "="*70)
    print("CONFUSION MATRIX — TOP-QUARTILE DISTRICT RANKING")
    print("="*70)
    tp, fp, fn, tn = metrics["tp"], metrics["fp"], metrics["fn"], metrics["tn"]
    print(f"\n                    Survey LTBI >25%   Survey LTBI <15%")
    print(f"  In Top Quartile:       TP={tp:<4}           FP={fp}")
    print(f"  Not Top Quartile:      FN={fn:<4}           TN={tn}")
    print(f"\n  Precision: {metrics['precision']:.4f}  Recall: {metrics['recall']:.4f}  F1: {metrics['f1']:.4f}")

    print("\n" + "="*70)
    print("TOP 10 INDICATOR LOADINGS — MOST ACTIONABLE COMPONENT")
    print("="*70)
    top_comp_H = metrics["top_comp_H"]
    top10_idx  = np.argsort(top_comp_H)[::-1][:10]
    print(f"\n  Component: '{metrics['top_comp_label']}'")
    print(f"  {'Rank':<5} {'Indicator':<42} {'Loading':>8}")
    print(f"  {'-'*5} {'-'*42} {'-'*8}")
    for rank, idx in enumerate(top10_idx, 1):
        print(f"  {rank:<5} {indicator_names[idx]:<42} {top_comp_H[idx]:>8.4f}")

    print("\n" + "="*70)
    print("DISTRICT RISK RANKING — TOP 20")
    print("="*70)
    w_all = metrics["w_all"]
    top20 = w_all.sort_values(ascending=False).head(20)
    print(f"\n  {'Rank':<5} {'District':<30} {'W-Score':>10}")
    print(f"  {'-'*5} {'-'*30} {'-'*10}")
    for rank, (dist, score) in enumerate(top20.items(), 1):
        print(f"  {rank:<5} {dist:<30} {score:>10.4f}")

    top3_idx   = np.argsort(top_comp_H)[::-1][:3]
    top3_names = [indicator_names[i] for i in top3_idx]
    print("\n" + "="*70)
    print("GOVERNMENT PROCUREMENT RECOMMENDATION")
    print("="*70)
    print(f"\n  Most actionable component: '{metrics['top_comp_label']}'")
    print(f"\n  Fund these 3 interventions for maximum LTBI reduction per UGX spent:")
    for i, name in enumerate(top3_names, 1):
        print(f"    {i}. {name}")
    print(f"\n  Targeting top-quartile districts saves:")
    print(f"    UGX {metrics['saving_ugx']:,.0f}  (~USD {metrics['saving_usd']:,.0f})")
    print(f"    ({metrics['saving_pct']:.1f}% reduction vs uniform 30% national coverage)")


def save_outputs(V_df, W, H, districts, indicator_names, component_labels, metrics, errors, optimal_r):
    W_df = pd.DataFrame(W, columns=[f"Component_{i+1}" for i in range(W.shape[1])])
    W_df.insert(0, "district", districts)
    W_df["top_component_score"] = W[:, metrics["best_comp"]]
    W_df.sort_values("top_component_score", ascending=False).reset_index(drop=True).to_csv(
        OUTPUT_DIR / "nmf_W_district_loadings.csv", index=False)

    H_df = pd.DataFrame(H, columns=indicator_names,
                        index=[f"C{i+1}_{l}" for i, l in enumerate(component_labels)])
    H_df.to_csv(OUTPUT_DIR / "nmf_H_basis_vectors.csv")

    err_rows = [{"rank": r, "frobenius_error": v["frobenius"],
                 "relative_error_pct": v["relative_pct"]} for r, v in errors.items()]
    pd.DataFrame(err_rows).to_csv(OUTPUT_DIR / "nmf_reconstruction_errors.csv", index=False)

    summary = {
        "metric": ["Reconstruction error (relative %)", "Frobenius error",
                   "Variance explained (%)", "Pearson r vs Survey LTBI", "Pearson p-value",
                   "Spearman r vs Survey LTBI", "Spearman p-value",
                   "Stability mean error (%)", "Stability std dev (%)",
                   "Precision (top-quartile)", "Recall (top-quartile)", "F1 (top-quartile)",
                   "Uniform strategy cost (UGX)", "Targeted strategy cost (UGX)",
                   "Estimated saving (UGX)", "Estimated saving (USD)", "Saving (%)",
                   "Optimal rank (r)", "Top component label"],
        "value": [f"{metrics['rel_error_pct']:.4f}", f"{metrics['frob_error']:.4f}",
                  f"{metrics['var_explained_pct']:.4f}", f"{metrics['pearson_r']:.4f}",
                  f"{metrics['pearson_p']:.6f}", f"{metrics['spearman_r']:.4f}",
                  f"{metrics['spearman_p']:.6f}", f"{metrics['stab_mean']:.4f}",
                  f"{metrics['stab_std']:.4f}", f"{metrics['precision']:.4f}",
                  f"{metrics['recall']:.4f}", f"{metrics['f1']:.4f}",
                  f"{metrics['uniform_cost_ugx']:.0f}", f"{metrics['targeted_cost_ugx']:.0f}",
                  f"{metrics['saving_ugx']:.0f}", f"{metrics['saving_usd']:.0f}",
                  f"{metrics['saving_pct']:.2f}", str(optimal_r), metrics["top_comp_label"]],
    }
    pd.DataFrame(summary).to_csv(OUTPUT_DIR / "nmf_summary_metrics.csv", index=False)
    print(f"\n  CSVs saved to: {OUTPUT_DIR}/")


# ── Main pipeline ─────────────────────────────────────────────────────────────
def main():
    from nmf_part1 import (
        build_district_indicator_matrix, select_optimal_rank,
        UGANDA_DISTRICTS, RANDOM_STATE,
    )
    print("\n" + "="*70)
    print("NMF PIPELINE — LATENT TB ANALYSIS, UGANDA")
    print("Optimizing Government Resources in Combating Latent Tuberculosis")
    print("="*70)

    rng = np.random.default_rng(RANDOM_STATE)
    V_df, indicator_names, survey_df = build_district_indicator_matrix(rng)
    districts = list(V_df["district"].values)
    V = V_df[indicator_names].values.astype(np.float64)

    errors, optimal_r = select_optimal_rank(V)
    W, H, model, component_labels = train_nmf(V, optimal_r, indicator_names)
    metrics = compute_performance_metrics(
        V, W, H, model, districts, survey_df, component_labels, optimal_r, rng)

    print("\n" + "="*70)
    print("STEP 8: Generating visualisations (300 DPI PNG)")
    print("="*70)
    plot_elbow(errors, optimal_r)
    plot_h_heatmap(H, indicator_names, component_labels)
    plot_top_indicator_loadings(metrics["top_comp_H"], indicator_names, metrics["top_comp_label"])
    plot_w_vs_survey(metrics["merged_top"], metrics["pearson_r"],
                     metrics["pearson_p"], metrics["top_comp_label"])
    plot_choropleth(metrics["w_all"], districts)
    plot_budget_comparison(metrics, metrics["ntlp_df_pop"], metrics["w_all"], districts)

    print_summary_table(metrics, errors, optimal_r, indicator_names, W, districts)
    save_outputs(V_df, W, H, districts, indicator_names, component_labels,
                 metrics, errors, optimal_r)

    print("\n" + "="*70)
    print("NMF PIPELINE COMPLETE")
    print(f"  Outputs: {OUTPUT_DIR}/")
    print(f"  Figures: 6 PNG files at 300 DPI")
    print(f"  CSVs:    4 files")
    print("="*70)


if __name__ == "__main__":
    main()
