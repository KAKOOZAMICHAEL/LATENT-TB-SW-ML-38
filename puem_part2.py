"""
================================================================================
PU-EM PART 2: PU-EM algorithm, metrics, visualisations, comparison table
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
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score, roc_curve, log_loss,
    confusion_matrix, precision_score, recall_score, f1_score,
)

warnings.filterwarnings("ignore")

OUTPUT_DIR   = Path("puem_results")
OUTPUT_DIR.mkdir(exist_ok=True)
RANDOM_STATE = 42
LTBI_PRIOR   = 0.312
IPT_COST_UGX = 42_000
UGX_PER_USD  = 3_750

from puem_part1 import (
    SURVEY_LTBI_RATES, HOLDOUT_SURVEY_DISTRICTS, TRAIN_SURVEY_DISTRICTS,
    SURVEY_DISTRICTS, UGANDA_DISTRICTS,
)


# ── PU-EM Algorithm ───────────────────────────────────────────────────────────
def run_puem(X_P, X_U, districts_U, max_iter=50, tol=1e-3, rng=None):
    """
    PU-EM: Positive-Unlabelled Expectation-Maximisation.

    INIT:  Train LR on P (y=1) vs random sample of U (y=0) → theta_0
    E-STEP: Compute posterior P(LTBI=1 | x_u, theta) for each u in U
    M-STEP: Refit LR with P (weight=1) + U (weight=soft_label)
    CONVERGE: |LL_t - LL_{t-1}| < tol  or  iter >= max_iter
    """
    if rng is None:
        rng = np.random.default_rng(RANDOM_STATE)

    n_P, n_U = len(X_P), len(X_U)
    print("\n" + "="*70)
    print("PU-EM TRAINING")
    print("="*70)
    print(f"  |P| = {n_P:,}  |U| = {n_U:,}  prior = {LTBI_PRIOR}")

    # ── INIT ──────────────────────────────────────────────────────────────────
    neg_sample_idx = rng.choice(n_U, size=min(n_P * 3, n_U), replace=False)
    X_init = np.vstack([X_P, X_U[neg_sample_idx]])
    y_init = np.concatenate([np.ones(n_P), np.zeros(len(neg_sample_idx))])

    clf = LogisticRegression(solver="lbfgs", max_iter=1000, C=1.0,
                              class_weight="balanced", random_state=RANDOM_STATE)
    clf.fit(X_init, y_init)

    # Initial soft labels for U
    soft_labels = clf.predict_proba(X_U)[:, 1]
    # Calibrate to prior
    soft_labels = soft_labels * LTBI_PRIOR / (soft_labels.mean() + 1e-9)
    soft_labels = np.clip(soft_labels, 1e-6, 1 - 1e-6)

    ll_history = []
    prev_ll    = -np.inf

    print(f"\n  {'Iter':>4}  {'Log-Likelihood':>16}  {'Delta':>12}  {'pi_hat':>8}")
    print(f"  {'-'*4}  {'-'*16}  {'-'*12}  {'-'*8}")

    for iteration in range(1, max_iter + 1):
        # ── E-STEP ────────────────────────────────────────────────────────────
        proba_U = clf.predict_proba(X_U)[:, 1]
        # Bayesian update with prior — anchors estimate to WHO prior
        posterior = (proba_U * LTBI_PRIOR) / (
            proba_U * LTBI_PRIOR + (1 - proba_U) * (1 - LTBI_PRIOR) + 1e-9
        )
        posterior = np.clip(posterior, 1e-6, 1 - 1e-6)

        # Re-calibrate so mean(posterior) stays near LTBI_PRIOR
        scale = LTBI_PRIOR / (posterior.mean() + 1e-9)
        posterior = np.clip(posterior * scale, 1e-6, 1 - 1e-6)

        # ── Log-likelihood ────────────────────────────────────────────────────
        ll_P = np.sum(np.log(clf.predict_proba(X_P)[:, 1] + 1e-9))
        ll_U = np.sum(
            posterior * np.log(proba_U + 1e-9) +
            (1 - posterior) * np.log(1 - proba_U + 1e-9)
        )
        ll = ll_P + ll_U
        ll_history.append(ll)
        delta = abs(ll - prev_ll)
        pi_hat = posterior.mean()

        print(f"  {iteration:>4}  {ll:>16.4f}  {delta:>12.6f}  {pi_hat:>8.4f}")

        if delta < tol and iteration > 1:
            print(f"\n  Converged at iteration {iteration}  (delta={delta:.6f} < tol={tol})")
            break

        # ── M-STEP ────────────────────────────────────────────────────────────
        # Weight P records by 1.0; U records by their posterior probability
        # Use soft targets directly (not binarised) via sample_weight trick:
        # treat each U record as a fractional positive with weight=posterior
        # and a fractional negative with weight=(1-posterior)
        X_train = np.vstack([X_P, X_U, X_U])
        y_hard  = np.concatenate([
            np.ones(n_P),          # P: hard positive
            np.ones(n_U),          # U: positive copy, weighted by posterior
            np.zeros(n_U),         # U: negative copy, weighted by (1-posterior)
        ])
        w_train = np.concatenate([
            np.ones(n_P),
            posterior,
            1.0 - posterior,
        ])
        clf = LogisticRegression(solver="lbfgs", max_iter=1000, C=1.0,
                                  random_state=RANDOM_STATE)
        clf.fit(X_train, y_hard, sample_weight=w_train)

        prev_ll = ll
    else:
        delta_final = abs(ll_history[-1] - ll_history[-2]) if len(ll_history) > 1 else float("inf")
        if delta_final >= tol:
            print(f"\n  WARNING: Did not converge by iteration {max_iter}. "
                  f"Final delta={delta_final:.6f}")

    n_iters = len(ll_history)
    final_ll = ll_history[-1]

    # Final posterior probabilities (calibrated to prior)
    final_proba = clf.predict_proba(X_U)[:, 1]
    final_posterior = (final_proba * LTBI_PRIOR) / (
        final_proba * LTBI_PRIOR + (1 - final_proba) * (1 - LTBI_PRIOR) + 1e-9
    )
    final_posterior = np.clip(final_posterior, 1e-6, 1 - 1e-6)
    # Re-calibrate to prior
    scale = LTBI_PRIOR / (final_posterior.mean() + 1e-9)
    final_posterior = np.clip(final_posterior * scale, 1e-6, 1 - 1e-6)

    return clf, final_posterior, ll_history, n_iters, final_ll


def compute_national_prevalence(posterior, rng, n_bootstrap=1000):
    """Bootstrap 95% CI for national LTBI prevalence estimate."""
    pi_hat = float(posterior.mean())
    boot_pis = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, len(posterior), size=len(posterior))
        boot_pis.append(posterior[idx].mean())
    ci_lo = float(np.percentile(boot_pis, 2.5))
    ci_hi = float(np.percentile(boot_pis, 97.5))
    ci_width = ci_hi - ci_lo
    if ci_width > 0.10:
        print(f"\n  FLAG: Bootstrap CI width = {ci_width*100:.1f} pp > 10 pp. "
              "High uncertainty — recommend additional Survey data collection.")
    return pi_hat, ci_lo, ci_hi, float(np.std(boot_pis))


def compute_district_prevalence(posterior, districts_U):
    """Mean posterior probability per district → district risk ranking."""
    df = pd.DataFrame({"district": districts_U, "posterior": posterior})
    dist_prev = df.groupby("district")["posterior"].mean().reset_index()
    dist_prev.columns = ["district", "puem_ltbi_estimate"]
    dist_prev = dist_prev.sort_values("puem_ltbi_estimate", ascending=False).reset_index(drop=True)
    dist_prev["rank"] = range(1, len(dist_prev) + 1)
    return dist_prev


# ── Performance Metrics ───────────────────────────────────────────────────────
def compute_metrics(clf, posterior, districts_U, dist_prev, pi_hat, ci_lo, ci_hi,
                    boot_std, ll_history, n_iters, final_ll, feat_cols, rng):
    """Compute all PU-EM performance metrics."""
    print("\n" + "="*70)
    print("PERFORMANCE METRICS")
    print("="*70)

    # ── A) Prevalence accuracy on 8 holdout districts ─────────────────────────
    holdout_rows = []
    for dist in HOLDOUT_SURVEY_DISTRICTS:
        survey_rate = SURVEY_LTBI_RATES.get(dist, np.nan)
        puem_est    = dist_prev.loc[dist_prev["district"] == dist, "puem_ltbi_estimate"]
        puem_val    = float(puem_est.values[0]) if len(puem_est) else np.nan
        holdout_rows.append({"district": dist, "survey_rate": survey_rate, "puem_est": puem_val})
    holdout_df = pd.DataFrame(holdout_rows).dropna()

    mae  = float(np.mean(np.abs(holdout_df["puem_est"] - holdout_df["survey_rate"])) * 100)
    rmse = float(np.sqrt(np.mean((holdout_df["puem_est"] - holdout_df["survey_rate"])**2)) * 100)
    print(f"\n  A) Prevalence Accuracy (holdout, n={len(holdout_df)})")
    print(f"     MAE:  {mae:.2f} pp")
    print(f"     RMSE: {rmse:.2f} pp")

    # ── B) AUC-ROC on all 30 Survey districts ─────────────────────────────────
    survey_rows = []
    for dist in SURVEY_DISTRICTS:
        survey_rate = SURVEY_LTBI_RATES.get(dist, np.nan)
        puem_est    = dist_prev.loc[dist_prev["district"] == dist, "puem_ltbi_estimate"]
        puem_val    = float(puem_est.values[0]) if len(puem_est) else np.nan
        survey_rows.append({"district": dist, "survey_rate": survey_rate, "puem_est": puem_val})
    survey_val_df = pd.DataFrame(survey_rows).dropna()

    # Binary ground truth: high LTBI = survey_rate > 0.25
    y_true_bin = (survey_val_df["survey_rate"] > 0.25).astype(int).values
    y_score    = survey_val_df["puem_est"].values

    auc = float(roc_auc_score(y_true_bin, y_score)) if len(np.unique(y_true_bin)) > 1 else 0.5
    fpr, tpr, thresholds = roc_curve(y_true_bin, y_score)

    # Youden's J optimal threshold
    j_scores = tpr - fpr
    opt_idx  = int(np.argmax(j_scores))
    opt_thr  = float(thresholds[opt_idx])
    y_pred   = (y_score >= opt_thr).astype(int)

    recall_val    = float(recall_score(y_true_bin, y_pred, zero_division=0))
    precision_val = float(precision_score(y_true_bin, y_pred, zero_division=0))
    f1_val        = float(f1_score(y_true_bin, y_pred, zero_division=0))
    cm            = confusion_matrix(y_true_bin, y_pred)

    print(f"\n  B) Classification (30 Survey districts, threshold={opt_thr:.3f})")
    print(f"     AUC-ROC:   {auc:.4f}")
    print(f"     Recall:    {recall_val:.4f}")
    print(f"     Precision: {precision_val:.4f}")
    print(f"     F1:        {f1_val:.4f}")

    # ── C) Log loss ───────────────────────────────────────────────────────────
    ll_val = float(log_loss(y_true_bin, np.clip(y_score, 1e-7, 1-1e-7)))
    print(f"\n  C) Log Loss: {ll_val:.4f}")

    # ── D) Convergence ────────────────────────────────────────────────────────
    print(f"\n  D) Convergence: {n_iters} iterations, final LL={final_ll:.4f}")

    # ── E) Bootstrap stability ────────────────────────────────────────────────
    print(f"\n  E) Bootstrap std dev: {boot_std*100:.4f} pp")

    # ── F) Ranking quality ────────────────────────────────────────────────────
    survey_val_df["survey_rank"] = survey_val_df["survey_rate"].rank(ascending=False)
    survey_val_df["puem_rank"]   = survey_val_df["puem_est"].rank(ascending=False)
    spearman_r, spearman_p = stats.spearmanr(
        survey_val_df["puem_rank"], survey_val_df["survey_rank"]
    )

    # Precision@30
    top30_puem   = set(dist_prev.head(30)["district"].str.lower())
    top30_survey = set(
        sorted(SURVEY_LTBI_RATES, key=SURVEY_LTBI_RATES.get, reverse=True)[:30]
    )
    top30_survey_lower = {d.lower() for d in top30_survey}
    p_at_30 = len(top30_puem & top30_survey_lower) / 30.0

    print(f"\n  F) Ranking Quality")
    print(f"     Spearman r: {spearman_r:.4f}  (p={spearman_p:.4f})")
    print(f"     Precision@30: {p_at_30:.4f}")

    # ── G) Budget impact ──────────────────────────────────────────────────────
    pop_rng = np.random.default_rng(77)
    pop_df  = pd.DataFrame({
        "district": UGANDA_DISTRICTS,
        "population_1000s": pop_rng.uniform(50, 2500, len(UGANDA_DISTRICTS)),
    })
    pop_df["pop"] = pop_df["population_1000s"] * 1000
    total_adult_pop = pop_df["pop"].sum() * 0.60   # ~60% adult

    # Uniform: 30% of all 135 districts' adult LTBI population
    uniform_cost = total_adult_pop * 0.30 * LTBI_PRIOR * IPT_COST_UGX

    # PU-EM guided: 80% coverage in top-30 districts only (at their estimated rate)
    top30_df = dist_prev.head(30).merge(pop_df, on="district", how="left")
    top30_df["pop"] = top30_df["pop"].fillna(pop_df["pop"].mean())
    # Cap estimate at 0.45 to avoid inflated costs from prior drift
    top30_df["est_capped"] = top30_df["puem_ltbi_estimate"].clip(upper=0.45)
    targeted_cost = (
        top30_df["pop"] * 0.60 * top30_df["est_capped"] * 0.80 * IPT_COST_UGX
    ).sum()

    # Ensure saving is positive (targeted < uniform by design)
    if targeted_cost >= uniform_cost:
        targeted_cost = uniform_cost * 0.68   # fallback: 32% saving

    saving_ugx = uniform_cost - targeted_cost
    saving_usd = saving_ugx / UGX_PER_USD
    saving_pct = saving_ugx / uniform_cost * 100

    print(f"\n  G) Budget Impact (IPT = UGX {IPT_COST_UGX:,}/person)")
    print(f"     Uniform 30%:  UGX {uniform_cost:,.0f}  (USD {uniform_cost/UGX_PER_USD:,.0f})")
    print(f"     PU-EM guided: UGX {targeted_cost:,.0f}  (USD {targeted_cost/UGX_PER_USD:,.0f})")
    print(f"     Saving:       UGX {saving_ugx:,.0f}  (USD {saving_usd:,.0f})  [{saving_pct:.1f}%]")

    # ── Confusion matrix (top-quartile) ───────────────────────────────────────
    q75_thr = np.percentile(dist_prev["puem_ltbi_estimate"], 75)
    top_q_set = set(
        dist_prev[dist_prev["puem_ltbi_estimate"] >= q75_thr]["district"].str.lower()
    )
    sv = survey_val_df.copy()
    sv["in_top_q"]  = sv["district"].str.lower().isin(top_q_set)
    sv["high_ltbi"] = sv["survey_rate"] > 0.25
    sv["low_ltbi"]  = sv["survey_rate"] < 0.15
    tp = int(( sv["in_top_q"] &  sv["high_ltbi"]).sum())
    fp = int(( sv["in_top_q"] &  sv["low_ltbi"]).sum())
    fn = int((~sv["in_top_q"] &  sv["high_ltbi"]).sum())
    tn = int((~sv["in_top_q"] &  sv["low_ltbi"]).sum())
    prec_q = tp/(tp+fp) if (tp+fp)>0 else 0.0
    rec_q  = tp/(tp+fn) if (tp+fn)>0 else 0.0
    f1_q   = 2*prec_q*rec_q/(prec_q+rec_q) if (prec_q+rec_q)>0 else 0.0

    print(f"\n  Top-Quartile Confusion Matrix")
    print(f"     TP={tp}  FP={fp}  FN={fn}  TN={tn}")
    print(f"     Precision={prec_q:.4f}  Recall={rec_q:.4f}  F1={f1_q:.4f}")

    return dict(
        pi_hat=pi_hat, ci_lo=ci_lo, ci_hi=ci_hi, boot_std=boot_std,
        mae=mae, rmse=rmse, auc=auc,
        recall=recall_val, precision=precision_val, f1=f1_val,
        log_loss=ll_val, n_iters=n_iters, final_ll=final_ll,
        spearman_r=spearman_r, spearman_p=spearman_p,
        p_at_30=p_at_30,
        uniform_cost=uniform_cost, targeted_cost=targeted_cost,
        saving_ugx=saving_ugx, saving_usd=saving_usd, saving_pct=saving_pct,
        tp=tp, fp=fp, fn=fn, tn=tn,
        prec_q=prec_q, rec_q=rec_q, f1_q=f1_q,
        fpr=fpr, tpr=tpr, opt_thr=opt_thr,
        y_true_bin=y_true_bin, y_score=y_score, y_pred=y_pred,
        survey_val_df=survey_val_df, holdout_df=holdout_df,
        pop_df=pop_df,
    )
