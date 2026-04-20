"""Knowledge distillation and feature pruning for the PUEM LTBI model.

This script trains a small student model to mimic the PUEM teacher's
district-level LTBI probability outputs and then evaluates the effect of
pruning low-importance features from the PUEM model.
"""

import pickle
import time
from pathlib import Path
import copy

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.feature_selection import mutual_info_regression
from sklearn.tree import DecisionTreeRegressor

from puem_part1 import (
    generate_survey_positives,
    generate_dhis2_unlabelled,
    preprocess,
    SURVEY_DISTRICTS,
    HOLDOUT_SURVEY_DISTRICTS,
    SURVEY_LTBI_RATES,
)
from puem_part2 import compute_district_prevalence, run_puem

STUDENT_OUTPUT_PATH = Path("student_model_distilled.pkl")
PRUNED_OUTPUT_PATH = Path("pruned_puem_model.pkl")
RANDOM_STATE = 42
INFERENCE_REPEATS = 20
PRUNE_FRACTION = 0.30


def prepare_puem_data(rng):
    survey_pos_df = generate_survey_positives(rng)
    dhis2_df = generate_dhis2_unlabelled(rng)

    X_P, X_U, districts_P, districts_U, feat_cols, scaler = preprocess(
        survey_pos_df, dhis2_df, rng
    )

    return X_P, X_U, districts_P, districts_U, feat_cols, scaler


def build_teacher_model(rng):
    X_P, X_U, districts_P, districts_U, feat_cols, scaler = prepare_puem_data(rng)

    clf, posterior, ll_history, n_iters, final_ll = run_puem(
        X_P, X_U, districts_U, max_iter=50, tol=1e-3, rng=rng
    )

    dist_prev = compute_district_prevalence(posterior, districts_U)
    dist_prev = dist_prev.set_index("district")["puem_ltbi_estimate"]
    pi_hat = float(posterior.mean())

    teacher_package = {
        "clf": clf,
        "scaler": scaler,
        "feat_cols": feat_cols,
    }

    return {
        "teacher": teacher_package,
        "X_P": X_P,
        "X_U": X_U,
        "districts_U": districts_U,
        "feat_cols": feat_cols,
        "teacher_dist_prev": dist_prev,
        "teacher_pi_hat": pi_hat,
        "teacher_n_iters": n_iters,
        "ll_history": ll_history,
    }


def build_district_feature_matrix(X_U, districts_U, feat_cols):
    df = pd.DataFrame(X_U, columns=feat_cols)
    df["district"] = districts_U
    dist_features = (
        df.groupby("district", as_index=False)[feat_cols]
        .mean()
        .sort_values("district")
        .reset_index(drop=True)
    )
    return dist_features


def rank_features_by_mutual_info(dist_features, teacher_scores, prune_fraction=PRUNE_FRACTION):
    X = dist_features.drop(columns=["district"]).values
    y = teacher_scores.values.astype(float)
    mi = mutual_info_regression(X, y, random_state=RANDOM_STATE)
    importance = pd.Series(mi, index=dist_features.drop(columns=["district"]).columns)
    importance = importance.sort_values(ascending=False)

    keep_n = max(1, int(np.ceil(len(importance) * (1.0 - prune_fraction))))
    selected_features = importance.index[:keep_n].tolist()
    return selected_features, importance


def run_pruned_puem(X_P, X_U, districts_U, feat_cols, selected_feat_cols, rng):
    selected_indices = [feat_cols.index(feat) for feat in selected_feat_cols]
    X_P_pruned = X_P[:, selected_indices]
    X_U_pruned = X_U[:, selected_indices]

    clf, posterior, ll_history, n_iters, final_ll = run_puem(
        X_P_pruned, X_U_pruned, districts_U, max_iter=50, tol=1e-3, rng=rng
    )

    dist_prev = compute_district_prevalence(posterior, districts_U)
    pi_hat = float(posterior.mean())

    pruned_package = {
        "clf": clf,
        "feat_cols": selected_feat_cols,
        "dist_prev": dist_prev,
        "pi_hat": pi_hat,
        "n_iters": n_iters,
        "ll_history": ll_history,
    }
    return pruned_package


def train_student(dist_features, teacher_scores):
    X = dist_features.drop(columns=["district"]).values
    y = teacher_scores.values.astype(float)

    student = DecisionTreeRegressor(max_depth=3, random_state=RANDOM_STATE)
    student.fit(X, y)
    return student


def evaluate_against_survey(predictions, name):
    if not isinstance(predictions, pd.Series):
        predictions = pd.Series(predictions)
    predictions.index = predictions.index.astype(str)

    holdout_truth = [SURVEY_LTBI_RATES[d] for d in HOLDOUT_SURVEY_DISTRICTS]
    holdout_pred = [predictions.get(d, np.nan) for d in HOLDOUT_SURVEY_DISTRICTS]
    holdout_mask = ~np.isnan(holdout_pred)

    if holdout_mask.sum() == 0:
        raise ValueError("No holdout districts available for evaluation")

    holdout_mae = float(np.mean(np.abs(np.array(holdout_pred)[holdout_mask] - np.array(holdout_truth)[holdout_mask])) * 100)

    survey_truth = [SURVEY_LTBI_RATES[d] for d in SURVEY_DISTRICTS]
    survey_pred = [predictions.get(d, np.nan) for d in SURVEY_DISTRICTS]
    survey_mask = ~np.isnan(survey_pred)
    survey_pred = np.array(survey_pred)[survey_mask]
    survey_truth = np.array(survey_truth)[survey_mask]

    if len(survey_pred) < 2:
        spearman_value = np.nan
    else:
        spearman_value = float(spearmanr(survey_pred, survey_truth).correlation)

    return holdout_mae, spearman_value


def measure_inference_time(fn, repeats=INFERENCE_REPEATS):
    start = time.perf_counter()
    for _ in range(repeats):
        fn()
    end = time.perf_counter()
    return (end - start) / repeats * 1000.0


def get_model_size_kb(obj):
    return len(pickle.dumps(obj)) / 1024.0


def print_comparison_table(rows):
    header = f"| {'Model':<12} | {'MAE':>8} | {'Spearman':>12} | {'Size (KB)':>10} | {'Inference (ms)':>15} |"
    sep = f"|{'-'*14}|{'-'*10}|{'-'*14}|{'-'*12}|{'-'*17}|"
    print(header)
    print(sep)
    for row in rows:
        print(f"| {row['model']:<12} | {row['mae']:>8.3f} | {row['spearman']:>12.3f} | {row['size_kb']:>10.2f} | {row['inference_ms']:>15.3f} |")


def print_pruning_table(rows):
    header = f"| {'Version':<14} | {'Features Used':<13} | {'Prevalence Est.':>16} | {'Spearman Corr':>14} | {'Size (KB)':>10} |"
    sep = f"|{'-'*16}|{'-'*15}|{'-'*18}|{'-'*16}|{'-'*12}|"
    print(header)
    print(sep)
    for row in rows:
        print(f"| {row['version']:<14} | {row['features_used']:<13} | {row['prevalence']:>16.3f} | {row['spearman']:>14.3f} | {row['size_kb']:>10.2f} |")


def quantize_model(clf, precision):
    """Quantize the LogisticRegression model to specified precision."""
    quantized_clf = copy.deepcopy(clf)
    
    if precision == 'float32':
        quantized_clf.coef_ = quantized_clf.coef_.astype(np.float32)
        quantized_clf.intercept_ = quantized_clf.intercept_.astype(np.float32)
    elif precision == 'float16':
        quantized_clf.coef_ = quantized_clf.coef_.astype(np.float16)
        quantized_clf.intercept_ = quantized_clf.intercept_.astype(np.float16)
    elif precision == 'int8':
        # Scale coefficients to 0-255 range
        coef_min, coef_max = quantized_clf.coef_.min(), quantized_clf.coef_.max()
        coef_scaled = ((quantized_clf.coef_ - coef_min) / (coef_max - coef_min)) * 255
        quantized_clf.coef_ = coef_scaled.astype(np.int8)
        # Store scaling factors for later use
        quantized_clf._coef_min = coef_min
        quantized_clf._coef_max = coef_max
        
        intercept_min, intercept_max = quantized_clf.intercept_.min(), quantized_clf.intercept_.max()
        intercept_scaled = ((quantized_clf.intercept_ - intercept_min) / (intercept_max - intercept_min)) * 255
        quantized_clf.intercept_ = intercept_scaled.astype(np.int8)
        quantized_clf._intercept_min = intercept_min
        quantized_clf._intercept_max = intercept_max
    else:
        raise ValueError(f"Unsupported precision: {precision}")
    
    return quantized_clf


def dequantize_model(quantized_clf, precision):
    """Dequantize the model for inference if needed."""
    if precision == 'int8':
        # Dequantize coefficients
        coef_range = quantized_clf._coef_max - quantized_clf._coef_min
        quantized_clf.coef_ = (quantized_clf.coef_.astype(np.float64) / 255) * coef_range + quantized_clf._coef_min
        
        intercept_range = quantized_clf._intercept_max - quantized_clf._intercept_min
        quantized_clf.intercept_ = (quantized_clf.intercept_.astype(np.float64) / 255) * intercept_range + quantized_clf._intercept_min
    
    return quantized_clf


def run_inference_with_quantized_model(quantized_clf, precision, X_U, districts_U, scaler):
    """Run inference with quantized model and compute district prevalence."""
    # Dequantize if necessary for inference
    clf_for_inference = dequantize_model(copy.deepcopy(quantized_clf), precision)
    
    X_U_scaled = scaler.transform(X_U)
    proba = clf_for_inference.predict_proba(X_U_scaled)[:, 1]
    dist_prev = pd.DataFrame({"district": districts_U, "posterior": proba}).groupby("district")["posterior"].mean().reset_index()
    dist_prev.columns = ["district", "puem_ltbi_estimate"]
    dist_prev = dist_prev.sort_values("puem_ltbi_estimate", ascending=False).reset_index(drop=True)
    dist_prev["rank"] = range(1, len(dist_prev) + 1)
    
    pi_hat = float(proba.mean())
    return dist_prev, pi_hat


def print_quantization_table(rows):
    header = f"| {'Precision':<10} | {'Prevalence Est.':>16} | {'Rank Corr':>10} | {'Size (KB)':>10} | {'Inference (ms)':>15} |"
    sep = f"|{'-'*12}|{'-'*18}|{'-'*12}|{'-'*12}|{'-'*17}|"
    print(header)
    print(sep)
    for row in rows:
        print(f"| {row['precision']:<10} | {row['prevalence']:>16.3f} | {row['rank_corr']:>10.3f} | {row['size_kb']:>10.2f} | {row['inference_ms']:>15.3f} |")


def main():
    rng = np.random.default_rng(RANDOM_STATE)

    teacher_data = build_teacher_model(rng)
    feat_cols = teacher_data["feat_cols"]
    teacher_dist_prev = teacher_data["teacher_dist_prev"]
    X_U = teacher_data["X_U"]
    districts_U = teacher_data["districts_U"]
    X_P = teacher_data["X_P"]

    dist_features = build_district_feature_matrix(X_U, districts_U, feat_cols)

    teacher_scores = teacher_dist_prev.reindex(dist_features["district"]).reset_index(drop=True)
    teacher_scores.index = dist_features["district"]
    teacher_scores = pd.Series(teacher_scores.values, index=dist_features["district"], name="teacher_score")

    student = train_student(dist_features, teacher_scores)

    student_inputs = dist_features.drop(columns=["district"]).values
    student_preds = student.predict(student_inputs)
    student_dist_prev = pd.Series(student_preds, index=dist_features["district"], name="student_score")

    teacher_mae, teacher_spearman = evaluate_against_survey(teacher_dist_prev, "PUEM Teacher")
    student_mae, student_spearman = evaluate_against_survey(student_dist_prev, "Student")

    teacher_size = get_model_size_kb({"scaler": teacher_data["teacher"]["scaler"], "clf": teacher_data["teacher"]["clf"], "feat_cols": feat_cols})
    student_size = get_model_size_kb(student)

    def teacher_inference():
        X_U_scaled = teacher_data["teacher"]["scaler"].transform(X_U)
        _ = teacher_data["teacher"]["clf"].predict_proba(X_U_scaled)[:, 1]
        _ = pd.DataFrame({"district": districts_U, "posterior": _}).groupby("district")["posterior"].mean()
        return _

    def student_inference():
        _ = student.predict(student_inputs)
        return _

    teacher_inference_ms = measure_inference_time(teacher_inference)
    student_inference_ms = measure_inference_time(student_inference)

    comparison_rows = [
        {
            "model": "PUEM Teacher",
            "mae": teacher_mae,
            "spearman": teacher_spearman,
            "size_kb": teacher_size,
            "inference_ms": teacher_inference_ms,
        },
        {
            "model": "Student",
            "mae": student_mae,
            "spearman": student_spearman,
            "size_kb": student_size,
            "inference_ms": student_inference_ms,
        },
    ]

    print("\nPUEM Distillation Comparison")
    print("===========================\n")
    print_comparison_table(comparison_rows)

    student_package = {"model": student, "feat_cols": feat_cols}
    with open(STUDENT_OUTPUT_PATH, "wb") as handle:
        pickle.dump(student_package, handle)

    print(f"\nSaved distilled student model to: {STUDENT_OUTPUT_PATH}")

    # PRUNING SECTION
    selected_feat_cols, importance = rank_features_by_mutual_info(dist_features, teacher_scores)
    pruned_data = run_pruned_puem(X_P, X_U, districts_U, feat_cols, selected_feat_cols, rng)

    pruned_dist_prev = pruned_data["dist_prev"].set_index("district")["puem_ltbi_estimate"]
    pruned_mae, pruned_spearman = evaluate_against_survey(pruned_dist_prev, "Pruned PUEM")
    pruned_size = get_model_size_kb({"clf": pruned_data["clf"], "feat_cols": selected_feat_cols})

    pruning_rows = [
        {
            "version": "Original PUEM",
            "features_used": "all",
            "prevalence": teacher_data["teacher_pi_hat"] * 100,
            "spearman": teacher_spearman,
            "size_kb": teacher_size,
        },
        {
            "version": "Pruned PUEM",
            "features_used": f"{len(selected_feat_cols)}/{len(feat_cols)}",
            "prevalence": pruned_data["pi_hat"] * 100,
            "spearman": pruned_spearman,
            "size_kb": pruned_size,
        },
    ]

    print("\nPUEM Feature Pruning Evaluation")
    print("================================\n")
    print_pruning_table(pruning_rows)
    print(f"\nFeature pruning removed {len(feat_cols) - len(selected_feat_cols)} of {len(feat_cols)} features.")
    print(f"Original PUEM convergence iterations: {teacher_data['teacher_n_iters']}")
    print(f"Pruned PUEM convergence iterations:   {pruned_data['n_iters']}")

    pruned_package = {
        "clf": pruned_data["clf"],
        "feat_cols": selected_feat_cols,
    }
    with open(PRUNED_OUTPUT_PATH, "wb") as handle:
        pickle.dump(pruned_package, handle)

    print(f"\nSaved pruned PUEM model to: {PRUNED_OUTPUT_PATH}")

    # QUANTIZATION SECTION
    original_clf = teacher_data["teacher"]["clf"]
    scaler = teacher_data["teacher"]["scaler"]
    precisions = ['float64', 'float32', 'float16', 'int8']
    quantization_results = []

    for precision in precisions:
        if precision == 'float64':
            quantized_clf = original_clf
        else:
            quantized_clf = quantize_model(original_clf, precision)
        
        # Measure size
        size_kb = get_model_size_kb(quantized_clf)
        
        # Measure inference time
        def inference_fn():
            return run_inference_with_quantized_model(quantized_clf, precision, X_U, districts_U, scaler)
        
        inference_ms = measure_inference_time(inference_fn)
        
        # Run inference and get results
        dist_prev_quant, pi_hat_quant = inference_fn()
        
        # Compute rank correlation with original
        original_ranks = teacher_dist_prev.rank(ascending=False)
        quant_ranks = dist_prev_quant.set_index("district")["puem_ltbi_estimate"].rank(ascending=False)
        common_districts = original_ranks.index.intersection(quant_ranks.index)
        if len(common_districts) > 1:
            rank_corr = spearmanr(original_ranks[common_districts], quant_ranks[common_districts]).correlation
        else:
            rank_corr = np.nan
        
        quantization_results.append({
            "precision": precision,
            "prevalence": pi_hat_quant * 100,
            "rank_corr": rank_corr,
            "size_kb": size_kb,
            "inference_ms": inference_ms,
            "clf": quantized_clf,
        })

    print("\nPUEM Quantization Evaluation")
    print("=============================\n")
    print_quantization_table(quantization_results)

    # Select best quantized model: smallest size with rank_corr > 0.95
    eligible = [r for r in quantization_results if r['rank_corr'] > 0.95 and r['precision'] != 'float64']
    if eligible:
        best = min(eligible, key=lambda x: x['size_kb'])
    else:
        # If none meet criteria, choose the one with highest rank_corr
        best = max(quantization_results[1:], key=lambda x: x['rank_corr'])  # Skip float64

    print(f"\nSelected best quantized model: {best['precision']} (size: {best['size_kb']:.2f} KB, rank_corr: {best['rank_corr']:.3f})")

    quantised_package = {
        "clf": best["clf"],
        "precision": best["precision"],
        "feat_cols": feat_cols,
    }
    with open("quantised_puem_model.pkl", "wb") as handle:
        pickle.dump(quantised_package, handle)

    print(f"\nSaved quantized PUEM model to: quantised_puem_model.pkl")


if __name__ == "__main__":
    main()
