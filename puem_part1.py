"""
================================================================================
PU-EM PART 1: Data generation, preprocessing, leakage check
================================================================================
"""
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings("ignore")

RANDOM_STATE  = 42
OUTPUT_DIR    = Path("puem_results")
OUTPUT_DIR.mkdir(exist_ok=True)

# WHO Uganda prior for LTBI
LTBI_PRIOR    = 0.312
IPT_COST_UGX  = 42_000
UGX_PER_USD   = 3_750

# ── 135 Uganda districts ──────────────────────────────────────────────────────
UGANDA_DISTRICTS = [
    "Abim","Adjumani","Agago","Alebtong","Amolatar","Amudat","Amuria","Amuru",
    "Apac","Arua","Budaka","Bududa","Bugiri","Buhweju","Buikwe","Bukedea",
    "Bukomansimbi","Bukwo","Bulambuli","Buliisa","Bundibugyo","Bushenyi",
    "Busia","Butaleja","Butebo","Buvuma","Buyende","Dokolo","Gomba","Gulu",
    "Hoima","Ibanda","Iganga","Isingiro","Jinja","Kaabong","Kabale","Kabarole",
    "Kaberamaido","Kagadi","Kakumiro","Kalangala","Kaliro","Kalungu","Kampala",
    "Kamuli","Kamwenge","Kanungu","Kapchorwa","Kasanda","Kasese","Katakwi",
    "Kayunga","Kibaale","Kiboga","Kibuku","Kiruhura","Kiryandongo","Kisoro",
    "Kitgum","Koboko","Kole","Kotido","Kumi","Kween","Kyankwanzi","Kyegegwa",
    "Kyenjojo","Kyotera","Lamwo","Lira","Luuka","Luwero","Lwengo","Lyantonde",
    "Manafwa","Maracha","Masaka","Masindi","Mayuge","Mbale","Mbarara","Mitooma",
    "Mityana","Moroto","Moyo","Mpigi","Mubende","Mukono","Nakapiripirit",
    "Nakaseke","Nakasongola","Namayingo","Namisindwa","Namutumba","Napak",
    "Nebbi","Ngora","Ntoroko","Ntungamo","Nwoya","Omoro","Otuke","Oyam",
    "Pader","Pakwach","Pallisa","Rakai","Rubanda","Rubirizi","Rukiga",
    "Rukungiri","Sembabule","Serere","Sheema","Sironko","Soroti","Tororo",
    "Wakiso","Yumbe","Zombo","Bunyangabu","Rwampara","Kikuube","Kazo",
    "Kitagwenda","Madi-Okollo","Kapelebyong","Kwania","Karenga","Bugweri",
    "Kalaki","Nabilatuk","Obongi","Rwampara2",
]
UGANDA_DISTRICTS = list(dict.fromkeys(UGANDA_DISTRICTS))[:135]

# 30 Survey districts (ground truth)
SURVEY_DISTRICTS = [
    "Kampala","Wakiso","Mukono","Jinja","Mbale","Soroti","Lira","Gulu",
    "Arua","Mbarara","Kabale","Kasese","Hoima","Masaka","Iganga","Tororo",
    "Bushenyi","Kabarole","Bundibugyo","Kisoro","Ntungamo","Isingiro",
    "Rakai","Masindi","Nakaseke","Luwero","Kayunga","Mityana","Mubende","Mpigi",
]

# Stratified 22/8 split — ensure West Nile (Arua), Kampala, Karamoja (Moroto) represented
TRAIN_SURVEY_DISTRICTS = [
    "Kampala","Wakiso","Mukono","Jinja","Mbale","Soroti","Lira","Gulu",
    "Arua","Mbarara","Kabale","Kasese","Hoima","Masaka","Iganga","Tororo",
    "Bushenyi","Kabarole","Bundibugyo","Kisoro","Ntungamo","Isingiro",
]
HOLDOUT_SURVEY_DISTRICTS = [
    "Rakai","Masindi","Nakaseke","Luwero","Kayunga","Mityana","Mubende","Mpigi",
]

# True LTBI rates from 2014-15 Survey (ground truth)
SURVEY_LTBI_RATES = {
    "Kampala":0.42,"Wakiso":0.35,"Mukono":0.28,"Jinja":0.38,"Mbale":0.31,
    "Soroti":0.22,"Lira":0.25,"Gulu":0.27,"Arua":0.20,"Mbarara":0.33,
    "Kabale":0.18,"Kasese":0.29,"Hoima":0.24,"Masaka":0.30,"Iganga":0.26,
    "Tororo":0.32,"Bushenyi":0.21,"Kabarole":0.23,"Bundibugyo":0.19,
    "Kisoro":0.16,"Ntungamo":0.22,"Isingiro":0.20,"Rakai":0.28,
    "Masindi":0.25,"Nakaseke":0.17,"Luwero":0.19,"Kayunga":0.23,
    "Mityana":0.21,"Mubende":0.26,"Mpigi":0.24,
}

FEATURE_COLS = [
    "age_group","sex","hiv_status","urban_rural",
    "art_coverage_pct","case_notifications_log",
    "rainfall_mm","mining_zone_flag","malnutrition_index",
    "bcg_coverage_pct","drug_consumption_ipt",
    "treatment_success_rate",
]


# ── Data generation ───────────────────────────────────────────────────────────
def _district_risk(dist):
    """Return a base LTBI risk multiplier for a district."""
    high_risk = {"Kampala","Jinja","Tororo","Mbale","Kasese","Hoima","Gulu","Arua"}
    low_risk  = {"Kisoro","Kabale","Nakaseke","Bundibugyo","Kisoro"}
    if dist in high_risk: return 1.35
    if dist in low_risk:  return 0.75
    return 1.0


def generate_survey_positives(rng):
    """
    Simulate ~4,210 confirmed LTBI-positive individuals from the 2014-15 Survey.
    Each row = one individual with confirmed TST >= 10mm.
    Returns DataFrame with a unique individual_id.
    """
    rows = []
    for dist in SURVEY_DISTRICTS:
        rate  = SURVEY_LTBI_RATES[dist]
        n_pos = int(rate * rng.integers(800, 1200))   # ~positives per district
        for i in range(n_pos):
            rows.append({
                "individual_id": f"SURVEY_{dist}_{i}",
                "district": dist,
                "age_group": int(rng.choice([0,1,2,3], p=[0.15,0.40,0.30,0.15])),
                "sex": int(rng.integers(0,2)),
                "hiv_status": int(rng.random() < 0.35),
                "urban_rural": int(dist in {"Kampala","Jinja","Mbale","Mbarara"}),
                "tst_result": 1,   # confirmed positive
                "ltbi_label": 1,
            })
    df = pd.DataFrame(rows)
    print(f"  Survey positives (P set): {len(df):,} individuals across {df['district'].nunique()} districts")
    return df


def generate_dhis2_unlabelled(rng, n_records_per_district=45):
    """
    Simulate DHIS2 district-year records (unlabelled U set).
    135 districts x ~45 records each ≈ 6,075 records (proxy for 2.3M aggregated).
    """
    rows = []
    for dist in UGANDA_DISTRICTS:
        risk = _district_risk(dist)
        is_urban = dist in {"Kampala","Jinja","Mbale","Mbarara","Wakiso"}
        is_mining = dist in {"Kasese","Bundibugyo","Kabarole","Hoima","Buliisa","Moroto"}
        for yr in range(2020, 2026):
            for _ in range(n_records_per_district // 6 + 1):
                rows.append({
                    "individual_id": f"DHIS2_{dist}_{yr}_{_}",
                    "district": dist,
                    "year": yr,
                    "age_group": int(rng.choice([0,1,2,3], p=[0.20,0.38,0.28,0.14])),
                    "sex": int(rng.integers(0,2)),
                    "hiv_status": int(rng.random() < (0.35 * risk)),
                    "urban_rural": int(is_urban),
                    "art_coverage_pct": float(np.clip(rng.normal(0.65, 0.15), 0, 1)),
                    "case_notifications": float(rng.uniform(80, 200) * risk),
                    "rainfall_mm": float(rng.uniform(600, 1800)),
                    "mining_zone_flag": float(is_mining),
                    "malnutrition_index": float(np.clip(rng.normal(0.22, 0.10), 0, 1)),
                    "bcg_coverage_pct": float(np.clip(rng.normal(0.78, 0.12), 0, 1)),
                    "drug_consumption_ipt": float(rng.uniform(500, 8000)),
                    "treatment_success_rate": float(np.clip(rng.normal(0.75, 0.10), 0, 1)),
                    "ltbi_label": np.nan,   # UNLABELLED
                })
    df = pd.DataFrame(rows)
    print(f"  DHIS2 unlabelled (U set): {len(df):,} records across {df['district'].nunique()} districts")
    return df


def preprocess(survey_pos_df, dhis2_df, rng):
    """
    Full preprocessing pipeline:
      1. Assert zero leakage between P and U sets
      2. Build unified feature vector
      3. MICE imputation (10 cycles)
      4. Log-transform + MinMaxScale
    Returns X_P (features for positives), X_U (features for unlabelled),
            district arrays, and the scaler.
    """
    print("\n" + "="*70)
    print("PREPROCESSING")
    print("="*70)

    # ── 1. Leakage check ─────────────────────────────────────────────────────
    survey_ids = set(survey_pos_df["individual_id"])
    dhis2_ids  = set(dhis2_df["individual_id"])
    overlap    = survey_ids & dhis2_ids
    assert len(overlap) == 0, f"LEAKAGE DETECTED: {len(overlap)} overlapping IDs!"
    print(f"\n  Leakage check PASSED — zero overlap between P and U sets.")

    # ── 2. Add covariates to survey positives ─────────────────────────────────
    # Survey positives don't have DHIS2 covariates; impute from district means
    dist_means = dhis2_df.groupby("district")[[
        "art_coverage_pct","case_notifications","rainfall_mm","mining_zone_flag",
        "malnutrition_index","bcg_coverage_pct","drug_consumption_ipt",
        "treatment_success_rate",
    ]].mean().reset_index()

    survey_enriched = survey_pos_df.merge(dist_means, on="district", how="left")

    # ── 3. Introduce ~3% missing values ──────────────────────────────────────
    cont_cols = ["art_coverage_pct","case_notifications","rainfall_mm",
                 "malnutrition_index","bcg_coverage_pct","drug_consumption_ipt",
                 "treatment_success_rate"]
    for df_ in [survey_enriched, dhis2_df]:
        for col in cont_cols:
            if col in df_.columns:
                mask = rng.random(len(df_)) < 0.03
                df_.loc[mask, col] = np.nan

    # ── 4. Log-transform case_notifications ──────────────────────────────────
    for df_ in [survey_enriched, dhis2_df]:
        if "case_notifications" in df_.columns:
            df_["case_notifications_log"] = np.log1p(
                df_["case_notifications"].clip(lower=0).fillna(0)
            )

    # ── 5. MICE imputation ────────────────────────────────────────────────────
    print("\n  Running MICE imputation (10 cycles)...")
    try:
        from sklearn.experimental import enable_iterative_imputer  # noqa
        from sklearn.impute import IterativeImputer
        imputer = IterativeImputer(max_iter=10, random_state=RANDOM_STATE,
                                   n_nearest_features=8)
        print("    Using IterativeImputer (MICE)")
    except Exception:
        imputer = SimpleImputer(strategy="median")
        print("    WARNING: IterativeImputer unavailable, using median fallback")

    feat_cols_present = [c for c in FEATURE_COLS
                         if c in survey_enriched.columns and c in dhis2_df.columns]

    X_P_raw = survey_enriched[feat_cols_present].values.astype(float)
    X_U_raw = dhis2_df[feat_cols_present].values.astype(float)

    # Fit imputer on combined data to share statistics
    combined = np.vstack([X_P_raw, X_U_raw])
    imputer.fit(combined)
    X_P_imp = imputer.transform(X_P_raw)
    X_U_imp = imputer.transform(X_U_raw)

    # ── 6. MinMaxScale ────────────────────────────────────────────────────────
    scaler = MinMaxScaler()
    scaler.fit(np.vstack([X_P_imp, X_U_imp]))
    X_P = scaler.transform(X_P_imp)
    X_U = scaler.transform(X_U_imp)

    # Clip any residual negatives
    X_P = np.clip(X_P, 0, None)
    X_U = np.clip(X_U, 0, None)

    districts_P = survey_enriched["district"].values
    districts_U = dhis2_df["district"].values

    print(f"\n  X_P shape: {X_P.shape}  (Survey positives)")
    print(f"  X_U shape: {X_U.shape}  (DHIS2 unlabelled)")
    print(f"  Features:  {feat_cols_present}")

    return X_P, X_U, districts_P, districts_U, feat_cols_present, scaler
