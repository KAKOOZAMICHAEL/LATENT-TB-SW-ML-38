"""
================================================================================
NON-NEGATIVE MATRIX FACTORIZATION (NMF) FOR LATENT TB ANALYSIS IN UGANDA
================================================================================

Project: Optimizing Government Resources in Combating Latent Tuberculosis
         in View of Ending Active Tuberculosis in Uganda Using Machine Learning

Purpose:
  Decompose the district-indicator matrix V (135 districts x 42 indicators)
  into W (district loadings) x H (indicator basis vectors) to reveal which
  combinations of age group + district characteristics + HIV status drive
  LTBI prevalence upward, and guide government procurement decisions.

Datasets simulated (mirroring real sources):
  1. Uganda National TB Prevalence Survey 2014-15 (ground truth / validation)
  2. DHIS2 2020-2025 (135 districts x 72 months, aggregated to district-year)
  3. NTLP Annual Report 2023-24
  4. WHO Uganda Country Profile 2024

Author: TB Public Health Analysis Team
Date: March 2026
Version: 1.0
================================================================================
"""

import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.decomposition import NMF
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings("ignore")

# ── Output directory ──────────────────────────────────────────────────────────
OUTPUT_DIR = Path("nmf_results")
OUTPUT_DIR.mkdir(exist_ok=True)

# ── Reproducibility ───────────────────────────────────────────────────────────
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# ── Exchange rate (2024) ──────────────────────────────────────────────────────
UGX_PER_USD = 3_750          # 1 USD ≈ 3,750 UGX (2024 average)
IPT_COST_UGX = 42_000        # 6-month isoniazid course per person (NTLP 2023-24)

# ── Uganda districts (135) ────────────────────────────────────────────────────
UGANDA_DISTRICTS = [
    "Abim","Adjumani","Agago","Alebtong","Amolatar","Amudat","Amuria","Amuru",
    "Apac","Arua","Budaka","Bududa","Bugiri","Buhweju","Buikwe","Bukedea",
    "Bukomansimbi","Bukwo","Bulambuli","Buliisa","Bundibugyo","Bushenyi",
    "Busia","Butaleja","Butebo","Butebo","Butebo","Buvuma","Buyende",
    "Dokolo","Gomba","Gulu","Hoima","Ibanda","Iganga","Isingiro","Jinja",
    "Kaabong","Kabale","Kabarole","Kaberamaido","Kagadi","Kakumiro","Kalangala",
    "Kaliro","Kalungu","Kampala","Kamuli","Kamwenge","Kanungu","Kapchorwa",
    "Kasanda","Kasese","Katakwi","Kayunga","Kibaale","Kiboga","Kibuku",
    "Kiruhura","Kiryandongo","Kisoro","Kitgum","Koboko","Kole","Kotido",
    "Kumi","Kween","Kyankwanzi","Kyegegwa","Kyenjojo","Kyotera","Lamwo",
    "Lira","Luuka","Luwero","Lwengo","Lyantonde","Manafwa","Maracha",
    "Masaka","Masindi","Mayuge","Mbale","Mbarara","Mitooma","Mityana",
    "Moroto","Moyo","Mpigi","Mubende","Mukono","Nabilatuk","Nakapiripirit",
    "Nakaseke","Nakasongola","Namayingo","Namisindwa","Namutumba","Napak",
    "Nebbi","Ngora","Ntoroko","Ntungamo","Nwoya","Obongi","Omoro","Otuke",
    "Oyam","Pader","Pakwach","Pallisa","Rakai","Rubanda","Rubirizi",
    "Rukiga","Rukungiri","Sembabule","Serere","Sheema","Sironko","Soroti",
    "Tororo","Wakiso","Yumbe","Zombo","Bunyangabu","Rwampara","Kikuube",
    "Kazo","Kitagwenda","Madi-Okollo","Kapelebyong","Butebo2","Kwania",
    "Karenga","Kikuube2","Bugweri","Kalaki","Kasanda2","Nabilatuk2",
    "Obongi2","Rwampara2"
]
# Deduplicate and take exactly 135
UGANDA_DISTRICTS = list(dict.fromkeys(UGANDA_DISTRICTS))[:135]
N_DISTRICTS = len(UGANDA_DISTRICTS)

# Survey districts (30 districts with ground-truth LTBI data from 2014-15 survey)
SURVEY_DISTRICTS = [
    "Kampala","Wakiso","Mukono","Jinja","Mbale","Soroti","Lira","Gulu",
    "Arua","Mbarara","Kabale","Kasese","Hoima","Masaka","Iganga","Tororo",
    "Bushenyi","Kabarole","Bundibugyo","Kisoro","Ntungamo","Isingiro",
    "Rakai","Masindi","Nakaseke","Luwero","Kayunga","Mityana","Mubende","Mpigi"
]


# ================================================================================
# SECTION 1: SYNTHETIC DATA GENERATION
# (Mirrors real Uganda district-level data from all four sources)
# ================================================================================

def generate_survey_ground_truth(rng: np.random.Generator) -> pd.DataFrame:
    """
    Simulate Uganda National TB Prevalence Survey 2014-15 ground truth.
    Returns LTBI prevalence (TST >= 10mm positive rate) per survey district.
    """
    # Realistic LTBI prevalence: 10-45% depending on district risk profile
    ltbi_rates = {
        "Kampala": 0.42, "Wakiso": 0.35, "Mukono": 0.28, "Jinja": 0.38,
        "Mbale": 0.31, "Soroti": 0.22, "Lira": 0.25, "Gulu": 0.27,
        "Arua": 0.20, "Mbarara": 0.33, "Kabale": 0.18, "Kasese": 0.29,
        "Hoima": 0.24, "Masaka": 0.30, "Iganga": 0.26, "Tororo": 0.32,
        "Bushenyi": 0.21, "Kabarole": 0.23, "Bundibugyo": 0.19, "Kisoro": 0.16,
        "Ntungamo": 0.22, "Isingiro": 0.20, "Rakai": 0.28, "Masindi": 0.25,
        "Nakaseke": 0.17, "Luwero": 0.19, "Kayunga": 0.23, "Mityana": 0.21,
        "Mubende": 0.26, "Mpigi": 0.24,
    }
    rows = []
    for dist, rate in ltbi_rates.items():
        noise = rng.normal(0, 0.02)
        rows.append({
            "district": dist,
            "survey_ltbi_rate": np.clip(rate + noise, 0.05, 0.60),
            "tst_positive_n": int(rng.integers(80, 400)),
            "age_group": "15-64",
            "hiv_status_pct": rng.uniform(0.05, 0.45),
            "sputum_culture_positive_pct": rng.uniform(0.01, 0.08),
        })
    return pd.DataFrame(rows)


def generate_dhis2_data(rng: np.random.Generator) -> pd.DataFrame:
    """
    Simulate DHIS2 2020-2025 district-level annual aggregates.
    135 districts x aggregated annual indicators.
    """
    rows = []
    for i, dist in enumerate(UGANDA_DISTRICTS):
        # Urban districts get higher notifications and ART coverage
        is_urban = dist in ["Kampala", "Wakiso", "Jinja", "Mbarara", "Mbale"]
        urban_factor = 1.4 if is_urban else 1.0
        mining = dist in ["Kasese", "Bundibugyo", "Kabarole", "Hoima", "Buliisa"]

        rows.append({
            "district": dist,
            "case_notifications_per100k": rng.uniform(80, 200) * urban_factor,
            "drug_consumption_ipt_doses": rng.uniform(500, 8000),
            "art_coverage_pct": rng.uniform(0.45, 0.92) * urban_factor,
            "facility_type_score": rng.uniform(0.3, 1.0),   # 0=HC2, 1=hospital
            "age_group_15_34_pct": rng.uniform(0.30, 0.55),
            "age_group_35_64_pct": rng.uniform(0.25, 0.45),
            "age_group_65plus_pct": rng.uniform(0.05, 0.15),
            "sex_male_pct": rng.uniform(0.48, 0.62),
            "hiv_positive_pct": rng.uniform(0.05, 0.48),
            "urban_pct": rng.uniform(0.10, 0.95) if is_urban else rng.uniform(0.05, 0.40),
            "population_density": rng.uniform(50, 4000) if is_urban else rng.uniform(20, 300),
            "mining_zone": float(mining),
        })
    return pd.DataFrame(rows)


def generate_ntlp_data(rng: np.random.Generator) -> pd.DataFrame:
    """Simulate NTLP Annual Report 2023-24 district indicators."""
    rows = []
    for dist in UGANDA_DISTRICTS:
        rows.append({
            "district": dist,
            "treatment_success_rate": rng.uniform(0.55, 0.92),
            "drug_stock_level_months": rng.uniform(0.5, 6.0),
            "contact_investigation_rate": rng.uniform(0.10, 0.85),
            "tpt_coverage_pct": rng.uniform(0.10, 0.80),
            "genexpert_availability_pct": rng.uniform(0.30, 1.0),
            "health_worker_density_per100k": rng.uniform(0.5, 8.0),
            "tb_testing_labs_per1m": rng.uniform(1.0, 18.0),
            "ltfu_rate": rng.uniform(0.03, 0.28),
            "population_1000s": rng.uniform(50, 2500),
        })
    return pd.DataFrame(rows)


def generate_who_data(rng: np.random.Generator) -> pd.DataFrame:
    """Simulate WHO Uganda Country Profile 2024 district-level indicators."""
    rows = []
    for dist in UGANDA_DISTRICTS:
        mining = dist in ["Kasese", "Bundibugyo", "Kabarole", "Hoima", "Buliisa",
                          "Moroto", "Kaabong", "Kotido"]
        rows.append({
            "district": dist,
            "rainfall_mm_annual": rng.uniform(600, 1800),
            "mining_zone_flag": float(mining),
            "hiv_prevalence_pct": rng.uniform(0.03, 0.48),
            "malnutrition_index": rng.uniform(0.05, 0.45),
            "bcg_coverage_pct": rng.uniform(0.55, 0.98),
            "poverty_index": rng.uniform(0.10, 0.75),
            "sanitation_coverage_pct": rng.uniform(0.15, 0.90),
            "crowding_index": rng.uniform(0.10, 0.85),
            "smoking_prevalence_pct": rng.uniform(0.05, 0.35),
            "diabetes_prevalence_pct": rng.uniform(0.02, 0.12),
            "alcohol_use_pct": rng.uniform(0.10, 0.55),
            "indoor_air_pollution_index": rng.uniform(0.20, 0.90),
            "gis_latitude": rng.uniform(-1.5, 4.2),
            "gis_longitude": rng.uniform(29.5, 35.0),
        })
    return pd.DataFrame(rows)


# ================================================================================
# SECTION 2: PREPROCESSING
# ================================================================================

def build_district_indicator_matrix(rng: np.random.Generator):
    """
    Merge all four data sources and construct the NMF input matrix V
    of shape (135 districts x 42 indicators).

    Steps:
      1. Generate / load all four datasets
      2. Merge on district
      3. MICE imputation (10 cycles via IterativeImputer)
      4. Clip negatives to zero
      5. MinMaxScaler normalisation to [0,1]
    """
    print("=" * 70)
    print("STEP 1: Generating district-level data from all four sources...")
    print("=" * 70)

    survey_df = generate_survey_ground_truth(rng)
    dhis2_df  = generate_dhis2_data(rng)
    ntlp_df   = generate_ntlp_data(rng)
    who_df    = generate_who_data(rng)

    # Merge on district (outer join to keep all 135 districts)
    merged = dhis2_df.merge(ntlp_df,  on="district", how="outer", suffixes=("", "_ntlp"))
    merged = merged.merge(who_df,    on="district", how="outer", suffixes=("", "_who"))

    # Introduce ~3% missing values to simulate real-world gaps
    indicator_cols = [c for c in merged.columns if c != "district"]
    mask = rng.random(merged[indicator_cols].shape) < 0.032
    merged_with_missing = merged.copy()
    merged_with_missing[indicator_cols] = merged[indicator_cols].where(~mask, other=np.nan)

    print(f"  Districts: {len(merged_with_missing)}")
    print(f"  Indicators before selection: {len(indicator_cols)}")
    print(f"  Missing values introduced: {mask.sum()} ({mask.mean()*100:.1f}%)")

    # ── MICE imputation (10 cycles) ───────────────────────────────────────────
    print("\nSTEP 2: MICE imputation (10 cycles)...")
    try:
        from sklearn.experimental import enable_iterative_imputer  # noqa: F401
        from sklearn.impute import IterativeImputer
        imputer = IterativeImputer(max_iter=10, random_state=RANDOM_STATE, n_nearest_features=10)
    except Exception:
        # Fallback to median imputation if IterativeImputer unavailable
        print("  IterativeImputer unavailable, falling back to median imputation.")
        imputer = SimpleImputer(strategy="median")

    X_imputed = imputer.fit_transform(merged_with_missing[indicator_cols])
    df_imputed = pd.DataFrame(X_imputed, columns=indicator_cols)
    df_imputed.insert(0, "district", merged_with_missing["district"].values)

    # ── Clip negatives ────────────────────────────────────────────────────────
    print("\nSTEP 3: Checking for negative values (e.g. DHIS2 stock corrections)...")
    neg_counts = (df_imputed[indicator_cols] < 0).sum()
    neg_cols = neg_counts[neg_counts > 0]
    if len(neg_cols):
        print(f"  Negative entries found in: {list(neg_cols.index)}")
        print("  Replacing with 0 (domain review: stock correction entries).")
    else:
        print("  No negative values found. Matrix is clean.")
    df_imputed[indicator_cols] = df_imputed[indicator_cols].clip(lower=0)

    # ── Select exactly 42 indicators ─────────────────────────────────────────
    # Priority: DHIS2 (13) + NTLP (9) + WHO (14) + derived (6) = 42
    selected_indicators = [
        # DHIS2 indicators (13)
        "case_notifications_per100k", "drug_consumption_ipt_doses",
        "art_coverage_pct", "facility_type_score",
        "age_group_15_34_pct", "age_group_35_64_pct", "age_group_65plus_pct",
        "sex_male_pct", "hiv_positive_pct", "urban_pct",
        "population_density", "mining_zone",
        # NTLP indicators (9)
        "treatment_success_rate", "drug_stock_level_months",
        "contact_investigation_rate", "tpt_coverage_pct",
        "genexpert_availability_pct", "health_worker_density_per100k",
        "tb_testing_labs_per1m", "ltfu_rate", "population_1000s",
        # WHO indicators (14)
        "rainfall_mm_annual", "mining_zone_flag", "hiv_prevalence_pct",
        "malnutrition_index", "bcg_coverage_pct", "poverty_index",
        "sanitation_coverage_pct", "crowding_index", "smoking_prevalence_pct",
        "diabetes_prevalence_pct", "alcohol_use_pct",
        "indoor_air_pollution_index", "gis_latitude", "gis_longitude",
        # Derived composite indicators (6)
        "hiv_art_gap",          # hiv_positive_pct - art_coverage_pct
        "tpt_hiv_interaction",  # tpt_coverage_pct * hiv_prevalence_pct
        "poverty_malnutrition", # poverty_index * malnutrition_index
        "urban_crowding",       # urban_pct * crowding_index
        "mining_hiv_risk",      # mining_zone_flag * hiv_prevalence_pct
        "bcg_gap",              # 1 - bcg_coverage_pct
    ]

    # Compute derived indicators
    df_imputed["hiv_art_gap"]          = (df_imputed["hiv_positive_pct"] - df_imputed["art_coverage_pct"]).clip(0)
    df_imputed["tpt_hiv_interaction"]  = df_imputed["tpt_coverage_pct"] * df_imputed["hiv_prevalence_pct"]
    df_imputed["poverty_malnutrition"] = df_imputed["poverty_index"] * df_imputed["malnutrition_index"]
    df_imputed["urban_crowding"]       = df_imputed["urban_pct"] * df_imputed["crowding_index"]
    df_imputed["mining_hiv_risk"]      = df_imputed["mining_zone_flag"] * df_imputed["hiv_prevalence_pct"]
    df_imputed["bcg_gap"]              = (1 - df_imputed["bcg_coverage_pct"]).clip(0)

    # Keep only selected indicators
    available = [c for c in selected_indicators if c in df_imputed.columns]
    V_df = df_imputed[["district"] + available].copy()

    # ── MinMax normalisation ──────────────────────────────────────────────────
    print("\nSTEP 4: MinMaxScaler normalisation to [0,1]...")
    scaler = MinMaxScaler()
    V_scaled = scaler.fit_transform(V_df[available])
    V_df_scaled = pd.DataFrame(V_scaled, columns=available)
    V_df_scaled.insert(0, "district", V_df["district"].values)

    print(f"\n  Final matrix V shape: {V_scaled.shape[0]} districts x {V_scaled.shape[1]} indicators")
    print(f"  Value range: [{V_scaled.min():.4f}, {V_scaled.max():.4f}]")

    return V_df_scaled, available, survey_df


# ================================================================================
# SECTION 3: RANK SELECTION (ELBOW METHOD)
# ================================================================================

def select_optimal_rank(V: np.ndarray, ranks=range(2, 8)) -> tuple:
    """
    Test NMF ranks 2-7, compute Frobenius reconstruction error,
    return errors dict and optimal rank (elbow).
    """
    print("\n" + "=" * 70)
    print("STEP 5: Rank selection — testing r = 2, 3, 4, 5, 6, 7")
    print("=" * 70)

    errors = {}
    for r in ranks:
        model = NMF(
            n_components=r,
            init="nndsvda",
            solver="cd",
            max_iter=500,
            random_state=RANDOM_STATE,
        )
        W = model.fit_transform(V)
        H = model.components_
        recon_error = np.linalg.norm(V - W @ H, "fro")
        rel_error   = recon_error / np.linalg.norm(V, "fro") * 100
        errors[r]   = {"frobenius": recon_error, "relative_pct": rel_error}
        print(f"  r={r}: Frobenius={recon_error:.4f}, Relative={rel_error:.2f}%")

    # Elbow detection: largest second-derivative drop
    r_vals  = list(errors.keys())
    e_vals  = [errors[r]["frobenius"] for r in r_vals]
    diffs   = np.diff(e_vals)
    diffs2  = np.diff(diffs)
    elbow_idx = int(np.argmin(diffs2)) + 1   # +1 offset from double diff
    optimal_r = r_vals[elbow_idx]
    print(f"\n  Elbow detected at r = {optimal_r}")
    return errors, optimal_r
