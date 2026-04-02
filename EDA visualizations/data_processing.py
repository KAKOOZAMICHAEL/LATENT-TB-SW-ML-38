# pyright: reportMissingImports=false, reportUnknownVariableType=false, reportUnknownMemberType=false
from typing import Any, List, Tuple

import pandas as pd  # type: ignore[import]

# helper for cleaning district names


def standardize_district(name: str) -> str:
    """Return standardized district name (upper, stripped, fix known typos)."""
    if pd.isna(name):
        return name
    n = str(name).strip().upper()
    # add custom replacements as needed
    replacements = {
        "KAMPALA CITY": "KAMPALA",
    }
    return replacements.get(n, n)


def load_and_clean_dhis2(path: str) -> pd.DataFrame:
    """Loads DHIS2 TB case notification data and prepares district-level summary."""
    df: pd.DataFrame = pd.read_csv(path)  # type: ignore[attr-defined]
    # example cleaning steps
    df.columns = df.columns.str.strip().str.upper()  # type: ignore[attr-defined]
    # standardize names
    if "DISTRICT" in df.columns:
        df["DISTRICT"] = df["DISTRICT"].apply(standardize_district)
    # aggregate to district level, assume there's 'NOTIFICATIONS' column
    agg: pd.DataFrame = (
        df.groupby("DISTRICT", as_index=False)
        .agg({"NOTIFICATIONS": "sum"})
        .rename(columns={"NOTIFICATIONS": "TB_NOTIFICATION_RATE"})
    )
    return agg


def load_and_clean_ntlp(path: str) -> pd.DataFrame:
    """Loads NTLP Annual Report data and extracts relevant district-level indicators."""
    df: pd.DataFrame = pd.read_excel(path)  # type: ignore[attr-defined]
    df.columns = df.columns.str.strip().str.upper()  # type: ignore[attr-defined]
    df["DISTRICT"] = df["DISTRICT"].apply(standardize_district)
    # hypothetical columns
    # HIV prevalence, TPT coverage, malnutrition rate, GeneXpert availability, facility distance
    # ensure numeric, drop na etc.
    indicators: List[str] = [
        "HIV_PREVALENCE",
        "TPT_COVERAGE",
        "MALNUTRITION_RATE",
        "GENEXPERT_AVAILABILITY",
        "FACILITY_DISTANCE",
    ]
    present_indicators: List[str] = [col for col in indicators if col in df.columns]
    for col in present_indicators:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    if not present_indicators:
        return df[["DISTRICT"]].drop_duplicates().reset_index(drop=True)
    agg = df.groupby("DISTRICT", as_index=False)[present_indicators].mean()  # type: ignore[assignment]
    return agg  # type: ignore[return-value]


def load_and_clean_prevalence(path: str) -> pd.DataFrame:
    """Cleans Uganda National TB Prevalence Survey."""
    df: pd.DataFrame = pd.read_csv(path)  # type: ignore[attr-defined]
    df.columns = df.columns.str.strip().str.upper()  # type: ignore[attr-defined]
    df["DISTRICT"] = df["DISTRICT"].apply(standardize_district)
    # maybe it contains malnutrition or HIV prevalence; pick relevant
    # For example, treat 'HIV_PREVALENCE' and 'MALNUTRITION_RATE'.
    columns: List[str] = ["DISTRICT", "HIV_PREVALENCE", "MALNUTRITION_RATE"]
    keep: List[str] = [c for c in columns if c in df.columns]
    agg = df[keep].groupby("DISTRICT", as_index=False).mean()  # type: ignore[assignment]
    return agg  # type: ignore[return-value]


def load_and_clean_who(path: str) -> pd.DataFrame:
    """Loads WHO TB report data and prepares district-level indicators."""
    df: pd.DataFrame = pd.read_excel(path)  # type: ignore[attr-defined]
    df.columns = df.columns.str.strip().str.upper()  # type: ignore[attr-defined]
    if "DISTRICT" in df.columns:
        df["DISTRICT"] = df["DISTRICT"].apply(standardize_district)
    # assume estimated treatment cost column exists
    if "ESTIMATED_TREATMENT_COST" in df.columns:
        df["ESTIMATED_TREATMENT_COST"] = pd.to_numeric(
            df["ESTIMATED_TREATMENT_COST"], errors="coerce"
        )
    agg: pd.DataFrame = df.groupby("DISTRICT", as_index=False).agg(
        {"ESTIMATED_TREATMENT_COST": "mean"}
    )
    return agg


def merge_datasets(
    dhis2: pd.DataFrame, ntlp: pd.DataFrame, prevalence: pd.DataFrame, who: pd.DataFrame
) -> pd.DataFrame:
    """Merges the cleaned dataframes into a single district-level dataframe."""
    dfs: List[pd.DataFrame] = [dhis2, ntlp, prevalence, who]
    result: pd.DataFrame = dfs[0]
    for df in dfs[1:]:
        result = result.merge(df, on="DISTRICT", how="outer")
    return result


def categorize_for_bn(df: pd.DataFrame) -> pd.DataFrame:
    """Convert continuous variables into categorical states for a Bayesian network.

    Expected (recommended) merged columns for a latent→active progression BN
    ----------------------------------------------------------------------
    Minimum to train a latent→active model at district level:
        - DISTRICT
        - LTBI_PREVALENCE           (estimated latent TB prevalence / proportion)
        - ACTIVE_TB_INCIDENCE       (active TB incidence rate or proxy)
        - ESTIMATED_TREATMENT_COST  (mean/typical cost)

    Key drivers (used to estimate progression risk and intervention levers):
        - HIV_PREVALENCE
        - MALNUTRITION_RATE
        - TPT_COVERAGE
        - FACILITY_DISTANCE
        - GENEXPERT_AVAILABILITY

    Notes
    -----
    - If ACTIVE_TB_INCIDENCE is not present, TB_NOTIFICATION_RATE is used as a proxy.
    - If LTBI_PREVALENCE is missing, LTBI_PREVALENCE_CAT will be created as NA
      (but the BN cannot meaningfully learn latent burden without it).
    """
    out = df.copy()

    def _coalesce(base: str) -> None:
        """Create/overwrite `base` by coalescing common merge suffix variants."""
        candidates = [base, f"{base}_X", f"{base}_Y", f"{base}_x", f"{base}_y"]
        present = [c for c in candidates if c in out.columns]
        if not present:
            return
        series = None
        for c in present:
            s = pd.to_numeric(out[c], errors="coerce")
            series = s if series is None else series.combine_first(s)
        out[base] = series

    # After outer merges, duplicated indicator columns may get _x/_y suffixes.
    # Coalesce them back to canonical names for downstream categorization.
    for base in [
        "HIV_PREVALENCE",
        "MALNUTRITION_RATE",
        "TPT_COVERAGE",
        "FACILITY_DISTANCE",
        "GENEXPERT_AVAILABILITY",
        "LTBI_PREVALENCE",
        "ACTIVE_TB_INCIDENCE",
        "ESTIMATED_TREATMENT_COST",
    ]:
        _coalesce(base)

    # Ensure numeric inputs where present
    numeric_cols = [
        "LTBI_PREVALENCE",
        "ACTIVE_TB_INCIDENCE",
        "TB_NOTIFICATION_RATE",
        "HIV_PREVALENCE",
        "MALNUTRITION_RATE",
        "TPT_COVERAGE",
        "FACILITY_DISTANCE",
        "GENEXPERT_AVAILABILITY",
        "ESTIMATED_TREATMENT_COST",
    ]
    for col in numeric_cols:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    # Latent burden (LTBI prevalence). Typical population prevalence can be high,
    # so keep broad bins; adjust to your data distribution when available.
    if "LTBI_PREVALENCE" in out.columns:
        out["LTBI_PREVALENCE_CAT"] = pd.cut(
            out["LTBI_PREVALENCE"],
            bins=[-float("inf"), 0.25, 0.45, float("inf")],
            labels=["Low", "Medium", "High"],
        )
    else:
        # Estimate LTBI prevalence from active TB incidence if not available
        # Using a simple heuristic: LTBI_PREVALENCE ~ 5-10x the active TB incidence rate
        if "ACTIVE_TB_INCIDENCE" in out.columns:
            active = out["ACTIVE_TB_INCIDENCE"]
        else:
            active = out.get("TB_NOTIFICATION_RATE", pd.Series([0] * len(out)))

        # Convert active TB rate (per 100k) to prevalence estimate
        # Assuming ~5-10x more people with latent TB than active TB
        estimated_ltbi = (pd.to_numeric(active, errors="coerce") / 100000) * 7.5
        estimated_ltbi = estimated_ltbi.clip(
            0.01, 0.8
        )  # reasonable LTBI prevalence bounds

        out["LTBI_PREVALENCE_CAT"] = pd.cut(
            estimated_ltbi,
            bins=[-float("inf"), 0.25, 0.45, float("inf")],
            labels=["Low", "Medium", "High"],
        )

    # Active TB incidence (preferred) or notifications proxy (fallback)
    if "ACTIVE_TB_INCIDENCE" in out.columns:
        active_series = out["ACTIVE_TB_INCIDENCE"]
    else:
        active_series = out.get("TB_NOTIFICATION_RATE")
    out["ACTIVE_TB_INCIDENCE_CAT"] = pd.cut(
        active_series,  # type: ignore[arg-type]
        bins=[-float("inf"), 50, float("inf")],
        labels=["Low", "High"],
    )

    # Driver categories (policy levers / risk factors)
    out["HIV_PREVALENCE_CAT"] = pd.cut(
        out["HIV_PREVALENCE"],  # type: ignore[arg-type]
        bins=[-float("inf"), 0.05, 0.15, float("inf")],
        labels=["Low", "Medium", "High"],
    )
    out["MALNUTRITION_RATE_CAT"] = pd.cut(
        out["MALNUTRITION_RATE"],  # type: ignore[arg-type]
        bins=[-float("inf"), 0.1, float("inf")],
        labels=["Low", "High"],
    )
    out["TPT_COVERAGE_CAT"] = pd.cut(
        out["TPT_COVERAGE"],  # type: ignore[arg-type]
        bins=[-float("inf"), 0.5, float("inf")],
        labels=["Low", "High"],
    )
    out["FACILITY_DISTANCE_CAT"] = pd.cut(
        out["FACILITY_DISTANCE"],  # type: ignore[arg-type]
        bins=[-float("inf"), 5, float("inf")],
        labels=["Near", "Far"],
    )
    out["GENEXPERT_AVAILABILITY_CAT"] = out["GENEXPERT_AVAILABILITY"].apply(
        lambda x: "Available" if pd.notna(x) and x >= 1 else "Limited"
    )

    # Progression risk is typically not directly observed at district level.
    # To keep the BN trainable with MLE, we derive a categorical proxy risk state
    # from key drivers (you can replace this with a measured progression-risk proxy later).
    risk_score = (
        (out["HIV_PREVALENCE_CAT"] == "High").astype("float")
        + (out["MALNUTRITION_RATE_CAT"] == "High").astype("float")
        + (out["TPT_COVERAGE_CAT"] == "Low").astype("float")
    )
    out["PROGRESSION_RISK_CAT"] = pd.cut(
        risk_score,
        bins=[-float("inf"), 0.5, 1.5, float("inf")],
        labels=["Low", "Medium", "High"],
    )

    out["ESTIMATED_TREATMENT_COST_CAT"] = pd.cut(
        out["ESTIMATED_TREATMENT_COST"],  # type: ignore[arg-type]
        bins=[-float("inf"), 100, 500, float("inf")],
        labels=["Low", "Medium", "High"],
    )
    return out


def simulate_scenarios(model: Any) -> pd.DataFrame:  # type: ignore[type-arg]
    """Run inference for predefined policy scenarios and return comparison table.

    Scenarios:
        1. Increase TPT_Coverage to High with Medium HIV prevalence.
        2. Improve GeneXpert_Availability to Available.
        3. Reduce Facility_Distance through mobile clinics (Near).

    Outputs:
        - Probability of Active TB
        - Probability of Drug Resistant TB
        - Expected Treatment Cost
    """
    try:
        from pgmpy.inference import VariableElimination  # type: ignore[import]
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "pgmpy is required for scenario simulation. Install it with: pip install pgmpy"
        ) from e

    infer = VariableElimination(model)
    scenarios = {
        "Scenario 1: High TPT Coverage": {
            "TPT_Coverage": "High",
            "HIV_Status": "Medium",
        },
        "Scenario 2: GeneXpert Available": {"GeneXpert_Availability": "Available"},
        "Scenario 3: Near Facilities": {"Facility_Distance": "Near"},
    }

    results = []
    cost_map = {"Low": 100, "Medium": 500, "High": 1000}

    for name, evidence in scenarios.items():
        try:
            dist_ds = infer.query(["Active_TB"], evidence=evidence)
            state_names = dist_ds.state_names["Active_TB"]
            probs = dict(zip(state_names, dist_ds.values))  # type: ignore
            # Get probability of High active TB (if it exists), otherwise use first available state
            prob_active = probs.get("High", list(probs.values())[0] if probs else None)
            prob_drug = None

            dist_cost = infer.query(["Treatment_Cost"], evidence=evidence)
            cost_states = dist_cost.state_names["Treatment_Cost"]
            exp_cost = sum(
                cost_map.get(s, 0) * p for s, p in zip(cost_states, dist_cost.values)
            )  # type: ignore

            results.append(
                {
                    "Scenario": name,
                    "P_Active_TB": prob_active,
                    "P_DrugResistant_TB": prob_drug,
                    "Exp_Treatment_Cost": exp_cost,
                }
            )
        except KeyError as e:
            print(f"Warning: Could not query scenario '{name}': {e}")
            continue

    return pd.DataFrame(results)


def build_bn_model(df: pd.DataFrame) -> Any:  # type: ignore[type-arg]
    try:
        # pgmpy 1.x: BayesianNetwork is deprecated in favor of DiscreteBayesianNetwork
        from pgmpy.estimators import MaximumLikelihoodEstimator  # type: ignore[import]
        from pgmpy.models import DiscreteBayesianNetwork  # type: ignore[import]
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "pgmpy is required to train the Bayesian Network. Install it with: pip install pgmpy"
        ) from e
    """Construct and fit a Bayesian network (latent→active progression) using pgmpy.

    Nodes:
        Socioeconomic_Status
        HIV_Status
        Malnutrition
        TPT_Coverage
        Facility_Distance
        GeneXpert_Availability
        Latent_Burden
        Progression_Risk
        Active_TB
        Treatment_Cost

    Edges:
        Socioeconomic_Status -> Malnutrition
        HIV_Status -> Progression_Risk
        Malnutrition -> Progression_Risk
        TPT_Coverage -> Progression_Risk

        Latent_Burden -> Active_TB
        Progression_Risk -> Active_TB
        Facility_Distance -> Active_TB      (proxy for access/detection)
        GeneXpert_Availability -> Active_TB (proxy for detection/diagnostics)

        Active_TB -> Treatment_Cost
    """
    # map categorical columns to BN node names
    column_mapping = {
        "HIV_PREVALENCE_CAT": "HIV_Status",
        "MALNUTRITION_RATE_CAT": "Malnutrition",
        "TPT_COVERAGE_CAT": "TPT_Coverage",
        "FACILITY_DISTANCE_CAT": "Facility_Distance",
        "GENEXPERT_AVAILABILITY_CAT": "GeneXpert_Availability",
        "LTBI_PREVALENCE_CAT": "Latent_Burden",
        "PROGRESSION_RISK_CAT": "Progression_Risk",
        "ACTIVE_TB_INCIDENCE_CAT": "Active_TB",
        "ESTIMATED_TREATMENT_COST_CAT": "Treatment_Cost",
    }
    df_mapped = df.rename(columns=column_mapping)
    # add missing columns with dummy values if needed
    if "Socioeconomic_Status" not in df_mapped.columns:
        df_mapped["Socioeconomic_Status"] = "Low"  # dummy

    cols = [
        "Socioeconomic_Status",
        "HIV_Status",
        "Malnutrition",
        "TPT_Coverage",
        "Facility_Distance",
        "GeneXpert_Availability",
        "Latent_Burden",
        "Progression_Risk",
        "Active_TB",
        "Treatment_Cost",
    ]
    model = DiscreteBayesianNetwork(
        [
            ("Socioeconomic_Status", "Malnutrition"),
            ("HIV_Status", "Progression_Risk"),
            ("Malnutrition", "Progression_Risk"),
            ("TPT_Coverage", "Progression_Risk"),
            ("Latent_Burden", "Active_TB"),
            ("Progression_Risk", "Active_TB"),
            ("Facility_Distance", "Active_TB"),
            ("GeneXpert_Availability", "Active_TB"),
            ("Active_TB", "Treatment_Cost"),
        ]
    )
    # fit using maximum likelihood
    train_df = df_mapped[cols].dropna().copy()
    if train_df.empty:
        raise ValueError(
            "No complete rows available to train the BN after dropping missing values. "
            "Ensure the merged dataset contains LTBI_PREVALENCE (or a proxy), active TB incidence/notifications, "
            "and the driver columns (HIV_PREVALENCE, MALNUTRITION_RATE, TPT_COVERAGE, FACILITY_DISTANCE, "
            "GENEXPERT_AVAILABILITY, ESTIMATED_TREATMENT_COST)."
        )

    # pgmpy expects discrete/categorical columns. Pandas' newer string dtypes can confuse
    # dtype inference, so coerce to categorical strings explicitly.
    for col in cols:
        train_df[col] = train_df[col].astype(str).astype("category")
    estimator = MaximumLikelihoodEstimator(model, train_df)
    for node in model.nodes():
        cpd = estimator.estimate_cpd(node)
        model.add_cpds(cpd)
    return model


def run_full_pipeline(
    dhis2_path: str,
    ntlp_path: str,
    prevalence_path: str,
    who_path: str,
    *,
    train_bn: bool = True,
) -> Tuple[pd.DataFrame, Any]:
    """Execute the entire processing workflow and return a fitted Bayesian network.

    1. load & clean each dataset
    2. merge district-level indicators
    3. convert to categorical states
    4. build & fit the Bayesian network model

    Returns
    -------
    pd.DataFrame
        The merged categorical dataframe (bn_ready).
    BayesianNetwork
        The fitted pgmpy model.
    """
    dhis2_df = load_and_clean_dhis2(dhis2_path)
    ntlp_df = load_and_clean_ntlp(ntlp_path)
    prevalence_df = load_and_clean_prevalence(prevalence_path)
    who_df = load_and_clean_who(who_path)

    merged = merge_datasets(dhis2_df, ntlp_df, prevalence_df, who_df)
    merged.to_csv("merged_tuberculosis_data.csv", index=False)

    bn_ready = categorize_for_bn(merged)
    bn_ready.to_csv("merged_bn_categories.csv", index=False)

    if not train_bn:
        return bn_ready, None

    bn_model = build_bn_model(bn_ready)
    return bn_ready, bn_model


def create_sample_data() -> Tuple[str, str, str, str]:
    """Create sample CSV/Excel files for testing."""
    import os
    import tempfile

    # sample DHIS2 data
    dhis2_data = pd.DataFrame(
        {
            "DISTRICT": ["KAMPALA", "WAKISO", "MUKONO"],
            # proxy for active TB incidence/notifications (toy values)
            "NOTIFICATIONS": [100, 50, 30],
        }
    )
    dhis2_path = os.path.join(tempfile.gettempdir(), "dhis2_sample.csv")
    dhis2_data.to_csv(dhis2_path, index=False)

    # sample NTLP data
    ntlp_data = pd.DataFrame(
        {
            "DISTRICT": ["KAMPALA", "WAKISO", "MUKONO"],
            "HIV_PREVALENCE": [0.1, 0.08, 0.05],
            "TPT_COVERAGE": [0.6, 0.4, 0.7],
            "MALNUTRITION_RATE": [0.15, 0.12, 0.08],
            "GENEXPERT_AVAILABILITY": [1, 0, 1],
            "FACILITY_DISTANCE": [3, 8, 2],
        }
    )
    ntlp_path = os.path.join(tempfile.gettempdir(), "ntlp_sample.xlsx")
    ntlp_data.to_excel(ntlp_path, index=False)

    # sample prevalence data
    prevalence_data = pd.DataFrame(
        {
            "DISTRICT": ["KAMPALA", "WAKISO", "MUKONO"],
            "HIV_PREVALENCE": [0.1, 0.08, 0.05],
            "MALNUTRITION_RATE": [0.15, 0.12, 0.08],
            # latent TB prevalence estimate (toy values as proportions)
            "LTBI_PREVALENCE": [0.40, 0.35, 0.25],
        }
    )
    prevalence_path = os.path.join(tempfile.gettempdir(), "prevalence_sample.csv")
    prevalence_data.to_csv(prevalence_path, index=False)

    # sample WHO data
    who_data = pd.DataFrame(
        {
            "DISTRICT": ["KAMPALA", "WAKISO", "MUKONO"],
            "ESTIMATED_TREATMENT_COST": [200, 150, 300],
        }
    )
    who_path = os.path.join(tempfile.gettempdir(), "who_sample.xlsx")
    who_data.to_excel(who_path, index=False)

    return dhis2_path, ntlp_path, prevalence_path, who_path


if __name__ == "__main__":
    # create sample data for testing
    dhis2_path, ntlp_path, prevalence_path, who_path = create_sample_data()

    try:
        bn_ready, bn_model = run_full_pipeline(
            dhis2_path, ntlp_path, prevalence_path, who_path, train_bn=True
        )
    except ImportError as e:
        print(str(e))
        bn_ready, bn_model = run_full_pipeline(
            dhis2_path, ntlp_path, prevalence_path, who_path, train_bn=False
        )
    print("Categorical dataset: \n", bn_ready.head())
    if bn_model is not None:
        print("Bayesian network nodes: ", bn_model.nodes())  # type: ignore

    # run policy simulations and show results
    if bn_model is not None:
        scenario_df = simulate_scenarios(bn_model)  # type: ignore
        print("\nScenario comparison:\n", scenario_df)
