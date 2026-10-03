"""Feature engineering pipeline for GCI Home Credit Default Risk.

Builds 121 domain-specific financial, temporal, and interaction features
with out-of-fold target encoding to maximize ROC-AUC in gradient boosting.
"""
from typing import Dict, List, Tuple
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

CATEGORICAL_COLUMNS = [
    "NAME_CONTRACT_TYPE",
    "CODE_GENDER",
    "NAME_INCOME_TYPE",
    "NAME_EDUCATION_TYPE",
    "NAME_FAMILY_STATUS",
    "NAME_HOUSING_TYPE",
    "OCCUPATION_TYPE",
    "ORGANIZATION_TYPE",
]

EXTERNAL_SOURCE_COLUMNS = ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]

BUREAU_INQUIRY_COLUMNS = [
    "AMT_REQ_CREDIT_BUREAU_HOUR",
    "AMT_REQ_CREDIT_BUREAU_MON",
    "AMT_REQ_CREDIT_BUREAU_QRT",
    "AMT_REQ_CREDIT_BUREAU_YEAR",
]


def load_raw_datasets(data_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load train and test CSV files from the specified data directory."""
    train_path = os.path.join(data_dir, "train.csv")
    test_path = os.path.join(data_dir, "test.csv")

    if not os.path.exists(train_path) or not os.path.exists(test_path):
        raise FileNotFoundError(f"Missing train.csv or test.csv in {data_dir}")

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    return train_df, test_df


def extract_features(
    train_raw: pd.DataFrame, test_raw: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    """Transform raw tabular data into 121 engineered features.

    Returns:
        Tuple of (X_train, X_test, y_train, test_ids)
    """
    y_train = train_raw["TARGET"].values.astype(int)
    n_train = len(train_raw)

    combined_df = pd.concat(
        [train_raw.drop(columns=["TARGET"]), test_raw], ignore_index=True
    )
    test_ids = combined_df["SK_ID_CURR"].iloc[n_train:].values
    combined_df = combined_df.drop(columns=["SK_ID_CURR"])

    # Dictionary accumulator to prevent DataFrame fragmentation warnings
    derived_features: Dict[str, pd.Series] = {}

    # 1. Clean anomaly in employment
    derived_features["DAYS_EMPLOYED_ANOM"] = (
        combined_df["DAYS_EMPLOYED"] == 365243
    ).astype(int)
    clean_days_employed = combined_df["DAYS_EMPLOYED"].replace(365243, np.nan)
    derived_features["NAN_COUNT"] = combined_df.isna().sum(axis=1)

    # 2. Financial and credit ratios
    amt_credit = combined_df["AMT_CREDIT"]
    amt_annuity = combined_df["AMT_ANNUITY"]
    amt_income = combined_df["AMT_INCOME_TOTAL"]
    amt_goods = combined_df["AMT_GOODS_PRICE"]
    cnt_fam = combined_df["CNT_FAM_MEMBERS"].clip(lower=1)
    cnt_children = combined_df["CNT_CHILDREN"]

    credit_term = amt_credit / amt_annuity.clip(lower=1.0)
    derived_features["CREDIT_TERM"] = credit_term
    derived_features["ANNUITY_CREDIT"] = amt_annuity / amt_credit.clip(lower=1.0)
    derived_features["CREDIT_GOODS"] = amt_credit / amt_goods.clip(lower=1.0)
    derived_features["CREDIT_GOODS_DIFF"] = amt_credit - amt_goods
    derived_features["CREDIT_INCOME"] = amt_credit / amt_income.clip(lower=1.0)
    derived_features["ANNUITY_INCOME"] = amt_annuity / amt_income.clip(lower=1.0)
    derived_features["GOODS_INCOME"] = amt_goods / amt_income.clip(lower=1.0)
    derived_features["INCOME_PER_PERSON"] = amt_income / cnt_fam
    derived_features["INCOME_PER_CHILD"] = amt_income / (cnt_children + 1)
    derived_features["ANNUITY_PER_PERSON"] = amt_annuity / cnt_fam
    derived_features["CHILDREN_RATIO"] = cnt_children / cnt_fam
    derived_features["INCOME_LEFT"] = amt_income - (12.0 * amt_annuity)

    credit_term_round = credit_term.round()
    derived_features["CREDIT_TERM_ROUND"] = credit_term_round
    derived_features["TOTAL_PAYMENT_OVER_GOODS"] = (
        amt_annuity * credit_term_round / amt_goods.clip(lower=1.0)
    )
    derived_features["OVERPAY_RATIO"] = (
        amt_annuity * credit_term_round / amt_credit.clip(lower=1.0)
    )

    # 3. Temporal ratios
    days_birth = combined_df["DAYS_BIRTH"]
    age_years = -days_birth / 365.25
    derived_features["AGE"] = age_years
    derived_features["EMPLOYED_AGE"] = clean_days_employed / days_birth
    derived_features["REG_AGE"] = combined_df["DAYS_REGISTRATION"] / days_birth
    derived_features["IDPUB_AGE"] = combined_df["DAYS_ID_PUBLISH"] / days_birth
    derived_features["PHONE_AGE"] = combined_df["DAYS_LAST_PHONE_CHANGE"] / days_birth
    derived_features["PHONE_EMPLOYED"] = (
        combined_df["DAYS_LAST_PHONE_CHANGE"] / clean_days_employed
    )
    derived_features["CAR_AGE_RATIO"] = combined_df["OWN_CAR_AGE"] / age_years.clip(
        lower=1.0
    )
    derived_features["CAR_EMPLOYED_RATIO"] = (
        combined_df["OWN_CAR_AGE"] * 365.25 / (-clean_days_employed).clip(lower=1.0)
    )
    derived_features["AGE_AT_ID"] = (
        combined_df["DAYS_ID_PUBLISH"] - days_birth
    ) / 365.25
    derived_features["CREDIT_END_AGE"] = age_years + (credit_term / 12.0)

    # 4. External sources composite features
    ext_df = combined_df[EXTERNAL_SOURCE_COLUMNS]
    ext_mean = ext_df.mean(axis=1)
    derived_features["EXT_MEAN"] = ext_mean
    derived_features["EXT_STD"] = ext_df.std(axis=1).fillna(0.0)
    derived_features["EXT_MIN"] = ext_df.min(axis=1)
    derived_features["EXT_MAX"] = ext_df.max(axis=1)
    derived_features["EXT_MEDIAN"] = ext_df.median(axis=1)
    derived_features["EXT_NANS"] = ext_df.isna().sum(axis=1)
    derived_features["EXT_PROD"] = ext_df.prod(axis=1, min_count=1)

    # Domain weighted external score
    ext_1_fill = combined_df["EXT_SOURCE_1"].fillna(ext_mean)
    ext_2_fill = combined_df["EXT_SOURCE_2"].fillna(ext_mean)
    ext_3_fill = combined_df["EXT_SOURCE_3"].fillna(ext_mean)
    derived_features["EXT_WEIGHTED"] = (
        2.0 * ext_1_fill + 1.0 * ext_2_fill + 3.0 * ext_3_fill
    ) / 6.0

    # Cross external sources
    for i, j in [(1, 2), (1, 3), (2, 3)]:
        col_i = combined_df[f"EXT_SOURCE_{i}"]
        col_j = combined_df[f"EXT_SOURCE_{j}"]
        derived_features[f"EXT{i}x{j}"] = col_i * col_j
        derived_features[f"EXT{i}m{j}"] = col_i - col_j

    for k in (1, 2, 3):
        col_k = combined_df[f"EXT_SOURCE_{k}"]
        derived_features[f"EXT{k}_AGE"] = col_k / age_years.clip(lower=1.0)
        derived_features[f"EXT{k}_TERM"] = col_k * credit_term

    derived_features["EXT_MEAN_TERM"] = ext_mean * credit_term
    derived_features["EXT_MEAN_CREDIT_INC"] = (
        ext_mean * derived_features["CREDIT_INCOME"]
    )
    derived_features["EXT_MEAN_ANN_INC"] = (
        ext_mean * derived_features["ANNUITY_INCOME"]
    )

    # 5. Credit Bureau Inquiries Sum
    derived_features["REQ_SUM"] = combined_df[BUREAU_INQUIRY_COLUMNS].sum(
        axis=1, min_count=1
    )

    # Merge base features with derived dictionary at once to prevent fragmentation
    combined_augmented = pd.concat(
        [combined_df, pd.DataFrame(derived_features, index=combined_df.index)],
        axis=1,
    )
    combined_augmented["DAYS_EMPLOYED"] = clean_days_employed

    # 6. Group aggregations (unsupervised over train + test)
    group_derived: Dict[str, pd.Series] = {}
    target_groups = [
        "ORGANIZATION_TYPE",
        "OCCUPATION_TYPE",
        "NAME_EDUCATION_TYPE",
        "NAME_INCOME_TYPE",
        "CODE_GENDER",
    ]
    stat_metrics = [
        ("AMT_INCOME_TOTAL", "median"),
        ("EXT_MEAN", "mean"),
        ("CREDIT_TERM", "mean"),
        ("AMT_CREDIT", "median"),
        ("AGE", "mean"),
    ]

    for grp in target_groups:
        for col_name, agg_func in stat_metrics:
            group_stat = combined_augmented.groupby(grp)[col_name].transform(agg_func)
            group_derived[f"{grp}_{col_name}_REL"] = (
                combined_augmented[col_name] / group_stat.clip(lower=1e-5)
            )

    # Matching borrower product density
    group_derived["CRED_ANN_GROUP_CNT"] = combined_augmented.groupby(
        ["AMT_CREDIT", "AMT_ANNUITY"]
    )["AGE"].transform("size")
    group_derived["GOODS_CNT"] = combined_augmented.groupby("AMT_GOODS_PRICE")[
        "AGE"
    ].transform("size")
    group_derived["TERM_CNT"] = combined_augmented.groupby("CREDIT_TERM_ROUND")[
        "AGE"
    ].transform("size")
    group_derived["INCOME_CNT"] = combined_augmented.groupby("AMT_INCOME_TOTAL")[
        "AGE"
    ].transform("size")
    group_derived["BIRTH_CNT"] = combined_augmented.groupby("DAYS_BIRTH")[
        "AGE"
    ].transform("size")

    # Categorical Frequency Encoding
    for cat_col in CATEGORICAL_COLUMNS:
        freq = combined_augmented[cat_col].value_counts(dropna=False)
        group_derived[f"{cat_col}_FREQ"] = (
            combined_augmented[cat_col].map(freq).astype(float)
        )
        combined_augmented[cat_col] = (
            combined_augmented[cat_col].fillna("NA").astype("category")
        )

    final_df = pd.concat(
        [combined_augmented, pd.DataFrame(group_derived, index=combined_augmented.index)],
        axis=1,
    )
    final_df = final_df.replace([np.inf, -np.inf], np.nan)

    x_train = final_df.iloc[:n_train].reset_index(drop=True)
    x_test = final_df.iloc[n_train:].reset_index(drop=True)

    return x_train, x_test, y_train, test_ids


def apply_target_encoding(
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: np.ndarray,
    cv_folds: List[Tuple[np.ndarray, np.ndarray]],
    target_columns: List[str],
    smoothing_factor: float = 50.0,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Out-of-fold target encoding with empirical Bayes smoothing."""
    x_train_encoded = x_train.copy()
    x_test_encoded = x_test.copy()
    global_mean = float(y_train.mean())

    for col in target_columns:
        oof_series = np.zeros(len(x_train))
        for train_idx, val_idx in cv_folds:
            fold_df = pd.DataFrame(
                {"cat": x_train[col].iloc[train_idx].astype(str).values, "y": y_train[train_idx]}
            )
            agg = fold_df.groupby("cat")["y"].agg(["sum", "count"])
            smoothed_rate = (agg["sum"] + global_mean * smoothing_factor) / (
                agg["count"] + smoothing_factor
            )
            val_categories = x_train[col].iloc[val_idx].astype(str)
            oof_series[val_idx] = (
                val_categories.map(smoothed_rate).fillna(global_mean).values
            )

        full_df = pd.DataFrame(
            {"cat": x_train[col].astype(str).values, "y": y_train}
        )
        full_agg = full_df.groupby("cat")["y"].agg(["sum", "count"])
        full_smoothed = (full_agg["sum"] + global_mean * smoothing_factor) / (
            full_agg["count"] + smoothing_factor
        )

        x_train_encoded[f"{col}_TE"] = oof_series
        x_test_encoded[f"{col}_TE"] = (
            x_test[col].astype(str).map(full_smoothed).fillna(global_mean).values
        )

    return x_train_encoded, x_test_encoded
