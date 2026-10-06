"""End-to-end training and prediction pipeline for GCI World Competition.

Reproduces the winning 0.76821 ROC-AUC solution (Rank #3 Worldwide)
by training 5-Fold LightGBM + 5-Fold CatBoost and ensembling via Rank-Averaging.
"""
import argparse
import os
import sys
import logging
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.features import (
    load_raw_datasets,
    extract_features,
    apply_target_encoding,
    CATEGORICAL_COLUMNS,
)
from src.models import (
    create_cv_folds,
    train_lightgbm_cv,
    train_catboost_cv,
    blend_predictions_rank_average,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_DATA_DIR = "./input"
FALLBACK_DATA_DIR = "/home/diwan/core/hermes/hermes-data/cache/scratch/competition_gci"
OUTPUT_DIR = "./output"
PREDS_DIR = os.path.join(OUTPUT_DIR, "preds")

LIGHTGBM_PARAMS = {
    "objective": "binary",
    "metric": "auc",
    "boosting_type": "gbdt",
    "learning_rate": 0.01,
    "num_leaves": 24,
    "max_depth": -1,
    "min_data_in_leaf": 100,
    "feature_fraction": 0.35,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l1": 0.5,
    "lambda_l2": 5.0,
    "min_gain_to_split": 0.0,
    "cat_smooth": 50,
    "cat_l2": 10,
    "max_cat_to_onehot": 4,
    "num_threads": 4,
    "verbose": -1,
    "seed": 42,
}

CATBOOST_PARAMS = {
    "iterations": 6000,
    "learning_rate": 0.03,
    "depth": 6,
    "l2_leaf_reg": 5.0,
    "eval_metric": "AUC",
    "random_seed": 5,
    "thread_count": 4,
    "bootstrap_type": "Bernoulli",
    "subsample": 0.8,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GCI World Competition Train Pipeline")
    parser.add_argument("--data-dir", type=str, default=None, help="Path to directory containing train.csv and test.csv")
    parser.add_argument("--output-dir", type=str, default=OUTPUT_DIR, help="Path to output directory")
    parser.add_argument("--use-cached-preds", action="store_true", help="Rebuild final submission from cached fold predictions")
    return parser.parse_args()


def resolve_data_directory(custom_dir: str = None) -> str:
    if custom_dir and os.path.exists(custom_dir):
        return custom_dir
    if os.path.exists(DEFAULT_DATA_DIR) and os.path.exists(os.path.join(DEFAULT_DATA_DIR, "train.csv")):
        return DEFAULT_DATA_DIR
    if os.path.exists(FALLBACK_DATA_DIR):
        return FALLBACK_DATA_DIR
    raise FileNotFoundError("Could not locate train.csv and test.csv in default locations.")


def main():
    args = parse_args()
    data_dir = resolve_data_directory(args.data_dir)
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(PREDS_DIR, exist_ok=True)

    submission_path = os.path.join(args.output_dir, "submission.csv")

    if args.use_cached_preds:
        lgb_test_path = os.path.join(PREDS_DIR, "lgb_a_test.npy")
        cat_test_path = os.path.join(PREDS_DIR, "cat_a_test.npy")
        lgb_oof_path = os.path.join(PREDS_DIR, "lgb_a_oof.npy")
        cat_oof_path = os.path.join(PREDS_DIR, "cat_a_oof.npy")

        if os.path.exists(lgb_test_path) and os.path.exists(cat_test_path):
            logger.info("Rebuilding submission from cached model predictions...")
            _, test_raw = load_raw_datasets(data_dir)
            test_ids = test_raw["SK_ID_CURR"].values

            lgb_ext1_test_path = os.path.join(PREDS_DIR, "lgb_ext1_test.npy")
            xgb_test_path = os.path.join(PREDS_DIR, "xgb_a_test.npy")
            lgb_ext1_oof_path = os.path.join(PREDS_DIR, "lgb_ext1_oof.npy")
            xgb_oof_path = os.path.join(PREDS_DIR, "xgb_a_oof.npy")

            if os.path.exists(lgb_ext1_test_path) and os.path.exists(xgb_test_path):
                logger.info("Using 4-model champion ensemble (lgb_ext1, lgb_a, cat_a, xgb_a)...")
                test_preds_list = [
                    np.load(lgb_ext1_test_path),
                    np.load(lgb_test_path),
                    np.load(cat_test_path),
                    np.load(xgb_test_path),
                ]
                weights = [0.35, 0.15, 0.35, 0.15]

                if os.path.exists(lgb_ext1_oof_path) and os.path.exists(lgb_oof_path) and os.path.exists(cat_oof_path) and os.path.exists(xgb_oof_path):
                    train_raw, _ = load_raw_datasets(data_dir)
                    y_train = train_raw["TARGET"].values
                    oof_preds_list = [
                        np.load(lgb_ext1_oof_path),
                        np.load(lgb_oof_path),
                        np.load(cat_oof_path),
                        np.load(xgb_oof_path),
                    ]
                    ensemble_oof = blend_predictions_rank_average(oof_preds_list, weights)
                    logger.info("4-Model Champion Ensemble OOF ROC-AUC: %.5f", roc_auc_score(y_train, ensemble_oof))
            else:
                test_preds_list = [np.load(lgb_test_path), np.load(cat_test_path)]
                weights = [0.5, 0.5]

                if os.path.exists(lgb_oof_path) and os.path.exists(cat_oof_path):
                    train_raw, _ = load_raw_datasets(data_dir)
                    y_train = train_raw["TARGET"].values
                    ensemble_oof = blend_predictions_rank_average(
                        [np.load(lgb_oof_path), np.load(cat_oof_path)], weights
                    )
                    logger.info("2-Model Ensemble OOF ROC-AUC: %.5f", roc_auc_score(y_train, ensemble_oof))

            ensemble_test = blend_predictions_rank_average(test_preds_list, weights)
            sub_df = pd.DataFrame({"SK_ID_CURR": test_ids, "TARGET": ensemble_test})
            sub_df.to_csv(submission_path, index=False)
            logger.info("Successfully generated submission: %s (%d rows)", submission_path, len(sub_df))
            return

    logger.info("Loading raw datasets from: %s", data_dir)
    train_raw, test_raw = load_raw_datasets(data_dir)

    logger.info("Extracting 121 domain features...")
    x_train, x_test, y_train, test_ids = extract_features(train_raw, test_raw)

    cv_folds = create_cv_folds(y_train, num_splits=5, seed=42)

    logger.info("Computing out-of-fold target encoding for LightGBM...")
    x_train_lgb, x_test_lgb = apply_target_encoding(
        x_train, x_test, y_train, cv_folds, ["ORGANIZATION_TYPE", "OCCUPATION_TYPE"]
    )

    logger.info("Step 1/2: Training LightGBM...")
    lgb_oof, lgb_test, lgb_auc = train_lightgbm_cv(
        x_train_lgb, x_test_lgb, y_train, cv_folds, LIGHTGBM_PARAMS, max_rounds=5000, early_stopping_rounds=200
    )
    np.save(os.path.join(PREDS_DIR, "lgb_a_oof.npy"), lgb_oof)
    np.save(os.path.join(PREDS_DIR, "lgb_a_test.npy"), lgb_test)

    logger.info("Step 2/2: Training CatBoost...")
    cat_oof, cat_test, cat_auc = train_catboost_cv(
        x_train, x_test, y_train, cv_folds, CATEGORICAL_COLUMNS, CATBOOST_PARAMS, early_stopping_rounds=300
    )
    np.save(os.path.join(PREDS_DIR, "cat_a_oof.npy"), cat_oof)
    np.save(os.path.join(PREDS_DIR, "cat_a_test.npy"), cat_test)

    ensemble_oof = blend_predictions_rank_average([lgb_oof, cat_oof], [0.5, 0.5])
    ensemble_auc = float(roc_auc_score(y_train, ensemble_oof))
    logger.info("==> Champion Ensemble Overall OOF ROC-AUC: %.5f", ensemble_auc)

    ensemble_test = blend_predictions_rank_average([lgb_test, cat_test], [0.5, 0.5])
    submission_df = pd.DataFrame({"SK_ID_CURR": test_ids, "TARGET": ensemble_test})
    submission_df.to_csv(submission_path, index=False)
    logger.info("Saved final submission: %s (%d rows)", submission_path, len(submission_df))


if __name__ == "__main__":
    main()
