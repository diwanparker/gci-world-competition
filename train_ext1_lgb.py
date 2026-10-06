"""Train 5-Fold LightGBM with Out-Of-Fold EXT_SOURCE_1 Imputation.
Evaluates OOF ROC-AUC and saves predictions to output/preds/lgb_ext1_oof.npy.
"""
import os
import sys
import time
import logging
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
import lightgbm as lgb

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.features import load_raw_datasets, extract_features, apply_target_encoding
from src.models import create_cv_folds

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

PREDS_DIR = "./output/preds"
os.makedirs(PREDS_DIR, exist_ok=True)

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

REG_PARAMS = {
    "objective": "regression",
    "metric": "rmse",
    "learning_rate": 0.03,
    "num_leaves": 31,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "verbose": -1,
    "n_jobs": 4,
    "seed": 42,
}


def main():
    t0 = time.time()
    logger.info("Loading datasets from ./input...")
    train_raw, test_raw = load_raw_datasets("./input")

    logger.info("Extracting baseline 121 domain features...")
    x_train, x_test, y_train, test_ids = extract_features(train_raw, test_raw)

    cv_folds = create_cv_folds(y_train, num_splits=5, seed=42)

    logger.info("Applying out-of-fold target encoding...")
    x_train_lgb, x_test_lgb = apply_target_encoding(
        x_train, x_test, y_train, cv_folds, ["ORGANIZATION_TYPE", "OCCUPATION_TYPE"]
    )

    pred_cols = [
        c for c in x_train_lgb.columns
        if not any(x in c for x in ["EXT", "TE", "TARGET"])
    ]

    oof_preds = np.zeros(len(x_train_lgb))
    test_preds = np.zeros(len(x_test_lgb))

    logger.info("Starting 5-Fold training with EXT_SOURCE_1 regression...")

    for fold_idx, (train_idx, val_idx) in enumerate(cv_folds):
        t_f0 = time.time()
        # 1. Train EXT_SOURCE_1 regressor strictly on train_idx where not null
        tr_notnull = train_idx[~x_train_lgb.iloc[train_idx]["EXT_SOURCE_1"].isna()]
        dtr_reg = lgb.Dataset(
            x_train_lgb.iloc[tr_notnull][pred_cols],
            label=x_train_lgb.iloc[tr_notnull]["EXT_SOURCE_1"]
        )
        reg = lgb.train(REG_PARAMS, dtr_reg, num_boost_round=300)

        # 2. Predict EXT_1
        ext1_tr = reg.predict(x_train_lgb.iloc[train_idx][pred_cols])
        ext1_va = reg.predict(x_train_lgb.iloc[val_idx][pred_cols])
        ext1_te = reg.predict(x_test_lgb[pred_cols])

        df_tr = x_train_lgb.iloc[train_idx].copy()
        df_va = x_train_lgb.iloc[val_idx].copy()
        df_te = x_test_lgb.copy()

        for df, e1_pred in [(df_tr, ext1_tr), (df_va, ext1_va), (df_te, ext1_te)]:
            df["EXT_1_PRED"] = e1_pred
            df["EXT_1_FILLED"] = df["EXT_SOURCE_1"].fillna(df["EXT_1_PRED"])
            df["EXT_1_DIFF"] = df["EXT_SOURCE_1"] - df["EXT_1_PRED"]
            e2 = df["EXT_SOURCE_2"]
            e3 = df["EXT_SOURCE_3"]
            df["EXT_FILLED_MEAN"] = (df["EXT_1_FILLED"] + e2.fillna(0.5) + e3.fillna(0.5)) / 3.0
            df["EXT_FILLED_PROD"] = df["EXT_1_FILLED"] * e2.fillna(1.0) * e3.fillna(1.0)
            df["EXT_FILLED_WEIGHTED"] = (
                2.0 * df["EXT_1_FILLED"] + 1.0 * e2.fillna(0.5) + 3.0 * e3.fillna(0.5)
            ) / 6.0

        # 3. Train Main Classifier
        dtrain = lgb.Dataset(df_tr, label=y_train[train_idx])
        dval = lgb.Dataset(df_va, label=y_train[val_idx], reference=dtrain)

        bst = lgb.train(
            LIGHTGBM_PARAMS,
            dtrain,
            num_boost_round=4000,
            valid_sets=[dval],
            callbacks=[lgb.early_stopping(150, verbose=False)]
        )

        best_it = bst.best_iteration
        val_pred = bst.predict(df_va, num_iteration=best_it)
        oof_preds[val_idx] = val_pred
        test_preds += bst.predict(df_te, num_iteration=best_it) / len(cv_folds)

        fold_auc = roc_auc_score(y_train[val_idx], val_pred)
        elapsed = time.time() - t_f0
        logger.info(
            "Fold %d/5: AUC = %.5f (Best iteration: %d, Time: %.1fs)",
            fold_idx + 1, fold_auc, best_it, elapsed
        )

    total_auc = roc_auc_score(y_train, oof_preds)
    logger.info("==========================================")
    logger.info("===> NEW 5-FOLD OOF AUC: %.5f", total_auc)
    logger.info("===> Baseline lgb_a AUC: 0.76365")
    logger.info("===> Delta: %+.5f", total_auc - 0.76365)
    logger.info("==========================================")

    np.save(os.path.join(PREDS_DIR, "lgb_ext1_oof.npy"), oof_preds)
    np.save(os.path.join(PREDS_DIR, "lgb_ext1_test.npy"), test_preds)
    logger.info("Saved predictions to %s", PREDS_DIR)


if __name__ == "__main__":
    main()
