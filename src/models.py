"""Model training and ensembling module for GCI World Competition.

Implements LightGBM and CatBoost 5-Fold Stratified Cross Validation with
Rank-Averaging ensembling to optimize ROC-AUC.
"""
from typing import Any, Dict, List, Tuple
import logging
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
import lightgbm as lgb
from catboost import CatBoostClassifier, Pool

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def create_cv_folds(
    y: np.ndarray, num_splits: int = 5, seed: int = 42
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Generate stratified K-fold train/validation indices."""
    skf = StratifiedKFold(n_splits=num_splits, shuffle=True, random_state=seed)
    return list(skf.split(np.zeros(len(y)), y))


def compute_rank_probabilities(predictions: np.ndarray) -> np.ndarray:
    """Normalize raw probabilities into uniform quantile ranks [0, 1]."""
    return rankdata(predictions) / len(predictions)


def train_lightgbm_cv(
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: np.ndarray,
    cv_folds: List[Tuple[np.ndarray, np.ndarray]],
    hyperparameters: Dict[str, Any],
    max_rounds: int = 5000,
    early_stopping_rounds: int = 200,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Train 5-fold LightGBM model and predict out-of-fold and test ranks."""
    oof_predictions = np.zeros(len(x_train))
    test_predictions = np.zeros(len(x_test))

    logger.info("Training LightGBM across %d folds...", len(cv_folds))

    for fold_idx, (train_idx, val_idx) in enumerate(cv_folds):
        x_tr, y_tr = x_train.iloc[train_idx], y_train[train_idx]
        x_va, y_va = x_train.iloc[val_idx], y_train[val_idx]

        train_data = lgb.Dataset(x_tr, label=y_tr)
        val_data = lgb.Dataset(x_va, label=y_va, reference=train_data)

        model = lgb.train(
            hyperparameters,
            train_data,
            num_boost_round=max_rounds,
            valid_sets=[val_data],
            callbacks=[lgb.early_stopping(early_stopping_rounds, verbose=False)],
        )

        best_iter = model.best_iteration
        oof_predictions[val_idx] = model.predict(x_va, num_iteration=best_iter)
        test_predictions += (
            model.predict(x_test, num_iteration=best_iter) / len(cv_folds)
        )

        fold_auc = roc_auc_score(y_va, oof_predictions[val_idx])
        logger.info("  LightGBM Fold %d: AUC = %.5f (Best iteration: %d)", fold_idx + 1, fold_auc, best_iter)

    total_auc = float(roc_auc_score(y_train, oof_predictions))
    logger.info("==> LightGBM Overall OOF AUC: %.5f", total_auc)
    return oof_predictions, test_predictions, total_auc


def train_catboost_cv(
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: np.ndarray,
    cv_folds: List[Tuple[np.ndarray, np.ndarray]],
    categorical_features: List[str],
    hyperparameters: Dict[str, Any],
    early_stopping_rounds: int = 300,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Train 5-fold CatBoost model and predict out-of-fold and test ranks."""
    oof_predictions = np.zeros(len(x_train))
    test_predictions = np.zeros(len(x_test))

    # Convert categoricals to strings for CatBoost native handler
    x_train_cat = x_train.copy()
    x_test_cat = x_test.copy()
    for col in categorical_features:
        if col in x_train_cat.columns:
            x_train_cat[col] = x_train_cat[col].astype(str)
            x_test_cat[col] = x_test_cat[col].astype(str)

    test_pool = Pool(x_test_cat, cat_features=categorical_features)
    logger.info("Training CatBoost across %d folds...", len(cv_folds))

    for fold_idx, (train_idx, val_idx) in enumerate(cv_folds):
        x_tr, y_tr = x_train_cat.iloc[train_idx], y_train[train_idx]
        x_va, y_va = x_train_cat.iloc[val_idx], y_train[val_idx]

        train_pool = Pool(x_tr, y_tr, cat_features=categorical_features)
        val_pool = Pool(x_va, y_va, cat_features=categorical_features)

        model = CatBoostClassifier(**hyperparameters)
        model.fit(
            train_pool,
            eval_set=val_pool,
            early_stopping_rounds=early_stopping_rounds,
            verbose=False,
        )

        best_iter = model.best_iteration_
        oof_predictions[val_idx] = model.predict_proba(x_va)[:, 1]
        test_predictions += model.predict_proba(test_pool)[:, 1] / len(cv_folds)

        fold_auc = roc_auc_score(y_va, oof_predictions[val_idx])
        logger.info("  CatBoost Fold %d: AUC = %.5f (Best iteration: %d)", fold_idx + 1, fold_auc, best_iter)

    total_auc = float(roc_auc_score(y_train, oof_predictions))
    logger.info("==> CatBoost Overall OOF AUC: %.5f", total_auc)
    return oof_predictions, test_predictions, total_auc


def blend_predictions_rank_average(
    predictions_list: List[np.ndarray], weights: List[float]
) -> np.ndarray:
    """Ensemble multiple prediction arrays via normalized weighted rank averaging."""
    if len(predictions_list) != len(weights):
        raise ValueError("predictions_list and weights must have identical length")

    normalized_weights = np.array(weights) / sum(weights)
    ensemble_ranks = np.zeros(len(predictions_list[0]))

    for preds, weight in zip(predictions_list, normalized_weights):
        ensemble_ranks += weight * compute_rank_probabilities(preds)

    return ensemble_ranks
