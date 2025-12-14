"""
Hyperparameter Tuning Module with Optuna

This module provides automated hyperparameter optimization for ML models
using Optuna with cross-validation and MLflow integration.
"""

import optuna
import pandas as pd
import numpy as np
import sys
from typing import Dict, Any, Optional, Callable, List, Tuple
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
import mlflow
import warnings

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

from utils.logger import logger
from utils.exception import CustomException


class HyperparameterTuner:
    """
    Automated hyperparameter tuning using Optuna with cross-validation.
    
    Attributes:
        n_trials: Number of optimization trials.
        cv_folds: Number of cross-validation folds.
        scoring: Scoring metric for optimization.
        random_state: Random seed for reproducibility.
    """

    def __init__(
        self,
        n_trials: int = 50,
        cv_folds: int = 5,
        scoring: str = "roc_auc",
        random_state: int = 42,
    ) -> None:
        """
        Initialize the HyperparameterTuner.

        Args:
            n_trials: Number of Optuna optimization trials.
            cv_folds: Number of folds for cross-validation.
            scoring: Sklearn scoring metric (e.g., 'roc_auc', 'f1', 'accuracy').
            random_state: Random seed for reproducibility.
        """
        self.n_trials = n_trials
        self.cv_folds = cv_folds
        self.scoring = scoring
        self.random_state = random_state
        self.best_params: Dict[str, Any] = {}
        self.study_results: Dict[str, optuna.Study] = {}
        logger.info(
            f"HyperparameterTuner initialized: trials={n_trials}, cv={cv_folds}, scoring={scoring}"
        )

    def _get_search_space(self, model_name: str, trial: optuna.Trial) -> Dict[str, Any]:
        """
        Define hyperparameter search spaces for each model type.

        Args:
            model_name: Name of the model ('logistic_regression', 'random_forest', etc.).
            trial: Optuna trial object.

        Returns:
            Dictionary of hyperparameters for the trial.

        Raises:
            ValueError: If model_name is not recognized.
        """
        if model_name == "logistic_regression":
            return {
                "C": trial.suggest_float("C", 0.01, 10.0, log=True),
                "max_iter": 1000,
                "solver": trial.suggest_categorical("solver", ["lbfgs", "liblinear"]),
                "random_state": self.random_state,
            }

        elif model_name == "random_forest":
            return {
                "n_estimators": trial.suggest_int("n_estimators", 50, 300),
                "max_depth": trial.suggest_int("max_depth", 5, 30),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
                "random_state": self.random_state,
                "n_jobs": -1,
            }

        elif model_name == "gradient_boosting":
            return {
                "n_estimators": trial.suggest_int("n_estimators", 50, 300),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
                "random_state": self.random_state,
            }

        elif model_name == "xgboost":
            return {
                "n_estimators": trial.suggest_int("n_estimators", 50, 300),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "random_state": self.random_state,
                "eval_metric": "logloss",
                "use_label_encoder": False,
            }

        else:
            raise ValueError(f"Unknown model: {model_name}")

    def _get_model_class(self, model_name: str) -> type:
        """
        Get the sklearn model class for the given model name.

        Args:
            model_name: Name of the model.

        Returns:
            The model class.

        Raises:
            ValueError: If model_name is not recognized.
        """
        model_classes = {
            "logistic_regression": LogisticRegression,
            "random_forest": RandomForestClassifier,
            "gradient_boosting": GradientBoostingClassifier,
            "xgboost": XGBClassifier,
        }
        if model_name not in model_classes:
            raise ValueError(f"Unknown model: {model_name}")
        return model_classes[model_name]

    def _create_objective(
        self, model_name: str, X: pd.DataFrame, y: pd.Series
    ) -> Callable[[optuna.Trial], float]:
        """
        Create an Optuna objective function for a specific model.

        Args:
            model_name: Name of the model to optimize.
            X: Feature DataFrame.
            y: Target Series.

        Returns:
            Objective function for Optuna optimization.
        """

        def objective(trial: optuna.Trial) -> float:
            params = self._get_search_space(model_name, trial)
            model_class = self._get_model_class(model_name)
            model = model_class(**params)

            cv = StratifiedKFold(
                n_splits=self.cv_folds, shuffle=True, random_state=self.random_state
            )

            try:
                scores = cross_val_score(model, X, y, cv=cv, scoring=self.scoring, n_jobs=-1)
                return scores.mean()
            except Exception as e:
                logger.warning(f"Trial failed: {e}")
                return 0.0

        return objective

    def tune_model(
        self, model_name: str, X: pd.DataFrame, y: pd.Series, direction: str = "maximize"
    ) -> Tuple[Dict[str, Any], float]:
        """
        Run hyperparameter optimization for a single model.

        Args:
            model_name: Name of the model to tune.
            X: Feature DataFrame.
            y: Target Series.
            direction: Optimization direction ('maximize' or 'minimize').

        Returns:
            Tuple of (best_params, best_score).

        Raises:
            CustomException: If optimization fails.
        """
        try:
            logger.info(f"Starting hyperparameter tuning for {model_name}")

            study = optuna.create_study(direction=direction)
            objective = self._create_objective(model_name, X, y)

            study.optimize(objective, n_trials=self.n_trials, show_progress_bar=False)

            best_params = study.best_params
            best_score = study.best_value

            self.best_params[model_name] = best_params
            self.study_results[model_name] = study

            logger.info(f"{model_name} tuning complete: best {self.scoring}={best_score:.4f}")
            logger.info(f"Best params: {best_params}")

            return best_params, best_score

        except Exception as e:
            logger.error(f"Hyperparameter tuning failed for {model_name}")
            raise CustomException(e, sys)

    def tune_all_models(
        self, X: pd.DataFrame, y: pd.Series
    ) -> Dict[str, Tuple[Dict[str, Any], float]]:
        """
        Run hyperparameter optimization for all supported models.

        Args:
            X: Feature DataFrame.
            y: Target Series.

        Returns:
            Dictionary mapping model names to (best_params, best_score) tuples.

        Raises:
            CustomException: If any optimization fails.
        """
        try:
            logger.info("Starting hyperparameter tuning for all models")

            models = ["logistic_regression", "random_forest", "gradient_boosting", "xgboost"]
            results: Dict[str, Tuple[Dict[str, Any], float]] = {}

            for model_name in models:
                best_params, best_score = self.tune_model(model_name, X, y)
                results[model_name] = (best_params, best_score)

            # Find best overall model
            best_model = max(results.items(), key=lambda x: x[1][1])
            logger.info(
                f"Best overall model: {best_model[0]} with {self.scoring}={best_model[1][1]:.4f}"
            )

            return results

        except Exception as e:
            logger.error("Failed to tune all models")
            raise CustomException(e, sys)

    def get_tuned_model(self, model_name: str) -> Any:
        """
        Get a model instance with the best tuned parameters.

        Args:
            model_name: Name of the model.

        Returns:
            Model instance with optimized hyperparameters.

        Raises:
            ValueError: If model hasn't been tuned yet.
        """
        if model_name not in self.best_params:
            raise ValueError(f"Model {model_name} has not been tuned yet. Run tune_model first.")

        model_class = self._get_model_class(model_name)
        params = self.best_params[model_name]

        # Add fixed params that might not be in search space
        if model_name == "logistic_regression":
            params["max_iter"] = 1000
            params["random_state"] = self.random_state
        elif model_name in ["random_forest", "gradient_boosting"]:
            params["random_state"] = self.random_state
        elif model_name == "xgboost":
            params["random_state"] = self.random_state
            params["eval_metric"] = "logloss"
            params["use_label_encoder"] = False

        return model_class(**params)


class CrossValidator:
    """
    Cross-validation utility for model evaluation.
    
    Provides k-fold cross-validation with detailed metrics reporting.
    """

    def __init__(self, n_folds: int = 5, random_state: int = 42) -> None:
        """
        Initialize CrossValidator.

        Args:
            n_folds: Number of cross-validation folds.
            random_state: Random seed for reproducibility.
        """
        self.n_folds = n_folds
        self.random_state = random_state
        self.cv_results: Dict[str, Any] = {}
        logger.info(f"CrossValidator initialized with {n_folds} folds")

    def cross_validate(
        self,
        model: Any,
        X: pd.DataFrame,
        y: pd.Series,
        metrics: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Perform stratified k-fold cross-validation.

        Args:
            model: Sklearn-compatible model to evaluate.
            X: Feature DataFrame.
            y: Target Series.
            metrics: List of scoring metrics (default: ['accuracy', 'roc_auc', 'f1']).

        Returns:
            Dictionary containing mean, std, and per-fold scores for each metric.

        Raises:
            CustomException: If cross-validation fails.
        """
        try:
            if metrics is None:
                metrics = ["accuracy", "roc_auc", "f1"]

            cv = StratifiedKFold(
                n_splits=self.n_folds, shuffle=True, random_state=self.random_state
            )

            results: Dict[str, Any] = {}

            for metric in metrics:
                scores = cross_val_score(model, X, y, cv=cv, scoring=metric, n_jobs=-1)
                results[metric] = {
                    "mean": float(scores.mean()),
                    "std": float(scores.std()),
                    "scores": scores.tolist(),
                }
                logger.info(f"CV {metric}: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")

            self.cv_results = results
            return results

        except Exception as e:
            logger.error("Cross-validation failed")
            raise CustomException(e, sys)

    def get_cv_summary(self) -> pd.DataFrame:
        """
        Get a summary DataFrame of cross-validation results.

        Returns:
            DataFrame with mean and std for each metric.
        """
        if not self.cv_results:
            logger.warning("No CV results available. Run cross_validate first.")
            return pd.DataFrame()

        summary_data = []
        for metric, values in self.cv_results.items():
            summary_data.append(
                {
                    "metric": metric,
                    "mean": values["mean"],
                    "std": values["std"],
                    "ci_95": f"{values['mean']:.4f} +/- {values['std'] * 1.96:.4f}",
                }
            )

        return pd.DataFrame(summary_data)
