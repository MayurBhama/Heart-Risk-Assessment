"""
Model Training Module - Industry-Grade Version

This module provides a comprehensive training pipeline with:
- Factory pattern for model creation (reduces code duplication)
- Cross-validation support with StratifiedKFold
- Optional hyperparameter tuning via Optuna integration
- MLflow experiment tracking
- Consistent train/val/test splits saved to disk

Author: Heart-Risk-Assessment Team
"""

import pandas as pd
import numpy as np
import yaml
import sys
import joblib
import mlflow
import mlflow.sklearn
import json
from pathlib import Path
from typing import Dict, Tuple, List, Any, Optional, Type
from datetime import datetime

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.base import BaseEstimator
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)

import warnings

warnings.filterwarnings("ignore")

from utils.logger import logger
from utils.exception import CustomException


# =============================================================================
# Model Factory - Reduces Code Duplication
# =============================================================================


class ModelFactory:
    """
    Factory class for creating ML models with standardized configurations.
    
    This eliminates code duplication by centralizing model creation logic.
    
    Attributes:
        MODEL_REGISTRY: Mapping of model names to their classes.
    """

    MODEL_REGISTRY: Dict[str, Type[BaseEstimator]] = {
        "Logistic_Regression": LogisticRegression,
        "Random_Forest": RandomForestClassifier,
        "Gradient_Boosting": GradientBoostingClassifier,
        "XGBoost": XGBClassifier,
    }

    @classmethod
    def create_model(cls, model_name: str, params: Dict[str, Any]) -> BaseEstimator:
        """
        Create a model instance with the given parameters.

        Args:
            model_name: Name of the model to create.
            params: Hyperparameters for the model.

        Returns:
            Configured model instance.

        Raises:
            ValueError: If model_name is not in the registry.
        """
        if model_name not in cls.MODEL_REGISTRY:
            raise ValueError(
                f"Unknown model: {model_name}. Available: {list(cls.MODEL_REGISTRY.keys())}"
            )

        model_class = cls.MODEL_REGISTRY[model_name]

        # Handle XGBoost-specific parameters
        if model_name == "XGBoost":
            params = params.copy()
            params.pop("use_label_encoder", None)  # Remove deprecated param

        return model_class(**params)

    @classmethod
    def get_config_key(cls, model_name: str) -> str:
        """
        Get the configuration key for a model name.

        Args:
            model_name: Display name of the model.

        Returns:
            Configuration key (snake_case).
        """
        key_mapping = {
            "Logistic_Regression": "logistic_regression",
            "Random_Forest": "random_forest",
            "Gradient_Boosting": "gradient_boosting",
            "XGBoost": "xgboost",
        }
        return key_mapping.get(model_name, model_name.lower())


# =============================================================================
# Main Model Trainer Class
# =============================================================================


class ModelTrainer:
    """
    Industry-grade training class with MLflow logging, cross-validation,
    and factory pattern for model creation.

    Attributes:
        config: Configuration dictionary loaded from YAML.
        models: Dictionary of trained model instances.
        results: Dictionary of evaluation results per model.
        scalers: Dictionary of fitted scalers.
        best_model: Best performing model instance.
        best_model_name: Name of the best model.
        cv_results: Cross-validation results if CV was performed.
    """

    def __init__(self, config_path: str = "configs/model_config.yaml") -> None:
        """
        Initialize ModelTrainer with configuration.

        Args:
            config_path: Path to the model configuration YAML file.

        Raises:
            CustomException: If initialization fails.
        """
        try:
            self.config: Dict[str, Any] = self._load_config(config_path)
            self.models: Dict[str, BaseEstimator] = {}
            self.results: Dict[str, Dict[str, Any]] = {}
            self.scalers: Dict[str, StandardScaler] = {}
            self.best_model: Optional[BaseEstimator] = None
            self.best_model_name: Optional[str] = None
            self.cv_results: Dict[str, Dict[str, Any]] = {}

            # Setup MLflow
            mlflow_tracking_uri = self.config.get("mlflow", {}).get("tracking_uri")
            mlflow_experiment = self.config.get("mlflow", {}).get("experiment_name")

            if mlflow_tracking_uri:
                mlflow.set_tracking_uri(mlflow_tracking_uri)
            if mlflow_experiment:
                mlflow.set_experiment(mlflow_experiment)

            logger.info("ModelTrainer initialized successfully")

        except Exception as e:
            raise CustomException(e, sys)

    def _load_config(self, path: str) -> Dict[str, Any]:
        """
        Load configuration from YAML file.

        Args:
            path: Path to configuration file.

        Returns:
            Configuration dictionary.

        Raises:
            CustomException: If loading fails.
        """
        try:
            with open(path, "r") as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Failed to load config from {path}")
            raise CustomException(e, sys)

    def prepare_data(
        self, df: pd.DataFrame, feature_columns: List[str]
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series, StandardScaler]:
        """
        Prepare data for training: split, scale, and save splits.

        Args:
            df: Input DataFrame with features and target.
            feature_columns: List of feature column names.

        Returns:
            Tuple of (X_train, X_val, X_test, y_train, y_val, y_test, scaler).

        Raises:
            CustomException: If data preparation fails.
        """
        try:
            logger.info("Preparing data for model training")

            features = [c for c in feature_columns if c in df.columns]
            if len(features) == 0:
                raise ValueError("No valid feature columns found")

            X = df[features].copy()
            y = df["cardio"].copy()

            # Get split parameters from config
            test_size = self.config["preprocessing"]["test_size"]
            val_size = self.config["preprocessing"]["validation_size"]
            random_state = self.config["preprocessing"]["random_state"]

            # First split: train+val vs test
            X_temp, X_test, y_temp, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state, stratify=y
            )

            # Second split: train vs val
            val_ratio = val_size / (1 - test_size)
            X_train, X_val, y_train, y_val = train_test_split(
                X_temp, y_temp, test_size=val_ratio, random_state=random_state, stratify=y_temp
            )

            logger.info(f"Data split: Train={X_train.shape}, Val={X_val.shape}, Test={X_test.shape}")

            # Save raw splits before scaling
            self._save_data_splits(X_train, y_train, X_val, y_val, X_test, y_test, features)

            # Scale features
            scaler = StandardScaler()
            X_train_s = pd.DataFrame(
                scaler.fit_transform(X_train), columns=features, index=X_train.index
            )
            X_val_s = pd.DataFrame(
                scaler.transform(X_val), columns=features, index=X_val.index
            )
            X_test_s = pd.DataFrame(
                scaler.transform(X_test), columns=features, index=X_test.index
            )

            logger.info("Data preparation completed")

            return (
                X_train_s,
                X_val_s,
                X_test_s,
                y_train.reset_index(drop=True),
                y_val.reset_index(drop=True),
                y_test.reset_index(drop=True),
                scaler,
            )

        except Exception as e:
            raise CustomException(e, sys)

    def _save_data_splits(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        features: List[str],
    ) -> None:
        """
        Save train/val/test splits to CSV for reproducible evaluation.

        Args:
            X_train, y_train: Training data.
            X_val, y_val: Validation data.
            X_test, y_test: Test data.
            features: List of feature names.

        Raises:
            CustomException: If saving fails.
        """
        try:
            logger.info("Saving data splits to CSV...")

            # Create directories
            for dir_name in ["data/train", "data/validation", "data/test"]:
                Path(dir_name).mkdir(parents=True, exist_ok=True)

            # Save each split
            splits = [
                ("data/train/train.csv", X_train, y_train),
                ("data/validation/validation.csv", X_val, y_val),
                ("data/test/test.csv", X_test, y_test),
            ]

            for path, X, y in splits:
                df = X.copy().reset_index(drop=True)
                df["cardio"] = y.reset_index(drop=True).values
                df.to_csv(path, index=False)
                logger.info(f"  Saved: {path} ({df.shape})")

            # Save feature metadata
            feature_metadata = {
                "features": features,
                "n_features": len(features),
                "saved_at": datetime.now().isoformat(),
            }

            Path("models/trained_models").mkdir(parents=True, exist_ok=True)
            with open("models/trained_models/feature_names.json", "w") as f:
                json.dump(feature_metadata, f, indent=4)

            logger.info(f"  Saved: feature_names.json ({len(features)} features)")

        except Exception as e:
            logger.error("Failed to save data splits!")
            raise CustomException(e, sys)

    def _evaluate_model(
        self,
        model: BaseEstimator,
        X_train: pd.DataFrame,
        X_val: pd.DataFrame,
        y_train: pd.Series,
        y_val: pd.Series,
    ) -> Dict[str, Any]:
        """
        Evaluate a model on training and validation sets.

        Args:
            model: Trained model to evaluate.
            X_train, y_train: Training data.
            X_val, y_val: Validation data.

        Returns:
            Dictionary with train_metrics, val_metrics, confusion_matrix, and report.

        Raises:
            CustomException: If evaluation fails.
        """
        try:
            # Training predictions
            y_train_pred = model.predict(X_train)
            y_train_proba = (
                model.predict_proba(X_train)[:, 1]
                if hasattr(model, "predict_proba")
                else np.zeros(len(y_train))
            )

            # Validation predictions
            y_val_pred = model.predict(X_val)
            y_val_proba = (
                model.predict_proba(X_val)[:, 1]
                if hasattr(model, "predict_proba")
                else np.zeros(len(y_val))
            )

            def calc_metrics(y_true: pd.Series, y_pred: np.ndarray, y_proba: np.ndarray) -> Dict[str, float]:
                return {
                    "accuracy": accuracy_score(y_true, y_pred),
                    "precision": precision_score(y_true, y_pred, zero_division=0),
                    "recall": recall_score(y_true, y_pred, zero_division=0),
                    "f1_score": f1_score(y_true, y_pred, zero_division=0),
                    "roc_auc": roc_auc_score(y_true, y_proba) if len(np.unique(y_proba)) > 1 else 0.0,
                }

            return {
                "train_metrics": calc_metrics(y_train, y_train_pred, y_train_proba),
                "val_metrics": calc_metrics(y_val, y_val_pred, y_val_proba),
                "confusion_matrix": confusion_matrix(y_val, y_val_pred).tolist(),
                "report": classification_report(y_val, y_val_pred, zero_division=0),
            }

        except Exception as e:
            raise CustomException(e, sys)

    def _train_single_model(
        self,
        model_name: str,
        X_train: pd.DataFrame,
        X_val: pd.DataFrame,
        y_train: pd.Series,
        y_val: pd.Series,
    ) -> None:
        """
        Train a single model using the factory pattern.

        Args:
            model_name: Name of the model to train.
            X_train, y_train: Training data.
            X_val, y_val: Validation data.

        Raises:
            CustomException: If training fails.
        """
        try:
            config_key = ModelFactory.get_config_key(model_name)
            params = self.config.get("models", {}).get(config_key, {})

            with mlflow.start_run(run_name=model_name):
                # Create and train model
                model = ModelFactory.create_model(model_name, params)

                # Handle XGBoost eval_set
                if model_name == "XGBoost" and params.get("early_stopping_rounds"):
                    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
                else:
                    model.fit(X_train, y_train)

                # Evaluate
                results = self._evaluate_model(model, X_train, X_val, y_train, y_val)

                # Log to MLflow
                mlflow.log_params(params)
                mlflow.log_metrics(results["val_metrics"])

                # Log feature importance for tree-based models
                if hasattr(model, "feature_importances_"):
                    feature_importance = dict(zip(X_train.columns, model.feature_importances_))
                    mlflow.log_dict(feature_importance, "feature_importance.json")

                mlflow.sklearn.log_model(model, "model")

                # Store results
                self.models[model_name] = model
                self.results[model_name] = results

                logger.info(f"{model_name} completed: AUC={results['val_metrics']['roc_auc']:.4f}")

        except Exception as e:
            logger.error(f"Training failed for {model_name}")
            raise CustomException(e, sys)

    def train_all_models(
        self,
        X_train: pd.DataFrame,
        X_val: pd.DataFrame,
        y_train: pd.Series,
        y_val: pd.Series,
    ) -> None:
        """
        Train all models in the registry.

        Args:
            X_train, y_train: Training data.
            X_val, y_val: Validation data.

        Raises:
            CustomException: If training fails.
        """
        try:
            logger.info("Starting training of all models")

            for model_name in ModelFactory.MODEL_REGISTRY.keys():
                self._train_single_model(model_name, X_train, X_val, y_train, y_val)

            self._select_best_model()
            logger.info("All models trained successfully")

        except Exception as e:
            raise CustomException(e, sys)

    def train_with_cross_validation(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        n_folds: int = 5,
        scoring: str = "roc_auc",
    ) -> Dict[str, Dict[str, Any]]:
        """
        Train all models using stratified k-fold cross-validation.

        Args:
            X: Feature DataFrame (scaled).
            y: Target Series.
            n_folds: Number of CV folds.
            scoring: Scoring metric for CV.

        Returns:
            Dictionary of CV results per model.

        Raises:
            CustomException: If CV training fails.
        """
        try:
            logger.info(f"Starting {n_folds}-fold cross-validation for all models")

            cv = StratifiedKFold(
                n_splits=n_folds,
                shuffle=True,
                random_state=self.config["preprocessing"]["random_state"],
            )

            cv_results: Dict[str, Dict[str, Any]] = {}

            for model_name in ModelFactory.MODEL_REGISTRY.keys():
                config_key = ModelFactory.get_config_key(model_name)
                params = self.config.get("models", {}).get(config_key, {})
                model = ModelFactory.create_model(model_name, params)

                scores = cross_val_score(model, X, y, cv=cv, scoring=scoring, n_jobs=-1)

                cv_results[model_name] = {
                    "mean": float(scores.mean()),
                    "std": float(scores.std()),
                    "scores": scores.tolist(),
                    "ci_95": f"{scores.mean():.4f} +/- {scores.std() * 1.96:.4f}",
                }

                logger.info(
                    f"{model_name} CV {scoring}: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})"
                )

            self.cv_results = cv_results

            # Find best model by CV score
            best_model_name = max(cv_results.items(), key=lambda x: x[1]["mean"])[0]
            logger.info(f"Best model by CV: {best_model_name}")

            return cv_results

        except Exception as e:
            logger.error("Cross-validation training failed")
            raise CustomException(e, sys)

    def _select_best_model(self) -> None:
        """
        Select the best performing model based on validation ROC-AUC.

        Raises:
            CustomException: If no models were trained.
        """
        try:
            best_score = -1.0
            best_name: Optional[str] = None

            for name, result in self.results.items():
                auc = result["val_metrics"].get("roc_auc", 0.0) or 0.0
                if auc > best_score:
                    best_score = auc
                    best_name = name

            if best_name is None:
                raise ValueError("No model was trained successfully")

            self.best_model_name = best_name
            self.best_model = self.models[best_name]

            logger.info(f"Best Model Selected -> {best_name} (ROC-AUC={best_score:.4f})")

        except Exception as e:
            raise CustomException(e, sys)

    def evaluate_on_test(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
        """
        Evaluate the best model on the test set.

        Args:
            X_test: Test features.
            y_test: Test target.

        Returns:
            Dictionary of test metrics.

        Raises:
            CustomException: If evaluation fails.
        """
        try:
            if self.best_model is None:
                raise ValueError("No best model available. Train models first.")

            logger.info(f"Evaluating best model: {self.best_model_name}")

            pred = self.best_model.predict(X_test)
            proba = (
                self.best_model.predict_proba(X_test)[:, 1]
                if hasattr(self.best_model, "predict_proba")
                else np.zeros(len(y_test))
            )

            metrics = {
                "accuracy": accuracy_score(y_test, pred),
                "precision": precision_score(y_test, pred, zero_division=0),
                "recall": recall_score(y_test, pred, zero_division=0),
                "f1": f1_score(y_test, pred, zero_division=0),
                "roc_auc": roc_auc_score(y_test, proba) if len(np.unique(proba)) > 1 else 0.0,
                "confusion_matrix": confusion_matrix(y_test, pred).tolist(),
            }

            logger.info(f"Test evaluation completed: AUC={metrics['roc_auc']:.4f}")
            return metrics

        except Exception as e:
            raise CustomException(e, sys)

    def save_best_model(self, model_path: str, scaler_path: str) -> None:
        """
        Save the best model and scaler to disk.

        Args:
            model_path: Path to save the model.
            scaler_path: Path to save the scaler.

        Raises:
            CustomException: If saving fails.
        """
        try:
            Path(model_path).parent.mkdir(parents=True, exist_ok=True)
            Path(scaler_path).parent.mkdir(parents=True, exist_ok=True)

            if self.best_model is None:
                raise ValueError("No best model to save")

            joblib.dump(self.best_model, model_path)
            logger.info(f"Best model saved at: {model_path}")

            scaler = self.scalers.get("scaler")
            if scaler is not None:
                joblib.dump(scaler, scaler_path)
                logger.info(f"Scaler saved at: {scaler_path}")
            else:
                logger.warning("No scaler found to save")

        except Exception as e:
            raise CustomException(e, sys)

    def get_training_summary(self) -> pd.DataFrame:
        """
        Get a summary DataFrame of all model results.

        Returns:
            DataFrame with model performance comparison.
        """
        if not self.results:
            logger.warning("No results available")
            return pd.DataFrame()

        summary_data = []
        for model_name, result in self.results.items():
            metrics = result["val_metrics"]
            summary_data.append(
                {
                    "model": model_name,
                    "accuracy": metrics["accuracy"],
                    "precision": metrics["precision"],
                    "recall": metrics["recall"],
                    "f1_score": metrics["f1_score"],
                    "roc_auc": metrics["roc_auc"],
                    "is_best": model_name == self.best_model_name,
                }
            )

        return pd.DataFrame(summary_data).sort_values("roc_auc", ascending=False)


# =============================================================================
# Main Execution
# =============================================================================


def main() -> ModelTrainer:
    """
    Main training pipeline execution.

    Returns:
        Trained ModelTrainer instance.

    Raises:
        CustomException: If pipeline fails.
    """
    try:
        df = pd.read_csv("data/processed/cardio_featured.csv")

        features = [
            "age_years", "gender", "height", "weight",
            "ap_hi", "ap_lo", "cholesterol", "gluc",
            "smoke", "alco", "active", "bmi", "bmi_category",
            "pulse_pressure", "mean_arterial_pressure",
            "bp_category", "age_group", "lifestyle_risk_score",
            "metabolic_risk_score", "combined_risk_score",
        ]

        trainer = ModelTrainer()

        X_train, X_val, X_test, y_train, y_val, y_test, scaler = trainer.prepare_data(df, features)
        trainer.scalers["scaler"] = scaler

        # Standard training
        trainer.train_all_models(X_train, X_val, y_train, y_val)

        # Optional: Cross-validation (uncomment to enable)
        # X_combined = pd.concat([X_train, X_val])
        # y_combined = pd.concat([y_train, y_val])
        # cv_results = trainer.train_with_cross_validation(X_combined, y_combined)

        test_results = trainer.evaluate_on_test(X_test, y_test)

        trainer.save_best_model(
            "models/trained_models/best_model.pkl",
            "models/trained_models/scaler.pkl",
        )

        # Print summary
        summary = trainer.get_training_summary()
        logger.info("\n" + "=" * 80)
        logger.info("TRAINING SUMMARY")
        logger.info("=" * 80)
        logger.info(f"\n{summary.to_string()}")
        logger.info("\n" + "=" * 80)
        logger.info("Training pipeline completed successfully")
        logger.info("=" * 80 + "\n")

        return trainer

    except Exception as e:
        raise CustomException(e, sys)


if __name__ == "__main__":
    main()
