"""
Feature Engineering Module

Creates new derived features to improve model performance:
1. BMI and body composition features
2. Blood pressure derived metrics
3. Composite risk scores
4. Age-related features
5. Interaction features

Author: Heart-Risk-Assessment Team
"""

import pandas as pd
import numpy as np
import yaml
import sys
from typing import Dict, List, Any, Callable
import warnings

warnings.filterwarnings("ignore")

from utils.logger import logger
from utils.exception import CustomException


class FeatureEngineer:
    """
    Feature engineering class for cardiovascular disease prediction.
    
    Creates clinically relevant features derived from patient data
    to improve model predictive performance.
    
    Attributes:
        config: Configuration dictionary loaded from YAML.
        feature_config: Feature engineering specific configuration.
        new_features: List of newly created feature names.
    """

    def __init__(self, config_path: str = "configs/model_config.yaml") -> None:
        """
        Initialize FeatureEngineer.

        Args:
            config_path: Path to model configuration file.

        Raises:
            CustomException: If initialization fails.
        """
        try:
            self.config: Dict[str, Any] = self._load_config(config_path)
            self.feature_config: Dict[str, Any] = self.config["feature_engineering"]
            self.new_features: List[str] = []
            logger.info("FeatureEngineer initialized successfully")
        except Exception as e:
            raise CustomException(e, sys)

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """
        Load configuration from YAML file.

        Args:
            config_path: Path to configuration file.

        Returns:
            Configuration dictionary.

        Raises:
            CustomException: If loading fails.
        """
        try:
            with open(config_path, "r") as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Failed to load config file: {config_path}")
            raise CustomException(e, sys)

    def create_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Execute the full feature engineering pipeline.

        Creates all configured features in sequence:
        1. BMI and body composition features
        2. Blood pressure derived features
        3. Age group features
        4. Composite risk scores
        5. Interaction features

        Args:
            df: Input DataFrame with cleaned patient data.

        Returns:
            DataFrame with all new engineered features.

        Raises:
            CustomException: If any feature creation fails.
        """
        try:
            logger.info("Starting feature engineering...")
            df_feat = df.copy()

            if self.feature_config.get("create_bmi"):
                df_feat = self._create_bmi_features(df_feat)

            if self.feature_config.get("create_bp_features"):
                df_feat = self._create_bp_features(df_feat)

            if self.feature_config.get("create_age_groups"):
                df_feat = self._create_age_features(df_feat)

            if self.feature_config.get("create_risk_scores"):
                df_feat = self._create_risk_scores(df_feat)

            df_feat = self._create_interaction_features(df_feat)

            logger.info(
                f"Feature engineering completed. Total new features: {len(self.new_features)}"
            )
            logger.info(f"New features: {self.new_features}")

            return df_feat

        except Exception as e:
            raise CustomException(e, sys)

    def _create_bmi_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create BMI and body composition features.

        Features created:
        - bmi: Body Mass Index = weight / height^2
        - bmi_category: WHO BMI categories (0-3)
        - bsa: Body Surface Area
        - weight_height_ratio: Simple weight/height ratio

        Args:
            df: Input DataFrame.

        Returns:
            DataFrame with BMI features added.

        Raises:
            CustomException: If creation fails.
        """
        try:
            logger.info("Creating BMI features...")

            # BMI calculation
            df["bmi"] = (df["weight"] / ((df["height"] / 100) ** 2)).round(2)
            self.new_features.append("bmi")

            # BMI categorization (WHO standards)
            def categorize_bmi(bmi: float) -> int:
                if bmi < 18.5:
                    return 0  # Underweight
                elif bmi < 25:
                    return 1  # Normal
                elif bmi < 30:
                    return 2  # Overweight
                return 3  # Obese

            df["bmi_category"] = df["bmi"].apply(categorize_bmi)
            self.new_features.append("bmi_category")

            # Body Surface Area (DuBois formula simplified)
            df["bsa"] = np.sqrt((df["height"] * df["weight"]) / 3600).round(2)
            self.new_features.append("bsa")

            # Weight-to-height ratio
            df["weight_height_ratio"] = (df["weight"] / df["height"] * 100).round(2)
            self.new_features.append("weight_height_ratio")

            return df

        except Exception as e:
            raise CustomException(e, sys)

    def _create_bp_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create blood pressure derived features.

        Features created:
        - pulse_pressure: Systolic - Diastolic
        - mean_arterial_pressure: DBP + (PP / 3)
        - bp_category: JNC-7 BP classification (0-4)
        - bp_ratio: Systolic / Diastolic ratio

        Args:
            df: Input DataFrame.

        Returns:
            DataFrame with BP features added.

        Raises:
            CustomException: If creation fails.
        """
        try:
            logger.info("Creating blood pressure features...")

            # Pulse pressure
            df["pulse_pressure"] = df["ap_hi"] - df["ap_lo"]
            self.new_features.append("pulse_pressure")

            # Mean Arterial Pressure
            df["mean_arterial_pressure"] = (
                df["ap_lo"] + (df["pulse_pressure"] / 3)
            ).round(1)
            self.new_features.append("mean_arterial_pressure")

            # BP Category (JNC-7 classification)
            def categorize_bp(row: pd.Series) -> int:
                s, d = row["ap_hi"], row["ap_lo"]
                if s < 120 and d < 80:
                    return 0  # Normal
                if s < 130 and d < 80:
                    return 1  # Elevated
                if s < 140 or d < 90:
                    return 2  # Stage 1 Hypertension
                if s < 180 or d < 120:
                    return 3  # Stage 2 Hypertension
                return 4  # Hypertensive Crisis

            df["bp_category"] = df.apply(categorize_bp, axis=1)
            self.new_features.append("bp_category")

            # BP ratio
            df["bp_ratio"] = (df["ap_hi"] / df["ap_lo"]).round(2)
            self.new_features.append("bp_ratio")

            return df

        except Exception as e:
            raise CustomException(e, sys)

    def _create_age_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create age-related features.

        Features created:
        - age_group: Age categories (0-3)
        - age_squared: Polynomial age feature

        Args:
            df: Input DataFrame.

        Returns:
            DataFrame with age features added.

        Raises:
            CustomException: If creation fails.
        """
        try:
            logger.info("Creating age features...")

            def categorize_age(age: int) -> int:
                if age < 40:
                    return 0  # Young adult
                if age < 50:
                    return 1  # Middle age early
                if age < 60:
                    return 2  # Middle age late
                return 3  # Senior

            df["age_group"] = df["age_years"].apply(categorize_age)
            self.new_features.append("age_group")

            # Age squared for non-linear relationships
            df["age_squared"] = df["age_years"] ** 2
            self.new_features.append("age_squared")

            return df

        except Exception as e:
            raise CustomException(e, sys)

    def _create_risk_scores(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create composite risk score features.

        Features created:
        - lifestyle_risk_score: Smoking + alcohol + inactivity
        - metabolic_risk_score: Cholesterol + glucose levels
        - combined_risk_score: Weighted composite of all risk factors
        - age_adjusted_risk: Risk score adjusted for age

        Args:
            df: Input DataFrame.

        Returns:
            DataFrame with risk score features added.

        Raises:
            CustomException: If creation fails.
        """
        try:
            logger.info("Creating risk score features...")

            # Lifestyle risk
            df["lifestyle_risk_score"] = (
                df["smoke"] + df["alco"] + (1 - df["active"])
            )
            self.new_features.append("lifestyle_risk_score")

            # Metabolic risk
            df["metabolic_risk_score"] = df["cholesterol"] + df["gluc"]
            self.new_features.append("metabolic_risk_score")

            # Combined risk (weighted composite)
            df["combined_risk_score"] = (
                (df["bp_category"] / 4 * 30)
                + (df["bmi_category"] / 3 * 20)
                + (df["lifestyle_risk_score"] / 3 * 25)
                + (df["metabolic_risk_score"] / 6 * 25)
            ).round(1)
            self.new_features.append("combined_risk_score")

            # Age-adjusted risk
            df["age_adjusted_risk"] = (
                df["combined_risk_score"] * (1 + (df["age_years"] - 30) / 100)
            ).round(1)
            self.new_features.append("age_adjusted_risk")

            return df

        except Exception as e:
            raise CustomException(e, sys)

    def _create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create second-order interaction features.

        Captures non-linear relationships between features:
        - age_bmi_interaction: Age × BMI
        - age_bp_interaction: Age × Mean BP
        - gender_bmi_interaction: Gender × BMI category
        - smoke_chol_interaction: Smoking × Cholesterol

        Args:
            df: Input DataFrame.

        Returns:
            DataFrame with interaction features added.

        Raises:
            CustomException: If creation fails.
        """
        try:
            logger.info("Creating interaction features...")

            df["age_bmi_interaction"] = (df["age_years"] * df["bmi"]).round(2)
            self.new_features.append("age_bmi_interaction")

            df["age_bp_interaction"] = (
                df["age_years"] * df["mean_arterial_pressure"]
            ).round(2)
            self.new_features.append("age_bp_interaction")

            df["gender_bmi_interaction"] = df["gender"] * df["bmi_category"]
            self.new_features.append("gender_bmi_interaction")

            df["smoke_chol_interaction"] = df["smoke"] * df["cholesterol"]
            self.new_features.append("smoke_chol_interaction")

            return df

        except Exception as e:
            raise CustomException(e, sys)

    def get_feature_importance_order(self) -> List[str]:
        """
        Get recommended feature order by expected importance.

        Returns:
            List of feature names ordered by expected importance.
        """
        return [
            "age_years", "bp_category", "mean_arterial_pressure",
            "ap_hi", "ap_lo", "bmi", "bmi_category",
            "cholesterol", "gluc", "combined_risk_score",
            "age_adjusted_risk", "lifestyle_risk_score",
            "metabolic_risk_score", "pulse_pressure",
            "gender", "smoke", "age_bmi_interaction",
            "age_bp_interaction", "weight", "height",
            "active", "alco",
        ]

    def get_feature_descriptions(self) -> Dict[str, str]:
        """
        Get descriptions of all engineered features.

        Returns:
            Dictionary mapping feature names to descriptions.
        """
        return {
            "bmi": "Body Mass Index (kg/m²)",
            "bmi_category": "BMI category: 0=Underweight, 1=Normal, 2=Overweight, 3=Obese",
            "bsa": "Body Surface Area (m²)",
            "weight_height_ratio": "Weight-to-height ratio (%)",
            "pulse_pressure": "Systolic - Diastolic BP (mmHg)",
            "mean_arterial_pressure": "Mean Arterial Pressure (mmHg)",
            "bp_category": "BP: 0=Normal, 1=Elevated, 2=Stage1, 3=Stage2, 4=Crisis",
            "bp_ratio": "Systolic / Diastolic ratio",
            "age_group": "Age: 0=<40, 1=40-49, 2=50-59, 3=60+",
            "age_squared": "Age squared (polynomial feature)",
            "lifestyle_risk_score": "Sum of smoking, alcohol, inactivity",
            "metabolic_risk_score": "Sum of cholesterol and glucose levels",
            "combined_risk_score": "Weighted composite risk (0-100)",
            "age_adjusted_risk": "Risk score adjusted for patient age",
            "age_bmi_interaction": "Age × BMI interaction",
            "age_bp_interaction": "Age × Mean BP interaction",
            "gender_bmi_interaction": "Gender × BMI category interaction",
            "smoke_chol_interaction": "Smoking × Cholesterol interaction",
        }

    def get_created_features(self) -> List[str]:
        """
        Get list of all newly created features.

        Returns:
            List of feature names created during engineering.
        """
        return self.new_features.copy()


def main() -> pd.DataFrame:
    """
    Run feature engineering pipeline standalone.

    Returns:
        DataFrame with engineered features.

    Raises:
        CustomException: If pipeline fails.
    """
    try:
        df = pd.read_csv("data/processed/cardio_cleaned.csv")
        engineer = FeatureEngineer()
        df_featured = engineer.create_all_features(df)
        df_featured.to_csv("data/processed/cardio_featured.csv", index=False)
        logger.info("Feature engineering completed and saved successfully")
        return df_featured
    except Exception as e:
        raise CustomException(e, sys)


if __name__ == "__main__":
    df_featured = main()
