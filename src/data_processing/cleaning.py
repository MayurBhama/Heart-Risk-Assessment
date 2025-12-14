"""
Data Cleaning Module

This module performs comprehensive data cleaning including:
- Outlier detection and handling using IQR method
- Missing value imputation
- Data type corrections
- Invalid value handling
- Blood pressure anomaly detection

Author: Heart-Risk-Assessment Team
"""

import pandas as pd
import numpy as np
import yaml
import sys
from typing import Tuple, Dict, List, Any, Optional
import warnings

warnings.filterwarnings("ignore")

from utils.logger import logger
from utils.exception import CustomException


class DataCleaner:
    """
    Comprehensive data cleaning class for cardiovascular dataset.
    
    Provides methods for outlier removal, duplicate handling,
    data type corrections, and blood pressure anomaly detection.
    
    Attributes:
        config: Configuration dictionary loaded from YAML.
        cleaning_report: Dictionary tracking all cleaning operations.
    """

    def __init__(self, config_path: str = "configs/data_config.yaml") -> None:
        """
        Initialize DataCleaner.

        Args:
            config_path: Path to data configuration file.

        Raises:
            CustomException: If initialization fails.
        """
        try:
            self.config: Dict[str, Any] = self._load_config(config_path)
            self.cleaning_report: Dict[str, Any] = {
                "initial_shape": None,
                "final_shape": None,
                "removed_rows": 0,
                "outliers_removed": {},
                "invalid_values_handled": {},
                "duplicates_removed": 0,
            }
            logger.info("DataCleaner initialized successfully")
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

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Execute the main cleaning pipeline.

        Applies all cleaning steps in sequence:
        1. Remove duplicates
        2. Convert age to years
        3. Handle invalid categorical values
        4. Remove outliers
        5. Handle blood pressure anomalies
        6. Handle missing values
        7. Ensure correct data types

        Args:
            df: Input DataFrame to clean.

        Returns:
            Cleaned DataFrame.

        Raises:
            CustomException: If any cleaning step fails.
        """
        try:
            logger.info("Starting data cleaning pipeline...")
            self.cleaning_report["initial_shape"] = df.shape

            df_clean = df.copy()

            df_clean = self._remove_duplicates(df_clean)
            df_clean = self._convert_age_to_years(df_clean)
            df_clean = self._handle_invalid_categorical(df_clean)
            df_clean = self._remove_outliers(df_clean)
            df_clean = self._handle_blood_pressure_anomalies(df_clean)
            df_clean = self._handle_missing_values(df_clean)
            df_clean = self._ensure_data_types(df_clean)

            self.cleaning_report["final_shape"] = df_clean.shape
            self.cleaning_report["removed_rows"] = (
                self.cleaning_report["initial_shape"][0] - df_clean.shape[0]
            )

            logger.info(
                f"Data cleaning completed. Total removed rows: {self.cleaning_report['removed_rows']}"
            )

            return df_clean

        except Exception as e:
            logger.error("Error occurred during data cleaning")
            raise CustomException(e, sys)

    def _remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove duplicate rows from DataFrame.

        Args:
            df: Input DataFrame.

        Returns:
            DataFrame with duplicates removed.

        Raises:
            CustomException: If removal fails.
        """
        try:
            initial_len = len(df)
            df_clean = df.drop_duplicates()
            removed = initial_len - len(df_clean)
            self.cleaning_report["duplicates_removed"] = removed
            logger.info(f"Removed {removed} duplicate rows")
            return df_clean
        except Exception as e:
            raise CustomException(e, sys)

    def _convert_age_to_years(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert age from days to years.

        Args:
            df: Input DataFrame with 'age' column in days.

        Returns:
            DataFrame with new 'age_years' column.

        Raises:
            CustomException: If conversion fails.
        """
        try:
            df["age_years"] = (df["age"] / 365.25).round().astype(int)
            logger.info("Converted age from days to years")
            return df
        except Exception as e:
            logger.error("Error converting age to years")
            raise CustomException(e, sys)

    def _handle_invalid_categorical(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove rows with invalid categorical values.

        Args:
            df: Input DataFrame.

        Returns:
            DataFrame with invalid categorical values removed.

        Raises:
            CustomException: If handling fails.
        """
        try:
            for col, valid_values in self.config["categorical_values"].items():
                if col in df.columns:
                    initial_len = len(df)
                    df = df[df[col].isin(valid_values)]
                    removed = initial_len - len(df)

                    if removed > 0:
                        self.cleaning_report["invalid_values_handled"][col] = removed
                        logger.info(f"Removed {removed} invalid values in {col}")

            return df
        except Exception as e:
            raise CustomException(e, sys)

    def _remove_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove outliers based on configured valid ranges.

        Args:
            df: Input DataFrame.

        Returns:
            DataFrame with outliers removed.

        Raises:
            CustomException: If removal fails.
        """
        try:
            valid_ranges = self.config["valid_ranges"]
            initial_len = len(df)

            # Define columns and their config keys
            columns_to_check = [
                ("age_years", "age_years", "age"),
                ("height", "height", "height"),
                ("weight", "weight", "weight"),
                ("ap_hi", "ap_hi", "systolic_bp"),
                ("ap_lo", "ap_lo", "diastolic_bp"),
            ]

            for col, range_key, report_key in columns_to_check:
                if col in df.columns and range_key in valid_ranges:
                    low, high = valid_ranges[range_key]
                    mask = (df[col] >= low) & (df[col] <= high)
                    removed = len(df) - mask.sum()
                    df = df[mask]
                    if removed > 0:
                        self.cleaning_report["outliers_removed"][report_key] = removed

            total_removed = initial_len - len(df)
            logger.info(f"Total outliers removed: {total_removed}")

            return df

        except Exception as e:
            raise CustomException(e, sys)

    def _handle_blood_pressure_anomalies(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove cases where diastolic BP >= systolic BP.

        Args:
            df: Input DataFrame.

        Returns:
            DataFrame with BP anomalies removed.

        Raises:
            CustomException: If handling fails.
        """
        try:
            if "ap_hi" in df.columns and "ap_lo" in df.columns:
                initial_len = len(df)
                df = df[df["ap_lo"] < df["ap_hi"]]
                removed = initial_len - len(df)
                if removed > 0:
                    self.cleaning_report["outliers_removed"]["bp_anomalies"] = removed
                    logger.info(f"Removed {removed} BP anomalies")
            return df
        except Exception as e:
            raise CustomException(e, sys)

    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Drop rows with missing values.

        Args:
            df: Input DataFrame.

        Returns:
            DataFrame with missing values removed.

        Raises:
            CustomException: If handling fails.
        """
        try:
            missing_count = df.isnull().sum().sum()
            if missing_count > 0:
                initial = len(df)
                df = df.dropna()
                logger.info(f"Removed {initial - len(df)} rows with missing values")
            else:
                logger.info("No missing values found")
            return df
        except Exception as e:
            raise CustomException(e, sys)

    def _ensure_data_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Ensure correct data types for all columns.

        Args:
            df: Input DataFrame.

        Returns:
            DataFrame with corrected data types.

        Raises:
            CustomException: If type conversion fails.
        """
        try:
            int_columns = [
                "id", "age", "gender", "height", "ap_hi", "ap_lo",
                "cholesterol", "gluc", "smoke", "alco", "active",
                "cardio", "age_years",
            ]

            for col in int_columns:
                if col in df.columns:
                    df[col] = df[col].astype(int)

            if "weight" in df.columns:
                df["weight"] = df["weight"].astype(float)

            logger.info("Data types ensured successfully")
            return df

        except Exception as e:
            raise CustomException(e, sys)

    def get_cleaning_report(self) -> Dict[str, Any]:
        """
        Get the cleaning summary report.

        Returns:
            Dictionary containing cleaning statistics.
        """
        return self.cleaning_report

    def get_data_statistics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate summary statistics for the DataFrame.

        Args:
            df: Input DataFrame.

        Returns:
            Dictionary containing numeric stats, categorical distributions,
            and target/gender distributions.

        Raises:
            CustomException: If statistics generation fails.
        """
        try:
            stats_dict: Dict[str, Any] = {
                "numeric_stats": df.describe().to_dict(),
                "categorical_distribution": {},
                "target_distribution": (
                    df["cardio"].value_counts().to_dict()
                    if "cardio" in df.columns
                    else None
                ),
                "gender_distribution": (
                    df["gender"].value_counts().to_dict()
                    if "gender" in df.columns
                    else None
                ),
            }

            categorical_cols = ["cholesterol", "gluc", "smoke", "alco", "active"]

            for col in categorical_cols:
                if col in df.columns:
                    stats_dict["categorical_distribution"][col] = (
                        df[col].value_counts().to_dict()
                    )

            return stats_dict

        except Exception as e:
            raise CustomException(e, sys)


def main() -> pd.DataFrame:
    """
    Run cleaning pipeline standalone for testing.

    Returns:
        Cleaned DataFrame.

    Raises:
        CustomException: If pipeline fails.
    """
    try:
        df = pd.read_csv("data/raw/cardio_train.csv", sep=";")
        cleaner = DataCleaner()
        df_clean = cleaner.clean_data(df)

        df_clean.to_csv("data/processed/cardio_cleaned.csv", index=False)
        logger.info("Cleaned data saved successfully")

        return df_clean
    except Exception as e:
        raise CustomException(e, sys)


if __name__ == "__main__":
    df_clean = main()