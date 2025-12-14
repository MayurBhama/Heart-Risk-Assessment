"""
Data Ingestion Module

This module handles loading raw data and initial validation including:
- Loading data from various sources (CSV, Excel)
- Schema validation against configuration
- Initial data quality checks
- Data profiling and metadata extraction

Author: Heart-Risk-Assessment Team
"""

import pandas as pd
import numpy as np
import yaml
import sys
from pathlib import Path
from typing import Tuple, Dict, List, Any, Optional
import warnings

warnings.filterwarnings("ignore")

from utils.logger import logger
from utils.exception import CustomException


class DataIngestion:
    """
    Class to handle data ingestion from various sources.
    
    This class provides methods for loading, validating, and profiling
    raw data before it enters the preprocessing pipeline.
    
    Attributes:
        config: Configuration dictionary loaded from YAML.
    """

    def __init__(self, config_path: str = "configs/data_config.yaml") -> None:
        """
        Initialize DataIngestion class.

        Args:
            config_path: Path to data configuration YAML file.

        Raises:
            CustomException: If initialization fails.
        """
        try:
            self.config: Dict[str, Any] = self._load_config(config_path)
            logger.info("DataIngestion initialized successfully")
        except Exception as e:
            raise CustomException(e, sys)

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """
        Load configuration from YAML file.

        Args:
            config_path: Path to the configuration file.

        Returns:
            Configuration dictionary.

        Raises:
            CustomException: If loading fails.
        """
        try:
            with open(config_path, "r") as f:
                config = yaml.safe_load(f)
            logger.info(f"Configuration loaded from {config_path}")
            return config

        except Exception as e:
            logger.error(f"Failed to load configuration file: {config_path}")
            raise CustomException(e, sys)

    def load_data(self, file_path: str, separator: str = ";") -> pd.DataFrame:
        """
        Load data from CSV file.

        Args:
            file_path: Path to the CSV file.
            separator: Column separator (default: ';').

        Returns:
            Loaded DataFrame.

        Raises:
            CustomException: If loading fails.
        """
        try:
            logger.info(f"Loading data from: {file_path}")
            df = pd.read_csv(file_path, sep=separator)
            logger.info(f"Data loaded successfully. Shape: {df.shape}")
            return df

        except Exception as e:
            logger.error(f"Failed to load data file: {file_path}")
            raise CustomException(e, sys)

    def get_data_info(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Extract dataset metadata and quality information.

        Args:
            df: Input DataFrame.

        Returns:
            Dictionary containing shape, columns, dtypes, missing values,
            duplicates, and memory usage information.

        Raises:
            CustomException: If extraction fails.
        """
        try:
            info: Dict[str, Any] = {
                "shape": df.shape,
                "columns": df.columns.tolist(),
                "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
                "missing_values": df.isnull().sum().to_dict(),
                "missing_percentage": (df.isnull().sum() / len(df) * 100).to_dict(),
                "duplicates": int(df.duplicated().sum()),
                "duplicate_percentage": float(df.duplicated().sum() / len(df) * 100),
                "memory_usage_mb": float(df.memory_usage(deep=True).sum() / 1024**2),
            }

            logger.info("Data information extracted successfully")
            return info

        except Exception as e:
            logger.error("Error extracting data information")
            raise CustomException(e, sys)

    def validate_data_schema(self, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """
        Validate DataFrame schema against configuration.

        Args:
            df: Input DataFrame to validate.

        Returns:
            Tuple of (is_valid, list of error messages).

        Raises:
            CustomException: If validation fails.
        """
        try:
            errors: List[str] = []

            # Check required columns
            expected_cols = list(self.config["columns"].values())
            missing_cols = set(expected_cols) - set(df.columns)

            if missing_cols:
                errors.append(f"Missing columns: {missing_cols}")

            # Check data types
            for col in df.columns:
                if col in expected_cols:
                    if not pd.api.types.is_numeric_dtype(df[col]):
                        if col not in ["id"]:
                            errors.append(f"Column {col} must be numeric")

            # Check for empty columns
            empty_cols = df.columns[df.isnull().all()].tolist()
            if empty_cols:
                errors.append(f"Empty columns found: {empty_cols}")

            is_valid = len(errors) == 0

            if is_valid:
                logger.info("Schema validation passed")
            else:
                logger.warning(f"Schema validation failed with {len(errors)} issues")
                for err in errors:
                    logger.warning(f"  - {err}")

            return is_valid, errors

        except Exception as e:
            logger.error("Error validating schema")
            raise CustomException(e, sys)

    def initial_data_quality_check(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Perform initial data quality checks.

        Includes missing value analysis, duplicate detection,
        and categorical value validation.

        Args:
            df: Input DataFrame.

        Returns:
            Dictionary containing quality report with pass/fail status
            for each check and an overall quality score.

        Raises:
            CustomException: If quality check fails.
        """
        try:
            quality_report: Dict[str, Any] = {}

            # Missing values check
            missing_percent = df.isnull().sum() / len(df) * 100
            max_missing_threshold = self.config["quality_thresholds"]["max_missing_percentage"]
            
            quality_report["missing_values"] = {
                "columns_with_missing": missing_percent[missing_percent > 0].to_dict(),
                "max_missing_percentage": float(missing_percent.max()),
                "passed": bool(missing_percent.max() < max_missing_threshold),
            }

            # Duplicates check
            duplicate_count = df.duplicated().sum()
            duplicate_percent = duplicate_count / len(df) * 100
            duplicate_threshold = self.config["quality_thresholds"]["duplicate_threshold"] * 100
            
            quality_report["duplicates"] = {
                "count": int(duplicate_count),
                "percentage": float(duplicate_percent),
                "passed": bool(duplicate_percent < duplicate_threshold),
            }

            # Categorical values validation
            categorical_checks: Dict[str, Dict[str, Any]] = {}
            for col, valid_vals in self.config["categorical_values"].items():
                if col in df.columns:
                    unique_vals = df[col].unique()
                    invalid_vals = set(unique_vals) - set(valid_vals) - {np.nan}
                    categorical_checks[col] = {
                        "unique_values": [v for v in unique_vals.tolist() if pd.notna(v)],
                        "expected_values": valid_vals,
                        "invalid_values": list(invalid_vals),
                        "passed": len(invalid_vals) == 0,
                    }

            quality_report["categorical_validation"] = categorical_checks

            # Calculate overall quality score
            passed_checks = sum([
                quality_report["missing_values"]["passed"],
                quality_report["duplicates"]["passed"],
                sum(v["passed"] for v in categorical_checks.values()),
            ])
            total_checks = 2 + len(categorical_checks)
            quality_report["quality_score"] = round((passed_checks / total_checks) * 100, 2)

            logger.info(f"Data quality score: {quality_report['quality_score']}%")
            return quality_report

        except Exception as e:
            logger.error("Error during data quality checks")
            raise CustomException(e, sys)

    def save_data(self, df: pd.DataFrame, output_path: str) -> None:
        """
        Save DataFrame to CSV file.

        Args:
            df: DataFrame to save.
            output_path: Output file path.

        Raises:
            CustomException: If saving fails.
        """
        try:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(output_path, index=False)
            logger.info(f"Data saved to: {output_path}")

        except Exception as e:
            logger.error(f"Error saving data to: {output_path}")
            raise CustomException(e, sys)

    def get_column_statistics(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate descriptive statistics for all columns.

        Args:
            df: Input DataFrame.

        Returns:
            DataFrame containing statistics for each column.
        """
        try:
            stats = df.describe(include="all").T
            stats["missing_count"] = df.isnull().sum()
            stats["missing_percent"] = (df.isnull().sum() / len(df) * 100).round(2)
            stats["unique_count"] = df.nunique()
            
            logger.info("Column statistics generated")
            return stats
            
        except Exception as e:
            logger.error("Error generating column statistics")
            raise CustomException(e, sys)


def main() -> pd.DataFrame:
    """
    Main function for standalone execution.

    Returns:
        Loaded and validated DataFrame.

    Raises:
        CustomException: If pipeline fails.
    """
    try:
        ingestion = DataIngestion()

        df = ingestion.load_data("data/raw/cardio_train.csv")

        info = ingestion.get_data_info(df)
        print("\n=== DATA INFORMATION ===")
        for k, v in info.items():
            print(f"{k}: {v}")

        is_valid, errors = ingestion.validate_data_schema(df)
        print("\n=== SCHEMA VALIDATION ===")
        print(f"Valid: {is_valid}")
        print(f"Errors: {errors}")

        quality = ingestion.initial_data_quality_check(df)
        print("\n=== QUALITY REPORT ===")
        print(f"Quality Score: {quality['quality_score']}%")

        return df

    except Exception as e:
        raise CustomException(e, sys)


if __name__ == "__main__":
    df = main()
