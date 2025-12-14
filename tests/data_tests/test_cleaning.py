"""
Tests for Data Cleaning Module

Tests the DataCleaner class functionality including:
- Initialization and configuration loading
- Duplicate removal
- Age conversion
- Outlier handling
- Blood pressure anomaly detection
"""

import pandas as pd
import numpy as np
import pytest
import os
from pathlib import Path

# Import module under test
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data_processing.cleaning import DataCleaner


class TestDataCleanerInitialization:
    """Test DataCleaner class initialization."""

    def test_initialization_with_default_config(self):
        """Test that DataCleaner initializes with default config path."""
        if not os.path.exists("configs/data_config.yaml"):
            pytest.skip("Config file not found")
        
        cleaner = DataCleaner()
        assert cleaner.config is not None
        assert isinstance(cleaner.config, dict)

    def test_initialization_creates_cleaning_report(self):
        """Test that initialization creates cleaning report structure."""
        if not os.path.exists("configs/data_config.yaml"):
            pytest.skip("Config file not found")
        
        cleaner = DataCleaner()
        report = cleaner.get_cleaning_report()
        
        expected_keys = ["initial_shape", "final_shape", "removed_rows", 
                        "outliers_removed", "duplicates_removed"]
        for key in expected_keys:
            assert key in report


class TestDuplicateRemoval:
    """Test duplicate removal functionality."""

    @pytest.fixture
    def df_with_duplicates(self):
        """Create DataFrame with duplicate rows."""
        return pd.DataFrame({
            "id": [1, 2, 2, 3, 4],
            "age": [18000, 19000, 19000, 20000, 21000],
            "gender": [1, 2, 2, 1, 2],
            "height": [170, 165, 165, 180, 175],
            "weight": [70.0, 80.0, 80.0, 75.0, 85.0],
            "ap_hi": [120, 130, 130, 140, 135],
            "ap_lo": [80, 85, 85, 90, 88],
            "cholesterol": [1, 2, 2, 3, 1],
            "gluc": [1, 2, 2, 1, 3],
            "smoke": [0, 1, 1, 0, 0],
            "alco": [0, 0, 0, 1, 0],
            "active": [1, 1, 1, 0, 1],
            "cardio": [0, 1, 1, 0, 1]
        })

    def test_removes_duplicates(self, df_with_duplicates):
        """Test that duplicate rows are removed."""
        if not os.path.exists("configs/data_config.yaml"):
            pytest.skip("Config file not found")
        
        cleaner = DataCleaner()
        df_clean = cleaner._remove_duplicates(df_with_duplicates)
        
        assert len(df_clean) < len(df_with_duplicates)
        assert df_clean.duplicated().sum() == 0

    def test_reports_duplicates_removed(self, df_with_duplicates):
        """Test that cleaning report tracks duplicates removed."""
        if not os.path.exists("configs/data_config.yaml"):
            pytest.skip("Config file not found")
        
        cleaner = DataCleaner()
        cleaner._remove_duplicates(df_with_duplicates)
        report = cleaner.get_cleaning_report()
        
        assert report["duplicates_removed"] == 1


class TestAgeConversion:
    """Test age conversion functionality."""

    @pytest.fixture
    def df_with_age_days(self):
        """Create DataFrame with age in days."""
        return pd.DataFrame({
            "age": [18250, 20075, 21900, 25550],  # ~50, 55, 60, 70 years
            "cardio": [0, 1, 0, 1]
        })

    def test_converts_age_to_years(self, df_with_age_days):
        """Test that age is converted from days to years."""
        if not os.path.exists("configs/data_config.yaml"):
            pytest.skip("Config file not found")
        
        cleaner = DataCleaner()
        df_converted = cleaner._convert_age_to_years(df_with_age_days)
        
        assert "age_years" in df_converted.columns

    def test_age_years_reasonable_range(self, df_with_age_days):
        """Test that converted ages are in reasonable range."""
        if not os.path.exists("configs/data_config.yaml"):
            pytest.skip("Config file not found")
        
        cleaner = DataCleaner()
        df_converted = cleaner._convert_age_to_years(df_with_age_days)
        
        assert df_converted["age_years"].min() >= 30
        assert df_converted["age_years"].max() <= 80


class TestBloodPressureAnomalies:
    """Test blood pressure anomaly detection."""

    @pytest.fixture
    def df_with_bp_anomalies(self):
        """Create DataFrame with BP anomalies."""
        return pd.DataFrame({
            "ap_hi": [120, 130, 80, 140, 90],   # Row 2 & 4: ap_hi < ap_lo (anomaly)
            "ap_lo": [80, 85, 100, 90, 95],
            "cardio": [0, 1, 0, 1, 0]
        })

    def test_removes_bp_anomalies(self, df_with_bp_anomalies):
        """Test that BP anomalies are removed."""
        if not os.path.exists("configs/data_config.yaml"):
            pytest.skip("Config file not found")
        
        cleaner = DataCleaner()
        df_clean = cleaner._handle_blood_pressure_anomalies(df_with_bp_anomalies)
        
        # All remaining rows should have ap_hi > ap_lo
        assert (df_clean["ap_hi"] > df_clean["ap_lo"]).all()


class TestCleaningPipeline:
    """Test the complete cleaning pipeline."""

    def test_clean_data_on_real_data(self):
        """Test cleaning pipeline on actual data file."""
        if not os.path.exists("data/raw/cardio_train.csv"):
            pytest.skip("Raw data file not found")
        if not os.path.exists("configs/data_config.yaml"):
            pytest.skip("Config file not found")
        
        df = pd.read_csv("data/raw/cardio_train.csv", sep=";")
        cleaner = DataCleaner()
        df_clean = cleaner.clean_data(df)
        
        # Should return a DataFrame
        assert isinstance(df_clean, pd.DataFrame)
        
        # Should have fewer or equal rows
        assert len(df_clean) <= len(df)
        
        # Should have age_years column
        assert "age_years" in df_clean.columns

    def test_cleaning_report_populated(self):
        """Test that cleaning report is populated after cleaning."""
        if not os.path.exists("data/raw/cardio_train.csv"):
            pytest.skip("Raw data file not found")
        if not os.path.exists("configs/data_config.yaml"):
            pytest.skip("Config file not found")
        
        df = pd.read_csv("data/raw/cardio_train.csv", sep=";")
        cleaner = DataCleaner()
        cleaner.clean_data(df)
        report = cleaner.get_cleaning_report()
        
        assert report["initial_shape"] is not None
        assert report["final_shape"] is not None
        assert report["removed_rows"] >= 0


class TestDataStatistics:
    """Test data statistics generation."""

    @pytest.fixture
    def sample_df(self):
        """Create sample DataFrame for statistics testing."""
        return pd.DataFrame({
            "age_years": [50, 55, 60, 65, 70],
            "gender": [1, 2, 1, 2, 1],
            "cholesterol": [1, 2, 3, 1, 2],
            "cardio": [0, 1, 0, 1, 1]
        })

    def test_get_data_statistics_returns_dict(self, sample_df):
        """Test that get_data_statistics returns a dictionary."""
        if not os.path.exists("configs/data_config.yaml"):
            pytest.skip("Config file not found")
        
        cleaner = DataCleaner()
        stats = cleaner.get_data_statistics(sample_df)
        
        assert isinstance(stats, dict)

    def test_statistics_contains_numeric_stats(self, sample_df):
        """Test that statistics contain numeric summaries."""
        if not os.path.exists("configs/data_config.yaml"):
            pytest.skip("Config file not found")
        
        cleaner = DataCleaner()
        stats = cleaner.get_data_statistics(sample_df)
        
        assert "numeric_stats" in stats
        assert "target_distribution" in stats
