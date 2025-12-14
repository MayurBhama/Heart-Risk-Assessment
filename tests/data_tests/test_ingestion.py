"""
Tests for Data Ingestion Module

Tests the DataIngestion class functionality including:
- Initialization and configuration loading
- Data loading from CSV files
- Schema validation
- Data quality checks
"""

import pandas as pd
import numpy as np
import pytest
import os
from pathlib import Path

# Import module under test
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data_ingestion.ingestion import DataIngestion


class TestDataIngestionInitialization:
    """Test DataIngestion class initialization."""

    def test_initialization_with_default_config(self):
        """Test that DataIngestion initializes with default config path."""
        if not os.path.exists("configs/data_config.yaml"):
            pytest.skip("Config file not found")
        
        ingestion = DataIngestion()
        assert ingestion.config is not None
        assert isinstance(ingestion.config, dict)

    def test_initialization_loads_expected_keys(self):
        """Test that config contains expected keys."""
        if not os.path.exists("configs/data_config.yaml"):
            pytest.skip("Config file not found")
        
        ingestion = DataIngestion()
        expected_keys = ["columns", "categorical_values", "valid_ranges", "quality_thresholds"]
        
        for key in expected_keys:
            assert key in ingestion.config, f"Missing expected config key: {key}"


class TestDataLoading:
    """Test data loading functionality."""

    def test_load_data_returns_dataframe(self):
        """Test that load_data returns a pandas DataFrame."""
        if not os.path.exists("data/raw/cardio_train.csv"):
            pytest.skip("Raw data file not found")
        
        ingestion = DataIngestion()
        df = ingestion.load_data("data/raw/cardio_train.csv")
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0

    def test_load_data_has_expected_columns(self):
        """Test that loaded data has expected columns."""
        if not os.path.exists("data/raw/cardio_train.csv"):
            pytest.skip("Raw data file not found")
        
        ingestion = DataIngestion()
        df = ingestion.load_data("data/raw/cardio_train.csv")
        
        expected_cols = ["age", "gender", "height", "weight", "ap_hi", "ap_lo", "cardio"]
        for col in expected_cols:
            assert col in df.columns, f"Missing expected column: {col}"


class TestDataInfo:
    """Test data info extraction."""

    @pytest.fixture
    def sample_df(self):
        """Create a sample DataFrame for testing."""
        return pd.DataFrame({
            "age": [50, 55, 60, 65, 70],
            "height": [170, 165, 180, 175, 168],
            "weight": [70.0, 80.0, 75.0, 85.0, 72.0],
            "cardio": [0, 1, 0, 1, 0]
        })

    def test_get_data_info_returns_dict(self, sample_df):
        """Test that get_data_info returns a dictionary."""
        if not os.path.exists("configs/data_config.yaml"):
            pytest.skip("Config file not found")
        
        ingestion = DataIngestion()
        info = ingestion.get_data_info(sample_df)
        
        assert isinstance(info, dict)

    def test_get_data_info_contains_shape(self, sample_df):
        """Test that info contains shape."""
        if not os.path.exists("configs/data_config.yaml"):
            pytest.skip("Config file not found")
        
        ingestion = DataIngestion()
        info = ingestion.get_data_info(sample_df)
        
        assert "shape" in info
        assert info["shape"] == (5, 4)

    def test_get_data_info_contains_columns(self, sample_df):
        """Test that info contains column list."""
        if not os.path.exists("configs/data_config.yaml"):
            pytest.skip("Config file not found")
        
        ingestion = DataIngestion()
        info = ingestion.get_data_info(sample_df)
        
        assert "columns" in info
        assert set(info["columns"]) == {"age", "height", "weight", "cardio"}


class TestSchemaValidation:
    """Test schema validation functionality."""

    @pytest.fixture
    def valid_df(self):
        """Create a DataFrame with valid schema."""
        return pd.DataFrame({
            "id": [1, 2, 3],
            "age": [18000, 19000, 20000],
            "gender": [1, 2, 1],
            "height": [170, 165, 180],
            "weight": [70.0, 80.0, 75.0],
            "ap_hi": [120, 130, 140],
            "ap_lo": [80, 85, 90],
            "cholesterol": [1, 2, 3],
            "gluc": [1, 2, 1],
            "smoke": [0, 1, 0],
            "alco": [0, 0, 1],
            "active": [1, 1, 0],
            "cardio": [0, 1, 0]
        })

    def test_validate_schema_returns_tuple(self, valid_df):
        """Test that validate_data_schema returns a tuple."""
        if not os.path.exists("configs/data_config.yaml"):
            pytest.skip("Config file not found")
        
        ingestion = DataIngestion()
        result = ingestion.validate_data_schema(valid_df)
        
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_validate_schema_valid_data(self, valid_df):
        """Test that valid data passes schema validation."""
        if not os.path.exists("configs/data_config.yaml"):
            pytest.skip("Config file not found")
        
        ingestion = DataIngestion()
        is_valid, errors = ingestion.validate_data_schema(valid_df)
        
        assert is_valid is True
        assert len(errors) == 0


class TestDataQualityCheck:
    """Test data quality check functionality."""

    @pytest.fixture
    def sample_df(self):
        """Create a sample DataFrame for quality testing."""
        return pd.DataFrame({
            "gender": [1, 2, 1, 2, 1],
            "cholesterol": [1, 2, 3, 1, 2],
            "gluc": [1, 1, 2, 3, 1],
            "smoke": [0, 1, 0, 0, 1],
            "alco": [0, 0, 1, 0, 0],
            "active": [1, 1, 1, 0, 1],
            "cardio": [0, 1, 0, 1, 0]
        })

    def test_quality_check_returns_dict(self, sample_df):
        """Test that quality check returns a dictionary."""
        if not os.path.exists("configs/data_config.yaml"):
            pytest.skip("Config file not found")
        
        ingestion = DataIngestion()
        quality = ingestion.initial_data_quality_check(sample_df)
        
        assert isinstance(quality, dict)

    def test_quality_check_has_quality_score(self, sample_df):
        """Test that quality check includes quality score."""
        if not os.path.exists("configs/data_config.yaml"):
            pytest.skip("Config file not found")
        
        ingestion = DataIngestion()
        quality = ingestion.initial_data_quality_check(sample_df)
        
        assert "quality_score" in quality
        assert 0 <= quality["quality_score"] <= 100
