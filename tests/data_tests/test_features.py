"""
Tests for Feature Engineering Module

Tests the FeatureEngineer class functionality including:
- Initialization and configuration loading
- BMI feature creation
- Blood pressure feature creation
- Risk score calculation
- Interaction feature generation
"""

import pandas as pd
import numpy as np
import pytest
import os
from pathlib import Path

# Import module under test
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.feature_engineering.features import FeatureEngineer


class TestFeatureEngineerInitialization:
    """Test FeatureEngineer class initialization."""

    def test_initialization_with_default_config(self):
        """Test that FeatureEngineer initializes with default config path."""
        if not os.path.exists("configs/model_config.yaml"):
            pytest.skip("Config file not found")
        
        engineer = FeatureEngineer()
        assert engineer.config is not None
        assert engineer.feature_config is not None

    def test_initialization_creates_empty_features_list(self):
        """Test that new_features list is initialized empty."""
        if not os.path.exists("configs/model_config.yaml"):
            pytest.skip("Config file not found")
        
        engineer = FeatureEngineer()
        assert engineer.new_features == []


class TestBMIFeatures:
    """Test BMI feature creation."""

    @pytest.fixture
    def sample_df(self):
        """Create sample DataFrame with height and weight."""
        return pd.DataFrame({
            "height": [170, 165, 180, 175, 160],
            "weight": [70.0, 80.0, 75.0, 100.0, 45.0],
            "gender": [1, 2, 1, 2, 1],
            "age_years": [50, 55, 45, 60, 35]
        })

    def test_creates_bmi_column(self, sample_df):
        """Test that BMI column is created."""
        if not os.path.exists("configs/model_config.yaml"):
            pytest.skip("Config file not found")
        
        engineer = FeatureEngineer()
        df_feat = engineer._create_bmi_features(sample_df)
        
        assert "bmi" in df_feat.columns

    def test_bmi_values_reasonable(self, sample_df):
        """Test that BMI values are in reasonable range."""
        if not os.path.exists("configs/model_config.yaml"):
            pytest.skip("Config file not found")
        
        engineer = FeatureEngineer()
        df_feat = engineer._create_bmi_features(sample_df)
        
        # BMI should typically be between 15 and 50
        assert df_feat["bmi"].min() >= 10
        assert df_feat["bmi"].max() <= 60

    def test_creates_bmi_category(self, sample_df):
        """Test that BMI category is created."""
        if not os.path.exists("configs/model_config.yaml"):
            pytest.skip("Config file not found")
        
        engineer = FeatureEngineer()
        df_feat = engineer._create_bmi_features(sample_df)
        
        assert "bmi_category" in df_feat.columns
        # Categories should be 0-3
        assert df_feat["bmi_category"].min() >= 0
        assert df_feat["bmi_category"].max() <= 3

    def test_bmi_calculation_correct(self):
        """Test that BMI calculation is mathematically correct."""
        if not os.path.exists("configs/model_config.yaml"):
            pytest.skip("Config file not found")
        
        df = pd.DataFrame({
            "height": [170],
            "weight": [70.0],
            "gender": [1],
            "age_years": [50]
        })
        
        engineer = FeatureEngineer()
        df_feat = engineer._create_bmi_features(df)
        
        # BMI = weight / (height in meters)^2
        expected_bmi = 70.0 / (1.70 ** 2)
        assert abs(df_feat["bmi"].iloc[0] - expected_bmi) < 0.1


class TestBloodPressureFeatures:
    """Test blood pressure feature creation."""

    @pytest.fixture
    def sample_df(self):
        """Create sample DataFrame with BP values."""
        return pd.DataFrame({
            "ap_hi": [120, 140, 160, 180, 110],
            "ap_lo": [80, 90, 100, 110, 70],
            "age_years": [50, 55, 60, 65, 45]
        })

    def test_creates_pulse_pressure(self, sample_df):
        """Test that pulse pressure is created."""
        if not os.path.exists("configs/model_config.yaml"):
            pytest.skip("Config file not found")
        
        engineer = FeatureEngineer()
        df_feat = engineer._create_bp_features(sample_df)
        
        assert "pulse_pressure" in df_feat.columns

    def test_pulse_pressure_correct(self, sample_df):
        """Test that pulse pressure is calculated correctly."""
        if not os.path.exists("configs/model_config.yaml"):
            pytest.skip("Config file not found")
        
        engineer = FeatureEngineer()
        df_feat = engineer._create_bp_features(sample_df)
        
        # Pulse pressure = systolic - diastolic
        expected_pp = sample_df["ap_hi"] - sample_df["ap_lo"]
        assert (df_feat["pulse_pressure"] == expected_pp).all()

    def test_creates_mean_arterial_pressure(self, sample_df):
        """Test that MAP is created."""
        if not os.path.exists("configs/model_config.yaml"):
            pytest.skip("Config file not found")
        
        engineer = FeatureEngineer()
        df_feat = engineer._create_bp_features(sample_df)
        
        assert "mean_arterial_pressure" in df_feat.columns

    def test_creates_bp_category(self, sample_df):
        """Test that BP category is created."""
        if not os.path.exists("configs/model_config.yaml"):
            pytest.skip("Config file not found")
        
        engineer = FeatureEngineer()
        df_feat = engineer._create_bp_features(sample_df)
        
        assert "bp_category" in df_feat.columns
        # Categories should be 0-4
        assert df_feat["bp_category"].min() >= 0
        assert df_feat["bp_category"].max() <= 4


class TestRiskScores:
    """Test risk score feature creation."""

    @pytest.fixture
    def sample_df(self):
        """Create sample DataFrame for risk score testing."""
        return pd.DataFrame({
            "age_years": [50, 55, 60],
            "height": [170, 165, 180],
            "weight": [70.0, 90.0, 80.0],
            "ap_hi": [120, 140, 160],
            "ap_lo": [80, 90, 100],
            "cholesterol": [1, 2, 3],
            "gluc": [1, 2, 3],
            "smoke": [0, 1, 0],
            "alco": [0, 1, 0],
            "active": [1, 0, 1],
            "gender": [1, 2, 1]
        })

    def test_creates_lifestyle_risk_score(self, sample_df):
        """Test that lifestyle risk score is created."""
        if not os.path.exists("configs/model_config.yaml"):
            pytest.skip("Config file not found")
        
        engineer = FeatureEngineer()
        # First create prerequisite features
        df_feat = engineer._create_bmi_features(sample_df)
        df_feat = engineer._create_bp_features(df_feat)
        df_feat = engineer._create_risk_scores(df_feat)
        
        assert "lifestyle_risk_score" in df_feat.columns

    def test_creates_metabolic_risk_score(self, sample_df):
        """Test that metabolic risk score is created."""
        if not os.path.exists("configs/model_config.yaml"):
            pytest.skip("Config file not found")
        
        engineer = FeatureEngineer()
        df_feat = engineer._create_bmi_features(sample_df)
        df_feat = engineer._create_bp_features(df_feat)
        df_feat = engineer._create_risk_scores(df_feat)
        
        assert "metabolic_risk_score" in df_feat.columns

    def test_creates_combined_risk_score(self, sample_df):
        """Test that combined risk score is created."""
        if not os.path.exists("configs/model_config.yaml"):
            pytest.skip("Config file not found")
        
        engineer = FeatureEngineer()
        df_feat = engineer._create_bmi_features(sample_df)
        df_feat = engineer._create_bp_features(df_feat)
        df_feat = engineer._create_risk_scores(df_feat)
        
        assert "combined_risk_score" in df_feat.columns


class TestFullPipeline:
    """Test the complete feature engineering pipeline."""

    def test_create_all_features_on_cleaned_data(self):
        """Test full pipeline on cleaned data file."""
        if not os.path.exists("data/processed/cardio_cleaned.csv"):
            pytest.skip("Cleaned data file not found")
        if not os.path.exists("configs/model_config.yaml"):
            pytest.skip("Config file not found")
        
        df = pd.read_csv("data/processed/cardio_cleaned.csv")
        engineer = FeatureEngineer()
        df_feat = engineer.create_all_features(df)
        
        # Should return a DataFrame
        assert isinstance(df_feat, pd.DataFrame)
        
        # Should have more columns than input
        assert len(df_feat.columns) > len(df.columns)

    def test_tracks_new_features(self):
        """Test that new features are tracked."""
        if not os.path.exists("data/processed/cardio_cleaned.csv"):
            pytest.skip("Cleaned data file not found")
        if not os.path.exists("configs/model_config.yaml"):
            pytest.skip("Config file not found")
        
        df = pd.read_csv("data/processed/cardio_cleaned.csv")
        engineer = FeatureEngineer()
        engineer.create_all_features(df)
        
        # Should have tracked multiple new features
        assert len(engineer.new_features) > 0


class TestUtilityMethods:
    """Test utility methods."""

    def test_get_feature_importance_order_returns_list(self):
        """Test that feature importance order returns a list."""
        if not os.path.exists("configs/model_config.yaml"):
            pytest.skip("Config file not found")
        
        engineer = FeatureEngineer()
        order = engineer.get_feature_importance_order()
        
        assert isinstance(order, list)
        assert len(order) > 0

    def test_get_feature_descriptions_returns_dict(self):
        """Test that feature descriptions returns a dictionary."""
        if not os.path.exists("configs/model_config.yaml"):
            pytest.skip("Config file not found")
        
        engineer = FeatureEngineer()
        descriptions = engineer.get_feature_descriptions()
        
        assert isinstance(descriptions, dict)
        assert len(descriptions) > 0
        assert "bmi" in descriptions
