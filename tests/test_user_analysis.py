"""Tests for the user analysis module."""

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
import matplotlib.pyplot as plt

from chatgpt_data.analysis.user_analysis import UserAnalysis


@pytest.fixture
def mock_user_data():
    """Create mock user data for testing."""
    return pd.DataFrame({
        "period_start": ["2025-01-12", "2025-01-19", "2025-01-26"],
        "period_end": ["2025-01-18", "2025-01-25", "2025-02-01"],
        "is_active": [1, 1, 1],
        "messages": [10, 20, 30],
        "gpt_messages": [5, 10, 15],
        "account_id": ["acc1", "acc1", "acc1"],
        "public_id": ["pub1", "pub1", "pub1"],
        "name": ["Test User", "Test User", "Test User"],
        "email": ["test@example.com", "test@example.com", "test@example.com"],
        "user_status": ["active", "active", "active"]
    })


class TestUserAnalysis:
    """Test the UserAnalysis class."""

    def test_load_data(self, mock_user_data, tmp_path):
        """Test loading user data."""
        # Create test files
        test_file = tmp_path / "proofpoint_user_engagement_test.csv"
        mock_user_data.to_csv(test_file, index=False)
        
        # Execute
        analyzer = UserAnalysis(tmp_path, tmp_path)
        
        # Assert
        assert analyzer.user_data is not None
        assert len(analyzer.user_data) == len(mock_user_data)

    def test_generate_active_users_trend(self, mock_user_data, tmp_path):
        """Test generating active users trend."""
        # Setup
        with patch.object(UserAnalysis, '_load_data'):
            analyzer = UserAnalysis(tmp_path, tmp_path)
            analyzer.user_data = mock_user_data.copy()
        
        # Execute
        fig = analyzer.generate_active_users_trend(save=False)
        
        # Assert
        assert fig is not None
        
    def test_generate_message_volume_trend(self, mock_user_data, tmp_path):
        """Test generating message volume trend."""
        # Setup
        with patch.object(UserAnalysis, '_load_data'):
            analyzer = UserAnalysis(tmp_path, tmp_path)
            analyzer.user_data = mock_user_data.copy()
        
        # Execute
        fig = analyzer.generate_message_volume_trend(save=False)
        
        # Assert
        assert fig is not None
        
    def test_generate_gpt_usage_trend(self, mock_user_data, tmp_path):
        """Test generating GPT usage trend."""
        # Setup
        with patch.object(UserAnalysis, '_load_data'):
            analyzer = UserAnalysis(tmp_path, tmp_path)
            analyzer.user_data = mock_user_data.copy()
        
        # Execute
        fig = analyzer.generate_gpt_usage_trend(save=False)
        
        # Assert
        assert fig is not None
        
    def test_generate_message_histogram(self, mock_user_data, tmp_path):
        """Test generating a message histogram."""
        with patch.object(UserAnalysis, '_load_data'):
            analysis = UserAnalysis(data_dir=tmp_path, output_dir=tmp_path)
            analysis.user_data = mock_user_data
            
            # Test with default parameters
            fig = analysis.generate_message_histogram(save=False)
            assert isinstance(fig, plt.Figure)
            
            # Test with custom parameters
            fig = analysis.generate_message_histogram(bins=10, max_value=50, save=False)
            assert isinstance(fig, plt.Figure)
            
            # Test saving the figure
            analysis.generate_message_histogram(save=True)
            assert (tmp_path / "message_histogram.png").exists()
