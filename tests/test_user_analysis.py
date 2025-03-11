"""Tests for the user analyzer module."""

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
import matplotlib.pyplot as plt

from chatgpt_data.analysis.user_analyzer import UserAnalyzer
from chatgpt_data.utils.data_loader import UserDataLoader


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
    """Test the UserAnalyzer class."""

    def test_data_loader(self, mock_user_data, tmp_path):
        """Test the data loader functionality."""
        # Create test files
        test_file = tmp_path / "proofpoint_user_engagement_test.csv"
        mock_user_data.to_csv(test_file, index=False)
        
        # Create a loader directly
        loader = UserDataLoader(tmp_path)
        data = loader.load_user_data()
        
        # Assert
        assert data is not None
        assert len(data) == len(mock_user_data)

    def test_generate_active_users_trend(self, mock_user_data, tmp_path):
        """Test generating active users trend."""
        # Setup - use patcher to avoid real loading
        with patch('chatgpt_data.utils.data_loader.UserDataLoader.load_user_data', return_value=mock_user_data.copy()):
            analyzer = UserAnalyzer(tmp_path, tmp_path)
        
        # Execute
        fig = analyzer.generate_active_users_trend(save=False)
        
        # Assert
        assert fig is not None
        
    def test_generate_message_volume_trend(self, mock_user_data, tmp_path):
        """Test generating message volume trend."""
        # Setup - use patcher to avoid real loading
        with patch('chatgpt_data.utils.data_loader.UserDataLoader.load_user_data', return_value=mock_user_data.copy()):
            analyzer = UserAnalyzer(tmp_path, tmp_path)
        
        # Execute
        fig = analyzer.generate_message_volume_trend(save=False)
        
        # Assert
        assert fig is not None
        
    # Note: Removed test_generate_gpt_usage_trend as we moved this feature to GPTAnalyzer
        
    def test_generate_message_histogram(self, mock_user_data, tmp_path):
        """Test generating a message histogram."""
        # Setup - use patcher to avoid real loading
        with patch('chatgpt_data.utils.data_loader.UserDataLoader.load_user_data', return_value=mock_user_data.copy()):
            analyzer = UserAnalyzer(tmp_path, tmp_path)
        
        # Test with default parameters
        fig = analyzer.generate_message_histogram(save=False)
        assert isinstance(fig, plt.Figure)
        
        # Test with custom parameters
        fig = analyzer.generate_message_histogram(bins=10, max_value=50, save=False)
        assert isinstance(fig, plt.Figure)
        
        # Test saving the figure
        analyzer.generate_message_histogram(save=True)
        assert (tmp_path / "message_histogram.png").exists()
