"""Tests for the user analysis module."""

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

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
    })


@patch("os.listdir")
@patch("pandas.read_csv")
def test_load_data(mock_read_csv, mock_listdir, mock_user_data, tmp_path):
    """Test loading user data."""
    # Setup
    mock_listdir.return_value = ["proofpoint_user_engagement_1.csv", "proofpoint_user_engagement_2.csv"]
    mock_read_csv.return_value = mock_user_data
    
    # Execute
    analyzer = UserAnalysis(tmp_path, tmp_path)
    
    # Assert
    assert mock_listdir.called
    assert mock_read_csv.call_count == 2
    assert analyzer.user_data is not None
    assert len(analyzer.user_data) == len(mock_user_data)


@patch("matplotlib.pyplot.savefig")
def test_generate_active_users_trend(mock_savefig, mock_user_data, tmp_path):
    """Test generating active users trend."""
    # Setup
    analyzer = UserAnalysis(tmp_path, tmp_path)
    analyzer.user_data = mock_user_data
    
    # Execute
    analyzer.generate_active_users_trend()
    
    # Assert
    mock_savefig.assert_called_once()
    assert "active_users_trend.png" in mock_savefig.call_args[0][0]


@patch("matplotlib.pyplot.savefig")
def test_generate_message_volume_trend(mock_savefig, mock_user_data, tmp_path):
    """Test generating message volume trend."""
    # Setup
    analyzer = UserAnalysis(tmp_path, tmp_path)
    analyzer.user_data = mock_user_data
    
    # Execute
    analyzer.generate_message_volume_trend()
    
    # Assert
    mock_savefig.assert_called_once()
    assert "message_volume_trend.png" in mock_savefig.call_args[0][0]


@patch("matplotlib.pyplot.savefig")
def test_generate_gpt_usage_trend(mock_savefig, mock_user_data, tmp_path):
    """Test generating GPT usage trend."""
    # Setup
    analyzer = UserAnalysis(tmp_path, tmp_path)
    analyzer.user_data = mock_user_data
    
    # Execute
    analyzer.generate_gpt_usage_trend()
    
    # Assert
    mock_savefig.assert_called_once()
    assert "gpt_usage_trend.png" in mock_savefig.call_args[0][0]
