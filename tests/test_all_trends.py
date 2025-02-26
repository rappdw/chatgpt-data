import os
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from chatgpt_data.cli import all_trends


class TestAllTrends:
    """Test the all_trends CLI."""

    @patch("chatgpt_data.cli.all_trends.UserAnalysis")
    @patch("chatgpt_data.cli.all_trends.GPTAnalysis")
    def test_main_default_options(self, mock_gpt_analysis, mock_user_analysis, tmp_path):
        """Test main function with default options."""
        # Setup
        mock_user_analyzer = MagicMock()
        mock_gpt_analyzer = MagicMock()
        mock_user_analysis.return_value = mock_user_analyzer
        mock_gpt_analysis.return_value = mock_gpt_analyzer
        
        # Execute
        with patch("sys.argv", ["all_trends", 
                               f"--data-dir={tmp_path}", 
                               f"--output-dir={tmp_path}"]):
            all_trends.main()
        
        # Assert
        mock_user_analysis.assert_called_once_with(str(tmp_path), tmp_path)
        mock_gpt_analysis.assert_called_once()
        
        # Check that all methods were called
        mock_user_analyzer.generate_all_trends.assert_called_once()
        mock_gpt_analyzer.generate_all_trends.assert_called_once()
        mock_user_analyzer.generate_engagement_report.assert_called_once()
        mock_user_analyzer.generate_non_engagement_report.assert_called_once()
        
    @patch("chatgpt_data.cli.all_trends.UserAnalysis")
    @patch("chatgpt_data.cli.all_trends.GPTAnalysis")
    def test_main_skip_options(self, mock_gpt_analysis, mock_user_analysis, tmp_path):
        """Test main function with skip options."""
        # Setup
        mock_user_analyzer = MagicMock()
        mock_gpt_analyzer = MagicMock()
        mock_user_analysis.return_value = mock_user_analyzer
        mock_gpt_analysis.return_value = mock_gpt_analyzer
        
        # Execute
        with patch("sys.argv", ["all_trends", 
                               f"--data-dir={tmp_path}", 
                               f"--output-dir={tmp_path}",
                               "--skip-user-trends",
                               "--skip-engagement-report"]):
            all_trends.main()
        
        # Assert
        # When --skip-user-trends is set, UserAnalysis is not instantiated
        mock_user_analysis.assert_not_called()
        mock_gpt_analysis.assert_called_once()
        
        # Since UserAnalysis is never instantiated, we don't check its methods
        mock_gpt_analyzer.generate_all_trends.assert_called_once()
        
    @patch("chatgpt_data.cli.all_trends.UserAnalysis")
    @patch("chatgpt_data.cli.all_trends.GPTAnalysis")
    def test_main_only_histogram(self, mock_gpt_analysis, mock_user_analysis):
        """Test main function with only histogram option."""
        # Setup
        mock_user_analyzer = MagicMock()
        mock_user_analysis.return_value = mock_user_analyzer
        
        # Execute
        with patch("sys.argv", ["all_trends", 
                               "--data-dir=data",
                               "--output-dir=output",
                               "--only-message-histogram"]):
            all_trends.main()
        
        # Assert
        mock_user_analysis.assert_called_once_with("data", Path("output"))
        mock_gpt_analysis.assert_not_called()
        
        # Check that only the histogram method was called
        mock_user_analyzer.generate_message_histogram.assert_called_once_with(bins=20, max_value=None)
        mock_user_analyzer.generate_all_trends.assert_not_called()
        mock_user_analyzer.generate_engagement_report.assert_not_called()
        mock_user_analyzer.generate_non_engaged_report.assert_not_called()
