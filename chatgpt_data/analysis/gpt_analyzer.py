"""GPT data analysis module."""

from pathlib import Path
from typing import Optional, Union

import matplotlib.pyplot as plt
import pandas as pd

from chatgpt_data.analysis.interfaces import DataAnalyzer
from chatgpt_data.utils.constants import DEFAULT_TIMEZONE
from chatgpt_data.utils.data_loader import GPTDataLoader
from chatgpt_data.utils.visualization import Visualizer


class GPTAnalyzer(DataAnalyzer):
    """Class for analyzing ChatGPT custom GPT engagement data."""

    def __init__(self, data_dir: Union[str, Path], output_dir: Union[str, Path]):
        """Initialize the GPTAnalyzer class.

        Args:
            data_dir: Directory containing the raw data files
            output_dir: Directory to save the output files
        """
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        
        # Initialize loaders and utilities
        self.data_loader = GPTDataLoader(data_dir)
        self.visualizer = Visualizer(output_dir)
        
        # Load data
        self.gpt_data = self.data_loader.load_gpt_data()
    
    def get_active_gpts_trend(self) -> pd.DataFrame:
        """Get the trend of active GPTs over time.
        
        Returns:
            DataFrame with period_start and active_gpts columns
        """
        if self.gpt_data is None:
            raise ValueError("GPT data not loaded")

        # Group by period and count active GPTs
        active_gpts = (
            self.gpt_data[self.gpt_data["is_active"] == 1]
            .groupby(["period_start"])
            .size()
            .reset_index(name="active_gpts")
        )
        
        # Convert period_start to datetime with consistent timezone handling
        active_gpts["period_start"] = pd.to_datetime(active_gpts["period_start"], utc=True).dt.tz_convert(DEFAULT_TIMEZONE)
        
        # Sort by date
        active_gpts = active_gpts.sort_values("period_start")
        
        return active_gpts
    
    def generate_active_gpts_trend(self, save: bool = True) -> Optional[plt.Figure]:
        """Generate a trend graph of active GPTs over time.

        Args:
            save: Whether to save the figure to the output directory

        Returns:
            The matplotlib figure if save is False, otherwise None
        """
        active_gpts = self.get_active_gpts_trend()
        
        # Create the visualization
        comment = """This graph shows the number of active custom GPTs for each time period.
An active GPT is defined as one that received at least one message during the period.
The trend indicates how custom GPT adoption and usage has evolved over time."""
        
        return self.visualizer.create_time_series_plot(
            data=active_gpts,
            x_col="period_start",
            y_col="active_gpts",
            title="Active GPTs Trend",
            xlabel="Period",
            ylabel="Number of Active GPTs",
            comment=comment,
            filename="active_gpts_trend.png",
            save=save
        )
    
    def get_gpt_messages_trend(self) -> pd.DataFrame:
        """Get the trend of GPT messages over time.
        
        Returns:
            DataFrame with period_start and messages_workspace columns
        """
        if self.gpt_data is None:
            raise ValueError("GPT data not loaded")

        # Group by period and sum messages
        message_volume = (
            self.gpt_data
            .groupby(["period_start"])
            ["messages_workspace"]
            .sum()
            .reset_index()
        )
        
        # Convert period_start to datetime with consistent timezone handling
        message_volume["period_start"] = pd.to_datetime(message_volume["period_start"], utc=True).dt.tz_convert(DEFAULT_TIMEZONE)
        
        # Sort by date
        message_volume = message_volume.sort_values("period_start")
        
        return message_volume
    
    def generate_gpt_messages_trend(self, save: bool = True) -> Optional[plt.Figure]:
        """Generate a trend graph of GPT messages over time.

        Args:
            save: Whether to save the figure to the output directory

        Returns:
            The matplotlib figure if save is False, otherwise None
        """
        message_volume = self.get_gpt_messages_trend()
        
        # Create the visualization
        comment = """This graph shows the total number of messages sent to custom GPTs for each time period.
The trend indicates overall GPT usage intensity and can help identify which periods
had the highest GPT engagement and interaction."""
        
        return self.visualizer.create_time_series_plot(
            data=message_volume,
            x_col="period_start",
            y_col="messages_workspace",
            title="GPT Messages Trend",
            xlabel="Period",
            ylabel="Number of GPT Messages",
            comment=comment,
            filename="gpt_messages_trend.png",
            save=save
        )
    
    def get_unique_messagers_trend(self) -> pd.DataFrame:
        """Get the trend of unique GPT messagers over time.
        
        Returns:
            DataFrame with period_start and unique_messagers_workspace columns
        """
        if self.gpt_data is None:
            raise ValueError("GPT data not loaded")

        # Group by period and sum unique messagers
        unique_messagers = (
            self.gpt_data
            .groupby(["period_start"])
            ["unique_messagers_workspace"]
            .sum()
            .reset_index()
        )
        
        # Convert period_start to datetime with consistent timezone handling
        unique_messagers["period_start"] = pd.to_datetime(unique_messagers["period_start"], utc=True).dt.tz_convert(DEFAULT_TIMEZONE)
        
        # Sort by date
        unique_messagers = unique_messagers.sort_values("period_start")
        
        return unique_messagers
    
    def generate_unique_messagers_trend(self, save: bool = True) -> Optional[plt.Figure]:
        """Generate a trend graph of unique GPT messagers over time.

        Args:
            save: Whether to save the figure to the output directory

        Returns:
            The matplotlib figure if save is False, otherwise None
        """
        unique_messagers = self.get_unique_messagers_trend()
        
        # Create the visualization
        comment = """This graph shows the number of unique users interacting with custom GPTs in each period.
The trend indicates how broadly GPTs are being adopted across the user base and
can help measure the effectiveness of GPT promotion and training efforts."""
        
        return self.visualizer.create_time_series_plot(
            data=unique_messagers,
            x_col="period_start",
            y_col="unique_messagers_workspace",
            title="Unique GPT Messagers Trend",
            xlabel="Period",
            ylabel="Number of Unique Messagers",
            comment=comment,
            filename="unique_messagers_trend.png",
            save=save
        )
    
    def generate_all_trends(self) -> None:
        """Generate all trend graphs and save them to the output directory."""
        self.generate_active_gpts_trend()
        self.generate_gpt_messages_trend()
        self.generate_unique_messagers_trend()