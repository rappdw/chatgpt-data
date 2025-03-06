"""GPT engagement analysis module."""

import os
import fnmatch
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import pandas as pd


class GPTAnalysis:
    """Class for analyzing ChatGPT custom GPT engagement data."""

    def __init__(self, data_dir: Union[str, Path], output_dir: Union[str, Path]):
        """Initialize the GPTAnalysis class.

        Args:
            data_dir: Directory containing the raw data files
            output_dir: Directory to save the output files
        """
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.gpt_data = None
        self._load_data()

    def _load_data(self) -> None:
        """Load all GPT engagement data files."""
        gpt_files = [f for f in os.listdir(self.data_dir) if fnmatch.fnmatch(f, '*gpt_engagement*.csv')]
        
        dfs = []
        for file in gpt_files:
            try:
                # Try different encodings if UTF-8 fails
                encodings = ['utf-8', 'latin1', 'ISO-8859-1', 'cp1252']
                for encoding in encodings:
                    try:
                        df = pd.read_csv(self.data_dir / file, encoding=encoding)
                        print(f"Successfully loaded {file} with encoding {encoding}")
                        dfs.append(df)
                        break
                    except UnicodeDecodeError:
                        continue
            except Exception as e:
                print(f"Error loading {file}: {str(e)}")
        
        if dfs:
            self.gpt_data = pd.concat(dfs, ignore_index=True)
        else:
            raise FileNotFoundError("No GPT engagement data files found or could not be loaded")

    def generate_active_gpts_trend(self, save: bool = True) -> Optional[plt.Figure]:
        """Generate a trend graph of active GPTs over time.

        Args:
            save: Whether to save the figure to the output directory

        Returns:
            The matplotlib figure if save is False, otherwise None
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
        
        # Convert period_start to datetime
        active_gpts["period_start"] = pd.to_datetime(active_gpts["period_start"])
        
        # Sort by date
        active_gpts = active_gpts.sort_values("period_start")
        
        # Create the plot
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(active_gpts["period_start"], active_gpts["active_gpts"], marker="o")
        ax.set_title("Active GPTs Trend")
        ax.set_xlabel("Period")
        ax.set_ylabel("Number of Active GPTs")
        ax.grid(True)
        
        # Add comment box with explanation
        comment = """This graph shows the number of active custom GPTs for each time period.
An active GPT is defined as one that received at least one message during the period.
The trend indicates how custom GPT adoption and usage has evolved over time."""
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax.text(0.05, 0.05, comment, transform=ax.transAxes, fontsize=9,
                verticalalignment='bottom', bbox=props)
        
        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        if save:
            output_path = self.output_dir / "active_gpts_trend.png"
            plt.savefig(output_path)
            plt.close(fig)
            return None
        
        return fig

    def generate_gpt_messages_trend(self, save: bool = True) -> Optional[plt.Figure]:
        """Generate a trend graph of GPT messages over time.

        Args:
            save: Whether to save the figure to the output directory

        Returns:
            The matplotlib figure if save is False, otherwise None
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
        
        # Convert period_start to datetime
        message_volume["period_start"] = pd.to_datetime(message_volume["period_start"])
        
        # Sort by date
        message_volume = message_volume.sort_values("period_start")
        
        # Create the plot
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(message_volume["period_start"], message_volume["messages_workspace"], marker="o")
        ax.set_title("GPT Messages Trend")
        ax.set_xlabel("Period")
        ax.set_ylabel("Number of GPT Messages")
        ax.grid(True)
        
        # Add comment box with explanation
        comment = """This graph shows the total number of messages sent to custom GPTs for each time period.
The trend indicates overall GPT usage intensity and can help identify which periods
had the highest GPT engagement and interaction."""
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax.text(0.05, 0.05, comment, transform=ax.transAxes, fontsize=9,
                verticalalignment='bottom', bbox=props)
        
        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        if save:
            output_path = self.output_dir / "gpt_messages_trend.png"
            plt.savefig(output_path)
            plt.close(fig)
            return None
        
        return fig

    def generate_unique_messagers_trend(self, save: bool = True) -> Optional[plt.Figure]:
        """Generate a trend graph of unique GPT messagers over time.

        Args:
            save: Whether to save the figure to the output directory

        Returns:
            The matplotlib figure if save is False, otherwise None
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
        
        # Convert period_start to datetime
        unique_messagers["period_start"] = pd.to_datetime(unique_messagers["period_start"])
        
        # Sort by date
        unique_messagers = unique_messagers.sort_values("period_start")
        
        # Create the plot
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(unique_messagers["period_start"], unique_messagers["unique_messagers_workspace"], marker="o")
        ax.set_title("Unique GPT Messagers Trend")
        ax.set_xlabel("Period")
        ax.set_ylabel("Number of Unique Messagers")
        ax.grid(True)
        
        # Add comment box with explanation
        comment = """This graph shows the number of unique users interacting with custom GPTs in each period.
The trend indicates how broadly GPTs are being adopted across the user base and
can help measure the effectiveness of GPT promotion and training efforts."""
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax.text(0.05, 0.05, comment, transform=ax.transAxes, fontsize=9,
                verticalalignment='bottom', bbox=props)
        
        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        if save:
            output_path = self.output_dir / "unique_messagers_trend.png"
            plt.savefig(output_path)
            plt.close(fig)
            return None
        
        return fig

    def generate_all_trends(self) -> None:
        """Generate all trend graphs and save them to the output directory."""
        self.generate_active_gpts_trend()
        self.generate_gpt_messages_trend()
        self.generate_unique_messagers_trend()
