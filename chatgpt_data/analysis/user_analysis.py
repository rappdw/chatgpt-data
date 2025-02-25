"""User engagement analysis module."""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class UserAnalysis:
    """Class for analyzing ChatGPT user engagement data."""

    def __init__(self, data_dir: Union[str, Path], output_dir: Union[str, Path]):
        """Initialize the UserAnalysis class.

        Args:
            data_dir: Directory containing the raw data files
            output_dir: Directory to save the output files
        """
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.user_data = None
        self._load_data()

    def _load_data(self) -> None:
        """Load all user engagement data files."""
        user_files = [f for f in os.listdir(self.data_dir) if f.startswith("proofpoint_user_engagement")]
        
        dfs = []
        for file in user_files:
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
            self.user_data = pd.concat(dfs, ignore_index=True)
        else:
            raise FileNotFoundError("No user engagement data files found or could not be loaded")

    def generate_active_users_trend(self, save: bool = True) -> Optional[plt.Figure]:
        """Generate a trend graph of active users over time.

        Args:
            save: Whether to save the figure to the output directory

        Returns:
            The matplotlib figure if save is False, otherwise None
        """
        if self.user_data is None:
            raise ValueError("User data not loaded")

        # Group by period and count active users
        active_users = (
            self.user_data[self.user_data["is_active"] == 1]
            .groupby(["period_start"])
            .size()
            .reset_index(name="active_users")
        )
        
        # Convert period_start to datetime
        active_users["period_start"] = pd.to_datetime(active_users["period_start"])
        
        # Sort by date
        active_users = active_users.sort_values("period_start")
        
        # Create the plot
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(active_users["period_start"], active_users["active_users"], marker="o")
        ax.set_title("Active Users Trend")
        ax.set_xlabel("Period")
        ax.set_ylabel("Number of Active Users")
        ax.grid(True)
        
        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        if save:
            output_path = self.output_dir / "active_users_trend.png"
            plt.savefig(output_path)
            plt.close(fig)
            return None
        
        return fig

    def generate_message_volume_trend(self, save: bool = True) -> Optional[plt.Figure]:
        """Generate a trend graph of message volume over time.

        Args:
            save: Whether to save the figure to the output directory

        Returns:
            The matplotlib figure if save is False, otherwise None
        """
        if self.user_data is None:
            raise ValueError("User data not loaded")

        # Group by period and sum messages
        message_volume = (
            self.user_data
            .groupby(["period_start"])
            ["messages"]
            .sum()
            .reset_index()
        )
        
        # Convert period_start to datetime
        message_volume["period_start"] = pd.to_datetime(message_volume["period_start"])
        
        # Sort by date
        message_volume = message_volume.sort_values("period_start")
        
        # Create the plot
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(message_volume["period_start"], message_volume["messages"], marker="o")
        ax.set_title("Message Volume Trend")
        ax.set_xlabel("Period")
        ax.set_ylabel("Number of Messages")
        ax.grid(True)
        
        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        if save:
            output_path = self.output_dir / "message_volume_trend.png"
            plt.savefig(output_path)
            plt.close(fig)
            return None
        
        return fig

    def generate_gpt_usage_trend(self, save: bool = True) -> Optional[plt.Figure]:
        """Generate a trend graph of GPT usage over time.

        Args:
            save: Whether to save the figure to the output directory

        Returns:
            The matplotlib figure if save is False, otherwise None
        """
        if self.user_data is None:
            raise ValueError("User data not loaded")

        # Group by period and sum GPT messages
        gpt_usage = (
            self.user_data
            .groupby(["period_start"])
            ["gpt_messages"]
            .sum()
            .reset_index()
        )
        
        # Convert period_start to datetime
        gpt_usage["period_start"] = pd.to_datetime(gpt_usage["period_start"])
        
        # Sort by date
        gpt_usage = gpt_usage.sort_values("period_start")
        
        # Create the plot
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(gpt_usage["period_start"], gpt_usage["gpt_messages"], marker="o")
        ax.set_title("GPT Usage Trend")
        ax.set_xlabel("Period")
        ax.set_ylabel("Number of GPT Messages")
        ax.grid(True)
        
        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        if save:
            output_path = self.output_dir / "gpt_usage_trend.png"
            plt.savefig(output_path)
            plt.close(fig)
            return None
        
        return fig

    def generate_all_trends(self) -> None:
        """Generate all trend graphs and save them to the output directory."""
        self.generate_active_users_trend()
        self.generate_message_volume_trend()
        self.generate_gpt_usage_trend()
        
    def get_engagement_levels(self, high_threshold: int = 20, low_threshold: int = 5) -> pd.DataFrame:
        """Categorize users by engagement level based on message count.
        
        Args:
            high_threshold: Minimum number of messages to be considered highly engaged
            low_threshold: Maximum number of messages to be considered low engaged
            
        Returns:
            DataFrame with user information and engagement level
        """
        if self.user_data is None:
            raise ValueError("User data not loaded")
            
        # Create a copy of the user data with relevant columns
        engagement_df = self.user_data[
            ["account_id", "public_id", "name", "email", "messages", "period_start", "period_end"]
        ].copy()
        
        # Add engagement level column
        conditions = [
            (engagement_df["messages"] >= high_threshold),
            (engagement_df["messages"] <= low_threshold) & (engagement_df["messages"] > 0),
            (engagement_df["messages"] == 0)
        ]
        choices = ["high", "low", "none"]
        engagement_df["engagement_level"] = np.select(conditions, choices, default="medium")
        
        return engagement_df
    
    def generate_engagement_report(self, output_file: str = None) -> pd.DataFrame:
        """Generate a report of user engagement levels.
        
        Args:
            output_file: Path to save the report CSV file (optional)
            
        Returns:
            DataFrame with user engagement report
        """
        if self.user_data is None:
            raise ValueError("User data not loaded")
            
        # Get engagement levels
        engagement_df = self.get_engagement_levels()
        
        # Group by user and get the most recent period data
        user_latest = (
            engagement_df
            .sort_values("period_end", ascending=False)
            .groupby(["account_id", "public_id", "email"])
            .first()
            .reset_index()
        )
        
        # Count users by engagement level
        engagement_counts = user_latest["engagement_level"].value_counts().to_dict()
        print(f"Engagement Levels Summary:")
        print(f"  High: {engagement_counts.get('high', 0)}")
        print(f"  Medium: {engagement_counts.get('medium', 0)}")
        print(f"  Low: {engagement_counts.get('low', 0)}")
        print(f"  None: {engagement_counts.get('none', 0)}")
        
        # Save to file if specified
        if output_file:
            user_latest.to_csv(output_file, index=False)
            print(f"Engagement report saved to {output_file}")
            
        return user_latest
    
    def get_non_engaged_users(self) -> pd.DataFrame:
        """Identify users who have never engaged across all tracked periods.
        
        Returns:
            DataFrame with non-engaged user information
        """
        if self.user_data is None:
            raise ValueError("User data not loaded")
            
        # Get all unique users
        all_users = self.user_data[["account_id", "public_id", "name", "email"]].drop_duplicates()
        
        # Get users who have been active at least once
        active_users = self.user_data[self.user_data["is_active"] == 1][
            ["account_id", "public_id", "name", "email"]
        ].drop_duplicates()
        
        # Find users who have never been active
        never_active = pd.merge(
            all_users, active_users, 
            on=["account_id", "public_id", "email"], 
            how="left", 
            indicator=True,
            suffixes=("", "_active")
        )
        never_active = never_active[never_active["_merge"] == "left_only"].drop(["_merge", "name_active"], axis=1, errors="ignore")
        
        # Get additional user information
        user_info = self.user_data[["account_id", "public_id", "email", "user_role", "role", "department", "user_status", "created_or_invited_date"]].drop_duplicates()
        never_active = pd.merge(
            never_active,
            user_info,
            on=["account_id", "public_id", "email"],
            how="left"
        )
        
        return never_active

    def generate_non_engagement_report(self, output_file: str = None) -> pd.DataFrame:
        """Generate a report of users who have never engaged.
        
        Args:
            output_file: Path to save the report CSV file (optional)
            
        Returns:
            DataFrame with non-engaged user report
        """
        # Get non-engaged users
        non_engaged = self.get_non_engaged_users()
        
        print(f"Non-Engaged Users Summary:")
        print(f"  Total non-engaged users: {len(non_engaged)}")
        
        # Count by user status
        status_counts = non_engaged["user_status"].value_counts().to_dict()
        for status, count in status_counts.items():
            print(f"  {status}: {count}")
        
        # Save to file if specified
        if output_file:
            non_engaged.to_csv(output_file, index=False)
            print(f"Non-engagement report saved to {output_file}")
            
        return non_engaged
