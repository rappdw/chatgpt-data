"""User data analysis module."""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from chatgpt_data.analysis.interfaces import DataAnalyzer
from chatgpt_data.utils.constants import DEFAULT_TIMEZONE
from chatgpt_data.utils.data_loader import UserDataLoader
from chatgpt_data.utils.name_matcher import NameMatcher
from chatgpt_data.utils.visualization import Visualizer


class UserAnalyzer(DataAnalyzer):
    """Class for analyzing ChatGPT user engagement data."""

    def __init__(self, data_dir: Union[str, Path], output_dir: Union[str, Path]):
        """Initialize the UserAnalyzer class.

        Args:
            data_dir: Directory containing the raw data files
            output_dir: Directory to save the output files
        """
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        
        # Initialize loaders and utilities
        self.data_loader = UserDataLoader(data_dir)
        self.visualizer = Visualizer(output_dir)
        
        # Load data
        self.user_data = self.data_loader.load_user_data()
        self.ad_data = self.data_loader.load_ad_data()
        self.management_chains = self.data_loader.load_management_chains()
        
        # Initialize name matcher if management chains are available
        self.name_matcher = NameMatcher(self.management_chains)
    
    def resolve_user_name_from_ad(self, email: str) -> str:
        """Resolve a user's display name from their email using AD data.
        
        Args:
            email: User's email address
            
        Returns:
            Display name from AD if found, otherwise the original email
        """
        if self.ad_data is None or email is None or pd.isna(email) or email == "":
            return email
            
        # Try to match on userPrincipalName first
        match = self.ad_data[self.ad_data["userPrincipalName"].str.lower() == email.lower()]
        
        # If no match, try the mail column
        if len(match) == 0:
            match = self.ad_data[self.ad_data["mail"].str.lower() == email.lower()]
            
        # Return the display name if found, otherwise return the email
        if len(match) > 0 and not pd.isna(match.iloc[0]["displayName"]):
            return match.iloc[0]["displayName"]
        else:
            return email
    
    def get_active_users_trend(self) -> pd.DataFrame:
        """Get the trend of active users over time.
        
        Returns:
            DataFrame with period_start and active_users columns
        """
        if self.user_data is None:
            raise ValueError("User data not loaded")

        # Define active users as those who have sent at least one message
        active_users = (
            self.user_data[self.user_data["messages"].fillna(0) > 0]
            .groupby(["period_start"])
            .size()
            .reset_index(name="active_users")
        )
        
        # If there are no active users based on messages, fall back to is_active flag
        if len(active_users) == 0 or active_users["active_users"].sum() == 0:
            print("No users with messages found, falling back to is_active flag")
            active_users = (
                self.user_data[self.user_data["is_active"] == 1]
                .groupby(["period_start"])
                .size()
                .reset_index(name="active_users")
            )
        
        # Convert period_start to datetime with consistent timezone handling
        active_users["period_start"] = pd.to_datetime(active_users["period_start"], utc=True).dt.tz_convert(DEFAULT_TIMEZONE)
        
        # Sort by date
        active_users = active_users.sort_values("period_start")
        
        return active_users
    
    def generate_active_users_trend(self, save: bool = True) -> Optional[plt.Figure]:
        """Generate a trend graph of active users over time.

        Args:
            save: Whether to save the figure to the output directory

        Returns:
            The matplotlib figure if save is False, otherwise None
        """
        active_users = self.get_active_users_trend()
        
        # Create the visualization
        comment = """This graph shows the number of active users for each time period.
An active user is defined as someone who sent at least one message during the period.
The trend indicates how user adoption and engagement has changed over time."""
        
        return self.visualizer.create_time_series_plot(
            data=active_users,
            x_col="period_start",
            y_col="active_users",
            title="Active Users Trend",
            xlabel="Period",
            ylabel="Number of Active Users",
            comment=comment,
            filename="active_users_trend.png",
            save=save
        )
    
    def get_message_volume_trend(self) -> pd.DataFrame:
        """Get the trend of message volume over time.
        
        Returns:
            DataFrame with period_start and messages columns
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
        message_volume["period_start"] = pd.to_datetime(message_volume["period_start"], utc=True).dt.tz_convert(DEFAULT_TIMEZONE)
        
        # Sort by date
        message_volume = message_volume.sort_values("period_start")
        
        return message_volume
    
    def generate_message_volume_trend(self, save: bool = True) -> Optional[plt.Figure]:
        """Generate a trend graph of message volume over time.

        Args:
            save: Whether to save the figure to the output directory

        Returns:
            The matplotlib figure if save is False, otherwise None
        """
        message_volume = self.get_message_volume_trend()
        
        # Create the visualization
        comment = """This graph shows the total number of messages sent across all users for each time period.
The trend indicates overall platform usage and can help identify seasonal patterns or
the impact of new features/promotions on engagement."""
        
        return self.visualizer.create_time_series_plot(
            data=message_volume,
            x_col="period_start",
            y_col="messages",
            title="Message Volume Trend",
            xlabel="Period",
            ylabel="Number of Messages",
            comment=comment,
            filename="message_volume_trend.png",
            save=save
        )
    
    def generate_message_histogram(self, bins: int = 20, max_value: Optional[int] = None, save: bool = True) -> Optional[plt.Figure]:
        """Generate a histogram of messages sent by users.
        
        Args:
            bins: Number of bins for the histogram
            max_value: Maximum value to include in the histogram (None for no limit)
            save: Whether to save the figure to the output directory
            
        Returns:
            The matplotlib figure if save is False, otherwise None
        """
        if self.user_data is None:
            raise ValueError("User data not loaded")
        
        comment = """This histogram shows the distribution of average messages sent by users.
The x-axis represents the average number of messages per user across all periods.
The y-axis shows how many users fall into each message count range.
This helps identify patterns in user engagement and message frequency."""
        
        return self.visualizer.create_histogram(
            data=self.user_data,
            value_col="messages",
            bins=bins,
            max_value=max_value,
            log_scale=False,
            title="Distribution of Messages per User",
            xlabel="Number of Messages",
            ylabel="Number of Users",
            comment=comment,
            filename="message_histogram.png",
            save=save
        )
    
    def generate_message_histogram_log(self, bins: int = 20, max_value: Optional[int] = None, save: bool = True) -> Optional[plt.Figure]:
        """Generate a histogram of messages sent by users with a logarithmic y-axis scale.
        
        Args:
            bins: Number of bins for the histogram
            max_value: Maximum value to include in the histogram (None for no limit)
            save: Whether to save the figure to the output directory
            
        Returns:
            The matplotlib figure if save is False, otherwise None
        """
        if self.user_data is None:
            raise ValueError("User data not loaded")
        
        comment = """This histogram shows the distribution of messages sent by users with a logarithmic y-axis scale.
The x-axis represents the number of messages per user.
The y-axis (log scale) shows how many users fall into each message count range.
Log scale helps visualize the distribution when there are large differences in frequency counts."""
        
        return self.visualizer.create_histogram(
            data=self.user_data,
            value_col="messages",
            bins=bins,
            max_value=max_value,
            log_scale=True,
            title="Distribution of Messages per User (Log Scale)",
            xlabel="Number of Messages",
            ylabel="Number of Users (Log Scale)",
            comment=comment,
            filename="message_histogram_log.png",
            save=save
        )
    
    def generate_all_trends(self) -> None:
        """Generate all trend graphs and save them to the output directory."""
        self.generate_active_users_trend()
        self.generate_message_volume_trend()
        self.generate_message_histogram()
        self.generate_message_histogram_log()
    
    def get_engagement_levels(self, high_threshold: int = 20, low_threshold: int = 5) -> pd.DataFrame:
        """Categorize users by engagement level based on average message count across all periods.
        
        Args:
            high_threshold: Minimum average number of messages to be considered highly engaged
            low_threshold: Maximum average number of messages to be considered low engaged
            
        Returns:
            DataFrame with user information and engagement level
        """
        if self.user_data is None:
            raise ValueError("User data not loaded")
            
        # Print data quality information for debugging
        print("\nMessage Data Quality Check:")
        print(f"Total rows in user_data: {len(self.user_data)}")
        print(f"Rows with null 'messages': {self.user_data['messages'].isna().sum()}")
        
        # Check if 'messages' column is numeric and convert if needed
        if not pd.api.types.is_numeric_dtype(self.user_data['messages']):
            print(f"Warning: 'messages' column is not numeric, attempting to convert")
            try:
                self.user_data['messages'] = pd.to_numeric(self.user_data['messages'], errors='coerce')
            except Exception as e:
                print(f"Error converting 'messages' to numeric: {e}")
                # Print sample values to help diagnose the issue
                print(f"Sample 'messages' values: {self.user_data['messages'].head().tolist()}")
        
        # Continue with debugging info
        print(f"Rows with zero 'messages': {len(self.user_data[self.user_data['messages'] == 0])}")
        print(f"Rows with positive 'messages': {len(self.user_data[self.user_data['messages'] > 0])}")
        
        # Filter to include rows with valid message data AND positive message count
        valid_data = self.user_data[
            (pd.notna(self.user_data["messages"])) & 
            (self.user_data["messages"] > 0)
        ].copy()
        
        print(f"Rows after filtering: {len(valid_data)}")
        
        # If we don't have enough data after filtering, use all data with messages >= 0
        if len(valid_data) < 2:
            print("Warning: Not enough data after filtering for positive messages. Using all valid message data.")
            valid_data = self.user_data[pd.notna(self.user_data["messages"])].copy()
            print(f"Rows after relaxed filtering: {len(valid_data)}")
            
        # Group by user identifiers and calculate statistics
        try:
            # First try with all identifiers
            engagement_df = (
                valid_data
                .groupby(["account_id", "public_id", "name", "email"])
                .agg({
                    "messages": "mean",
                    "period_start": "min",
                    "period_end": "max"
                })
                .reset_index()
            )
            
            # If we don't have enough data, try with just public_id
            if len(engagement_df) < 2:
                print("Warning: Not enough data after grouping. Trying with just public_id.")
                engagement_df = (
                    valid_data
                    .groupby(["public_id"])
                    .agg({
                        "messages": "mean",
                        "period_start": "min",
                        "period_end": "max",
                        "name": "first",
                        "email": "first",
                        "account_id": "first"
                    })
                    .reset_index()
                )
                
            print(f"Final engagement_df rows: {len(engagement_df)}")
        except Exception as e:
            print(f"Error in groupby operation: {e}")
            # Create a minimal DataFrame with just the message data
            engagement_df = pd.DataFrame({
                "public_id": valid_data["public_id"],
                "name": valid_data["name"],
                "email": valid_data["email"],
                "account_id": valid_data["account_id"],
                "avg_messages": valid_data["messages"],
                "first_period": valid_data["period_start"],
                "last_period": valid_data["period_end"]
            })
        
        # Rename columns for clarity
        engagement_df = engagement_df.rename(columns={
            "messages": "avg_messages",
            "period_start": "first_period",
            "period_end": "last_period"
        })
        
        # Add engagement level column
        conditions = [
            (engagement_df["avg_messages"] >= high_threshold),
            (engagement_df["avg_messages"] <= low_threshold) & (engagement_df["avg_messages"] > 0),
            (engagement_df["avg_messages"] == 0)
        ]
        choices = ["high", "low", "none"]
        engagement_df["engagement_level"] = np.select(conditions, choices, default="medium")
        
        return engagement_df