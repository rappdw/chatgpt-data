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
        # UserDataLoader will load and cache all data during initialization
        self.data_loader = UserDataLoader(data_dir, load_all=True)
        self.visualizer = Visualizer(output_dir)
        
        # Initialize name matcher if management chains are available
        self.name_matcher = NameMatcher(self.data_loader.get_management_chains())
    
    def resolve_user_name_from_ad(self, email: str) -> str:
        """Resolve a user's display name from their email using AD data.
        
        Args:
            email: User's email address
            
        Returns:
            Display name from AD if found, otherwise the original email
        """
        # Use the refactored method from UserDataLoader
        return self.data_loader.resolve_user_name_from_ad(email)
    
    def get_active_users_trend(self) -> pd.DataFrame:
        """Get the trend of active users over time.
        
        Returns:
            DataFrame with period_start and active_users columns
        """
        user_data = self.data_loader.get_user_data()
        if user_data is None:
            raise ValueError("No user data available")
            
        # Mark users as active if they have at least one message
        active_users = (
            user_data[user_data["messages"].fillna(0) > 0]
            .groupby("period_start")
            .agg(active_users=pd.NamedAgg(column="email", aggfunc="nunique"))
            .reset_index()
        )
        
        # If there's an 'is_active' column, use that instead
        if "is_active" in user_data.columns:
            active_users = (
                user_data[user_data["is_active"] == 1]
                .groupby("period_start")
                .agg(active_users=pd.NamedAgg(column="email", aggfunc="nunique"))
                .reset_index()
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
            DataFrame with period_start and total_messages columns
        """
        user_data = self.data_loader.get_user_data()
        if user_data is None:
            raise ValueError("No user data available")
            
        # Sum messages by period
        message_volume = (
            user_data
            .groupby("period_start")
            .agg(total_messages=pd.NamedAgg(column="messages", aggfunc="sum"))
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
            y_col="total_messages",
            title="Message Volume Trend",
            xlabel="Period",
            ylabel="Number of Messages",
            comment=comment,
            filename="message_volume_trend.png",
            save=save
        )
    
    def generate_message_histogram(self, output_path: Optional[Union[str, Path]] = None) -> None:
        """Generate a histogram of average messages per user.
        
        Args:
            output_path: Optional path to save the histogram image
        """
        user_data = self.data_loader.get_user_data()
        if user_data is None:
            raise ValueError("No user data available")
            
        # Default output path if not provided
        if output_path is None:
            output_path = self.output_dir / "message_histogram.png"
            
        # Generate the histogram
        self.visualizer.plot_message_histogram(
            data=user_data,
            x="messages",
            output_path=output_path,
            title="Distribution of Messages per User",
            xlabel="Messages",
            ylabel="Number of Users",
            log_scale=False
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
        user_data = self.data_loader.get_user_data()
        if user_data is None:
            raise ValueError("No user data available")
        
        comment = """This histogram shows the distribution of messages sent by users with a logarithmic y-axis scale.
The x-axis represents the number of messages per user.
The y-axis (log scale) shows how many users fall into each message count range.
Log scale helps visualize the distribution when there are large differences in frequency counts."""
        
        return self.visualizer.create_histogram(
            data=user_data,
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
    
    def get_engagement_levels(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Get user engagement levels by analyzing message patterns.
        
        Returns:
            Tuple of (engaged_users, non_engaged_users) DataFrames
        """
        user_data = self.data_loader.get_user_data()
        if user_data is None:
            raise ValueError("No user data available")
            
        # Print some diagnostic information
        print(f"Total rows in user_data: {len(user_data)}")
        
        # Ensure messages column is numeric
        if not pd.api.types.is_numeric_dtype(user_data['messages']):
            try:
                print("Converting 'messages' column to numeric...")
                user_data['messages'] = pd.to_numeric(user_data['messages'], errors='coerce')
                print(f"Conversion successful. NaN values: {user_data['messages'].isna().sum()}")
            except Exception as e:
                print(f"Error converting 'messages' to numeric: {str(e)}")
                print(f"Sample 'messages' values: {user_data['messages'].head().tolist()}")
        
        # More diagnostics
        print(f"Rows with zero 'messages': {len(user_data[user_data['messages'] == 0])}")
        print(f"Rows with positive 'messages': {len(user_data[user_data['messages'] > 0])}")
        
        # Filter to valid data (non-null, positive messages)
        valid_data = user_data[
            (pd.notna(user_data["messages"])) & 
            (user_data["messages"] > 0)
        ]
        
        # If we have no valid data after filtering, try a different approach
        if len(valid_data) == 0:
            print("No valid data after filtering for positive messages. Using all non-null data.")
            valid_data = user_data[pd.notna(user_data["messages"])].copy()
            print(f"Rows after relaxed filtering: {len(valid_data)}")
            
        # Group by user identifiers and calculate statistics
        try:
            # First try with all identifiers
            engagement_df = (
                valid_data
                .groupby(["public_id", "name", "email"])
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