"""User engagement analysis module."""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import re

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
        self.ad_data = None
        self._load_data()
        self._load_ad_data()

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

    def _load_ad_data(self) -> None:
        """Load Active Directory export data for name resolution."""
        ad_file = self.data_dir / "AD_export.csv"
        
        if not ad_file.exists():
            print("AD export file not found. User name resolution from AD will not be available.")
            return
            
        try:
            # Try different encodings if UTF-8 fails
            encodings = ['utf-8', 'utf-8-sig', 'latin1', 'ISO-8859-1', 'cp1252']
            for encoding in encodings:
                try:
                    self.ad_data = pd.read_csv(ad_file, encoding=encoding)
                    print(f"Successfully loaded AD export data with encoding {encoding}")
                    print(f"AD data contains {len(self.ad_data)} records")
                    break
                except UnicodeDecodeError:
                    continue
                    
            if self.ad_data is None:
                print("Could not load AD export data with any encoding.")
        except Exception as e:
            print(f"Error loading AD export data: {str(e)}")
            
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
        
        # Add comment box with explanation
        comment = """This graph shows the number of active users for each time period.
An active user is defined as someone who sent at least one message during the period.
The trend indicates how user adoption and engagement has changed over time."""
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax.text(0.05, 0.05, comment, transform=ax.transAxes, fontsize=9,
                verticalalignment='bottom', bbox=props)
        
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
        
        # Add comment box with explanation
        comment = """This graph shows the total number of messages sent across all users for each time period.
The trend indicates overall platform usage and can help identify seasonal patterns or
the impact of new features/promotions on engagement."""
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax.text(0.05, 0.05, comment, transform=ax.transAxes, fontsize=9,
                verticalalignment='bottom', bbox=props)
        
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
        
        # Add comment box with explanation
        comment = """This graph shows the total number of messages sent to GPTs for each time period.
The trend indicates how GPT adoption has changed over time and can help measure
the impact of new GPT features or training initiatives."""
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax.text(0.05, 0.05, comment, transform=ax.transAxes, fontsize=9,
                verticalalignment='bottom', bbox=props)
        
        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        if save:
            output_path = self.output_dir / "gpt_usage_trend.png"
            plt.savefig(output_path)
            plt.close(fig)
            return None
        
        return fig

    def generate_message_histogram(self, bins: int = 20, max_value: Optional[int] = None, save: bool = True) -> Optional[plt.Figure]:
        """Generate a histogram of average messages sent by users.
        
        Args:
            bins: Number of bins for the histogram
            max_value: Maximum value to include in the histogram (None for no limit)
            save: Whether to save the figure to the output directory
            
        Returns:
            The matplotlib figure if save is False, otherwise None
        """
        if self.user_data is None:
            raise ValueError("User data not loaded")
            
        # Get engagement data which already has average messages calculated
        engagement_df = self.get_engagement_levels()
        
        # Filter out extreme values if max_value is specified
        if max_value is not None:
            plot_data = engagement_df[engagement_df["avg_messages"] <= max_value]
        else:
            plot_data = engagement_df
        
        # Create the histogram
        fig, ax = plt.subplots(figsize=(12, 7))
        
        # Plot histogram with density=False to show counts
        n, bins_array, patches = ax.hist(
            plot_data["avg_messages"], 
            bins=bins, 
            edgecolor='black', 
            alpha=0.7
        )
        
        # Add a vertical line for the mean
        mean_value = plot_data["avg_messages"].mean()
        median_value = plot_data["avg_messages"].median()
        ax.axvline(mean_value, color='red', linestyle='dashed', linewidth=1, label=f'Mean: {mean_value:.2f}')
        ax.axvline(median_value, color='green', linestyle='dashed', linewidth=1, label=f'Median: {median_value:.2f}')
        
        # Add labels and title
        ax.set_title("Distribution of Average Messages per User")
        ax.set_xlabel("Average Number of Messages")
        ax.set_ylabel("Number of Users")
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Add comment box with explanation
        total_users = len(engagement_df)
        filtered_users = len(plot_data)
        filtered_text = ""
        if max_value is not None and filtered_users < total_users:
            filtered_text = f"\nNote: {total_users - filtered_users} users with >{max_value} messages are not shown."
            
        comment = f"""This histogram shows the distribution of average messages sent by users.
The x-axis represents the average number of messages per user across all periods.
The y-axis shows how many users fall into each message count range.
This helps identify patterns in user engagement and message frequency.{filtered_text}"""
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax.text(0.05, 0.95, comment, transform=ax.transAxes, fontsize=9,
                verticalalignment='top', bbox=props)
        
        plt.tight_layout()
        
        if save:
            output_path = self.output_dir / "message_histogram.png"
            plt.savefig(output_path)
            plt.close(fig)
            return None
        
        return fig

    def generate_all_trends(self) -> None:
        """Generate all trend graphs and save them to the output directory."""
        self.generate_active_users_trend()
        self.generate_message_volume_trend()
        self.generate_gpt_usage_trend()
        self.generate_message_histogram()

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
            
        # Filter to only include rows with valid message data
        valid_data = self.user_data[pd.notna(self.user_data["messages"])].copy()
        
        # Calculate average messages per user across all periods
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
    
    def generate_engagement_report(self, output_file: str = None) -> pd.DataFrame:
        """Generate a report of user engagement levels based on average message count.
        
        Args:
            output_file: Path to save the report CSV file (optional)
            
        Returns:
            DataFrame with user engagement report
        """
        if self.user_data is None:
            raise ValueError("User data not loaded")
            
        # Print data quality information for debugging
        print("\nData Quality Check:")
        print(f"Total rows in user_data: {len(self.user_data)}")
        print(f"Rows with null 'name': {self.user_data['name'].isna().sum()}")
        print(f"Rows with null 'email': {self.user_data['email'].isna().sum()}")
        print(f"Rows with null 'public_id': {self.user_data['public_id'].isna().sum()}")
        
        # Get the latest period data to identify active users
        latest_period = self.user_data["period_end"].max()
        latest_data = self.user_data[self.user_data["period_end"] == latest_period]
        
        # Filter out users with pending status
        active_users = latest_data[latest_data["user_status"] != "pending"]
        active_user_ids = set(active_users["public_id"].unique())
        
        # Filter to only include rows with valid message data and active users
        valid_data = self.user_data[
            (pd.notna(self.user_data["messages"])) & 
            (self.user_data["public_id"].isin(active_user_ids))
        ].copy()
        
        # Create a mapping of email to AD display name for faster processing
        email_to_display_name = {}
        
        # Only create the mapping if AD data is available
        if self.ad_data is not None:
            # Count rows with valid emails that we can potentially resolve
            rows_with_email = valid_data["email"].notna().sum()
            
            print(f"\nAttempting to resolve display names from AD data for {rows_with_email} users with emails")
            
            # Process userPrincipalName column
            for _, row in self.ad_data.iterrows():
                if pd.notna(row["userPrincipalName"]) and pd.notna(row["displayName"]):
                    email_to_display_name[row["userPrincipalName"].lower()] = row["displayName"]
                
                # Also process the mail column if it exists and is different from userPrincipalName
                if pd.notna(row["mail"]) and pd.notna(row["displayName"]):
                    if row["mail"].lower() not in email_to_display_name:
                        email_to_display_name[row["mail"].lower()] = row["displayName"]
            
            print(f"Created mapping for {len(email_to_display_name)} email addresses from AD data")
        
        # Fill missing names with a placeholder
        valid_data["name"] = valid_data["name"].fillna("Unknown User")
        
        # Calculate average messages per user across all periods
        user_avg = (
            valid_data
            .groupby(["account_id", "public_id", "name", "email"])
            .agg({
                "messages": "mean",
                "period_start": "min",
                "period_end": "max",
                "is_active": "sum"
            })
            .reset_index()
        )
        
        # Print information about the grouped data
        print(f"\nAfter grouping:")
        print(f"Total unique users: {len(user_avg)}")
        print(f"Users with 'Unknown User' as name: {(user_avg['name'] == 'Unknown User').sum()}")
        
        # Get the created date for each user to calculate periods they could have been active
        user_created_dates = self.user_data[["public_id", "created_or_invited_date"]].drop_duplicates()
        
        # Print information about user_created_dates
        print(f"\nUser created dates:")
        print(f"Total rows: {len(user_created_dates)}")
        print(f"Unique public_ids: {user_created_dates['public_id'].nunique()}")
        print(f"Rows with null public_id: {user_created_dates['public_id'].isna().sum()}")
        
        # Remove rows with null public_id before merging
        user_created_dates = user_created_dates.dropna(subset=["public_id"])
        
        # Merge with inner join instead of left join to avoid creating rows with missing data
        user_avg = pd.merge(user_created_dates, user_avg, on="public_id", how="inner")
        
        # Check for any rows with missing critical data after merge
        print(f"\nAfter merging with created dates:")
        print(f"Total rows: {len(user_avg)}")
        print(f"Rows with null name: {user_avg['name'].isna().sum()}")
        print(f"Rows with null email: {user_avg['email'].isna().sum()}")
        
        # Convert dates to datetime objects for comparison
        user_avg["first_period_dt"] = pd.to_datetime(user_avg["period_start"])
        user_avg["last_period_dt"] = pd.to_datetime(user_avg["period_end"])
        user_avg["created_date"] = pd.to_datetime(user_avg["created_or_invited_date"])
        
        # Get all periods in the dataset
        all_periods = sorted(pd.to_datetime(self.user_data["period_start"].unique()))
        
        # Calculate active periods and eligible periods for each user
        active_eligible_periods = []
        
        for _, row in user_avg.iterrows():
            # Get all periods the user appears in the dataset
            user_data = self.user_data[self.user_data["public_id"] == row["public_id"]]
            user_periods = pd.to_datetime(user_data["period_start"].unique())
            
            # Count active periods (where is_active=1)
            active_data = user_data[user_data["is_active"] == 1]
            active_periods = len(active_data)
            
            # Determine eligible periods (periods after user creation)
            if pd.isna(row["created_date"]):
                eligible_periods = len(all_periods)
            else:
                eligible_periods = sum(1 for period in all_periods if period >= row["created_date"])
                eligible_periods = max(eligible_periods, 1)  # Ensure we don't divide by zero
            
            # Ensure active periods don't exceed eligible periods
            active_periods = min(active_periods, eligible_periods)
            
            active_eligible_periods.append({
                "public_id": row["public_id"],
                "active_periods": active_periods,
                "eligible_periods": eligible_periods
            })
        
        # Convert to DataFrame and merge
        periods_df = pd.DataFrame(active_eligible_periods)
        user_avg = user_avg.drop(columns=["is_active"], errors="ignore")
        user_avg = pd.merge(user_avg, periods_df, on="public_id", how="left")
        
        # Calculate active period percentage
        user_avg["active_period_pct"] = (user_avg["active_periods"] / user_avg["eligible_periods"] * 100).round(1)
        
        # Rename columns for clarity and drop temporary columns
        user_avg = user_avg.rename(columns={
            "messages": "avg_messages",
        })
        
        # Drop temporary columns used for calculation
        user_avg = user_avg.drop(columns=[
            "created_date", 
            "created_or_invited_date", 
            "first_period_dt", 
            "last_period_dt"
        ], errors="ignore")
        
        # Add engagement level column based on average message count
        conditions = [
            (user_avg["avg_messages"] >= 20),
            (user_avg["avg_messages"] <= 5) & (user_avg["avg_messages"] > 0),
            (user_avg["avg_messages"] == 0)
        ]
        choices = ["high", "low", "none"]
        user_avg["engagement_level"] = np.select(conditions, choices, default="medium")
        
        # Create a custom sort order for engagement levels
        engagement_order = {"high": 0, "medium": 1, "low": 2, "none": 3}
        user_avg["engagement_sort"] = user_avg["engagement_level"].map(engagement_order)
        
        # Sort by engagement level (high to low) and then by average messages (high to low)
        user_avg = user_avg.sort_values(
            by=["engagement_sort", "avg_messages"], 
            ascending=[True, False]
        ).drop(columns=["engagement_sort"])
        
        # Count users by engagement level
        engagement_counts = user_avg["engagement_level"].value_counts().to_dict()
        print(f"Engagement Levels Summary (based on average message count):")
        print(f"  High: {engagement_counts.get('high', 0)}")
        print(f"  Medium: {engagement_counts.get('medium', 0)}")
        print(f"  Low: {engagement_counts.get('low', 0)}")
        print(f"  None: {engagement_counts.get('none', 0)}")
        
        # Save to file if specified
        if output_file:
            # Format avg_messages as a number with fixed decimal places to ensure Excel recognizes it as a number
            output_df = user_avg.copy()
            output_df["avg_messages"] = output_df["avg_messages"].round(2)
            
            # Create a display name column using AD data for all users
            output_df["display_name"] = output_df.apply(
                lambda row: email_to_display_name.get(row["email"].lower(), row["name"]) 
                if pd.notna(row["email"]) else row["name"],
                axis=1
            )
            
            # For any rows where display_name is still an email address, try to find a better name
            email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
            
            # Print diagnostic information about potential email addresses in display_name
            email_mask = output_df["display_name"].str.match(email_pattern, na=False)
            email_count = email_mask.sum()
            if email_count > 0:
                print(f"\nFound {email_count} display names that appear to be email addresses")
                
                # For rows where display_name is an email, try to extract a name from the email
                for idx, row in output_df[email_mask].iterrows():
                    email = row["display_name"]
                    # Extract the username part of the email (before @)
                    username = email.split('@')[0]
                    # Convert username to a more readable format (e.g., john.doe -> John Doe)
                    name_parts = re.split(r'[._-]', username)
                    formatted_name = ' '.join([part.capitalize() for part in name_parts])
                    output_df.at[idx, "display_name"] = formatted_name
                    
                print(f"Converted {email_count} email addresses to formatted names")
            
            # Ensure we don't have any rows with missing critical data
            output_df = output_df.dropna(subset=["email"])
            
            # Select only the columns we want to include in the report
            columns_to_include = [
                "display_name", "email", "avg_messages", "period_start", "period_end", 
                "active_periods", "eligible_periods", "active_period_pct", "engagement_level"
            ]
            # Only include columns that exist in the DataFrame
            columns_to_include = [col for col in columns_to_include if col in output_df.columns]
            
            output_df = output_df[columns_to_include]
            
            # Final check for blank rows
            print(f"\nFinal output check:")
            print(f"Total rows: {len(output_df)}")
            print(f"Rows with blank display_name: {(output_df['display_name'] == '').sum()}")
            
            output_df.to_csv(output_file, index=False)
            print(f"Engagement report saved to {output_file}")
            
        return user_avg

    def get_non_engaged_users(self, only_latest_period: bool = False) -> pd.DataFrame:
        """Identify users who have never engaged across all tracked periods or just the latest period.
        
        Args:
            only_latest_period: If True, only consider the latest period for non-engagement
                                If False, identify users who have never engaged across all periods
        
        Returns:
            DataFrame with non-engaged user information
        """
        if self.user_data is None:
            raise ValueError("User data not loaded")
            
        if only_latest_period:
            # Get the latest period data
            latest_period = self.user_data["period_end"].max()
            latest_data = self.user_data[self.user_data["period_end"] == latest_period]
            
            # Find users who were not active in the latest period
            non_engaged_latest = latest_data[
                (latest_data["is_active"] == 0) | 
                (pd.isna(latest_data["is_active"]))
            ]
            
            # Make sure we have all necessary columns and remove duplicates
            # This ensures we only have one row per user with their latest status
            result = non_engaged_latest.drop_duplicates(subset=["account_id", "public_id", "email"]).copy()
            
            return result
        else:
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
            
            # Get additional user information from the latest period for each user
            # First, sort by period_end to get the latest data for each user
            sorted_data = self.user_data.sort_values("period_end", ascending=False)
            
            # Then, get the first occurrence of each user (which will be their latest status)
            user_info = sorted_data[["account_id", "public_id", "email", "user_role", "role", "department", "user_status", "created_or_invited_date"]].drop_duplicates(
                subset=["account_id", "public_id", "email"]
            )
            
            # Merge with never_active users
            never_active = pd.merge(
                never_active,
                user_info,
                on=["account_id", "public_id", "email"],
                how="left"
            )
            
            return never_active

    def generate_non_engagement_report(self, output_file: str = None, only_latest_period: bool = False) -> pd.DataFrame:
        """Generate a report of users who have never engaged.
        
        Args:
            output_file: Path to save the report CSV file (optional)
            only_latest_period: If True, only consider the latest period for non-engagement
                                If False, identify users who have never engaged across all periods
            
        Returns:
            DataFrame with non-engaged user report
        """
        # Get non-engaged users
        non_engaged = self.get_non_engaged_users(only_latest_period=only_latest_period)
        
        period_text = "in the latest period" if only_latest_period else "across all tracked periods"
        print(f"Non-Engaged Users Summary {period_text}:")
        print(f"  Total non-engaged users: {len(non_engaged)}")
        
        # Count by user status if the column exists
        if 'user_status' in non_engaged.columns:
            status_counts = non_engaged["user_status"].value_counts().to_dict()
            for status, count in status_counts.items():
                print(f"  {status}: {count}")
        
        # Sort by user_status and created_or_invited_date
        if 'user_status' in non_engaged.columns and 'created_or_invited_date' in non_engaged.columns:
            # Convert created_or_invited_date to datetime for proper sorting
            non_engaged['created_or_invited_date'] = pd.to_datetime(
                non_engaged['created_or_invited_date'], errors='coerce'
            )
            
            # Define custom sort order for user_status
            status_order = {
                'enabled': 0,
                'pending': 1,
                'deleted': 2
            }
            
            # Create a helper column for sorting by status
            non_engaged['status_order'] = non_engaged['user_status'].map(
                lambda x: status_order.get(x, 999)  # Default high value for unknown statuses
            )
            
            # Sort by status_order and then by created_or_invited_date (oldest first)
            non_engaged = non_engaged.sort_values(
                by=['status_order', 'created_or_invited_date'],
                ascending=[True, True]
            )
            
            # Drop the helper column
            non_engaged = non_engaged.drop('status_order', axis=1)
        
        # Save to file if specified
        if output_file:
            # Create a copy for output formatting
            output_df = non_engaged.copy()
            
            # Create a display name column (use name if available, otherwise email)
            output_df["display_name"] = output_df["name"].fillna(output_df["email"])
            
            # Select only the columns we want to include in the report
            columns_to_include = [
                "display_name", "user_role", "role", "department", 
                "user_status", "created_or_invited_date"
            ]
            # Only include columns that exist in the DataFrame
            columns_to_include = [col for col in columns_to_include if col in output_df.columns]
            
            output_df = output_df[columns_to_include]
            output_df.to_csv(output_file, index=False)
            print(f"Non-engagement report saved to {output_file}")
            
        return non_engaged
