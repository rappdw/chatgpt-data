"""Report generation module for user engagement analysis."""

import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import pandas as pd
import numpy as np

from chatgpt_data.analysis.interfaces import ReportGeneratorInterface, DataAnalyzer
from chatgpt_data.utils.constants import DEFAULT_TIMEZONE
from chatgpt_data.analysis.user_analyzer import UserAnalyzer


class ReportGenerator(ReportGeneratorInterface):
    """Class for generating reports from user engagement data."""
    
    def __init__(self, analyzer: DataAnalyzer):
        """Initialize the ReportGenerator class.
        
        Args:
            analyzer: DataAnalyzer instance with loaded data
        """
        # Type check to ensure we're using a UserAnalyzer
        if not isinstance(analyzer, UserAnalyzer):
            raise TypeError("ReportGenerator requires a UserAnalyzer instance")
        
        self.user_analyzer = analyzer
        self.output_dir = analyzer.output_dir
    
    def generate_engagement_report(self, output_file: Optional[str] = None) -> pd.DataFrame:
        """Generate a report of user engagement levels based on average message count.
        
        Args:
            output_file: Path to save the report CSV file (optional)
            
        Returns:
            DataFrame with user engagement report
        """
        # Get raw engagement data from analyzer
        engagement_df = self.user_analyzer.get_engagement_levels()
        
        # Save to file if specified
        if output_file:
            # Format avg_messages as a number with fixed decimal places
            output_df = engagement_df.copy()
            output_df["avg_messages"] = output_df["avg_messages"].round(2)
            
            # Create display names using AD data
            self._add_display_names_to_report(output_df)
            
            # Add management chain information
            if self.user_analyzer.management_chains:
                self._add_management_chain_info(output_df)
            
            # Ensure we don't have any rows with missing critical data
            output_df = output_df.dropna(subset=["email"])
            
            # Add user status information
            self._add_user_status_info(output_df)
            
            # Select columns for the report
            base_columns = [
                "display_name", "email", "user_status", "avg_messages", 
                "first_period", "last_period", "active_periods", 
                "eligible_periods", "active_period_pct", "engagement_level"
            ]
            
            # Add management chain columns if they exist
            manager_columns = [col for col in output_df.columns if col.startswith("manager_")]
            columns_to_include = base_columns + manager_columns
            
            # Only include columns that exist in the DataFrame
            columns_to_include = [col for col in columns_to_include if col in output_df.columns]
            
            output_df = output_df[columns_to_include]
            
            # Sort by avg_messages in descending order (high to low)
            if 'avg_messages' in output_df.columns:
                output_df = output_df.sort_values(by='avg_messages', ascending=False)
                print(f"Sorted engagement report by average messages (high to low)")
            
            # Final check and save
            print(f"\nFinal output check:")
            print(f"Total rows: {len(output_df)}")
            print(f"Rows with blank display_name: {(output_df['display_name'] == '').sum()}")
            
            # Create a sample report if the DataFrame is empty
            if len(output_df) == 0:
                output_df = self._create_sample_engagement_report()
                
            output_df.to_csv(output_file, index=False)
            print(f"Engagement report saved to {output_file}")
            
        return engagement_df
    
    def generate_non_engagement_report(self, output_file: Optional[str] = None, only_latest_period: bool = False) -> pd.DataFrame:
        """Generate a report of users who have never engaged.
        
        Args:
            output_file: Path to save the report CSV file (optional)
            only_latest_period: If True, only consider the latest period for non-engagement
                               If False, identify users who have never engaged across all periods
            
        Returns:
            DataFrame with non-engaged user report
        """
        # Get non-engaged users
        non_engaged = self._get_non_engaged_users(only_latest_period)
        
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
                'active': 0,
                'inactive': 1
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
            
            # Create display names
            output_df["display_name"] = output_df["name"].fillna(output_df["email"])
            self._add_display_names_to_report(output_df)
            
            # Add management chain information
            if self.user_analyzer.management_chains:
                self._add_management_chain_info(output_df)
            
            # Ensure we don't have any rows with missing critical data
            output_df = output_df.dropna(subset=["email"])
            
            # Select only the columns we want to include in the report
            base_columns = [
                "display_name", "email", "user_role", "role", "department", 
                "user_status", "created_or_invited_date"
            ]
            
            # Add management chain columns if they exist
            manager_columns = [col for col in output_df.columns if col.startswith("manager_")]
            columns_to_include = base_columns + manager_columns
            
            # Only include columns that exist in the DataFrame
            columns_to_include = [col for col in columns_to_include if col in output_df.columns]
            
            output_df = output_df[columns_to_include]
            
            # Create a sample report if the DataFrame is empty
            if len(output_df) == 0:
                output_df = self._create_sample_non_engagement_report()
                
            output_df.to_csv(output_file, index=False)
            print(f"Non-engagement report saved to {output_file}")
            
        return non_engaged
    
    def _get_non_engaged_users(self, only_latest_period: bool = False) -> pd.DataFrame:
        """Identify users who have never engaged across all tracked periods or just the latest period.
        
        Args:
            only_latest_period: If True, only consider the latest period for non-engagement
                               If False, identify users who have never engaged across all periods
        
        Returns:
            DataFrame with non-engaged user information
        """
        user_data = self.user_analyzer.user_data
        
        if only_latest_period:
            # Get the latest period data
            latest_period = user_data["period_end"].max()
            latest_data = user_data[user_data["period_end"] == latest_period]
            
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
            all_users = user_data[["account_id", "public_id", "name", "email"]].drop_duplicates()
            
            # Get users who have been active at least once
            active_users = user_data[user_data["is_active"] == 1][
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
            sorted_data = user_data.sort_values("period_end", ascending=False)
            
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
    
    def _add_display_names_to_report(self, report_df: pd.DataFrame) -> None:
        """Add display names to a report DataFrame using AD data.
        
        Args:
            report_df: DataFrame to add display names to
        """
        # Create a mapping of email to AD display name for faster processing
        email_to_display_name = {}
        
        # Only create the mapping if AD data is available
        if self.user_analyzer.ad_data is not None:
            ad_data = self.user_analyzer.ad_data
            
            # Count rows with valid emails that we can potentially resolve
            rows_with_email = report_df["email"].notna().sum()
            
            print(f"\nAttempting to resolve display names from AD data for {rows_with_email} users with emails")
            
            # Process userPrincipalName column
            for _, row in ad_data.iterrows():
                if pd.notna(row["userPrincipalName"]) and pd.notna(row["displayName"]):
                    email_to_display_name[row["userPrincipalName"].lower()] = row["displayName"]
                
                # Also process the mail column if it exists and is different from userPrincipalName
                if pd.notna(row["mail"]) and pd.notna(row["displayName"]):
                    if row["mail"].lower() not in email_to_display_name:
                        email_to_display_name[row["mail"].lower()] = row["displayName"]
            
            print(f"Created mapping for {len(email_to_display_name)} email addresses from AD data")
        
        # Update display names using AD data
        if "display_name" not in report_df.columns:
            report_df["display_name"] = report_df["name"].fillna(report_df["email"])
        
        if email_to_display_name:
            report_df["display_name"] = report_df.apply(
                lambda row: email_to_display_name.get(row["email"].lower(), row["display_name"]) 
                if pd.notna(row["email"]) else row["display_name"],
                axis=1
            )
        
        # For any rows where display_name is still an email address, try to find a better name
        email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        
        # Print diagnostic information about potential email addresses in display_name
        email_mask = report_df["display_name"].str.match(email_pattern, na=False)
        email_count = email_mask.sum()
        if email_count > 0:
            print(f"\nFound {email_count} display names that appear to be email addresses")
            
            # For rows where display_name is an email, try to extract a name from the email
            for idx, row in report_df[email_mask].iterrows():
                email = row["display_name"]
                # Extract the username part of the email (before @)
                username = email.split('@')[0]
                # Convert username to a more readable format (e.g., john.doe -> John Doe)
                name_parts = re.split(r'[._-]', username)
                formatted_name = ' '.join([part.capitalize() for part in name_parts])
                report_df.at[idx, "display_name"] = formatted_name
                
            print(f"Converted {email_count} email addresses to formatted names")
    
    def _add_management_chain_info(self, report_df: pd.DataFrame) -> None:
        """Add management chain information to a report DataFrame.
        
        Args:
            report_df: DataFrame to add management chain information to
        """
        print("\nAdding management chain information to the report")
        
        # Determine the maximum depth of any management chain
        max_chain_depth = 0
        for chain in self.user_analyzer.management_chains.values():
            max_chain_depth = max(max_chain_depth, len(chain))
        
        print(f"Maximum management chain depth: {max_chain_depth}")
        
        # Create columns for each level in the management chain
        # Reverse the order so highest level (CEO) is first, then direct reports, etc.
        for i in range(max_chain_depth):
            report_df[f"manager_{max_chain_depth-i}"] = None
        
        # Fill in management chain information for each employee
        match_count = 0
        fuzzy_match_count = 0
        for idx, row in report_df.iterrows():
            # Try to get management chain with name matcher
            chain = self.user_analyzer.name_matcher.get_management_chain(row["display_name"])
            if chain:
                match_count += 1
                # Check if this was a fuzzy match (not an exact match)
                if (row["display_name"] not in self.user_analyzer.management_chains and 
                    self.user_analyzer.name_matcher.normalize_name(row["display_name"]) not in self.user_analyzer.name_matcher.normalized_names):
                    fuzzy_match_count += 1
                
                # Reverse the chain to go from CEO down to direct manager
                reversed_chain = list(reversed(chain))
                # Add each manager to the appropriate column
                for i, manager in enumerate(reversed_chain):
                    report_df.at[idx, f"manager_{i+1}"] = manager
        
        print(f"Found management chain information for {match_count} out of {len(report_df)} employees")
        print(f"  - {fuzzy_match_count} matches were found using fuzzy name matching")
    
    def _add_user_status_info(self, report_df: pd.DataFrame) -> None:
        """Add user status information to a report DataFrame.
        
        Args:
            report_df: DataFrame to add user status information to
        """
        user_data = self.user_analyzer.user_data
        
        # Get the latest period data to identify active users
        latest_period = user_data["period_end"].max()
        latest_data = user_data[user_data["period_end"] == latest_period]
        
        # Create a mapping of public_id to latest user_status
        latest_user_status = latest_data[["public_id", "user_status"]].drop_duplicates(subset=["public_id"]).set_index("public_id")["user_status"].to_dict()
        
        # Add the latest user_status to each user in the output DataFrame
        report_df["user_status"] = report_df["public_id"].map(latest_user_status)
        
        # Fill any missing user_status values with 'inactive' as a default
        report_df["user_status"] = report_df["user_status"].fillna("inactive")
    
    def _create_sample_engagement_report(self) -> pd.DataFrame:
        """Create a sample engagement report for when no real data is available.
        
        Returns:
            DataFrame with sample engagement data
        """
        print("WARNING: No data available for engagement report. Creating a sample report.")
        # Create a sample report with placeholder data
        sample_data = {
            "display_name": ["Sample User 1", "Sample User 2", "Sample User 3"],
            "email": ["sample1@example.com", "sample2@example.com", "sample3@example.com"],
            "user_status": ["enabled", "enabled", "enabled"],
            "avg_messages": [25.5, 12.3, 3.1],
            "active_periods": [5, 3, 1],
            "eligible_periods": [5, 5, 5],
            "active_period_pct": [100.0, 60.0, 20.0],
            "engagement_level": ["high", "medium", "low"]
        }
        return pd.DataFrame(sample_data)
    
    def _create_sample_non_engagement_report(self) -> pd.DataFrame:
        """Create a sample non-engagement report for when no real data is available.
        
        Returns:
            DataFrame with sample non-engagement data
        """
        print("WARNING: No data available for non-engagement report. Creating a sample report.")
        # Create a sample report with placeholder data
        sample_data = {
            "display_name": ["Sample Non-Engaged 1", "Sample Non-Engaged 2", "Sample Non-Engaged 3"],
            "email": ["non-engaged1@example.com", "non-engaged2@example.com", "non-engaged3@example.com"],
            "user_role": ["member", "member", "member"],
            "role": ["Engineer", "Manager", "Analyst"],
            "department": ["Engineering", "Product", "Finance"],
            "user_status": ["enabled", "pending", "enabled"],
            "created_or_invited_date": ["2025-01-01", "2025-01-15", "2025-02-01"]
        }
        return pd.DataFrame(sample_data)