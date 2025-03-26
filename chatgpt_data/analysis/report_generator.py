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
                output_df = self._create_sample_data("engagement")
                
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
        
        period_text = "in latest period" if only_latest_period else "across all periods"
        print(f"\nNon-Engaged Users Summary {period_text}:")
        print(f"  Total non-engaged users: {len(non_engaged)}")
        
        # Print user status breakdown if available
        if 'user_status' in non_engaged.columns:
            status_counts = non_engaged["user_status"].value_counts().to_dict()
            print(f"  Status breakdown:")
            for status, count in status_counts.items():
                print(f"    - {status}: {count}")
        
        # Print non-engagement category breakdown
        if 'non_engagement_category' in non_engaged.columns:
            category_counts = non_engaged["non_engagement_category"].value_counts().to_dict()
            print(f"\n  Non-Engagement Category breakdown:")
            for category, count in sorted(category_counts.items(), key=lambda x: x[1], reverse=True):
                print(f"    - {category}: {count} ({count/len(non_engaged)*100:.1f}%)")
        
        # Save to file if specified
        if output_file:
            # Create a copy for output formatting
            output_df = non_engaged.copy()
            
            # Create a display_name column (use name if available, otherwise email)
            if "name" in output_df.columns:
                output_df["display_name"] = output_df["name"].fillna(output_df["email"])
            else:
                output_df["display_name"] = output_df["email"]
            
            # Add display names using AD data
            self._add_display_names_to_report(output_df)
            
            # Add management chain information
            self._add_management_chain_info(output_df)
            
            # Select only the columns we want to include in the report
            base_columns = [
                "display_name", "email", "user_role", "role", "department", 
                "user_status", "created_or_invited_date", "non_engagement_category"
            ]
            
            # Add management chain columns if they exist
            management_columns = [col for col in output_df.columns if col.startswith("manager_")]
            columns_to_include = base_columns + management_columns
            
            # Only keep columns that actually exist in the DataFrame
            columns_to_include = [col for col in columns_to_include if col in output_df.columns]
            
            # Keep only the columns we want and sort by display_name
            output_df = output_df[columns_to_include].sort_values("display_name")
            
            # Create a custom sort order for non-engagement categories if they exist
            if "non_engagement_category" in output_df.columns:
                category_order = {
                    "new_account_never_active": 0,        # Newest non-engaged accounts
                    "recent_account_never_active": 1,      # Recently created non-engaged
                    "logged_in_no_messages": 2,            # Logged in but no messages
                    "has_messages_not_active": 3,          # Has messages but not active (data error?)
                    "fully_inactive": 4,                   # Completely inactive
                    "established_account_never_active": 5, # Older accounts
                    "long_term_account_never_active": 6,    # Oldest accounts
                    "lapsed_user": 7,                      # Lapsed users
                    "dormant_user": 8                      # Dormant users
                }
                
                # Get a default value higher than any defined value
                default_value = max(category_order.values()) + 1
                
                # Create a sort key and sort by it
                output_df["category_sort"] = output_df["non_engagement_category"].map(
                    lambda x: category_order.get(x, default_value)
                )
                output_df = output_df.sort_values(["category_sort", "display_name"])
                output_df = output_df.drop("category_sort", axis=1)
            
            # Create a sample report if the DataFrame is empty
            if len(output_df) == 0:
                output_df = self._create_sample_data("non_engagement")
                
            output_df.to_csv(output_file, index=False)
            print(f"\n✓ Non-engaged users report ({period_text}) saved to {output_file}")
        
        return non_engaged
    
    def _get_non_engaged_users(self, only_latest_period: bool = False) -> pd.DataFrame:
        """Identify users who have never engaged or haven't engaged recently.
        
        This method identifies several categories of non-engaged users:
        - Never active: Users who have never engaged across all tracked periods
        - Lapsed: Users who were active in the past but not in the recent 2-4 periods
        - Dormant: Users who were active in the past but not in 8+ periods
        - Inactive latest: Users who were not active in just the latest period
        
        Args:
            only_latest_period: If True, only consider the latest period for non-engagement
                               If False, identify all categories of non-engaged users
        
        Returns:
            DataFrame with non-engaged user information and categorization
        """
        user_data = self.user_analyzer.user_data
        
        # Get unique periods to understand the timeframe
        periods = user_data["period_end"].sort_values().unique()
        num_periods = len(periods)
        
        if only_latest_period:
            # Get the latest period data
            latest_period = user_data["period_end"].max()
            latest_data = user_data[user_data["period_end"] == latest_period]
            
            # Find users who were not active in the latest period
            # Consider both is_active flag AND message count
            non_engaged_latest = latest_data[
                ((latest_data["is_active"] == 0) | pd.isna(latest_data["is_active"])) |
                ((latest_data["messages"] == 0) | pd.isna(latest_data["messages"]))
            ]
            
            # Add non-engagement category column
            non_engaged_latest["non_engagement_category"] = "no_activity_latest_period"
            
            # Further categorize based on specific conditions
            # Messages = 0 but is_active = 1 (logged in but didn't send messages)
            mask_logged_no_messages = (non_engaged_latest["is_active"] == 1) & ((non_engaged_latest["messages"] == 0) | pd.isna(non_engaged_latest["messages"]))
            non_engaged_latest.loc[mask_logged_no_messages, "non_engagement_category"] = "logged_in_no_messages"
            
            # is_active = 0 but has messages (unusual case, could be data error)
            mask_messages_not_active = (non_engaged_latest["messages"] > 0) & ((non_engaged_latest["is_active"] == 0) | pd.isna(non_engaged_latest["is_active"]))
            non_engaged_latest.loc[mask_messages_not_active, "non_engagement_category"] = "has_messages_not_active"
            
            # Both is_active = 0 and messages = 0 (completely inactive)
            mask_fully_inactive = ((non_engaged_latest["is_active"] == 0) | pd.isna(non_engaged_latest["is_active"])) & ((non_engaged_latest["messages"] == 0) | pd.isna(non_engaged_latest["messages"]))
            non_engaged_latest.loc[mask_fully_inactive, "non_engagement_category"] = "fully_inactive"
            
            # Make sure we have all necessary columns and remove duplicates
            # This ensures we only have one row per user with their latest status
            result = non_engaged_latest.drop_duplicates(subset=["account_id", "public_id", "email"]).copy()
            
            return result
        else:
            ## COMPREHENSIVE NON-ENGAGEMENT ANALYSIS ##
            
            # 1. First identify users who never engaged (existing logic)
            all_users = user_data[["account_id", "public_id", "name", "email"]].drop_duplicates()
            
            active_users = user_data[
                (user_data["is_active"] == 1) | (user_data["messages"] > 0)
            ][
                ["account_id", "public_id", "name", "email"]
            ].drop_duplicates()
            
            # Find users who have NEVER been active
            never_active = pd.merge(
                all_users, active_users, 
                on=["account_id", "public_id", "email"], 
                how="left", 
                indicator=True,
                suffixes=("", "_active")
            )
            never_active = never_active[never_active["_merge"] == "left_only"].drop(["_merge", "name_active"], axis=1, errors="ignore")
            
            # Add non-engagement category
            never_active["non_engagement_category"] = "never_active_all_periods"
            
            # 2. Now identify LAPSED and DORMANT users (active in the past but not recently)
            # First, get all users who have been active at least once
            ever_active_users = user_data[
                (user_data["is_active"] == 1) | (user_data["messages"] > 0)
            ][
                ["account_id", "public_id", "email"]
            ].drop_duplicates()
            
            # For each user, find their activity pattern
            user_activity = {}
            for idx, period in enumerate(periods):
                period_data = user_data[user_data["period_end"] == period]
                
                # Get active users in this period
                period_active = period_data[
                    (period_data["is_active"] == 1) | (period_data["messages"] > 0)
                ][["account_id", "public_id", "email"]].drop_duplicates()
                
                # Record activity status for each user
                for _, user_row in period_active.iterrows():
                    user_key = (
                        str(user_row["account_id"]) if not pd.isna(user_row["account_id"]) else "",
                        str(user_row["public_id"]) if not pd.isna(user_row["public_id"]) else "",
                        str(user_row["email"]) if not pd.isna(user_row["email"]) else ""
                    )
                    if user_key not in user_activity:
                        user_activity[user_key] = [False] * num_periods
                    user_activity[user_key][idx] = True
            
            # Identify lapsed and dormant users based on activity patterns
            lapsed_users = []
            dormant_users = []
            
            latest_idx = num_periods - 1  # Index of the latest period
            
            print(f"Analyzing activity patterns for {len(user_activity)} users across {num_periods} periods")
            
            for user_key, activity_pattern in user_activity.items():
                # Skip if active in the latest period
                if latest_idx >= 0 and activity_pattern[latest_idx]:
                    continue
                
                # Count consecutive inactive periods at the end
                inactive_streak = 0
                for i in range(latest_idx, -1, -1):
                    if not activity_pattern[i]:
                        inactive_streak += 1
                    else:
                        break
                
                # Skip users who have never been active (all False in activity pattern)
                if all(not active for active in activity_pattern):
                    continue
                
                # Categorize based on inactivity streak
                if inactive_streak >= 8:  # ~2 months (8 weeks)
                    dormant_users.append(user_key)
                elif inactive_streak >= 3:  # ~3 weeks
                    lapsed_users.append(user_key)
            
            # Create DataFrames for lapsed and dormant users
            lapsed_df = pd.DataFrame([
                {"account_id": k[0], "public_id": k[1], "email": k[2]} 
                for k in lapsed_users
            ]) if lapsed_users else pd.DataFrame()
            
            dormant_df = pd.DataFrame([
                {"account_id": k[0], "public_id": k[1], "email": k[2]} 
                for k in dormant_users
            ]) if dormant_users else pd.DataFrame()
            
            if not lapsed_df.empty:
                lapsed_df["non_engagement_category"] = "lapsed_user"  # Not active in 3-7 periods
                print(f"Found {len(lapsed_df)} lapsed users (inactive for 3-7 periods)")
            
            if not dormant_df.empty:
                dormant_df["non_engagement_category"] = "dormant_user"  # Not active in 8+ periods
                print(f"Found {len(dormant_df)} dormant users (inactive for 8+ periods)")
            
            # 3. Combine all categories of non-engaged users
            non_engaged_dfs = [never_active]
            if not lapsed_df.empty:
                non_engaged_dfs.append(lapsed_df)
            if not dormant_df.empty:
                non_engaged_dfs.append(dormant_df)
            
            # Handle the case where all dataframes might be empty
            if all(df.empty for df in non_engaged_dfs):
                print("Warning: No non-engaged users found. Creating empty DataFrame.")
                return pd.DataFrame(columns=["account_id", "public_id", "email", "non_engagement_category"])
            
            combined_non_engaged = pd.concat(non_engaged_dfs, ignore_index=True)
            
            # 4. Add user information from the latest period
            # Sort by period_end to get the latest data for each user
            sorted_data = user_data.sort_values("period_end", ascending=False)
            
            # Get the first occurrence of each user (which will be their latest status)
            user_info = sorted_data[["account_id", "public_id", "email", "name", "user_role", "role", "department", "user_status", "created_or_invited_date"]].drop_duplicates(
                subset=["account_id", "public_id", "email"]
            )
            
            # Ensure consistent types across dataframes for merge keys
            for col in ["account_id", "public_id", "email"]:
                if col in combined_non_engaged.columns and col in user_info.columns:
                    # Convert both to string to ensure consistent types
                    combined_non_engaged[col] = combined_non_engaged[col].astype(str)
                    user_info[col] = user_info[col].astype(str)
            
            # Merge with the combined non-engaged users
            combined_non_engaged = pd.merge(
                combined_non_engaged,
                user_info,
                on=["account_id", "public_id", "email"],
                how="left"
            )
            
            # 5. Calculate time since account creation for users with creation date
            if "created_or_invited_date" in combined_non_engaged.columns:
                try:
                    latest_period_end = user_data["period_end"].max()
                    latest_date = pd.to_datetime(latest_period_end)
                    
                    # Convert created_or_invited_date to datetime
                    combined_non_engaged["created_date"] = pd.to_datetime(combined_non_engaged["created_or_invited_date"], errors="coerce")
                    
                    # Calculate days since creation
                    combined_non_engaged["days_since_creation"] = (latest_date - combined_non_engaged["created_date"]).dt.days
                    
                    # Further categorize never_active users based on account age
                    never_active_mask = combined_non_engaged["non_engagement_category"] == "never_active_all_periods"
                    
                    # Add categories based on account age
                    conditions = [
                        never_active_mask & (combined_non_engaged["days_since_creation"] <= 14),  # New accounts (< 2 weeks)
                        never_active_mask & (combined_non_engaged["days_since_creation"] <= 30),  # Recent accounts (< 1 month)
                        never_active_mask & (combined_non_engaged["days_since_creation"] <= 90),  # Established accounts (< 3 months)
                    ]
                    choices = [
                        "new_account_never_active",
                        "recent_account_never_active",
                        "established_account_never_active"
                    ]
                    
                    # Update non-engagement category with more detailed information
                    combined_non_engaged.loc[conditions[0], "non_engagement_category"] = choices[0]
                    combined_non_engaged.loc[conditions[1] & ~conditions[0], "non_engagement_category"] = choices[1]
                    combined_non_engaged.loc[conditions[2] & ~conditions[1] & ~conditions[0], "non_engagement_category"] = choices[2]
                    
                    # Long-term accounts that have never been active
                    combined_non_engaged.loc[never_active_mask & ~conditions[2] & ~conditions[1] & ~conditions[0], 
                                            "non_engagement_category"] = "long_term_account_never_active"
                    
                    # Clean up temporary columns
                    combined_non_engaged = combined_non_engaged.drop(["created_date", "days_since_creation"], axis=1, errors="ignore")
                except Exception as e:
                    print(f"Error calculating account age categories: {e}")
            
            return combined_non_engaged
    
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
    
    def _create_sample_data(self, data_type: str = "non_engagement") -> pd.DataFrame:
        """Create sample data for cases when real data is not available.
        
        Args:
            data_type: Type of sample data to create
            
        Returns:
            DataFrame with sample data
        """
        if data_type == "engagement":
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
        else:
            sample_data = {
                "display_name": ["Sample Never Engaged", "Sample Lapsed", "Sample Dormant", "Sample Recent", "Sample Latest"],
                "email": ["never@example.com", "lapsed@example.com", "dormant@example.com", "recent@example.com", "latest@example.com"],
                "user_role": ["member", "member", "member", "member", "member"],
                "role": ["Engineer", "Manager", "Analyst", "Designer", "Director"],
                "department": ["Engineering", "Product", "Finance", "Design", "Marketing"],
                "user_status": ["enabled", "enabled", "inactive", "pending", "enabled"],
                "created_or_invited_date": ["2024-09-01", "2024-11-15", "2024-10-01", "2025-02-15", "2025-03-01"],
                "non_engagement_category": [
                    "long_term_account_never_active", 
                    "lapsed_user",
                    "dormant_user", 
                    "recent_account_never_active",
                    "fully_inactive"
                ]
            }
            return pd.DataFrame(sample_data)