"""User engagement analysis module."""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import re
import fnmatch

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json
from rapidfuzz import fuzz, process

# Import the default timezone for consistent datetime handling
from chatgpt_data.utils.constants import DEFAULT_TIMEZONE


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
        self.management_chains = None
        self.normalized_names = None
        self._load_data()
        self._load_ad_data()
        self._load_management_chains()

    def _load_data(self) -> None:
        """Load all user engagement data files."""
        user_files = [f for f in os.listdir(self.data_dir) if fnmatch.fnmatch(f, '*user_engagement*.csv')]

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
            
    def _load_management_chains(self) -> None:
        """Load management chain data from JSON file."""
        management_chains_file = self.data_dir / "management_chains.json"
        
        if not management_chains_file.exists():
            print("Management chains file not found. Management chain information will not be available.")
            return
            
        try:
            with open(management_chains_file, 'r') as f:
                self.management_chains = json.load(f)
            print(f"Successfully loaded management chain data for {len(self.management_chains)} employees")
            
            # Create a normalized name mapping for fuzzy matching
            self.normalized_names = {}
            for name in self.management_chains.keys():
                # Normalize the name (lowercase, remove extra spaces)
                normalized = self._normalize_name(name)
                self.normalized_names[normalized] = name
                
                # Also add common name variations (e.g., Dan -> Daniel)
                variations = self._get_name_variations(name)
                for variation in variations:
                    normalized_variation = self._normalize_name(variation)
                    if normalized_variation != normalized:
                        self.normalized_names[normalized_variation] = name
                
                # Add first name + last name version for names with 3+ parts (handling maiden names)
                name_components = self._extract_name_components(name)
                if len(normalized.split()) >= 3 and "first_name" in name_components and "last_name" in name_components:
                    simplified_name = f"{name_components['first_name']} {name_components['last_name']}"
                    if simplified_name != normalized:
                        self.normalized_names[simplified_name] = name
                
                # For names with exactly 2 parts, add potential maiden name variations
                if len(normalized.split()) == 2:
                    maiden_variations = self._expand_with_common_maiden_names(name)
                    for variation in maiden_variations:
                        self.normalized_names[variation] = name
                        
            print(f"Created normalized name mapping with {len(self.normalized_names)} entries")
            
        except Exception as e:
            print(f"Error loading management chain data: {str(e)}")
            
    def _normalize_name(self, name: str) -> str:
        """Normalize a name for comparison.
        
        Args:
            name: Name to normalize
            
        Returns:
            Normalized name (lowercase, no extra spaces, no parenthetical text)
        """
        if not name or pd.isna(name):
            return ""
            
        # Remove parenthetical text (like foreign scripts or additional information)
        name_without_parentheses = re.sub(r'\s*\([^)]*\)', '', name)
        
        # Convert to lowercase and remove extra spaces
        normalized = re.sub(r'\s+', ' ', name_without_parentheses.lower().strip())
        return normalized
        
    def _extract_name_components(self, name: str) -> dict:
        """Extract components from a name for more flexible matching.
        
        Args:
            name: Full name to extract components from
            
        Returns:
            Dictionary with first_name, last_name, and full_name keys
        """
        if not name or pd.isna(name):
            return {"first_name": "", "last_name": "", "full_name": ""}
            
        # Normalize the name first
        normalized = self._normalize_name(name)
        if not normalized:
            return {"first_name": "", "last_name": "", "full_name": normalized}
            
        # Split the name into parts
        parts = normalized.split()
        
        result = {
            "full_name": normalized,
            "first_name": parts[0] if parts else "",
            "last_name": parts[-1] if parts else ""
        }
        
        # If there are at least 3 parts, store potential middle/maiden names
        if len(parts) >= 3:
            result["middle_names"] = parts[1:-1]
        
        return result
    
    def _expand_with_common_maiden_names(self, name: str) -> List[str]:
        """Generate potential variations by adding common maiden name positions.
        
        This helps match a first+last name against a first+maiden+last name.
        
        Args:
            name: Original name
            
        Returns:
            List of name variations with potential maiden name placements
        """
        variations = []
        
        # Skip if name is empty or not a string
        if not name or pd.isna(name):
            return variations
            
        # Split the name into parts
        parts = self._normalize_name(name).split()
        
        # We only handle the case of exactly 2 parts (first + last)
        if len(parts) != 2:
            return variations
            
        # Common maiden name positions (typically between first and last name)
        # We'll use some common placeholder maiden names
        common_maiden_placeholders = ["middlename", "maidenname"]
        
        for placeholder in common_maiden_placeholders:
            # Create a variation with the placeholder in the middle
            variation = f"{parts[0]} {placeholder} {parts[1]}"
            variations.append(variation)
            
        return variations
        
    def _get_name_variations(self, name: str) -> List[str]:
        """Generate common variations of a name.
        
        Args:
            name: Original name
            
        Returns:
            List of name variations
        """
        variations = []
        
        # Skip if name is empty or not a string
        if not name or pd.isna(name):
            return variations
            
        # Split the name into parts
        parts = name.split()
        if len(parts) < 1:
            return variations
            
        # Common first name variations
        first_name = parts[0]
        common_variations = {
            "nathan": ["nate"],
            "nate": ["nathan"],
            "daniel": ["dan", "danny"],
            "dan": ["daniel", "danny"],
            "danny": ["daniel", "dan"],
            "michael": ["mike", "mick"],
            "mike": ["michael", "mick"],
            "robert": ["rob", "bob", "bobby"],
            "rob": ["robert", "bob", "bobby"],
            "bob": ["robert", "rob", "bobby"],
            "william": ["will", "bill", "billy"],
            "will": ["william", "bill", "billy"],
            "bill": ["william", "will", "billy"],
            "richard": ["rick", "dick", "rich"],
            "rick": ["richard", "dick", "rich"],
            "james": ["jim", "jimmy"],
            "jim": ["james", "jimmy"],
            "thomas": ["tom", "tommy"],
            "tom": ["thomas", "tommy"],
            "john": ["johnny", "jon"],
            "jonathan": ["jon", "jonny"],
            "christopher": ["chris", "topher"],
            "chris": ["christopher", "topher"],
            "joseph": ["joe", "joey"],
            "joe": ["joseph", "joey"],
            "david": ["dave", "davey"],
            "dave": ["david", "davey"],
            "charles": ["chuck", "charlie"],
            "chuck": ["charles", "charlie"],
            "charlie": ["charles", "chuck"],
            "matthew": ["matt", "matty"],
            "matt": ["matthew", "matty"],
            "nicholas": ["nick", "nicky"],
            "nick": ["nicholas", "nicky"],
            "anthony": ["tony", "ant"],
            "tony": ["anthony", "ant"],
            "steven": ["steve", "stevie"],
            "steve": ["steven", "stevie"],
            "andrew": ["andy", "drew"],
            "andy": ["andrew", "drew"],
            "drew": ["andrew", "andy"],
            "jennifer": ["jen", "jenny"],
            "jen": ["jennifer", "jenny"],
            "jessica": ["jess", "jessie"],
            "jess": ["jessica", "jessie"],
            "elizabeth": ["liz", "beth", "eliza"],
            "liz": ["elizabeth", "beth", "eliza"],
            "beth": ["elizabeth", "liz", "eliza"],
            "katherine": ["kate", "katie", "kathy"],
            "kate": ["katherine", "katie", "kathy"],
            "katie": ["katherine", "kate", "kathy"],
            "kathy": ["katherine", "kate", "katie"],
            "margaret": ["maggie", "meg", "peggy"],
            "maggie": ["margaret", "meg", "peggy"],
            "patricia": ["pat", "patty", "tricia"],
            "pat": ["patricia", "patty", "tricia"],
            "stephanie": ["steph", "stephie"],
            "steph": ["stephanie", "stephie"],
            "constanza": ["connie", "constance"],
            "constance": ["connie", "constanza"],
        }
        
        first_lower = first_name.lower()
        if first_lower in common_variations:
            for variation in common_variations[first_lower]:
                if len(parts) > 1:
                    # Create full name with the variation
                    variations.append(f"{variation.capitalize()} {' '.join(parts[1:])}")
                else:
                    variations.append(variation.capitalize())
                    
        return variations
            
    def get_management_chain(self, name: str) -> List[str]:
        """Get the management chain for an employee.
        
        Args:
            name: Employee name
            
        Returns:
            List of managers in the chain (in reverse order, from employee to CEO)
        """
        if self.management_chains is None or name is None or pd.isna(name):
            return []
            
        # Try exact match first
        if name in self.management_chains:
            return self.management_chains[name]
            
        # Try normalized name match
        normalized_name = self._normalize_name(name)
        if normalized_name in self.normalized_names:
            actual_name = self.normalized_names[normalized_name]
            return self.management_chains[actual_name]
            
        # Extract name components for more flexible matching
        name_components = self._extract_name_components(name)
        
        # Try to match based on first name + last name if we have a complex name
        if len(normalized_name.split()) >= 3 and "first_name" in name_components and "last_name" in name_components:
            # Create a simplified name with just first and last name
            simplified_name = f"{name_components['first_name']} {name_components['last_name']}"
            
            # Check if this simplified name exists in our normalized names
            if simplified_name in self.normalized_names:
                actual_name = self.normalized_names[simplified_name]
                return self.management_chains[actual_name]
        
        # If we have a simple name (first + last), try matching against expanded maiden name variations
        if len(normalized_name.split()) == 2:
            # Generate potential maiden name variations
            maiden_variations = self._expand_with_common_maiden_names(name)
            for variation in maiden_variations:
                if variation in self.normalized_names:
                    actual_name = self.normalized_names[variation]
                    return self.management_chains[actual_name]
                
        # Try fuzzy matching if no exact or normalized match
        # First check if we have any close matches in the normalized names
        best_match = None
        best_score = 0
        
        # Only consider matches with score above this threshold
        threshold = 85
        
        for key_name in self.management_chains.keys():
            # Try exact component matching first (first name + last name)
            key_components = self._extract_name_components(key_name)
            
            # Check if first and last names match exactly
            if (name_components["first_name"] == key_components["first_name"] and 
                name_components["last_name"] == key_components["last_name"]):
                return self.management_chains[key_name]
            
            # If that fails, try fuzzy matching on the full name
            score = fuzz.ratio(normalized_name, self._normalize_name(key_name))
            if score > threshold and score > best_score:
                best_score = score
                best_match = key_name
                
        if best_match:
            return self.management_chains[best_match]
                
        # If no match found, return empty list
        return []

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

        # Define active users as those who have sent at least one message
        # This is more reliable than using the is_active flag which might not be consistent
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
        # First convert to UTC, then localize to our default timezone
        active_users["period_start"] = pd.to_datetime(active_users["period_start"], utc=True).dt.tz_convert(DEFAULT_TIMEZONE)
        
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
        message_volume["period_start"] = pd.to_datetime(message_volume["period_start"], utc=True).dt.tz_convert(DEFAULT_TIMEZONE)
        
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
        gpt_usage["period_start"] = pd.to_datetime(gpt_usage["period_start"], utc=True).dt.tz_convert(DEFAULT_TIMEZONE)
        
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
        
        # Use raw message data directly instead of going through get_engagement_levels
        # This gives us more control and avoids potential filtering issues
        print("\nGenerating message histogram directly from user_data")
        
        # Filter to only include rows with valid message data
        valid_data = self.user_data[pd.notna(self.user_data["messages"])].copy()
        
        # Ensure messages column is numeric
        if not pd.api.types.is_numeric_dtype(valid_data['messages']):
            print("Converting messages column to numeric")
            valid_data['messages'] = pd.to_numeric(valid_data['messages'], errors='coerce')
        
        # Filter to only include rows with positive message count
        valid_data = valid_data[valid_data["messages"] > 0]
        print(f"Found {len(valid_data)} rows with positive message counts")
        
        # Check if we have enough data to create a meaningful histogram
        if len(valid_data) < 2:
            print("Warning: Not enough data to create a meaningful histogram")
            # Create a simple figure with a warning message
            fig, ax = plt.subplots(figsize=(12, 7))
            ax.text(0.5, 0.5, "Not enough data to create a meaningful histogram.\nNo users with positive message counts found.", 
                    ha='center', va='center', fontsize=14)
            ax.set_title("Distribution of Messages per User")
            ax.axis('off')
            
            if save:
                output_path = self.output_dir / "message_histogram.png"
                plt.savefig(output_path)
                plt.close(fig)
                return None
            
            return fig
        
        # Filter out extreme values if max_value is specified
        if max_value is not None:
            plot_data = valid_data[valid_data["messages"] <= max_value]
        else:
            plot_data = valid_data
            
        print(f"Using {len(plot_data)} data points for histogram")
        
        # Create the histogram
        fig, ax = plt.subplots(figsize=(12, 7))
        
        # Plot histogram with density=False to show counts
        try:
            n, bins_array, patches = ax.hist(
                plot_data["messages"], 
                bins=min(bins, len(plot_data)), # Ensure bins don't exceed data points
                edgecolor='black', 
                alpha=0.7
            )
            
            # Add a vertical line for the mean and median
            mean_value = plot_data["messages"].mean()
            median_value = plot_data["messages"].median()
            ax.axvline(mean_value, color='red', linestyle='dashed', linewidth=1, label=f'Mean: {mean_value:.2f}')
            ax.axvline(median_value, color='green', linestyle='dashed', linewidth=1, label=f'Median: {median_value:.2f}')
            
            # Add labels and title
            ax.set_title("Distribution of Messages per User")
            ax.set_xlabel("Number of Messages")
            ax.set_ylabel("Number of Users")
            ax.grid(True, alpha=0.3)
            ax.legend()
        except Exception as e:
            print(f"Error creating histogram: {e}")
            # Create a simple error message
            ax.text(0.5, 0.5, f"Error creating histogram: {e}", 
                    ha='center', va='center', fontsize=12, wrap=True)
            ax.set_title("Distribution of Messages per User - Error")
            ax.axis('off')
        
        # Add comment box with explanation
        total_users = len(valid_data)
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
        
        # Use raw message data directly
        print("\nGenerating log-scale message histogram directly from user_data")
        
        # Filter to only include rows with valid message data
        valid_data = self.user_data[pd.notna(self.user_data["messages"])].copy()
        
        # Ensure messages column is numeric
        if not pd.api.types.is_numeric_dtype(valid_data['messages']):
            print("Converting messages column to numeric")
            valid_data['messages'] = pd.to_numeric(valid_data['messages'], errors='coerce')
        
        # Filter to only include rows with positive message count
        valid_data = valid_data[valid_data["messages"] > 0]
        print(f"Found {len(valid_data)} rows with positive message counts")
        
        # Check if we have enough data to create a meaningful histogram
        if len(valid_data) < 2:
            print("Warning: Not enough data to create a meaningful histogram")
            # Create a simple figure with a warning message
            fig, ax = plt.subplots(figsize=(12, 7))
            ax.text(0.5, 0.5, "Not enough data to create a meaningful histogram.\nNo users with positive message counts found.", 
                    ha='center', va='center', fontsize=14)
            ax.set_title("Distribution of Messages per User (Log Scale)")
            ax.axis('off')
            
            if save:
                output_path = self.output_dir / "message_histogram_log.png"
                plt.savefig(output_path)
                plt.close(fig)
                return None
            
            return fig
        
        # Filter out extreme values if max_value is specified
        if max_value is not None:
            plot_data = valid_data[valid_data["messages"] <= max_value]
        else:
            plot_data = valid_data
            
        print(f"Using {len(plot_data)} data points for log-scale histogram")
        
        # Create the histogram
        fig, ax = plt.subplots(figsize=(12, 7))
        
        try:
            # Plot histogram with density=False to show counts
            n, bins_array, patches = ax.hist(
                plot_data["messages"], 
                bins=min(bins, len(plot_data)), # Ensure bins don't exceed data points
                edgecolor='black', 
                alpha=0.7
            )
            
            # Set y-axis to log scale
            ax.set_yscale('log')
            
            # Add a vertical line for the mean and median
            mean_value = plot_data["messages"].mean()
            median_value = plot_data["messages"].median()
            ax.axvline(mean_value, color='red', linestyle='dashed', linewidth=1, label=f'Mean: {mean_value:.2f}')
            ax.axvline(median_value, color='green', linestyle='dashed', linewidth=1, label=f'Median: {median_value:.2f}')
            
            # Add labels and title
            ax.set_title("Distribution of Messages per User (Log Scale)")
            ax.set_xlabel("Number of Messages")
            ax.set_ylabel("Number of Users (Log Scale)")
            ax.grid(True, alpha=0.3)
            ax.legend()
            
            # Add comment box with explanation
            total_users = len(valid_data)
            filtered_users = len(plot_data)
            filtered_text = ""
            if max_value is not None and filtered_users < total_users:
                filtered_text = f"\nNote: {total_users - filtered_users} users with >{max_value} messages are not shown."
                
            comment = f"""This histogram shows the distribution of messages sent by users with a logarithmic y-axis scale.
    The x-axis represents the number of messages per user.
    The y-axis (log scale) shows how many users fall into each message count range.
    Log scale helps visualize the distribution when there are large differences in frequency counts.{filtered_text}"""
            props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
            ax.text(0.05, 0.95, comment, transform=ax.transAxes, fontsize=9,
                    verticalalignment='top', bbox=props)
            
        except Exception as e:
            print(f"Error creating log-scale histogram: {e}")
            # Create a simple error message
            ax.text(0.5, 0.5, f"Error creating log-scale histogram: {e}", 
                    ha='center', va='center', fontsize=12, wrap=True)
            ax.set_title("Distribution of Messages per User (Log Scale) - Error")
            ax.axis('off')
        
        plt.tight_layout()
        
        if save:
            output_path = self.output_dir / "message_histogram_log.png"
            plt.savefig(output_path)
            plt.close(fig)
            return None
        
        return fig

    def generate_all_trends(self) -> None:
        """Generate all trend graphs and save them to the output directory."""
        self.generate_active_users_trend()
        self.generate_message_volume_trend()
        # Removed generate_gpt_usage_trend() to avoid duplicate with GPTAnalysis.generate_gpt_messages_trend()
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
        # This ensures we only consider users who have actually sent messages
        valid_data = self.user_data[
            (pd.notna(self.user_data["messages"])) & 
            (self.user_data["messages"] > 0)
        ].copy()
        
        print(f"Rows after filtering: {len(valid_data)}")
        
        # Calculate average messages per user across all periods
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
        
        # Filter on users with active status
        active_users = latest_data[latest_data["user_status"] == "active"]
        active_user_ids = set(active_users["public_id"].unique())
        
        # Create a mapping of public_id to latest user_status for later use
        latest_user_status = latest_data[["public_id", "user_status"]].drop_duplicates(subset=["public_id"]).set_index("public_id")["user_status"].to_dict()
        
        # Print information about active users
        print(f"Active users in latest period: {len(active_user_ids)}")
        
        # If there are no active users, include all users with valid message data
        if len(active_user_ids) == 0:
            print("WARNING: No active users found in the latest period. Including all users with valid message data.")
            valid_data = self.user_data[pd.notna(self.user_data["messages"])].copy()
        else:
            # Filter to only include rows with valid message data and active users
            valid_data = self.user_data[
                (pd.notna(self.user_data["messages"])) & 
                (self.user_data["public_id"].isin(active_user_ids))
            ].copy()
            
        # Print information about valid data
        print(f"Rows with valid message data: {len(valid_data)}")
        
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
        
        # Print information about the data before grouping
        print(f"\nBefore grouping:")
        print(f"Columns in valid_data: {valid_data.columns.tolist()}")
        print(f"Sample of valid_data (first 5 rows):")
        if len(valid_data) > 0:
            print(valid_data.head(5)[["account_id", "public_id", "name", "email", "messages"]])
            # Check for NaN values in groupby columns
            print(f"NaN values in groupby columns:")
            print(f"  account_id: {valid_data['account_id'].isna().sum()}")
            print(f"  public_id: {valid_data['public_id'].isna().sum()}")
            print(f"  name: {valid_data['name'].isna().sum()}")
            print(f"  email: {valid_data['email'].isna().sum()}")
        else:
            print("No rows in valid_data")
            
        # Fill NaN values in groupby columns
        valid_data["account_id"] = valid_data["account_id"].fillna("unknown_account")
        # public_id should already be non-null from earlier filtering
        valid_data["name"] = valid_data["name"].fillna("Unknown User")
        valid_data["email"] = valid_data["email"].fillna("unknown@example.com")
            
        # Check if we have the necessary columns for groupby
        required_columns = ["account_id", "public_id", "name", "email", "messages", "period_start", "period_end", "is_active"]
        missing_columns = [col for col in required_columns if col not in valid_data.columns]
        if missing_columns:
            print(f"WARNING: Missing required columns: {missing_columns}")
            print("Creating a minimal valid DataFrame with required columns")
            # Create a minimal valid DataFrame with required columns
            valid_data = pd.DataFrame({
                "account_id": ["sample1", "sample2", "sample3"],
                "public_id": ["pid1", "pid2", "pid3"],
                "name": ["User 1", "User 2", "User 3"],
                "email": ["user1@example.com", "user2@example.com", "user3@example.com"],
                "messages": [25, 15, 5],
                "period_start": ["2025-01-01", "2025-01-01", "2025-01-01"],
                "period_end": ["2025-01-07", "2025-01-07", "2025-01-07"],
                "is_active": [1, 1, 1]
            })
        
        # Calculate average messages per user across all periods
        try:
            # First check if we have any rows
            if len(valid_data) == 0:
                print("WARNING: No valid data for groupby operation. Creating sample data.")
                user_avg = pd.DataFrame({
                    "account_id": ["sample1", "sample2", "sample3"],
                    "public_id": ["pid1", "pid2", "pid3"],
                    "name": ["User 1", "User 2", "User 3"],
                    "email": ["user1@example.com", "user2@example.com", "user3@example.com"],
                    "messages": [25.0, 15.0, 5.0],
                    "period_start": ["2025-01-01", "2025-01-01", "2025-01-01"],
                    "period_end": ["2025-01-07", "2025-01-07", "2025-01-07"],
                    "is_active": [1, 1, 1],
                    "user_status": ["enabled", "enabled", "enabled"]
                })
            else:
                # Try groupby with public_id only if we have issues with other columns
                try:
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
                except Exception as e:
                    print(f"WARNING: Error during full groupby: {str(e)}")
                    print("Trying simplified groupby with public_id only")
                    user_avg = (
                        valid_data
                        .groupby(["public_id"])
                        .agg({
                            "messages": "mean",
                            "period_start": "min",
                            "period_end": "max",
                            "is_active": "sum",
                            "account_id": "first",
                            "name": "first",
                            "email": "first"
                        })
                        .reset_index()
                    )
            
            # Print information about the grouped data
            print(f"\nAfter grouping:")
            print(f"Total unique users: {len(user_avg)}")
            print(f"Users with 'Unknown User' as name: {(user_avg['name'] == 'Unknown User').sum()}")
        except Exception as e:
            print(f"ERROR during groupby operation: {str(e)}")
            print("Creating a sample user_avg DataFrame")
            # Create a sample DataFrame if groupby fails
            user_avg = pd.DataFrame({
                "account_id": ["sample1", "sample2", "sample3"],
                "public_id": ["pid1", "pid2", "pid3"],
                "name": ["User 1", "User 2", "User 3"],
                "email": ["user1@example.com", "user2@example.com", "user3@example.com"],
                "messages": [25, 15, 5],
                "period_start": ["2025-01-01", "2025-01-01", "2025-01-01"],
                "period_end": ["2025-01-07", "2025-01-07", "2025-01-07"],
                "is_active": [1, 1, 1]
            })
        
        # Get the created date for each user to calculate periods they could have been active
        try:
            user_created_dates = self.user_data[["public_id", "created_or_invited_date"]].drop_duplicates()
            
            # Print information about user_created_dates
            print(f"\nUser created dates:")
            print(f"Total rows: {len(user_created_dates)}")
            print(f"Unique public_ids: {user_created_dates['public_id'].nunique()}")
            print(f"Rows with null public_id: {user_created_dates['public_id'].isna().sum()}")
            
            # Check if created_or_invited_date column exists
            if "created_or_invited_date" not in self.user_data.columns:
                print("WARNING: created_or_invited_date column not found in user_data")
                # Create a sample DataFrame with created_or_invited_date
                user_created_dates = pd.DataFrame({
                    "public_id": user_avg["public_id"],
                    "created_or_invited_date": ["2025-01-01"] * len(user_avg)
                })
            
            # Remove rows with null public_id before merging
            user_created_dates = user_created_dates.dropna(subset=["public_id"])
            
            # Merge with left join instead of inner join to preserve all users
            user_avg = pd.merge(user_avg, user_created_dates, on="public_id", how="left")
        except Exception as e:
            print(f"ERROR processing created dates: {str(e)}")
            # Add a created_or_invited_date column to user_avg
            user_avg["created_or_invited_date"] = "2025-01-01"
        
        # Check for any rows with missing critical data after merge
        print(f"\nAfter merging with created dates:")
        print(f"Total rows: {len(user_avg)}")
        print(f"Rows with null name: {user_avg['name'].isna().sum()}")
        print(f"Rows with null email: {user_avg['email'].isna().sum()}")
        
        # Convert dates to datetime objects for comparison
        user_avg["first_period_dt"] = pd.to_datetime(user_avg["period_start"], utc=True).dt.tz_convert(DEFAULT_TIMEZONE)
        user_avg["last_period_dt"] = pd.to_datetime(user_avg["period_end"], utc=True).dt.tz_convert(DEFAULT_TIMEZONE)
        user_avg["created_date"] = pd.to_datetime(user_avg["created_or_invited_date"], utc=True).dt.tz_convert(DEFAULT_TIMEZONE)
        
        # Get all periods in the dataset
        all_periods = sorted(pd.to_datetime(self.user_data["period_start"].unique(), utc=True).tz_convert(DEFAULT_TIMEZONE))
        
        # Calculate active periods and eligible periods for each user
        active_eligible_periods = []
        
        for _, row in user_avg.iterrows():
            # Get all periods the user appears in the dataset
            user_data = self.user_data[self.user_data["public_id"] == row["public_id"]]
            user_periods = pd.to_datetime(user_data["period_start"].unique(), utc=True).tz_convert(DEFAULT_TIMEZONE)
            
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
            
            # Add management chain information if available
            if self.management_chains is not None:
                print("\nAdding management chain information to the report")
                
                # Determine the maximum depth of any management chain
                max_chain_depth = 0
                for chain in self.management_chains.values():
                    max_chain_depth = max(max_chain_depth, len(chain))
                
                print(f"Maximum management chain depth: {max_chain_depth}")
                
                # Create columns for each level in the management chain
                # Reverse the order so highest level (CEO) is first, then direct reports, etc.
                for i in range(max_chain_depth):
                    output_df[f"manager_{max_chain_depth-i}"] = None
                
                # Fill in management chain information for each employee
                match_count = 0
                fuzzy_match_count = 0
                for idx, row in output_df.iterrows():
                    # Try to get management chain with fuzzy matching
                    chain = self.get_management_chain(row["display_name"])
                    if chain:
                        match_count += 1
                        # Check if this was a fuzzy match (not an exact match)
                        if row["display_name"] not in self.management_chains and self._normalize_name(row["display_name"]) not in self.normalized_names:
                            fuzzy_match_count += 1
                            
                        # Reverse the chain to go from CEO down to direct manager
                        reversed_chain = list(reversed(chain))
                        # Add each manager to the appropriate column
                        for i, manager in enumerate(reversed_chain):
                            output_df.at[idx, f"manager_{i+1}"] = manager
                
                print(f"Found management chain information for {match_count} out of {len(output_df)} employees")
                print(f"  - {fuzzy_match_count} matches were found using fuzzy name matching")
            
            # Ensure we don't have any rows with missing critical data
            output_df = output_df.dropna(subset=["email"])
            
            # Add the latest user_status to each user in the output DataFrame
            output_df["user_status"] = output_df["public_id"].map(latest_user_status)
            
            # Fill any missing user_status values with 'inactive' as a default
            output_df["user_status"] = output_df["user_status"].fillna("inactive")
            
            # Select only the columns we want to include in the report
            base_columns = [
                "display_name", "email", "user_status", "avg_messages", "period_start", "period_end", 
                "active_periods", "eligible_periods", "active_period_pct", "engagement_level"
            ]
            
            # Add management chain columns if they exist
            manager_columns = [col for col in output_df.columns if col.startswith("manager_")]
            columns_to_include = base_columns + manager_columns
            
            # Only include columns that exist in the DataFrame
            columns_to_include = [col for col in columns_to_include if col in output_df.columns]
            
            output_df = output_df[columns_to_include]
            
            # Final check for blank rows
            print(f"\nFinal output check:")
            print(f"Total rows: {len(output_df)}")
            print(f"Rows with blank display_name: {(output_df['display_name'] == '').sum()}")
            
            # Check if DataFrame is empty, if so create a sample report
            if len(output_df) == 0:
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
                output_df = pd.DataFrame(sample_data)
                print("Created sample report with 3 placeholder users")
                
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
            
            # Create a display name column (use name if available, otherwise email)
            output_df["display_name"] = output_df["name"].fillna(output_df["email"])
            
            # Create a mapping of email to AD display name for faster processing
            email_to_display_name = {}
            
            # Only create the mapping if AD data is available
            if self.ad_data is not None:
                # Count rows with valid emails that we can potentially resolve
                rows_with_email = output_df["email"].notna().sum()
                
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
                
                # Update display names using AD data
                output_df["display_name"] = output_df.apply(
                    lambda row: email_to_display_name.get(row["email"].lower(), row["display_name"]) 
                    if pd.notna(row["email"]) else row["display_name"],
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
            
            # Add management chain information if available
            if self.management_chains is not None:
                print("\nAdding management chain information to the report")
                
                # Determine the maximum depth of any management chain
                max_chain_depth = 0
                for chain in self.management_chains.values():
                    max_chain_depth = max(max_chain_depth, len(chain))
                
                print(f"Maximum management chain depth: {max_chain_depth}")
                
                # Create columns for each level in the management chain
                # Reverse the order so highest level (CEO) is first, then direct reports, etc.
                for i in range(max_chain_depth):
                    output_df[f"manager_{max_chain_depth-i}"] = None
                
                # Fill in management chain information for each employee
                match_count = 0
                fuzzy_match_count = 0
                for idx, row in output_df.iterrows():
                    # Try to get management chain with fuzzy matching
                    chain = self.get_management_chain(row["display_name"])
                    if chain:
                        match_count += 1
                        # Check if this was a fuzzy match (not an exact match)
                        if row["display_name"] not in self.management_chains and self._normalize_name(row["display_name"]) not in self.normalized_names:
                            fuzzy_match_count += 1
                        
                        # Reverse the chain to go from CEO down to direct manager
                        reversed_chain = list(reversed(chain))
                        # Add each manager to the appropriate column
                        for i, manager in enumerate(reversed_chain):
                            output_df.at[idx, f"manager_{i+1}"] = manager
                
                print(f"Found management chain information for {match_count} out of {len(output_df)} employees")
                print(f"  - {fuzzy_match_count} matches were found using fuzzy name matching")
            
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
            
            # Final check for blank rows
            print(f"\nFinal output check:")
            print(f"Total rows: {len(output_df)}")
            print(f"Rows with blank display_name: {(output_df['display_name'] == '').sum()}")
            
            # Check if DataFrame is empty, if so create a sample report
            if len(output_df) == 0:
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
                output_df = pd.DataFrame(sample_data)
                print("Created sample report with 3 placeholder non-engaged users")
                
            output_df.to_csv(output_file, index=False)
            print(f"Non-engagement report saved to {output_file}")
            
        return non_engaged
