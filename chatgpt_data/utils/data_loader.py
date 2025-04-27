"""Data loading utilities for ChatGPT data analysis."""

import os
import fnmatch
from pathlib import Path
from typing import Dict, List, Optional, Union

import pandas as pd

from chatgpt_data.analysis.interfaces import DataLoader as DataLoaderInterface


class DataLoader(DataLoaderInterface):
    """Base class for loading data from files."""
    
    def __init__(self, data_dir: Union[str, Path]):
        """Initialize the DataLoader.
        
        Args:
            data_dir: Directory containing the raw data files
        """
        self.data_dir = Path(data_dir)
        
    def load_csv_with_fallback_encoding(self, file_path: Path) -> Optional[pd.DataFrame]:
        """Load a CSV file with fallback encoding options.
        
        Args:
            file_path: Path to the CSV file
            
        Returns:
            DataFrame if successful, None otherwise
        """
        try:
            # Try different encodings if UTF-8 fails
            encodings = ['utf-8', 'utf-8-sig', 'latin1', 'ISO-8859-1', 'cp1252']
            for encoding in encodings:
                try:
                    df = pd.read_csv(file_path, encoding=encoding)
                    print(f"Successfully loaded {file_path.name} with encoding {encoding}")
                    return df
                except UnicodeDecodeError:
                    continue
            
            print(f"Could not load {file_path.name} with any encoding")
            return None
        except Exception as e:
            print(f"Error loading {file_path.name}: {str(e)}")
            return None


class UserDataLoader(DataLoader):
    """Class for loading user engagement data."""
    
    def __init__(self, data_dir: Union[str, Path], load_all: bool = True):
        """Initialize the UserDataLoader.
        
        Args:
            data_dir: Directory containing the raw data files
            load_all: Whether to load all data during initialization
        """
        super().__init__(data_dir)
        self._ad_data = None
        self._email_to_display_name = None
        self._management_chains = None
        self._user_data = None
        
        # Load all data during initialization if requested
        if load_all:
            self.load_ad_data()
            self.load_management_chains()
            try:
                self.load_user_data()
            except FileNotFoundError:
                # User data might not be available yet, that's okay
                pass
                
    def get_user_data(self) -> pd.DataFrame:
        """Get the user engagement data.
        
        Returns:
            DataFrame containing user engagement data
        """
        if self._user_data is None:
            self.load_user_data()
        return self._user_data
    
    def get_ad_data(self) -> Optional[pd.DataFrame]:
        """Get the AD data.
        
        Returns:
            DataFrame containing AD data if available, None otherwise
        """
        if self._ad_data is None:
            self.load_ad_data()
        return self._ad_data
    
    def get_management_chains(self) -> Optional[Dict]:
        """Get the management chain data.
        
        Returns:
            Dictionary containing management chain data if available, None otherwise
        """
        if self._management_chains is None:
            self.load_management_chains()
        return self._management_chains
    
    def load_user_data(self) -> pd.DataFrame:
        """Load all user engagement data files.
        
        Returns:
            DataFrame containing user engagement data
        """
        # Return cached data if available
        if self._user_data is not None:
            return self._user_data
            
        user_files = [f for f in os.listdir(self.data_dir) if fnmatch.fnmatch(f, '*user_engagement*.csv')]
        
        dfs = []
        for file in user_files:
            df = self.load_csv_with_fallback_encoding(self.data_dir / file)
            if df is not None:
                dfs.append(df)
        
        if dfs:
            self._user_data = pd.concat(dfs, ignore_index=True)
            return self._user_data
        else:
            raise FileNotFoundError("No user engagement data files found or could not be loaded")
    
    def load_ad_data(self) -> Optional[pd.DataFrame]:
        """Load Active Directory export data for name resolution.
        
        Returns:
            DataFrame containing AD data if available, None otherwise
        """
        # Return cached data if available
        if self._ad_data is not None:
            return self._ad_data
            
        ad_file = self.data_dir / "AD_export.csv"
        
        if not ad_file.exists():
            print("AD export file not found. User name resolution from AD will not be available.")
            self._ad_data = None
            return None
            
        df = self.load_csv_with_fallback_encoding(ad_file)
        if df is not None:
            print(f"AD data contains {len(df)} records")
            self._ad_data = df
        
        return self._ad_data
    
    def load_management_chains(self) -> Optional[Dict]:
        """Load management chain data from JSON file.
        
        Returns:
            Dictionary containing management chain data if available, None otherwise
        """
        # Return cached data if available
        if self._management_chains is not None:
            return self._management_chains
            
        import json
        
        management_chains_file = self.data_dir / "management_chains.json"
        
        if not management_chains_file.exists():
            print("Management chains file not found. Management chain information will not be available.")
            self._management_chains = None
            return None
            
        try:
            with open(management_chains_file, 'r') as f:
                chains = json.load(f)
            print(f"Successfully loaded management chain data for {len(chains)} employees")
            self._management_chains = chains
            return self._management_chains
        except Exception as e:
            print(f"Error loading management chain data: {str(e)}")
            self._management_chains = None
            return None
            
    def resolve_user_name_from_ad(self, email: str) -> str:
        """Resolve a user's display name from their email using AD data.
        
        Args:
            email: User's email address
            
        Returns:
            Display name from AD if found, otherwise the original email
        """
        # If email is invalid, return it as is
        if email is None or pd.isna(email) or email == "":
            return email
            
        # Load AD data if not already loaded
        if self._ad_data is None:
            self.load_ad_data()
            
        # If no AD data available, return the original email
        if self._ad_data is None or len(self._ad_data) == 0:
            return email
            
        # Try to match on userPrincipalName first
        match = self._ad_data[self._ad_data["userPrincipalName"].str.lower() == email.lower()]
        
        # If no match, try the mail column
        if len(match) == 0 and "mail" in self._ad_data.columns:
            match = self._ad_data[self._ad_data["mail"].str.lower() == email.lower()]
            
        # Return the display name if found, otherwise return the email
        if len(match) > 0 and not pd.isna(match.iloc[0]["displayName"]):
            return match.iloc[0]["displayName"]
        else:
            return email
            
    def create_email_to_display_name_mapping(self) -> Dict[str, str]:
        """Create a mapping from email addresses to display names using AD data.
        
        Returns:
            Dictionary mapping email addresses to display names
        """
        # Return cached mapping if available
        if self._email_to_display_name is not None:
            return self._email_to_display_name
            
        # Load AD data if not already loaded
        if self._ad_data is None:
            self.load_ad_data()
            
        # If no AD data available, return an empty dictionary
        if self._ad_data is None or len(self._ad_data) == 0:
            self._email_to_display_name = {}
            return {}
            
        email_to_display_name = {}
        
        # Process userPrincipalName column (primary email)
        for _, row in self._ad_data.iterrows():
            if pd.notna(row.get("userPrincipalName")) and pd.notna(row.get("displayName")):
                email_to_display_name[row["userPrincipalName"].lower()] = row["displayName"]
            
            # Also process the mail column if it exists and is different
            if pd.notna(row.get("mail")) and pd.notna(row.get("displayName")):
                if row["mail"].lower() not in email_to_display_name:
                    email_to_display_name[row["mail"].lower()] = row["displayName"]
        
        print(f"Created mapping for {len(email_to_display_name)} email addresses to display names")
        self._email_to_display_name = email_to_display_name
        return self._email_to_display_name
        
    def add_display_names_to_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add display names to a DataFrame based on email addresses.
        
        Args:
            df: DataFrame to add display names to
            
        Returns:
            DataFrame with display names added
        """
        if df is None or len(df) == 0:
            return df
        
        # Determine the email column name
        email_col = None
        if "email" in df.columns:
            email_col = "email"
        elif "user_email" in df.columns:
            email_col = "user_email"
        
        if email_col is None:
            return df
        
        # Create a copy to avoid modifying the original
        result_df = df.copy()
        
        # Create email to display name mapping
        email_to_display_name = self.create_email_to_display_name_mapping()
        
        # If no mapping available, return the original DataFrame
        if not email_to_display_name:
            return df
        
        # Add display_name column if it doesn't exist
        if "display_name" not in result_df.columns:
            # Map emails to display names
            display_names = []
            for email in result_df[email_col]:
                if pd.isna(email):
                    display_names.append(None)
                elif email.lower() in email_to_display_name:
                    display_names.append(email_to_display_name[email.lower()])
                else:
                    # Use email as fallback
                    display_names.append(email)
            
            # Add display_name column
            result_df.insert(1, "display_name", display_names)
        
        return result_df
        
    def format_email_to_display_name(self, email: str) -> str:
        """Format an email address into a display name.
        
        Args:
            email: Email address to format
            
        Returns:
            Formatted display name
        """
        import re
        
        # If email is invalid, return it as is
        if email is None or pd.isna(email) or email == "" or not isinstance(email, str) or '@' not in email:
            return email
        
        # Extract the username part of the email (before @)
        username = email.split('@')[0]
        # Convert username to a more readable format (e.g., john.doe -> John Doe)
        name_parts = re.split(r'[._-]', username)
        formatted_name = ' '.join([part.capitalize() for part in name_parts])
        return formatted_name
    
    def get_display_name_for_email(self, email: str) -> str:
        """Get the display name for an email address.
        
        Args:
            email: Email address to get display name for
            
        Returns:
            Display name if available, otherwise None
        """
        # If email is invalid, return None
        if email is None or pd.isna(email) or email == "":
            return None
            
        # Ensure email_to_display_name mapping is loaded
        if self._email_to_display_name is None:
            self.create_email_to_display_name_mapping()
            
        # Check if email exists in mapping (case-insensitive)
        if self._email_to_display_name and email.lower() in self._email_to_display_name:
            return self._email_to_display_name[email.lower()]
            
        # If not in mapping, try to resolve from AD data directly
        display_name = self.resolve_user_name_from_ad(email)
        if display_name != email:  # If we got a real display name (not the fallback email)
            return display_name
            
        return None
    
    def get_email_to_display_name_mapping(self) -> Dict[str, str]:
        """Get the email to display name mapping.
        
        Returns:
            Dictionary mapping email addresses to display names
        """
        if self._email_to_display_name is None:
            self.create_email_to_display_name_mapping()
        return self._email_to_display_name


class GPTDataLoader(DataLoader):
    """Class for loading GPT engagement data."""
    
    def load_gpt_data(self) -> pd.DataFrame:
        """Load all GPT engagement data files.
        
        Returns:
            DataFrame containing GPT engagement data
        """
        gpt_files = [f for f in os.listdir(self.data_dir) if fnmatch.fnmatch(f, '*gpt_engagement*.csv')]
        
        dfs = []
        for file in gpt_files:
            df = self.load_csv_with_fallback_encoding(self.data_dir / file)
            if df is not None:
                dfs.append(df)
        
        if dfs:
            return pd.concat(dfs, ignore_index=True)
        else:
            raise FileNotFoundError("No GPT engagement data files found or could not be loaded")