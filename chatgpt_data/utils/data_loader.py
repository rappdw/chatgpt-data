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
    
    def load_user_data(self) -> pd.DataFrame:
        """Load all user engagement data files.
        
        Returns:
            DataFrame containing user engagement data
        """
        user_files = [f for f in os.listdir(self.data_dir) if fnmatch.fnmatch(f, '*user_engagement*.csv')]
        
        dfs = []
        for file in user_files:
            df = self.load_csv_with_fallback_encoding(self.data_dir / file)
            if df is not None:
                dfs.append(df)
        
        if dfs:
            return pd.concat(dfs, ignore_index=True)
        else:
            raise FileNotFoundError("No user engagement data files found or could not be loaded")
    
    def load_ad_data(self) -> Optional[pd.DataFrame]:
        """Load Active Directory export data for name resolution.
        
        Returns:
            DataFrame containing AD data if available, None otherwise
        """
        ad_file = self.data_dir / "AD_export.csv"
        
        if not ad_file.exists():
            print("AD export file not found. User name resolution from AD will not be available.")
            return None
            
        df = self.load_csv_with_fallback_encoding(ad_file)
        if df is not None:
            print(f"AD data contains {len(df)} records")
        
        return df
    
    def load_management_chains(self) -> Optional[Dict]:
        """Load management chain data from JSON file.
        
        Returns:
            Dictionary containing management chain data if available, None otherwise
        """
        import json
        
        management_chains_file = self.data_dir / "management_chains.json"
        
        if not management_chains_file.exists():
            print("Management chains file not found. Management chain information will not be available.")
            return None
            
        try:
            with open(management_chains_file, 'r') as f:
                chains = json.load(f)
            print(f"Successfully loaded management chain data for {len(chains)} employees")
            return chains
        except Exception as e:
            print(f"Error loading management chain data: {str(e)}")
            return None


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