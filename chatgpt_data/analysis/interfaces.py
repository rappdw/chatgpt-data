"""Interface definitions for the analysis modules."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Union

import matplotlib.pyplot as plt
import pandas as pd


class DataAnalyzer(ABC):
    """Interface for data analyzers."""
    
    @abstractmethod
    def __init__(self, data_dir: Union[str, Path], output_dir: Union[str, Path]):
        """Initialize the data analyzer.
        
        Args:
            data_dir: Directory containing the raw data files
            output_dir: Directory to save the output files
        """
        pass
    
    @abstractmethod
    def generate_all_trends(self) -> None:
        """Generate all trend graphs and save them to the output directory."""
        pass


class DataLoader(ABC):
    """Interface for data loaders."""
    
    @abstractmethod
    def __init__(self, data_dir: Union[str, Path]):
        """Initialize the data loader.
        
        Args:
            data_dir: Directory containing the raw data files
        """
        pass


class ReportGeneratorInterface(ABC):
    """Interface for report generators."""
    
    @abstractmethod
    def __init__(self, analyzer: DataAnalyzer):
        """Initialize the report generator.
        
        Args:
            analyzer: DataAnalyzer instance
        """
        pass
    
    @abstractmethod
    def generate_engagement_report(self, output_file: Optional[str] = None) -> pd.DataFrame:
        """Generate a report of user engagement levels.
        
        Args:
            output_file: Path to save the report CSV file (optional)
            
        Returns:
            DataFrame with user engagement report
        """
        pass