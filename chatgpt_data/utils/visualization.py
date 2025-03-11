"""Visualization utilities for ChatGPT data analysis."""

from pathlib import Path
from typing import Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Import the default timezone for consistent datetime handling
from chatgpt_data.utils.constants import DEFAULT_TIMEZONE


class Visualizer:
    """Base class for data visualization."""
    
    def __init__(self, output_dir: Union[str, Path]):
        """Initialize the Visualizer.
        
        Args:
            output_dir: Directory to save the output files
        """
        self.output_dir = Path(output_dir)
    
    def create_figure(self, figsize: Tuple[int, int] = (10, 6)) -> Tuple[plt.Figure, plt.Axes]:
        """Create a figure with a single set of axes.
        
        Args:
            figsize: Figure size (width, height) in inches
            
        Returns:
            Figure and axes objects
        """
        fig, ax = plt.subplots(figsize=figsize)
        return fig, ax
    
    def add_comment_box(self, ax: plt.Axes, comment: str, 
                        position: Tuple[float, float] = (0.05, 0.05), 
                        fontsize: int = 9, 
                        valign: str = 'bottom') -> None:
        """Add a comment box to a plot.
        
        Args:
            ax: Axes to add the comment box to
            comment: Comment text
            position: Position of the comment box (x, y) in axes coordinates
            fontsize: Font size
            valign: Vertical alignment ('bottom' or 'top')
        """
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax.text(position[0], position[1], comment, transform=ax.transAxes, fontsize=fontsize,
                verticalalignment=valign, bbox=props)
    
    def finalize_plot(self, fig: plt.Figure, ax: plt.Axes, title: str, 
                     xlabel: str, ylabel: str, save_path: Optional[str] = None,
                     rotate_xticks: bool = True) -> None:
        """Finalize a plot with common settings.
        
        Args:
            fig: Figure object
            ax: Axes object
            title: Plot title
            xlabel: X-axis label
            ylabel: Y-axis label
            save_path: Path to save the figure (optional)
            rotate_xticks: Whether to rotate x-axis tick labels
        """
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(True)
        
        if rotate_xticks:
            plt.xticks(rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            plt.close(fig)
    
    def create_time_series_plot(self, data: pd.DataFrame, x_col: str, y_col: str, 
                               title: str, xlabel: str, ylabel: str, comment: str,
                               filename: str, save: bool = True) -> Optional[plt.Figure]:
        """Create a time series plot.
        
        Args:
            data: DataFrame containing the data
            x_col: Column name for x-axis (usually a date)
            y_col: Column name for y-axis
            title: Plot title
            xlabel: X-axis label
            ylabel: Y-axis label
            comment: Comment to add to the plot
            filename: Filename to save the plot (without directory)
            save: Whether to save the figure to the output directory
            
        Returns:
            The matplotlib figure if save is False, otherwise None
        """
        # Create the plot
        fig, ax = self.create_figure()
        ax.plot(data[x_col], data[y_col], marker="o")
        
        # Add comment box
        self.add_comment_box(ax, comment)
        
        # Finalize the plot
        save_path = self.output_dir / filename if save else None
        self.finalize_plot(fig, ax, title, xlabel, ylabel, save_path)
        
        if save:
            return None
        return fig
    
    def create_histogram(self, data: pd.DataFrame, value_col: str, bins: int = 20,
                        max_value: Optional[int] = None, log_scale: bool = False,
                        title: str = None, xlabel: str = None, ylabel: str = None,
                        comment: str = None, filename: str = None, 
                        save: bool = True) -> Optional[plt.Figure]:
        """Create a histogram.
        
        Args:
            data: DataFrame containing the data
            value_col: Column to create histogram from
            bins: Number of bins for the histogram
            max_value: Maximum value to include (None for no limit)
            log_scale: Whether to use a logarithmic y-axis scale
            title: Plot title
            xlabel: X-axis label
            ylabel: Y-axis label
            comment: Comment to add to the plot
            filename: Filename to save the plot (without directory)
            save: Whether to save the figure to the output directory
            
        Returns:
            The matplotlib figure if save is False, otherwise None
        """
        # Set default values if not provided
        if title is None:
            title = f"Distribution of {value_col}" + (" (Log Scale)" if log_scale else "")
        if xlabel is None:
            xlabel = f"{value_col}"
        if ylabel is None:
            ylabel = "Number of Items" + (" (Log Scale)" if log_scale else "")
        if comment is None:
            comment = f"This histogram shows the distribution of {value_col.lower()}."
        if filename is None:
            filename = f"{value_col.lower()}_histogram" + ("_log" if log_scale else "") + ".png"
        
        # Filter to only include rows with valid data
        valid_data = data[pd.notna(data[value_col])].copy()
        
        # Ensure data is numeric
        if not pd.api.types.is_numeric_dtype(valid_data[value_col]):
            valid_data[value_col] = pd.to_numeric(valid_data[value_col], errors='coerce')
        
        # Filter to positive values
        valid_data = valid_data[valid_data[value_col] > 0]
        print(f"Found {len(valid_data)} rows with positive {value_col}")
        
        # Check if we have enough data to create a meaningful histogram
        if len(valid_data) < 2:
            print(f"Warning: Not enough data to create a meaningful histogram for {value_col}")
            fig, ax = plt.subplots(figsize=(12, 7))
            ax.text(0.5, 0.5, f"Not enough data to create a meaningful histogram.\nNo items with positive {value_col} found.", 
                    ha='center', va='center', fontsize=14)
            ax.set_title(title)
            ax.axis('off')
            
            if save:
                output_path = self.output_dir / filename
                plt.savefig(output_path)
                plt.close(fig)
                return None
            
            return fig
        
        # Filter out extreme values if max_value is specified
        if max_value is not None:
            plot_data = valid_data[valid_data[value_col] <= max_value]
        else:
            plot_data = valid_data
            
        print(f"Using {len(plot_data)} data points for histogram")
        
        # Create the histogram
        fig, ax = plt.subplots(figsize=(12, 7))
        
        try:
            # Plot histogram
            n, bins_array, patches = ax.hist(
                plot_data[value_col], 
                bins=min(bins, len(plot_data)), # Ensure bins don't exceed data points
                edgecolor='black', 
                alpha=0.7
            )
            
            # Set y-axis to log scale if requested
            if log_scale:
                ax.set_yscale('log')
            
            # Add a vertical line for the mean and median
            mean_value = plot_data[value_col].mean()
            median_value = plot_data[value_col].median()
            ax.axvline(mean_value, color='red', linestyle='dashed', linewidth=1, label=f'Mean: {mean_value:.2f}')
            ax.axvline(median_value, color='green', linestyle='dashed', linewidth=1, label=f'Median: {median_value:.2f}')
            
            ax.legend()
            
            # Add filter information to comment if applicable
            total_items = len(valid_data)
            filtered_items = len(plot_data)
            if max_value is not None and filtered_items < total_items:
                filter_text = f"\nNote: {total_items - filtered_items} items with >{max_value} {value_col} are not shown."
                comment += filter_text
                
            # Add comment box
            self.add_comment_box(ax, comment, position=(0.05, 0.95), valign='top')
            
        except Exception as e:
            print(f"Error creating histogram: {e}")
            ax.text(0.5, 0.5, f"Error creating histogram: {e}", 
                   ha='center', va='center', fontsize=12, wrap=True)
            ax.set_title(f"{title} - Error")
            ax.axis('off')
        
        # Finalize the plot
        save_path = self.output_dir / filename if save else None
        self.finalize_plot(fig, ax, title, xlabel, ylabel, save_path, rotate_xticks=False)
        
        if save:
            return None
        return fig