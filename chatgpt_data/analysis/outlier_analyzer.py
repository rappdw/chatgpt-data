"""Outlier analysis module for ChatGPT usage data."""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from chatgpt_data.analysis.interfaces import DataAnalyzer
from chatgpt_data.utils.constants import DEFAULT_TIMEZONE
from chatgpt_data.utils.data_loader import UserDataLoader
from chatgpt_data.utils.visualization import Visualizer


class OutlierAnalyzer(DataAnalyzer):
    """Class for analyzing outliers in ChatGPT usage data."""

    def __init__(self, data_dir: Union[str, Path], output_dir: Union[str, Path]):
        """Initialize the OutlierAnalyzer class.

        Args:
            data_dir: Directory containing the outlier data files
            output_dir: Directory to save the output files
        """
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.visualizer = Visualizer(output_dir)
        
        # Initialize UserDataLoader for data loading and display name resolution
        self.data_loader = UserDataLoader(data_dir, load_all=False)
        
        # Load outlier data
        self.period_outliers = self._load_outlier_data("message_outliers_by_period.csv")
        self.overall_outliers = self._load_outlier_data("message_outliers_overall.csv")
        self.pattern_outliers = self._load_outlier_data("message_pattern_outliers.csv")
        self.power_users = self._load_outlier_data("power_users.csv")
        
        # Load user engagement data for comparison
        self.user_data = self._load_outlier_data("user_engagement_report.csv")
        
        # Get email to display name mapping from UserDataLoader
        self.email_to_name = self.data_loader.get_email_to_display_name_mapping()
        
    def _load_outlier_data(self, filename: str) -> Optional[pd.DataFrame]:
        """Load outlier data from CSV file.
        
        Args:
            filename: Name of the CSV file to load
            
        Returns:
            DataFrame with outlier data or None if file not found
        """
        file_path = self.data_dir / filename
        if not file_path.exists():
            print(f"Warning: {file_path} not found")
            return None
            
        try:
            df = pd.read_csv(file_path)
            print(f"Loaded {len(df)} rows from {file_path}")
            return df
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return None
    
    def _get_display_name(self, email: str) -> str:
        """Get the display name for an email address using UserDataLoader.
        
        Args:
            email: Email address to get display name for
            
        Returns:
            Display name if available, otherwise formatted email
        """
        # Check for direct match in our local mapping first
        if email in self.email_to_name:
            return self.email_to_name[email]
        
        # Check for case-insensitive match in our local mapping
        if email.lower() in self.email_to_name:
            return self.email_to_name[email.lower()]
        
        # Try to get the display name from UserDataLoader
        display_name = self.data_loader.get_display_name_for_email(email)
        if display_name:
            # Cache this for future lookups
            self.email_to_name[email] = display_name
            self.email_to_name[email.lower()] = display_name
            return display_name
        
        # If UserDataLoader doesn't have it, format the email using its helper method
        formatted_name = self.data_loader.format_email_to_display_name(email)
        
        # Cache this for future lookups
        self.email_to_name[email] = formatted_name
        self.email_to_name[email.lower()] = formatted_name
        return formatted_name
    
    def generate_outlier_distribution_chart(self, save: bool = True) -> Optional[plt.Figure]:
        """Generate a bar chart showing the number of outliers per period.
        
        Args:
            save: Whether to save the figure to the output directory
            
        Returns:
            The matplotlib figure if save is False, otherwise None
        """
        if self.period_outliers is None:
            print("No period outliers data available")
            return None
            
        # Count outliers per period
        outlier_counts = self.period_outliers.groupby("period").size().reset_index(name="outlier_count")
        
        # Sort by period
        outlier_counts = outlier_counts.sort_values("period")
        
        # Create the figure
        fig, ax = plt.subplots(figsize=(12, 6))
        bars = ax.bar(outlier_counts["period"], outlier_counts["outlier_count"], color="#0F70C6")
        
        # Add data labels on top of bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f"{int(height)}", ha="center", va="bottom")
        
        # Set labels and title
        ax.set_xlabel("Period")
        ax.set_ylabel("Number of Outliers")
        ax.set_title("Number of Message Usage Outliers per Period")
        
        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45, ha="right")
        
        # Add grid lines
        ax.grid(axis="y", linestyle="--", alpha=0.7)
        
        # Tight layout
        plt.tight_layout()
        
        # Save or return the figure
        if save:
            output_path = self.output_dir / "outlier_distribution.png"
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
            plt.close(fig)
            print(f"Saved outlier distribution chart to {output_path}")
            return None
        else:
            return fig
    
    def generate_top_outliers_heatmap(self, save: bool = True) -> Optional[plt.Figure]:
        """Generate a heatmap showing the most frequent outliers across periods.
        
        Args:
            save: Whether to save the figure to the output directory
            
        Returns:
            The matplotlib figure if save is False, otherwise None
        """
        if self.period_outliers is None:
            print("No period outliers data available")
            return None
        
        # Get email column (could be 'email' or 'user_email')
        email_col = "email" if "email" in self.period_outliers.columns else "user_email"
        
        # Count occurrences of each user across periods
        user_period_counts = self.period_outliers.groupby([email_col, "period"]).size().reset_index(name="count")
        
        # Get the top 20 users by total occurrences
        top_users = (
            self.period_outliers.groupby(email_col).size()
            .sort_values(ascending=False)
            .head(20)
            .index.tolist()
        )
        
        # Create a mapping of emails to display names for the top users
        email_to_display = {}
        for email in top_users:
            email_to_display[email] = self._get_display_name(email)
        
        # Add display name column to user_period_counts
        user_period_counts["display_name"] = user_period_counts[email_col].map(email_to_display)
        
        # Filter to include only top users
        user_period_counts = user_period_counts[user_period_counts[email_col].isin(top_users)]
        
        # Pivot to create a matrix of users by periods using display names
        pivot_df = user_period_counts.pivot(index="display_name", columns="period", values="count").fillna(0)
        
        # Create the figure
        fig, ax = plt.subplots(figsize=(14, 10))
        
        # Create heatmap
        sns.heatmap(
            pivot_df, 
            cmap="YlOrRd", 
            linewidths=0.5, 
            ax=ax,
            cbar_kws={"label": "Occurrences"}
        )
        
        # Set labels and title
        ax.set_title("Top 20 Message Usage Outliers Across Periods")
        ax.set_xlabel("Period")
        ax.set_ylabel("User")
        
        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45, ha="right")
        
        # Tight layout
        plt.tight_layout()
        
        # Save or return the figure
        if save:
            output_path = self.output_dir / "top_outliers_heatmap.png"
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
            plt.close(fig)
            print(f"Saved top outliers heatmap to {output_path}")
            return None
        else:
            return fig
    
    def generate_outlier_impact_analysis(self, save: bool = True) -> Optional[plt.Figure]:
        """Generate a line chart comparing total message volume with message volume contributed by significant outliers.
        
        Args:
            save: Whether to save the figure to the output directory
            
        Returns:
            The matplotlib figure if save is False, otherwise None
        """
        if self.period_outliers is None:
            print("Missing required data for outlier impact analysis")
            return None
        
        try:
            # Get email column (could be 'email' or 'user_email')
            email_col = "email" if "email" in self.period_outliers.columns else "user_email"
            
            # First, identify the most significant outliers across all periods
            # We'll define these as users who appear as outliers most frequently
            top_outliers = (
                self.period_outliers.groupby(email_col).size()
                .sort_values(ascending=False)
                .head(10)  # Top 10 most frequent outliers
                .index.tolist()
            )
            
            # Create a flag for top outliers
            self.period_outliers["is_top_outlier"] = self.period_outliers[email_col].isin(top_outliers)
            
            # Group by period and calculate message counts
            outlier_by_period = (
                self.period_outliers.groupby(["period", "is_top_outlier"])
                .agg(
                    outlier_count=pd.NamedAgg(column=email_col, aggfunc="count"),
                    outlier_messages=pd.NamedAgg(column="messages", aggfunc="sum")
                )
                .reset_index()
            )
            
            # Pivot to get separate columns for top outliers and other outliers
            pivot_df = outlier_by_period.pivot(index="period", columns="is_top_outlier", values=["outlier_count", "outlier_messages"])
            
            # Flatten the multi-index columns
            pivot_df.columns = [f"{col[0]}_{col[1]}" for col in pivot_df.columns]
            pivot_df = pivot_df.reset_index()
            
            # Ensure we have all columns, even if some combinations don't exist
            for col in ["outlier_messages_True", "outlier_messages_False"]:
                if col not in pivot_df.columns:
                    pivot_df[col] = 0
            
            # Calculate total outlier messages
            pivot_df["total_outlier_messages"] = pivot_df["outlier_messages_True"].fillna(0) + pivot_df["outlier_messages_False"].fillna(0)
            
            # Sort by period
            pivot_df["period"] = pd.to_datetime(pivot_df["period"])
            pivot_df = pivot_df.sort_values("period")
            
            # Create the figure
            fig, ax = plt.subplots(figsize=(12, 6))
            
            # Plot total outlier messages
            ax.plot(
                pivot_df["period"], 
                pivot_df["total_outlier_messages"], 
                marker="o", 
                linewidth=2,
                color="#0F70C6",
                label="All Outlier Messages"
            )
            
            # Plot top outlier messages
            ax.plot(
                pivot_df["period"], 
                pivot_df["outlier_messages_True"].fillna(0), 
                marker="s", 
                linewidth=2,
                color="#FF6B00",
                label="Top 10 Outliers' Messages"
            )
            
            # Fill the area between the two lines to highlight the impact of top outliers
            ax.fill_between(
                pivot_df["period"],
                0,
                pivot_df["outlier_messages_True"].fillna(0),
                alpha=0.3,
                color="#FF6B00",
                label="Top Outliers' Contribution"
            )
            
            # Add annotations showing percentage contribution of top outliers
            for i, row in pivot_df.iterrows():
                if row["total_outlier_messages"] > 0:
                    pct = (row["outlier_messages_True"] / row["total_outlier_messages"]) * 100
                    if not pd.isna(pct) and pct > 0:
                        ax.annotate(
                            f"{pct:.0f}%",
                            (row["period"], row["outlier_messages_True"]),
                            textcoords="offset points",
                            xytext=(0, 10),
                            ha='center',
                            fontsize=8,
                            color="#FF6B00"
                        )
            
            # Set labels and title
            ax.set_xlabel("Period")
            ax.set_ylabel("Message Count")
            ax.set_title("Impact of Most Significant Outliers on Total Outlier Message Volume")
            
            # Add legend
            ax.legend(loc='upper left')
            
            # Add grid lines
            ax.grid(linestyle="--", alpha=0.3)
            
            # Rotate x-axis labels for better readability
            plt.xticks(rotation=45, ha="right")
            
            # Add a text box with the names of the top outliers
            top_outlier_names = []
            for email in top_outliers:
                display_name = self._get_display_name(email)
                top_outlier_names.append(f"{display_name} ({email})")
            
            # Create a string with the top outlier names
            top_outlier_text = "Top Outliers:\n" + "\n".join(top_outlier_names[:5])
            if len(top_outlier_names) > 5:
                top_outlier_text += f"\n...and {len(top_outlier_names) - 5} more"
            
            # Add a text box with the top outlier names
            props = dict(boxstyle='round', facecolor='white', alpha=0.7)
            ax.text(0.02, 0.98, top_outlier_text, transform=ax.transAxes, fontsize=9,
                    verticalalignment='top', bbox=props)
            
            # Tight layout
            plt.tight_layout()
            
            # Save or return the figure
            if save:
                output_path = self.output_dir / "outlier_impact_analysis.png"
                plt.savefig(output_path, dpi=300, bbox_inches="tight")
                plt.close(fig)
                print(f"Saved outlier impact analysis to {output_path}")
                return None
            else:
                return fig
                
        except Exception as e:
            print(f"Error generating outlier impact analysis: {e}")
            return None
    
    def generate_pattern_outliers_visualization(self, save: bool = True) -> Optional[plt.Figure]:
        """Generate a scatter plot showing coefficient of variation vs. max personal deviation.
        
        Args:
            save: Whether to save the figure to the output directory
            
        Returns:
            The matplotlib figure if save is False, otherwise None
        """
        if self.pattern_outliers is None or len(self.pattern_outliers) == 0:
            print("No pattern outliers data available")
            # Create a sample visualization with placeholder data
            fig, ax = plt.subplots(figsize=(12, 8))
            ax.text(0.5, 0.5, "No pattern outliers data available", 
                   horizontalalignment='center', verticalalignment='center',
                   transform=ax.transAxes, fontsize=14)
            ax.set_xlabel("Coefficient of Variation (std/mean)")
            ax.set_ylabel("Max Personal Deviation (std deviations)")
            ax.set_title("Pattern Outliers: Users with Unusual Usage Patterns")
            
            # Save or return the figure
            if save:
                output_path = self.output_dir / "pattern_outliers.png"
                plt.savefig(output_path, dpi=300, bbox_inches="tight")
                plt.close(fig)
                print(f"Saved empty pattern outliers visualization to {output_path}")
                return None
            else:
                return fig
            
        # Create the figure
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Get email column (could be 'email' or 'user_email')
        email_col = "email" if "email" in self.pattern_outliers.columns else "user_email"
        
        # Ensure numeric data types for plotting
        try:
            self.pattern_outliers["coefficient_of_variation"] = pd.to_numeric(
                self.pattern_outliers["coefficient_of_variation"], errors="coerce")
            self.pattern_outliers["max_personal_deviation"] = pd.to_numeric(
                self.pattern_outliers["max_personal_deviation"], errors="coerce")
            self.pattern_outliers["mean_messages"] = pd.to_numeric(
                self.pattern_outliers["mean_messages"], errors="coerce")
            
            # Drop rows with NaN values after conversion
            self.pattern_outliers = self.pattern_outliers.dropna(subset=[
                "coefficient_of_variation", "max_personal_deviation", "mean_messages"])
            
            if len(self.pattern_outliers) == 0:
                print("No valid pattern outliers data after conversion to numeric")
                ax.text(0.5, 0.5, "No valid pattern outliers data", 
                       horizontalalignment='center', verticalalignment='center',
                       transform=ax.transAxes, fontsize=14)
            else:
                # Create scatter plot
                scatter = ax.scatter(
                    self.pattern_outliers["coefficient_of_variation"],
                    self.pattern_outliers["max_personal_deviation"],
                    c=self.pattern_outliers["mean_messages"],
                    cmap="viridis",
                    alpha=0.7,
                    s=100,
                    edgecolors="w"
                )
                
                # Add colorbar
                cbar = plt.colorbar(scatter)
                cbar.set_label("Average Messages per Period")
                
                # Add grid lines
                ax.grid(linestyle="--", alpha=0.7)
                
                # Add annotations for top outliers (up to 10, but only if we have enough data)
                num_to_annotate = min(10, len(self.pattern_outliers))
                if num_to_annotate > 0:
                    # Sort by max_personal_deviation and take top N
                    sorted_outliers = self.pattern_outliers.sort_values(
                        by="max_personal_deviation", ascending=False).head(num_to_annotate)
                    
                    for _, row in sorted_outliers.iterrows():
                        email = row[email_col]
                        display_name = self._get_display_name(email)
                        ax.annotate(
                            display_name,
                            (row["coefficient_of_variation"], row["max_personal_deviation"]),
                            xytext=(5, 5),
                            textcoords="offset points",
                            fontsize=8
                        )
        except Exception as e:
            print(f"Error creating pattern outliers visualization: {e}")
            ax.text(0.5, 0.5, f"Error: {str(e)}", 
                   horizontalalignment='center', verticalalignment='center',
                   transform=ax.transAxes, fontsize=12)
        
        # Set labels and title
        ax.set_xlabel("Coefficient of Variation (std/mean)")
        ax.set_ylabel("Max Personal Deviation (std deviations)")
        ax.set_title("Pattern Outliers: Users with Unusual Usage Patterns")
        
        # Tight layout
        plt.tight_layout()
        
        # Save or return the figure
        if save:
            output_path = self.output_dir / "pattern_outliers.png"
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
            plt.close(fig)
            print(f"Saved pattern outliers visualization to {output_path}")
            return None
        else:
            return fig
    
    def generate_power_users_dashboard(self, save: bool = True) -> Optional[plt.Figure]:
        """Generate a dashboard for power users showing their impact on overall usage.
        
        Args:
            save: Whether to save the figure to the output directory
            
        Returns:
            The matplotlib figure if save is False, otherwise None
        """
        if self.power_users is None or len(self.power_users) == 0:
            print("No power users data available")
            # Create a sample visualization with placeholder data
            fig = plt.figure(figsize=(15, 10))
            plt.text(0.5, 0.5, "No power users data available", 
                   horizontalalignment='center', verticalalignment='center',
                   transform=plt.gca().transAxes, fontsize=14)
            plt.title("Power Users Analysis Dashboard")
            
            # Save or return the figure
            if save:
                output_path = self.output_dir / "power_users_dashboard.png"
                plt.savefig(output_path, dpi=300, bbox_inches="tight")
                plt.close(fig)
                print(f"Saved empty power users dashboard to {output_path}")
                return None
            else:
                return fig
        
        try:
            # Create the figure with subplots
            fig = plt.figure(figsize=(15, 10))
            
            # Define grid for subplots
            gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
            
            # Get email column (could be 'email' or 'user_email')
            email_col = "email" if "email" in self.power_users.columns else "user_email"
            
            # 1. Top power users by average messages
            ax1 = fig.add_subplot(gs[0, 0])
            top_users = self.power_users.nlargest(10, "avg_messages")
            # Map emails to display names
            display_names = [self._get_display_name(email) for email in top_users[email_col]]
            bars1 = ax1.barh(
                display_names,
                top_users["avg_messages"],
                color="#0F70C6"
            )
            ax1.set_xlabel("Average Messages per Period")
            ax1.set_title("Top 10 Power Users by Average Messages")
            ax1.invert_yaxis()  # To have the highest at the top
            
            # Add data labels
            for bar in bars1:
                width = bar.get_width()
                ax1.text(width + 1, bar.get_y() + bar.get_height()/2, f"{width:.1f}", 
                        ha="left", va="center")
            
            # 2. Distribution of times above population average
            ax2 = fig.add_subplot(gs[0, 1])
            ax2.hist(
                self.power_users["times_above_population_avg"],
                bins=10,
                color="#FF6B00",
                edgecolor="black"
            )
            ax2.set_xlabel("Times Above Population Average")
            ax2.set_ylabel("Number of Power Users")
            ax2.set_title("Distribution of Power User Message Volume\nRelative to Population Average")
            
            # Check if we have the period consistency columns
            has_period_data = all(col in self.power_users.columns for col in ["pct_periods_above_avg", "period_count"])
            
            if has_period_data:
                # 3. Percentage of periods above average
                ax3 = fig.add_subplot(gs[1, 0])
                ax3.hist(
                    self.power_users["pct_periods_above_avg"],
                    bins=10,
                    range=(70, 100),  # Power users are defined as being above avg in at least 75% of periods
                    color="#58595B",
                    edgecolor="black"
                )
                ax3.set_xlabel("Percentage of Periods Above Average")
                ax3.set_ylabel("Number of Power Users")
                ax3.set_title("Consistency of Power Users\n(% of Periods Above Average)")
                
                # 4. Scatter plot of consistency vs. volume
                ax4 = fig.add_subplot(gs[1, 1])
                scatter = ax4.scatter(
                    self.power_users["pct_periods_above_avg"],
                    self.power_users["times_above_population_avg"],
                    c=self.power_users["period_count"],
                    cmap="viridis",
                    alpha=0.7,
                    s=100,
                    edgecolors="w"
                )
                ax4.set_xlabel("Consistency (% Periods Above Avg)")
                ax4.set_ylabel("Volume (Times Above Population Avg)")
                ax4.set_title("Power User Consistency vs. Volume")
                
                # Add colorbar
                cbar = plt.colorbar(scatter, ax=ax4)
                cbar.set_label("Number of Active Periods")
                
                # Add annotations for extreme power users
                extreme_users = self.power_users.nlargest(5, "times_above_population_avg")
                for _, row in extreme_users.iterrows():
                    email = row[email_col]
                    display_name = self._get_display_name(email)
                    ax4.annotate(
                        display_name,
                        (row["pct_periods_above_avg"], row["times_above_population_avg"]),
                        xytext=(5, 5),
                        textcoords="offset points",
                        fontsize=8
                    )
            else:
                # Alternative visualizations for the bottom row if period data is missing
                
                # 3. Top power users by times above population average
                ax3 = fig.add_subplot(gs[1, 0])
                top_ratio_users = self.power_users.nlargest(10, "times_above_population_avg")
                # Map emails to display names
                display_names_ratio = [self._get_display_name(email) for email in top_ratio_users[email_col]]
                bars3 = ax3.barh(
                    display_names_ratio,
                    top_ratio_users["times_above_population_avg"],
                    color="#58595B"
                )
                ax3.set_xlabel("Times Above Population Average")
                ax3.set_title("Top 10 Power Users by Relative Usage")
                ax3.invert_yaxis()  # To have the highest at the top
                
                # Add data labels
                for bar in bars3:
                    width = bar.get_width()
                    ax3.text(width + 0.1, bar.get_y() + bar.get_height()/2, f"{width:.1f}x", 
                            ha="left", va="center")
                
                # 4. Scatter plot of average messages vs. times above population average
                ax4 = fig.add_subplot(gs[1, 1])
                scatter = ax4.scatter(
                    self.power_users["avg_messages"],
                    self.power_users["times_above_population_avg"],
                    c=self.power_users["times_above_population_avg"],
                    cmap="viridis",
                    alpha=0.7,
                    s=100,
                    edgecolors="w"
                )
                ax4.set_xlabel("Average Messages per Period")
                ax4.set_ylabel("Times Above Population Average")
                ax4.set_title("Power User Volume vs. Relative Usage")
                
                # Add colorbar
                cbar = plt.colorbar(scatter, ax=ax4)
                cbar.set_label("Times Above Population Average")
                
                # Add annotations for extreme power users
                extreme_users = self.power_users.nlargest(5, "times_above_population_avg")
                for _, row in extreme_users.iterrows():
                    email = row[email_col]
                    display_name = self._get_display_name(email)
                    ax4.annotate(
                        display_name,
                        (row["avg_messages"], row["times_above_population_avg"]),
                        xytext=(5, 5),
                        textcoords="offset points",
                        fontsize=8
                    )
            
            # Create a figure with an extra space at the top for criteria text
            fig.suptitle("Power Users Analysis Dashboard", fontsize=16, y=0.98)
            
            # Add text box explaining power user criteria
            criteria_text = (
                "Power User Criteria: "
                "1) Active in 3+ time periods  "
                "2) Above average usage in 75%+ of active periods  "
                "3) At least 2x the population average message volume"
            )
            
            # Add text as a subtitle below the main title
            plt.figtext(0.5, 0.93, criteria_text, ha="center", fontsize=9, 
                      bbox={"facecolor": "#f0f0f0", "alpha": 0.7, "pad": 5, "boxstyle": "round,pad=0.5"})
            
            # Tight layout - adjust for the overall title and criteria text at top
            # Increased vertical space by adjusting the rect parameter
            plt.tight_layout(rect=[0, 0, 1, 0.89])  # Leave more space at top
            
            # Save or return the figure
            if save:
                output_path = self.output_dir / "power_users_dashboard.png"
                plt.savefig(output_path, dpi=300, bbox_inches="tight")
                plt.close(fig)
                print(f"Saved power users dashboard to {output_path}")
                return None
            else:
                return fig
                
        except Exception as e:
            print(f"Error generating power users dashboard: {e}")
            # Create a simple error figure
            fig = plt.figure(figsize=(15, 10))
            plt.text(0.5, 0.5, f"Error generating power users dashboard: {e}", 
                   horizontalalignment='center', verticalalignment='center',
                   transform=plt.gca().transAxes, fontsize=12)
            plt.title("Power Users Analysis Dashboard - Error")
            
            # Save or return the figure
            if save:
                output_path = self.output_dir / "power_users_dashboard.png"
                plt.savefig(output_path, dpi=300, bbox_inches="tight")
                plt.close(fig)
                print(f"Saved error power users dashboard to {output_path}")
                return None
            else:
                return fig
    
    def generate_all_trends(self) -> None:
        """Generate all trend visualizations (required by DataAnalyzer interface)."""
        self.generate_all_visualizations()
    
    def _copy_csv_files(self) -> None:
        """Copy outlier CSV files from data_dir to output_dir."""
        # List of CSV files to copy
        csv_files = [
            "message_outliers_by_period.csv",
            "message_outliers_overall.csv",
            "message_pattern_outliers.csv",
            "power_users.csv"
        ]
        
        for filename in csv_files:
            source_path = self.data_dir / filename
            target_path = self.output_dir / filename
            
            if source_path.exists():
                try:
                    # Special handling for power_users.csv to reorder columns
                    if filename == "power_users.csv":
                        self._reorder_power_users_csv(source_path, target_path)
                    else:
                        import shutil
                        shutil.copy2(source_path, target_path)
                    print(f"Copied {filename} to {self.output_dir}")
                except Exception as e:
                    print(f"Error copying {filename}: {e}")
            else:
                print(f"Warning: Source file {source_path} not found, cannot copy")
                
    def _reorder_power_users_csv(self, source_path: Path, target_path: Path) -> None:
        """Reorder columns in power_users.csv to have display_name as the first column.
        
        Args:
            source_path: Path to the source power_users.csv file
            target_path: Path to the target power_users.csv file
        """
        try:
            # Read the CSV file
            power_users_df = pd.read_csv(source_path)
            
            # Add display names if not already present
            if 'display_name' not in power_users_df.columns:
                power_users_df = self.data_loader.add_display_names_to_dataframe(power_users_df)
            
            # Get the email column name (could be 'email' or 'user_email')
            email_col = 'email' if 'email' in power_users_df.columns else 'user_email'
            
            # Reorder columns to have display_name first, then email, then the rest
            cols = ['display_name', email_col]
            other_cols = [col for col in power_users_df.columns if col not in cols]
            power_users_df = power_users_df[cols + other_cols]
            
            # Save the reordered DataFrame
            power_users_df.to_csv(target_path, index=False)
        except Exception as e:
            print(f"Error reordering power_users.csv: {e}")
            # Fall back to simple copy if reordering fails
            import shutil
            shutil.copy2(source_path, target_path)
    
    def generate_all_visualizations(self) -> None:
        """Generate all outlier visualizations and save them to the output directory."""
        print("\nGenerating outlier visualizations...")
        # Copy CSV files first
        self._copy_csv_files()
        
        # Generate visualizations
        self.generate_outlier_distribution_chart()
        self.generate_top_outliers_heatmap()
        self.generate_outlier_impact_analysis()
        self.generate_pattern_outliers_visualization()
        self.generate_power_users_dashboard()
        print("All outlier visualizations generated successfully")
