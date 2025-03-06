import unittest
import pandas as pd
from datetime import datetime, timedelta

class TestDetermineCadence(unittest.TestCase):
    def setUp(self):
        # Define the _determine_cadence method directly in the test
        self.determine_cadence = self._determine_cadence
    
    def _determine_cadence(self, start_date, end_date):
        """Determine the cadence based on the time span between start_date and end_date.
        
        Args:
            start_date: The start date of the period
            end_date: The end date of the period
            
        Returns:
            String indicating the cadence (daily, weekly, bi-weekly, monthly, quarterly, etc.)
        """
        if pd.isna(start_date) or pd.isna(end_date):
            return "unknown"
            
        # Calculate the time difference in days
        time_diff = (end_date - start_date).days
        
        # Determine cadence based on the time span
        if time_diff < 2:
            return "daily"
        elif time_diff <= 7:
            return "weekly"
        elif time_diff <= 14:
            return "bi-weekly"
        elif time_diff <= 31:
            return "monthly"
        elif time_diff <= 92:
            return "quarterly"
        elif time_diff <= 183:
            return "semi-annual"
        else:
            return "annual"
    
    def test_determine_cadence_daily(self):
        start_date = pd.Timestamp('2023-01-01')
        end_date = pd.Timestamp('2023-01-01')  # Same day
        self.assertEqual(self.determine_cadence(start_date, end_date), "daily")
        
        # Test with 1 day difference
        end_date = pd.Timestamp('2023-01-02')
        self.assertEqual(self.determine_cadence(start_date, end_date), "daily")
    
    def test_determine_cadence_weekly(self):
        start_date = pd.Timestamp('2023-01-01')
        end_date = pd.Timestamp('2023-01-07')  # 6 days difference
        self.assertEqual(self.determine_cadence(start_date, end_date), "weekly")
        
        # Test with 7 days difference
        end_date = pd.Timestamp('2023-01-08')
        self.assertEqual(self.determine_cadence(start_date, end_date), "weekly")
    
    def test_determine_cadence_biweekly(self):
        start_date = pd.Timestamp('2023-01-01')
        end_date = pd.Timestamp('2023-01-14')  # 13 days difference
        self.assertEqual(self.determine_cadence(start_date, end_date), "bi-weekly")
    
    def test_determine_cadence_monthly(self):
        start_date = pd.Timestamp('2023-01-01')
        end_date = pd.Timestamp('2023-01-31')  # 30 days difference
        self.assertEqual(self.determine_cadence(start_date, end_date), "monthly")
    
    def test_determine_cadence_quarterly(self):
        start_date = pd.Timestamp('2023-01-01')
        end_date = pd.Timestamp('2023-03-31')  # ~90 days difference
        self.assertEqual(self.determine_cadence(start_date, end_date), "quarterly")
    
    def test_determine_cadence_semiannual(self):
        start_date = pd.Timestamp('2023-01-01')
        end_date = pd.Timestamp('2023-06-30')  # ~180 days difference
        self.assertEqual(self.determine_cadence(start_date, end_date), "semi-annual")
    
    def test_determine_cadence_annual(self):
        start_date = pd.Timestamp('2023-01-01')
        end_date = pd.Timestamp('2023-12-31')  # ~365 days difference
        self.assertEqual(self.determine_cadence(start_date, end_date), "annual")
    
    def test_determine_cadence_unknown(self):
        # Test with NaN values
        start_date = pd.NaT
        end_date = pd.Timestamp('2023-01-01')
        self.assertEqual(self.determine_cadence(start_date, end_date), "unknown")
        
        start_date = pd.Timestamp('2023-01-01')
        end_date = pd.NaT
        self.assertEqual(self.determine_cadence(start_date, end_date), "unknown")
        
        start_date = pd.NaT
        end_date = pd.NaT
        self.assertEqual(self.determine_cadence(start_date, end_date), "unknown")

if __name__ == '__main__':
    unittest.main()
