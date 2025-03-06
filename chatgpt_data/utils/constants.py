"""Constants used throughout the chatgpt-data package."""

from datetime import timezone, datetime
import zoneinfo

# Define the standard timezone to use for all datetime conversions (US Pacific Time)
DEFAULT_TIMEZONE = zoneinfo.ZoneInfo("America/Los_Angeles")

# Maximum allowed time difference (in seconds) between message timestamp and conversation last_active_at
# This accounts for potential timestamp inconsistencies in the API (10 minutes)
MESSAGE_TIMESTAMP_TOLERANCE = 10 * 60  # 10 minutes in seconds

# Helper function to convert a date string to a Unix timestamp
def date_to_timestamp(date_str, end_of_day=False):
    """Convert a date string in the format 'yyyy-mm-dd' to a Unix timestamp.
    
    Args:
        date_str: Date string in the format 'yyyy-mm-dd'
        end_of_day: If True, set the time to 23:59:59, otherwise 00:00:00
        
    Returns:
        Unix timestamp (seconds since epoch)
    """
    year, month, day = map(int, date_str.split('-'))
    if end_of_day:
        dt = datetime(year, month, day, 23, 59, 59, tzinfo=DEFAULT_TIMEZONE)
    else:
        dt = datetime(year, month, day, tzinfo=DEFAULT_TIMEZONE)
    return int(dt.timestamp())

# Earliest expected timestamp for conversations and messages (Oct 31, 2024 end of day Pacific Time)
# Any timestamps before this date are considered suspicious and will be logged
EARLIEST_EXPECTED_TIMESTAMP = date_to_timestamp('2024-10-31', end_of_day=True)
