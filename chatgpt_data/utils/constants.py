"""Constants used throughout the chatgpt-data package."""

from datetime import timezone
import zoneinfo

# Define the standard timezone to use for all datetime conversions (US Pacific Time)
DEFAULT_TIMEZONE = zoneinfo.ZoneInfo("America/Los_Angeles")
