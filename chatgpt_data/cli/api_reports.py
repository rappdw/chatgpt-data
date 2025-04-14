"""CLI tool to fetch ChatGPT usage reports via the Enterprise Compliance API."""

import argparse
from operator import gt
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List, Optional, Union, Tuple, Dict, Set, Any, Literal
from dataclasses import dataclass, field
import csv
import re
import glob
import pickle
import json
from pathlib import Path
from tqdm import tqdm
import sys

from chatgpt_data.api.compliance_api import EnterpriseComplianceAPI, User, GPT, Project
from chatgpt_data.cli.all_trends import main as run_all_trends
from chatgpt_data.utils.constants import DEFAULT_TIMEZONE
from dotenv import load_dotenv

def parse_date(date_str: str) -> datetime:
    """Parse a date string in YYYY-MM-DD format.

    Args:
        date_str: Date string to parse

    Returns:
        Parsed datetime object
    """
    try:
        # Parse the date and attach our default timezone
        dt = datetime.strptime(date_str, "%Y-%m-%d")
        return dt.replace(tzinfo=DEFAULT_TIMEZONE)
    except ValueError:
        raise argparse.ArgumentTypeError(f"Invalid date format: {date_str}. Use YYYY-MM-DD")


def get_default_dates() -> Tuple[datetime, datetime]:
    """Get default start and end dates for reports.

    Returns:
        Tuple of (start_date, end_date) as datetime objects
    """
    # Default end date is today
    end_date = datetime.now(DEFAULT_TIMEZONE)
    
    # Default start date is 7 days ago
    start_date = end_date - timedelta(days=7)
    
    # Set time to midnight for consistent behavior
    start_date = start_date.replace(hour=0, minute=0, second=0, microsecond=0)
    end_date = end_date.replace(hour=23, minute=59, second=59, microsecond=999999)
    
    return start_date, end_date


def datetime_to_unix_timestamp(dt: datetime) -> int:
    """Convert a datetime object to a Unix timestamp.

    Args:
        dt: Datetime object to convert

    Returns:
        Unix timestamp (seconds since epoch)
    """
    # Ensure the datetime is timezone-aware using our default timezone
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=DEFAULT_TIMEZONE)
    return int(dt.timestamp())


def determine_cadence(start_date: datetime, end_date: datetime) -> str:
    """Determine the cadence based on the time span between start_date and end_date.
    
    Args:
        start_date: The start date of the period
        end_date: The end date of the period
        
    Returns:
        String indicating the cadence (daily, weekly, bi-weekly, monthly, quarterly, etc.)
    """
    if not start_date or not end_date:
        return "unknown"
        
    # Calculate the time difference in days
    time_diff = (end_date - start_date).days
    
    # Determine cadence based on the time span
    if time_diff < 2:
        return "Daily"
    elif time_diff <= 7:
        return "Weekly"
    elif time_diff <= 14:
        return "Bi-weekly"
    elif time_diff <= 31:
        return "Monthly"
    elif time_diff <= 92:
        return "Quarterly"
    elif time_diff <= 183:
        return "Semi-annual"
    else:
        return "Annual"


@dataclass
class RawData:
    users: Dict[str, User]
    earliest_message_timestamp: int
    latest_message_timestamp: int
    project_map: Dict[str, Project]
    gpt_map: Dict[str, GPT]

@dataclass
class UserMetrics:
    conversation_count: int = 0
    message_count: int = 0
    user_message_count: int = 0
    gpt_message_count: int = 0
    gpt_set: Set[str] = field(default_factory=set)
    tool_message_count: int = 0
    tool_set: Set[str] = field(default_factory=set)
    project_message_count: int = 0
    project_set: Set[str] = field(default_factory=set)
    system_message_count: int = 0
    assistant_message_count: int = 0
    first_day_active: int = 0
    last_day_active: int = 0

@dataclass
class GPTMetrics:
    conversation_count: int = 0
    message_count: int = 0
    user_set: Set[str] = field(default_factory=set)
    gpt_set: Set[str] = field(default_factory=set)
    first_day_active: Optional[int] = None
    last_day_active: Optional[int] = None

@dataclass
class EngagementMetrics:
    users: Dict[str, User]
    user_activity: Dict[str, UserMetrics] = field(default_factory=dict)
    gpts: Dict[str, GPT] = field(default_factory=dict)
    gpt_activity: Dict[str, GPTMetrics] = field(default_factory=dict)

@dataclass
class Message:
    id: str
    created_at: int
    gpt_id: Optional[str] = None
    project_id: Optional[str] = None
    role: Literal["user", "system", "assistant", "tool"] = "user"
    tool_name: Optional[str] = None

@dataclass
class Conversation:
    id: str
    created_at: int
    last_active_at: int
    title: str
    messages: List[Message] = field(default_factory=list)

def get_engagement_metrics(raw_data: RawData, engagement_metrics: Optional[EngagementMetrics] = None) -> EngagementMetrics:
    """Get engagement metrics for a set of users and conversations.
    
    Args:
        raw_data: Raw data containing users, conversations, and messages
        engagement_metrics: Optional existing EngagementMetrics object to populate
        
    Returns:
        Populated EngagementMetrics object
    """
    # Create new EngagementMetrics if none provided
    if engagement_metrics is None:
        engagement_metrics = EngagementMetrics(users=raw_data.users)

    total_conversation_count = 0
    total_message_count = 0

    for user_id, user in raw_data.users.items():
        user_metrics = UserMetrics()
        engagement_metrics.user_activity[user_id] = user_metrics
        for conversation in user.conversations:
            if conversation.last_active_at < raw_data.earliest_message_timestamp or conversation.created_at > raw_data.latest_message_timestamp:
                continue
            total_conversation_count += 1
            conversation_in_range = False
            for message in conversation.messages:
                if message.created_at is None or message.created_at < raw_data.earliest_message_timestamp or message.created_at > raw_data.latest_message_timestamp:
                    continue
                # if message.created_at < conversation.created_at or message.created_at > conversation.last_active_at:
                #     print(f"Skipping message from conversation out of range: {message.id}")
                #     continue
                total_message_count += 1
                conversation_in_range = True
                user_metrics.message_count += 1

                if user_metrics.first_day_active == 0 or message.created_at < user_metrics.first_day_active:
                    user_metrics.first_day_active = message.created_at                   
                if message.created_at > user_metrics.last_day_active:
                    user_metrics.last_day_active = message.created_at

                if message.role == "user":
                    user_metrics.user_message_count += 1
                elif message.role == "assistant":
                    user_metrics.assistant_message_count += 1
                elif message.role == "system":
                    user_metrics.system_message_count += 1
                elif message.role == "tool":
                    user_metrics.tool_message_count += 1
                    if message.tool_name:
                        user_metrics.tool_set.add(message.tool_name)
                    else:
                        user_metrics.tool_set.add("unknown")
                if message.tool_name and message.role not in {"tool", "assistant"}:
                    print(f"WARNING: Message with role {message.role} has tool name {message.tool_name}")
                    user_metrics.tool_set.add(message.tool_name)
                if message.gpt_id:
                    # if role != "user":
                    #     print(f" Non-user gpt message. role, '{role}'\nmsg: {msg}\n\n")
                    user_metrics.gpt_message_count += 1
                    user_metrics.gpt_set.add(message.gpt_id)
                    gpt_metrics = engagement_metrics.gpt_activity.get(message.gpt_id, GPTMetrics())
                    gpt_metrics.message_count += 1
                    gpt_metrics.user_set.add(user_id)
                    
                    # Update first and last active day for the GPT
                    message_day = int(message.created_at / 86400) * 86400
                    if gpt_metrics.first_day_active is None or message_day < gpt_metrics.first_day_active:
                        gpt_metrics.first_day_active = message_day
                    if gpt_metrics.last_day_active is None or message_day > gpt_metrics.last_day_active:
                        gpt_metrics.last_day_active = message_day
                        
                    # Store the updated metrics
                    engagement_metrics.gpt_activity[message.gpt_id] = gpt_metrics
                if message.project_id:
                    # if role != "user":
                    #     print(f" Non-user project message. role, '{role}'\nmsg: {msg}\n\n")
                    user_metrics.project_message_count += 1
                    user_metrics.project_set.add(message.project_id)


            if conversation_in_range:
                user_metrics.conversation_count += 1
    
    # start_date = datetime.fromtimestamp(raw_data.earliest_message_timestamp, tz=DEFAULT_TIMEZONE).strftime('%Y-%m-%d')
    # end_date = datetime.fromtimestamp(raw_data.latest_message_timestamp, tz=DEFAULT_TIMEZONE).strftime('%Y-%m-%d')
    # print(f"Data set in scope: {start_date} to {end_date}")
    # print(f"Total conversations: {total_conversation_count}")
    # print(f"Total messages: {total_message_count}")
    return engagement_metrics

def get_users_conversations(api: EnterpriseComplianceAPI, debug_logging: bool = False) -> RawData:
    """
    Get users and conversations they have engaged in.
    
    Args:
        api: The EnterpriseComplianceAPI instance
        debug_logging: Whether to print detailed debug logs
    
    Returns:
        A dictionary of users and their engagement data
    """
    
    print("Fetching users...")
    # Get all users with progress tracking
    users_dict = api.get_all_users(use_tqdm=True)
    conversation_count = 0
    message_count = 0
    project_map = dict()
    gpt_map = dict()
    # get the unix timestamp of current time
    earliest_message_timestamp = datetime.now(DEFAULT_TIMEZONE).timestamp()
    most_recent_message_timestamp = 0

    # Process all conversations with a callback function
    def process_conversation(conversation):
        nonlocal users_dict, conversation_count, message_count, project_map, gpt_map, earliest_message_timestamp, most_recent_message_timestamp
        conversation_id = conversation.get("id", "unknown")
        user_id = conversation.get("user_id")
        last_active = conversation.get("last_active_at")
        title = conversation.get("title")

        user = users_dict.get(user_id)
        if not user:
            print(f"\n\nWARNING: Could not find user with id {user_id} in conversation {conversation_id}\n\n")
            return
        
        conversation_data = Conversation(
            id=conversation_id,
            created_at=conversation.get("created_at"),
            last_active_at=last_active,
            title=title,
        )
        user.conversations.append(conversation_data)
        conversation_count += 1

        messages_response = conversation.get("messages", {})
        if messages_response.get("has_more"):
            print(f"\n\nWARNING: Conversation {conversation_id} 'has_more' is set!!!! Don't know how to handle this\n\n")
        
        messages = messages_response.get("data", [])
        for msg in messages:
            gpt_id = msg.get("gpt_id")
            project_id = msg.get("project_id")
            if gpt_id and not gpt_id in gpt_map:
                gpt_map[gpt_id] = api.get_gpt(gpt_id)
            if project_id and not project_id in project_map:
                project_map[project_id] = api.get_project(project_id)
            msg_timestamp = msg.get("created_at")
            if msg_timestamp is not None:
                if msg_timestamp < earliest_message_timestamp:
                    earliest_message_timestamp = msg_timestamp
                if msg_timestamp > most_recent_message_timestamp:
                    most_recent_message_timestamp = msg_timestamp
            message_data = Message(
                id=msg.get("id", "unknown"),
                created_at=msg_timestamp,
                gpt_id=gpt_id,
                project_id=project_id,
                role=msg.get("author", {}).get("role"),
                tool_name=msg.get("author", {}).get("tool_name"),
            )
            conversation_data.messages.append(message_data)
            message_count += 1
    
    # Process all conversations with progress tracking
    print("Processing conversations...")
    api.process_all_conversations(
        callback_fn=process_conversation,
        debug_logging=debug_logging,
        use_tqdm=True
    )
    
    # Print summary statistics for verification
    print(f"Retrieved {len(users_dict)} users for analysis")
    print(f"\nProcessed {conversation_count} total conversations")
    print(f"Processed {message_count} total messages")
    
    return RawData(
        users=users_dict,
        project_map=project_map,
        gpt_map=gpt_map,
        earliest_message_timestamp=earliest_message_timestamp,
        latest_message_timestamp=most_recent_message_timestamp
    )

def find_existing_csv_files(output_dir: str) -> List[str]:
    """
    Find existing CSV files in the output directory that match engagement report patterns.
    
    Args:
        output_dir: Directory to search for CSV files
        
    Returns:
        List of paths to existing CSV files
    """
    patterns = [
        "user_engagement_*.csv",
        "non_engagement_*.csv",
        "gpt_engagement_*.csv",
        "gpt_non_engagement_*.csv"
    ]
    
    existing_files = []
    for pattern in patterns:
        existing_files.extend(glob.glob(os.path.join(output_dir, pattern)))
    
    return existing_files

def save_engagement_metrics_to_csv(metrics: EngagementMetrics, output_dir: str, end_date: datetime, start_date: datetime) -> None:
    save_user_engagement_metrics_to_csv(metrics, output_dir, end_date, start_date)
    save_gpt_engagement_metrics_to_csv(metrics, output_dir, end_date, start_date)


def save_user_engagement_metrics_to_csv(metrics: EngagementMetrics, output_dir: str, end_date: datetime, start_date: datetime) -> None:
    """Save user engagement metrics to a CSV file.
    
    Args:
        metrics: EngagementMetrics object with user activity data
        output_dir: Directory to save output files
        end_date: Optional end date of the reporting period. If provided, users created after this date
                 will be excluded from the non-engaged users report.
        start_date: Optional start date of the reporting period. Used for determining cadence.
        
    This function creates two CSV files:
    1. The main engagement metrics file with the name provided in output_file
    2. A non-engagement file for users who had no activity during the reporting period
       with the name pattern: non_engagement_YYYYMMDD_YYYYMMDD.csv
    """
    date_range_str = f"{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}"
    # Format period start and end dates
    period_start_str = start_date.strftime("%Y-%m-%d")
    period_end_str = end_date.strftime("%Y-%m-%d")
    
    # Create rows for active users
    active_rows = []
    
    # Convert user activity data to rows
    for user_id, activity in metrics.user_activity.items():
        # Get user info - handle the case when user might be None
        user = metrics.users.get(user_id)
        
        # Create display name (use name if available, otherwise email)
        display_name = user.name
        if not display_name:
            display_name = user.email
        
        # Get last active timestamp
        last_active_timestamp = activity.last_day_active
        last_active_str = datetime.fromtimestamp(last_active_timestamp, tz=DEFAULT_TIMEZONE).strftime("%Y-%m-%d") if last_active_timestamp else ""
        
        # Get first active timestamp
        first_active_timestamp = activity.first_day_active
        first_active_str = datetime.fromtimestamp(first_active_timestamp, tz=DEFAULT_TIMEZONE).strftime("%Y-%m-%d") if first_active_timestamp else ""
        
        # Get created_at timestamp
        created_at = user.created_at
        created_at_str = datetime.fromtimestamp(created_at, tz=DEFAULT_TIMEZONE).strftime("%Y-%m-%d") if created_at else ""
        
        # Determine cadence based on start_date and end_date
        cadence = determine_cadence(start_date, end_date) if start_date and end_date else "unknown"
        
        # Get organization ID (account_id)
        account_id = ""
        
        row = {
            "cadence": cadence,
            "period_start": period_start_str,
            "period_end": period_end_str,
            "account_id": account_id,
            "public_id": user_id,
            "name": display_name,
            "email": user.email,
            "role": "",
            "user_role": user.role,
            "department": "",  # Empty string as requested
            "user_status": user.status,
            "created_or_invited_date": created_at_str,
            "is_active": 1 if (activity.user_message_count > 0 and 
                               activity.last_day_active >= datetime_to_unix_timestamp(end_date - timedelta(days=30))) else 0,  # Only active if user has sent messages in the last month
            "first_day_active_in_period": first_active_str,
            "last_day_active_in_period": last_active_str,
            "messages": activity.user_message_count,
            "messages_rank": 0,
            "gpt_messages": activity.gpt_message_count,
            "gpts_messaged": len(activity.gpt_set),
            "tool_messages": activity.tool_message_count,
            "tools_messaged": len(activity.tool_set),
            "last_day_active": last_active_str,
            "project_messages": activity.project_message_count,
            "projects_messaged": len(activity.project_set),
            "assistant_message_count": activity.assistant_message_count,
            "system_message_count": activity.system_message_count,
            "total_message_count": activity.message_count,
            "conversation_count": activity.conversation_count,
            "gpt_models": ",".join(activity.gpt_set),
            "tools": ",".join(activity.tool_set),
            "projects": ",".join(activity.project_set),
        }
        active_rows.append(row)
    
    # Sort rows by messages (descending)
    active_rows.sort(key=lambda x: x["messages"], reverse=True)
    
    # Add message rank based on sort order
    for i, row in enumerate(active_rows):
        row["message_rank"] = i + 1  # 1-based ranking
    
    # Write to CSV
    if active_rows:
        output_file = os.path.join(output_dir, f"user_engagement_{date_range_str}.csv")
        with open(output_file, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=active_rows[0].keys())
            writer.writeheader()
            writer.writerows(active_rows)
        print(f"Saved engagement metrics for {len(active_rows)} users to {output_file}")
    else:
        print("No user engagement data to save")

def save_gpt_engagement_metrics_to_csv(metrics: EngagementMetrics, output_dir: str, end_date: datetime, start_date: datetime) -> None:
    """Save gpt engagement metrics to a CSV file.
    
    Args:
        metrics: EngagementMetrics object with gpt activity data
        output_dir: Directory to save output files
        end_date: Optional end date of the reporting period. If provided, users created after this date
                 will be excluded from the non-engaged users report.
        start_date: Optional start date of the reporting period. Used for determining cadence.
        
    This function creates two CSV files:
    1. The main engagement metrics file with the name provided in output_file
    2. A non-engagement file for users who had no activity during the reporting period
       with the name pattern: non_engagement_YYYYMMDD_YYYYMMDD.csv
    """
    date_range_str = f"{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}"
    # Format period start and end dates
    period_start_str = start_date.strftime("%Y-%m-%d")
    period_end_str = end_date.strftime("%Y-%m-%d")
    
    # Create rows for active users
    active_rows = []
    
    # Convert gpt activity data to rows
    for gpt_id, activity in metrics.gpt_activity.items():
        # Get gpt info - handle the case when gpt might be None
        gpt = metrics.gpts.get(gpt_id)
        if not gpt:
            continue
        
        # Create display name (use name if available, otherwise "unknown")
        display_name = gpt.name
        
        # Get last active timestamp
        last_active_timestamp = activity.last_day_active
        last_active_str = datetime.fromtimestamp(last_active_timestamp, tz=DEFAULT_TIMEZONE).strftime("%Y-%m-%d") if last_active_timestamp else ""
        
        # Get first active timestamp
        first_active_timestamp = activity.first_day_active
        first_active_str = datetime.fromtimestamp(first_active_timestamp, tz=DEFAULT_TIMEZONE).strftime("%Y-%m-%d") if first_active_timestamp else ""
        
        # Determine cadence based on start_date and end_date
        cadence = determine_cadence(start_date, end_date) if start_date and end_date else "unknown"
        
        # Get organization ID (account_id)
        account_id = ""
        
        # Map user status to expected values
        row = {
            "cadence": cadence,
            "period_start": period_start_str,
            "period_end": period_end_str,
            "account_id": account_id,
            "gpt_id": gpt_id,
            "gpt_name": display_name,
            "gpt_description": gpt.description,
            "gpt_url": f"https://chatgpt.com/g/{gpt_id}",
            "gpt_creator": gpt.creator_id,
            "gpt_creator_email": gpt.creator_email,
            "is_active": 1 if (activity.message_count > 0 and 
                              activity.last_day_active >= datetime_to_unix_timestamp(end_date - timedelta(days=30))) else 0,  # Only active if gpt has received messages in the last month
            "first_day_active_in_period": first_active_str,
            "last_day_active_in_period": last_active_str,
            "messages_workspace": activity.message_count,
            "unique_messagers_workspace": len(activity.user_set),
        }
        active_rows.append(row)
    
    # Sort rows by messages (descending)
    active_rows.sort(key=lambda x: x["messages_workspace"], reverse=True)
    
    # Write to CSV
    if active_rows:
        output_file = os.path.join(output_dir, f"gpt_engagement_{date_range_str}.csv")
        with open(output_file, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=active_rows[0].keys())
            writer.writeheader()
            writer.writerows(active_rows)
        print(f"Saved engagement metrics for {len(active_rows)} gpts to {output_file}")
    else:
        print("No gpt engagement data to save")
    


def process_engagement_data(raw_data: RawData, output_dir: str) -> None:
    # convert rawdata.earliest_message_timestamp and raw_data.latest_message_timestamp to datetime
    # Add GPTs and Projects to the engagement metrics
    start_date = datetime.fromtimestamp(raw_data.earliest_message_timestamp, tz=DEFAULT_TIMEZONE)
    end_date = datetime.fromtimestamp(raw_data.latest_message_timestamp, tz=DEFAULT_TIMEZONE)   
    print(f"Processing engagement data for date range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    
    # Create EngagementMetrics object with users and GPTs
    user_metrics = EngagementMetrics(
        users=raw_data.users,
        gpts=raw_data.gpt_map
    )
    
    # Process message data to populate metrics
    user_metrics = get_engagement_metrics(raw_data, user_metrics)

    # Save engagement metrics to CSV
    save_engagement_metrics_to_csv(
        metrics=user_metrics,
        output_dir=output_dir,
        end_date=end_date,
        start_date=start_date
    )

def process_data_in_weekly_chunks(raw_data: RawData, output_dir: str, allow_partial_weeks: bool = False) -> None:
    """
    Process data in weekly chunks from the earliest to latest timestamp in raw_data.
    
    Args:
        raw_data: The raw data containing users and timestamp information
        output_dir: Directory to save output files
        allow_partial_weeks: If True, process partial weeks at the end of the date range
        
    Returns:
        None
    """
    # Check for existing CSV files before processing any chunks
    existing_files = find_existing_csv_files(output_dir)
    if existing_files:
        print(f"Found {len(existing_files)} existing CSV files in {output_dir}")
        for file in existing_files[:5]:  # Show up to 5 files
            print(f"  - {os.path.basename(file)}")
        if len(existing_files) > 5:
            print(f"  - ... and {len(existing_files) - 5} more")
            
        # Ask user for confirmation
        response = input("\nDelete these files before proceeding with weekly processing? [y/N] ").strip().lower()
        if response == 'y' or response == 'yes':
            for file in existing_files:
                try:
                    os.remove(file)
                    print(f"Deleted: {os.path.basename(file)}")
                except Exception as e:
                    print(f"Error deleting {file}: {str(e)}")
            print(f"Deleted {len(existing_files)} existing files")
        else:
            print("Keeping existing files")
    
    # Get datetime objects from timestamps
    raw_start_dt = datetime.fromtimestamp(raw_data.earliest_message_timestamp, tz=DEFAULT_TIMEZONE)
    raw_end_dt = datetime.fromtimestamp(raw_data.latest_message_timestamp, tz=DEFAULT_TIMEZONE)
    
    # Normalize to start of day for start date and end of day for end date
    current_start = datetime(raw_start_dt.year, raw_start_dt.month, raw_start_dt.day, 0, 0, 0, tzinfo=DEFAULT_TIMEZONE)
    end_date = datetime(raw_end_dt.year, raw_end_dt.month, raw_end_dt.day, 23, 59, 59, tzinfo=DEFAULT_TIMEZONE)
    
    while current_start <= end_date:
        # Calculate end of current week (or use end_date if it's sooner)
        week_end = current_start + timedelta(days=6)
        # Set to end of day (23:59:59)
        current_end = datetime(week_end.year, week_end.month, week_end.day, 23, 59, 59, tzinfo=DEFAULT_TIMEZONE)
        # Don't go beyond the overall end date
        if current_end > end_date:
            current_end = end_date
            
            # Check if this is a partial week and skip if not allowed
            days_in_chunk = (current_end - current_start).days + 1
            if not allow_partial_weeks and days_in_chunk < 7:
                print(f"Skipping partial week with only {days_in_chunk} days (from {current_start.strftime('%Y-%m-%d')} to {current_end.strftime('%Y-%m-%d')})")
                break
        
        # Create a copy of raw_data with updated timestamps for this chunk
        # Ensure timezone-aware datetime objects for timestamp conversion
        if current_start.tzinfo is None:
            current_start = current_start.replace(tzinfo=DEFAULT_TIMEZONE)
        if current_end.tzinfo is None:
            current_end = current_end.replace(tzinfo=DEFAULT_TIMEZONE)
            
        chunk_data = RawData(
            users=raw_data.users,
            earliest_message_timestamp=current_start.timestamp(),
            latest_message_timestamp=current_end.timestamp(),
            project_map=raw_data.project_map,
            gpt_map=raw_data.gpt_map
        )
        
        # Process this week's data
        print(f"Processing data for week: {current_start.strftime('%Y-%m-%d')} to {current_end.strftime('%Y-%m-%d')}")
        process_engagement_data(chunk_data, output_dir)
        
        # Move to next week (start of next day)
        current_start = (current_end + timedelta(seconds=1)).replace(hour=0, minute=0, second=0)

def apply_date_filters(raw_data: RawData, start_date: Optional[datetime] = None, end_date: Optional[datetime] = None) -> RawData:
    """
    Apply date filters to raw data by updating the timestamp boundaries.
    
    Args:
        raw_data: The raw data to filter
        start_date: Optional start date to filter from
        end_date: Optional end date to filter to
        
    Returns:
        RawData: The filtered raw data
    """
    if start_date:
        # Normalize to start of day using our default timezone
        start_date = datetime(start_date.year, start_date.month, start_date.day, 0, 0, 0, tzinfo=DEFAULT_TIMEZONE)
        # Ensure timezone-aware datetime for timestamp conversion
        if start_date.tzinfo is None:
            start_date = start_date.replace(tzinfo=DEFAULT_TIMEZONE)
        raw_data.earliest_message_timestamp = start_date.timestamp()
    if end_date:
        # Normalize to end of day using our default timezone
        end_date = datetime(end_date.year, end_date.month, end_date.day, 23, 59, 59, tzinfo=DEFAULT_TIMEZONE)
        # Ensure timezone-aware datetime for timestamp conversion
        if end_date.tzinfo is None:
            end_date = end_date.replace(tzinfo=DEFAULT_TIMEZONE)
        raw_data.latest_message_timestamp = end_date.timestamp()
    
    return raw_data

def save_raw_data(raw_data: RawData, output_dir: str) -> tuple[str, str]:
    """
    Save raw data to disk in both pickle and JSON formats.
    
    Args:
        raw_data: The raw data to save
        output_dir: Directory to save the files
        loaded_data: Whether the data was loaded from existing files
        
    Returns:
        tuple: Paths to the pickle and JSON files
    """
    # Create a timestamp for the filenames
    timestamp = datetime.now(DEFAULT_TIMEZONE).strftime("%Y%m%d_%H%M%S")
    
    # Save as pickle (preserves all object types)
    pickle_path = os.path.join(output_dir, f"raw_data_{timestamp}.pkl")
    with open(pickle_path, 'wb') as f:
        pickle.dump(raw_data, f)
    print(f"Raw data saved to {pickle_path}")
    
    # Save as JSON (more human-readable)
    # Convert raw_data to a serializable format
    json_data = {
        "users": {
            user_id: {
                "id": user_id,
                "email": user.email,
                "name": user.name,
                "status": user.status,
                "conversations": [
                    {
                        "id": conv.id,
                        "title": conv.title,
                        "created_at": conv.created_at,
                        "last_active_at": conv.last_active_at,
                        "messages": [
                            {
                                "id": msg.id,
                                "created_at": msg.created_at,
                                "role": msg.role,
                                "gpt_id": msg.gpt_id,
                                "project_id": msg.project_id,
                                "tool_name": msg.tool_name
                            } for msg in conv.messages
                        ]
                    } for conv in user.conversations
                ]
            } for user_id, user in raw_data.users.items()
        },
        "earliest_message_timestamp": raw_data.earliest_message_timestamp,
        "latest_message_timestamp": raw_data.latest_message_timestamp
    }
    
    json_path = os.path.join(output_dir, f"raw_data_{timestamp}.json")
    with open(json_path, 'w') as f:
        json.dump(json_data, f, indent=2)
    print(f"Raw data saved to {json_path}")
    
    return pickle_path, json_path

def load_existing_data(output_dir: str, force: bool = False) -> tuple[Optional[RawData], bool]:
    """
    Check for existing pickle files and prompt user to load one.
    
    Args:
        output_dir: Directory to search for pickle files
        force: If True, skip loading from pickle and return None
        
    Returns:
        tuple: (raw_data, loaded_data) where raw_data is the loaded data or None,
               and loaded_data is a boolean indicating if data was loaded
    """
    pkl_files = glob.glob(os.path.join(output_dir, "raw_data_*.pkl"))
    raw_data = None
    loaded_data = False
    
    if pkl_files and not force:
        # Sort files by modification time (newest first)
        pkl_files.sort(key=os.path.getmtime, reverse=True)
        
        # Display the available pickle files
        print("\nFound existing data files:")
        for i, file in enumerate(pkl_files[:5]):  # Show at most 5 most recent files
            file_time = datetime.fromtimestamp(os.path.getmtime(file), tz=DEFAULT_TIMEZONE)
            file_size = os.path.getsize(file) / (1024 * 1024)  # Size in MB
            print(f"{i+1}. {os.path.basename(file)} - {file_time.strftime('%Y-%m-%d %H:%M:%S')} ({file_size:.2f} MB)")
        
        # Ask user if they want to use an existing file
        response = input("\nUse existing data file? (y/yes or file number 1-5, n/no to use API): ").lower().strip()
        
        # Check if response is a number (direct file selection)
        file_choice = None
        if response.isdigit():
            file_choice = int(response)
            if 1 <= file_choice <= min(5, len(pkl_files)):
                use_existing = True
            else:
                print(f"Invalid file number. Must be between 1 and {min(5, len(pkl_files))}.")
                use_existing = False
        else:
            # Check if response is yes/y
            use_existing = response in ['y', 'yes']
        
        if use_existing:
            # If user didn't specify a file number directly but said yes
            if file_choice is None:
                file_choice = 1  # Default to most recent
                if len(pkl_files) > 1:
                    choice = input(f"Enter file number (1-{min(5, len(pkl_files))}), or press Enter for most recent: ").strip()
                    if choice and choice.isdigit():
                        file_choice = int(choice)
                        if file_choice < 1 or file_choice > min(5, len(pkl_files)):
                            file_choice = 1
                            print("Invalid choice, using most recent file.")
            
            selected_file = pkl_files[file_choice-1]
            print(f"Loading data from {selected_file}...")
            
            try:
                with open(selected_file, 'rb') as f:
                    raw_data = pickle.load(f)
                print(f"Successfully loaded data with {len(raw_data.users)} users.")
                loaded_data = True
                
                # Display date range from the loaded data
                start_dt = datetime.fromtimestamp(raw_data.earliest_message_timestamp, tz=DEFAULT_TIMEZONE)
                end_dt = datetime.fromtimestamp(raw_data.latest_message_timestamp, tz=DEFAULT_TIMEZONE)
                print(f"Data covers period: {start_dt.strftime('%Y-%m-%d')} to {end_dt.strftime('%Y-%m-%d')}")
            except Exception as e:
                print(f"Error loading pickle file: {str(e)}")
                print("Will fetch data from API instead.")
                raw_data = None
    
    return raw_data, loaded_data