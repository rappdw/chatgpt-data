"""CLI tool to fetch ChatGPT usage reports via the Enterprise Compliance API."""

import argparse
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional, Union, Tuple, Dict, Set, Any
from dataclasses import dataclass, field
import csv
import re

from chatgpt_data.api.compliance_api import EnterpriseComplianceAPI
from chatgpt_data.cli.all_trends import main as run_all_trends
from dotenv import load_dotenv


def parse_date(date_str: str) -> datetime:
    """Parse a date string in YYYY-MM-DD format.

    Args:
        date_str: Date string to parse

    Returns:
        Parsed datetime object
    """
    try:
        return datetime.strptime(date_str, "%Y-%m-%d")
    except ValueError:
        raise argparse.ArgumentTypeError(f"Invalid date format: {date_str}. Use YYYY-MM-DD")


def get_default_dates() -> Tuple[datetime, datetime]:
    """Get default start and end dates for reports.

    Returns:
        Tuple of (start_date, end_date) as datetime objects
    """
    # Default end date is today
    end_date = datetime.now()
    
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


def map_user_status(status: str) -> str:
    """Map API status values to the expected values.
    
    Args:
        status: Status value from the API
        
    Returns:
        Mapped status value (enabled, pending, deleted)
    """
    status_mapping = {
        "active": "enabled",
        "enabled": "enabled",
        "pending": "pending",
        "deleted": "deleted",
        "disabled": "deleted",
        "suspended": "deleted",
    }
    
    return status_mapping.get(status.lower(), "enabled")  # Default to enabled if unknown


@dataclass
class UserActivity:
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
class EngagementMetrics:
    users: Dict[str, Any]
    user_activity: Dict[str, UserActivity] = field(default_factory=dict)
    gpts: Dict[str, Any] = field(default_factory=dict)
    projects: Dict[str, Any] = field(default_factory=dict)


def get_user_engagement(api: EnterpriseComplianceAPI, start_date: datetime, end_date: datetime, debug_logging: bool = False) -> EngagementMetrics:
    """
    Get user engagement metrics by processing all conversations.
    
    Args:
        api: The EnterpriseComplianceAPI instance
        start_date: The start date for the engagement period
        end_date: The end date for the engagement period
        debug_logging: Whether to print detailed debug logs
    
    Returns:
        An EngagementMetrics object with user engagement data
    """
    print("\nCalculating user engagement metrics...")
    
    # Get all users
    users_dict = api.get_all_users()
    print(f"Retrieved {len(users_dict)} users for engagement analysis")

    # Initialize engagement metrics
    engagement_metrics = EngagementMetrics(users=users_dict)
    
    # Determine timestamp filter if start_date is provided
    since_timestamp = datetime_to_unix_timestamp(start_date)
    before_timestamp = datetime_to_unix_timestamp(end_date)
    
    if debug_logging:
        print(f"Date range: {start_date.isoformat()} to {end_date.isoformat()}")
        print(f"Timestamp range: {since_timestamp} to {before_timestamp}")
    
    # Track conversation and message counts for verification
    total_conversations = 0
    total_conversations_in_range = 0
    total_messages = 0
    total_messages_in_range = 0
    
    # Process all conversations with a callback function
    def process_conversation(conversation):
        nonlocal total_conversations, total_conversations_in_range, total_messages, total_messages_in_range
        
        total_conversations += 1
        conversation_id = conversation.get("id", "unknown")
        user_id = conversation.get("user_id")
        last_active = conversation.get("last_active_at")
        
        if debug_logging:
            print(f"\nProcessing conversation: {conversation_id} for user: {user_id}")
            print(f"  Last active: {last_active} ({datetime.fromtimestamp(last_active).isoformat() if last_active else 'None'})")
        
        # Initialize user metrics if not already present
        if user_id not in engagement_metrics.user_activity:
            engagement_metrics.user_activity[user_id] = UserActivity()
            
        # Update user metrics
        user_metrics = engagement_metrics.user_activity[user_id]

        # Get messages
        messages = conversation.get("messages", {}).get("data", [])
        
        if debug_logging:
            print(f"  Found {len(messages)} messages in conversation")
        
        convo_in_range = False
        # iterate messages, skip any that are not in the specified date range
        for msg in messages:
            total_messages += 1
            
            msg_id = msg.get("id", "unknown")
            created_at = msg.get("created_at")
            
            if debug_logging:
                print(f"  Message {msg_id} created at: {created_at} ({datetime.fromtimestamp(created_at).isoformat() if created_at else 'None'})")
            
            if not created_at or created_at < since_timestamp or created_at > before_timestamp:
                if debug_logging:
                    print(f"    Skipping message: outside date range")
                continue
                
            total_messages_in_range += 1
            convo_in_range = True
            user_metrics.message_count += 1
            
            # Update first and last active timestamps
            if user_metrics.first_day_active == 0 or created_at < user_metrics.first_day_active:
                user_metrics.first_day_active = created_at
                
            if user_metrics.last_day_active < created_at:
                user_metrics.last_day_active = created_at

            gpt_id = msg.get("gpt_id")
            project_id = msg.get("project_id")
            role = msg.get("author", {}).get("role")
            tool_name = msg.get("author", {}).get("tool_name")
            
            if debug_logging:
                print(f"    Message role: {role}, GPT: {gpt_id}, Project: {project_id}, Tool: {tool_name}")
            
            # Explicitly track user messages
            if role == "user":
                user_metrics.user_message_count += 1
            elif role == "assistant":
                user_metrics.assistant_message_count += 1
            elif role == "system":
                user_metrics.system_message_count += 1
            elif role == "tool":
                user_metrics.tool_message_count += 1
                if tool_name:
                    user_metrics.tool_set.add(tool_name)
                else:
                    user_metrics.tool_set.add("unknown")
            else:
                print(f"    Unknown message role, '{role}'. msg: '{msg}'\n\n")
            if gpt_id:
                # if role != "user":
                #     print(f" Non-user gpt message. role, '{role}'\nmsg: {msg}\n\n")
                user_metrics.gpt_message_count += 1
                user_metrics.gpt_set.add(gpt_id)
                if gpt_id not in engagement_metrics.gpts:
                    gpt_builder_name = api.get_gpt_name(gpt_id)
                    engagement_metrics.gpts[gpt_id] = gpt_builder_name
            if project_id:
                # if role != "user":
                #     print(f" Non-user project message. role, '{role}'\nmsg: {msg}\n\n")
                user_metrics.project_message_count += 1
                user_metrics.project_set.add(project_id)
                if project_id not in engagement_metrics.projects:
                    project_name = api.get_project_name(project_id)
                    engagement_metrics.projects[project_id] = project_name

        
        if convo_in_range:
            total_conversations_in_range += 1
            user_metrics.conversation_count += 1
            if debug_logging:
                print(f"  Conversation in range: Yes")
        elif debug_logging:
            print(f"  Conversation in range: No")
    
    # Process all conversations
    api.process_all_conversations(
        callback_fn=process_conversation,
        since_timestamp=since_timestamp,
        debug_logging=debug_logging
    )
    
    # Print summary statistics for verification
    print(f"\nProcessed {total_conversations} total conversations ({total_conversations_in_range} in date range)")
    print(f"Processed {total_messages} total messages ({total_messages_in_range} in date range)")
    print(f"Found {len(engagement_metrics.user_activity)} users with activity in the date range")
    
    return engagement_metrics


def save_engagement_metrics_to_csv(metrics: EngagementMetrics, output_file: str, end_date: Optional[datetime] = None, start_date: Optional[datetime] = None) -> None:
    """Save user engagement metrics to a CSV file.
    
    Args:
        metrics: EngagementMetrics object with user activity data
        output_file: Path to output CSV file
        end_date: Optional end date of the reporting period. If provided, users created after this date
                 will be excluded from the non-engaged users report.
        start_date: Optional start date of the reporting period. Used for determining cadence.
        
    This function creates two CSV files:
    1. The main engagement metrics file with the name provided in output_file
    2. A non-engagement file for users who had no activity during the reporting period
       with the name pattern: non_engagement_YYYYMMDD_YYYYMMDD.csv
    """
    # Create rows for active users
    active_rows = []
    
    # Convert user activity data to rows
    for user_id, activity in metrics.user_activity.items():
        # Get user info - handle the case when user might be None
        user = metrics.users.get(user_id)
        
        # Create display name (use name if available, otherwise email)
        display_name = getattr(user, 'name', "") if user else ""
        if not display_name and user:
            display_name = getattr(user, 'email', "")
        
        # Convert GPT IDs to names
        gpt_names = [metrics.gpts.get(gpt_id, str(gpt_id)) for gpt_id in activity.gpt_set if gpt_id is not None]
        
        # Convert project IDs to names
        project_names = [metrics.projects.get(project_id, str(project_id)) for project_id in activity.project_set if project_id is not None]
        
        # Convert tool IDs to names (no mapping needed)
        tool_names = [str(tool_id) for tool_id in activity.tool_set if tool_id is not None]
        
        # Get last active timestamp
        last_active_timestamp = activity.last_day_active
        last_active_str = datetime.fromtimestamp(last_active_timestamp).strftime("%Y-%m-%d") if last_active_timestamp else ""
        
        # Get first active timestamp
        first_active_timestamp = activity.first_day_active
        first_active_str = datetime.fromtimestamp(first_active_timestamp).strftime("%Y-%m-%d") if first_active_timestamp else ""
        
        # Get created_at timestamp
        created_at = getattr(user, 'created_at', None) if user else None
        created_at_str = datetime.fromtimestamp(created_at).strftime("%Y-%m-%d") if created_at else ""
        
        # Determine cadence based on start_date and end_date
        cadence = determine_cadence(start_date, end_date) if start_date and end_date else "unknown"
        
        # Get organization ID (account_id)
        account_id = getattr(user, 'organization_id', "") if user else ""
        
        # Format period start and end dates
        period_start = start_date if start_date else None
        period_end = end_date if end_date else None
        period_start_str = period_start.strftime("%Y-%m-%d") if period_start else ""
        period_end_str = period_end.strftime("%Y-%m-%d") if period_end else ""
        
        # Map user status to expected values
        user_status = getattr(user, 'status', "") if user else ""
        mapped_status = map_user_status(user_status)
        
        row = {
            "cadence": cadence,
            "period_start": period_start_str,
            "period_end": period_end_str,
            "account_id": account_id,
            "public_id": user_id,
            "name": display_name,
            "email": getattr(user, 'email', "") if user else "",
            "role": "",
            "user_role": getattr(user, 'role', "") if user else "",
            "department": "",  # Empty string as requested
            "user_status": mapped_status,
            "created_or_invited_date": created_at_str,
            "is_active": 1,  # User has messages in the period
            "first_day_active_in_period": first_active_str,
            "last_day_active_in_period": last_active_str,
            "messages": activity.user_message_count,
            "message_rank": 0,
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
            "gpt_models": ",".join(gpt_names) if gpt_names else "",
            "tools": ",".join(tool_names) if tool_names else "",
            "projects": ",".join(project_names) if project_names else "",
        }
        active_rows.append(row)
    
    # Sort rows by messages (descending)
    active_rows.sort(key=lambda x: x["messages"], reverse=True)
    
    # Add message rank based on sort order
    for i, row in enumerate(active_rows):
        row["message_rank"] = i + 1  # 1-based ranking
    
    # Write to CSV
    if active_rows:
        with open(output_file, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=active_rows[0].keys())
            writer.writeheader()
            writer.writerows(active_rows)
        print(f"Saved engagement metrics for {len(active_rows)} users to {output_file}")
    else:
        print("No user engagement data to save")
    
    # Create a second CSV for non-engaged users
    # Extract date range from the output filename for the non-engagement filename
    base_filename = os.path.basename(output_file)
    date_parts = re.findall(r'\d{8}', base_filename)
    date_suffix = "_".join(date_parts) if date_parts else "report"
    
    non_engagement_file = os.path.join(os.path.dirname(output_file), f"non_engagement_{date_suffix}.csv")
    
    # Calculate end timestamp if end_date is provided
    end_timestamp = None
    if end_date:
        end_timestamp = datetime_to_unix_timestamp(end_date)
    
    # Find users who were not active during the reporting period
    non_engaged_rows = []
    for user_id, user in metrics.users.items():
        # Skip users who were created after the reporting period
        if end_timestamp and hasattr(user, 'created_at') and user.created_at > end_timestamp:
            continue
            
        if user_id not in metrics.user_activity:
            # Create display name (use name if available, otherwise email)
            display_name = getattr(user, 'name', "") if user else ""
            if not display_name and user:
                display_name = getattr(user, 'email', "")
                
            # Get created_at timestamp
            created_at = getattr(user, 'created_at', None) if user else None
            created_at_str = datetime.fromtimestamp(created_at).strftime("%Y-%m-%d") if created_at else ""
            
            # Determine cadence based on start_date and end_date
            cadence = determine_cadence(start_date, end_date) if start_date and end_date else "unknown"
            
            # Get organization ID (account_id)
            account_id = getattr(user, 'organization_id', "") if user else ""
            
            # Format period start and end dates
            period_start = start_date if start_date else None
            period_end = end_date if end_date else None
            period_start_str = period_start.strftime("%Y-%m-%d") if period_start else ""
            period_end_str = period_end.strftime("%Y-%m-%d") if period_end else ""
            
            # Map user status to expected values
            user_status = getattr(user, 'status', "") if user else ""
            mapped_status = map_user_status(user_status)
            
            row = {
                "cadence": cadence,
                "period_start": period_start_str,
                "period_end": period_end_str,
                "account_id": account_id,
                "public_id": user_id,
                "name": display_name,
                "email": getattr(user, 'email', "") if user else "",
                "role": "",
                "user_role": getattr(user, 'role', "") if user else "",
                "department": "",  # Empty string as requested
                "user_status": mapped_status,
                "created_or_invited_date": created_at_str,
                "is_active": 0,  # User has no messages in the period
                "first_day_active_in_period": "",
                "last_day_active_in_period": "",
                "messages": 0,
                "message_rank": 0,
                "gpt_messages": 0,
                "gpts_messaged": 0,
                "tool_messages": 0,
                "tools_messaged": 0,
                "last_day_active": "",
                "project_messages": 0,
                "projects_messaged": 0,
                "conversation_count": 0,
                "user_message_count": 0,
                "assistant_message_count": 0,
                "system_message_count": 0,
                "gpt_models": "",
                "tools": "",
                "projects": "",
            }
            non_engaged_rows.append(row)
    
    # Sort non-engaged rows by name
    non_engaged_rows.sort(key=lambda x: x["name"])
    
    # Write non-engaged users to CSV
    if non_engaged_rows:
        with open(non_engagement_file, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=non_engaged_rows[0].keys())
            writer.writeheader()
            writer.writerows(non_engaged_rows)
        print(f"Saved non-engagement data for {len(non_engaged_rows)} users to {non_engagement_file}")
    else:
        print("No non-engaged users to report")


def main() -> None:
    """Main entry point for the CLI tool."""
    # Load environment variables from .env file if it exists
    load_dotenv()
    
    parser = argparse.ArgumentParser(
        description="Fetch ChatGPT usage reports via the Enterprise Compliance API"
    )
    
    # Data options
    data_group = parser.add_argument_group("Data Options")
    data_group.add_argument(
        "--workspace-id",
        help="Workspace ID to fetch data from (default: from environment variable)",
    )
    data_group.add_argument(
        "--api-key",
        help="Enterprise API key with compliance_export scope (default: from environment variable)",
    )
    data_group.add_argument(
        "--org-id",
        help="Organization ID (default: from environment variable)",
    )
    data_group.add_argument(
        "--start-date",
        type=parse_date,
        help="Start date for reports (YYYY-MM-DD, default: 7 days ago)",
    )
    data_group.add_argument(
        "--end-date",
        type=parse_date,
        help="End date for reports (YYYY-MM-DD, default: today)",
    )
    data_group.add_argument(
        "--output-dir",
        default="./reports",
        help="Directory to save reports (default: ./reports)",
    )
    data_group.add_argument(
        "--test",
        action="store_true",
        help="Run in test mode with mock data",
    )
    data_group.add_argument(
        "--debug",
        action="store_true",
        help="Enable detailed debug logging for verification",
    )
    
    # Analysis options
    analysis_group = parser.add_argument_group("Analysis Options")
    analysis_group.add_argument(
        "--run-analysis",
        action="store_true",
        help="Run data analysis after downloading reports"
    )
    analysis_group.add_argument(
        "--analysis-output-dir",
        default="./data",
        help="Directory to save analysis output (default: ./data)"
    )
    
    # Add page_size parameter
    parser.add_argument(
        "--page-size",
        type=int,
        default=200,
        help="Number of items to request per API page (default: 200, max: 200)"
    )
    
    args = parser.parse_args()
    
    # Run the command
    try:
        # Initialize API client
        api_key = args.api_key or os.environ.get("API_KEY")
        org_id = args.org_id or os.environ.get("ORG_ID")
        workspace_id = args.workspace_id or os.environ.get("WORKSPACE_ID")
        
        # Validate required credentials
        missing_credentials = False
        if not api_key:
            print("ERROR: API key is required. Provide via --api-key or API_KEY env var")
            missing_credentials = True
            
        if not org_id:
            print("WARNING: Organization ID not provided. Using default value")
            org_id = "org_default"
            
        if not workspace_id:
            print("WARNING: Workspace ID not provided. Using default value")
            workspace_id = "05a09bbb-00b5-4224-bee8-739bb86ec062"
        
        if missing_credentials:
            print("\nExiting due to missing credentials")
            return 1
        
        # Initialize the API client
        api = EnterpriseComplianceAPI(
            api_key=api_key,
            org_id=org_id,
            workspace_id=workspace_id,
            output_dir=args.output_dir,
            page_size=min(args.page_size, 200)  # Ensure page_size doesn't exceed API limit
        )
        
        # Print date range
        print(f"Date range: {args.start_date.strftime('%Y-%m-%d')} to {args.end_date.strftime('%Y-%m-%d')}")
        
        # Download reports
        downloaded_files = []
        
        # Get users data
        try:
            # Get user engagement metrics
            engagement_metrics = get_user_engagement(api, args.start_date, args.end_date, debug_logging=args.debug)
            
            # Save engagement metrics to CSV
            output_file = os.path.join(args.output_dir, f"user_engagement_{args.start_date.strftime('%Y%m%d')}_{args.end_date.strftime('%Y%m%d')}.csv")
            save_engagement_metrics_to_csv(
                metrics=engagement_metrics,
                output_file=output_file,
                end_date=args.end_date,
                start_date=args.start_date
            )
            downloaded_files.append(output_file)
            
        except Exception as e:
            print(f"Error fetching workspace users: {str(e)}")
            print("This could be due to:")
            print("  - Invalid API credentials")
            print("  - Insufficient permissions for the compliance API")
            print("  - Network connectivity issues")
            print("  - API rate limiting")
        
        # Run analysis if requested
        if args.run_analysis and downloaded_files:
            print("Running data analysis...")
            try:
                # Use modified sys.argv to call all_trends with the right parameters
                import sys
                original_argv = sys.argv.copy()
                sys.argv = [
                    "all_trends",
                    "--data-dir", args.output_dir,
                    "--output-dir", args.analysis_output_dir
                ]
                
                run_all_trends()
                
                # Restore original argv
                sys.argv = original_argv
                
                print(f"Analysis complete. Results saved to {args.analysis_output_dir}")
            except Exception as e:
                print(f"Error running analysis: {str(e)}")
        
        if downloaded_files:
            print("\nDownloaded files:")
            for file_path in downloaded_files:
                print(f"  - {file_path}")
        
        print("\nNote: If you're experiencing API rate limiting, try using a smaller page size:")
        print("  python -m chatgpt_data.cli.api_reports --page-size 50")
        
        return 0
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return 1


if __name__ == "__main__":
    main()