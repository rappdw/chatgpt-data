#!/usr/bin/env python
"""Example script to fetch conversations from the Enterprise Compliance API."""

import os
import sys
import argparse
from datetime import datetime, timedelta
import pandas as pd

# Add the parent directory to the path so we can import the package
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from chatgpt_data.api.compliance_api import EnterpriseComplianceAPI


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Fetch conversations from the Enterprise Compliance API"
    )
    
    parser.add_argument(
        "--api-key", 
        help="Enterprise API key (defaults to OPENAI_ENTERPRISE_API_KEY env var)"
    )
    parser.add_argument(
        "--org-id", 
        help="Organization ID (defaults to OPENAI_ORG_ID env var)"
    )
    parser.add_argument(
        "--workspace-id", 
        help="Workspace ID (defaults to OPENAI_WORKSPACE_ID env var)"
    )
    parser.add_argument(
        "--output-file", 
        default="conversations.csv",
        help="Path to save the CSV file (default: conversations.csv)"
    )
    parser.add_argument(
        "--days", 
        type=int,
        default=7,
        help="Number of days to look back (default: 7)"
    )
    parser.add_argument(
        "--include-messages",
        action="store_true",
        help="Include message content in the CSV"
    )
    parser.add_argument(
        "--filter-users",
        nargs="+",
        help="List of user IDs to filter conversations by"
    )
    parser.add_argument(
        "--allow-mock-data",
        action="store_true",
        help="Allow fallback to mock data when API calls fail"
    )
    
    return parser.parse_args()


def main():
    """Main entry point for the script."""
    args = parse_args()
    
    # Get API credentials
    api_key = args.api_key or os.environ.get("API_KEY")
    org_id = args.org_id or os.environ.get("ORG_ID")
    workspace_id = args.workspace_id or os.environ.get("WORKSPACE_ID")
    
    # Validate required credentials
    if not api_key:
        print("ERROR: API key is required. Provide via --api-key or API_KEY env var")
        return 1
        
    if not org_id:
        print("WARNING: Organization ID not provided. Using default value")
        org_id = "org_default"
        
    if not workspace_id:
        print("WARNING: Workspace ID not provided. Using default value")
        workspace_id = "05a09bbb-00b5-4224-bee8-739bb86ec062"
    
    # Calculate timestamp for filtering
    since_date = datetime.now() - timedelta(days=args.days)
    since_timestamp = int(since_date.timestamp())
    print(f"Fetching conversations since: {since_date.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Initialize API client
    api = EnterpriseComplianceAPI(
        api_key=api_key,
        org_id=org_id,
        workspace_id=workspace_id,
        allow_mock_data=args.allow_mock_data
    )
    
    # Prepare CSV data
    conversations_data = []
    
    def process_conversation(conversation):
        """Process a conversation and add it to the CSV data."""
        # Extract basic conversation metadata
        conv_data = {
            "id": conversation.get("id"),
            "workspace_id": conversation.get("workspace_id"),
            "user_id": conversation.get("user_id"),
            "user_email": conversation.get("user_email"),
            "created_at": conversation.get("created_at"),
            "last_active_at": conversation.get("last_active_at"),
            "title": conversation.get("title"),
            "message_count": len(conversation.get("messages", {}).get("data", [])),
        }
        
        # Add message data if requested
        if args.include_messages:
            messages = conversation.get("messages", {}).get("data", [])
            
            # Add first user message if available
            user_messages = [m for m in messages if m.get("role") == "user"]
            if user_messages:
                first_user_msg = user_messages[0]
                conv_data["first_user_message"] = extract_message_text(first_user_msg)
                conv_data["first_user_message_time"] = first_user_msg.get("created_at")
            else:
                conv_data["first_user_message"] = ""
                conv_data["first_user_message_time"] = ""
            
            # Add first assistant message if available
            assistant_messages = [m for m in messages if m.get("role") == "assistant"]
            if assistant_messages:
                first_assistant_msg = assistant_messages[0]
                conv_data["first_assistant_message"] = extract_message_text(first_assistant_msg)
                conv_data["first_assistant_message_time"] = first_assistant_msg.get("created_at")
            else:
                conv_data["first_assistant_message"] = ""
                conv_data["first_assistant_message_time"] = ""
        
        conversations_data.append(conv_data)
    
    # Helper function to extract message text
    def extract_message_text(message):
        """Extract text content from a message."""
        content = message.get("content", {})
        if content.get("content_type") == "text":
            parts = content.get("parts", [])
            if parts:
                return parts[0]
        return ""
    
    try:
        # Process all conversations
        conversation_count = api.process_all_conversations(
            callback_fn=process_conversation,
            since_timestamp=since_timestamp,
            users=args.filter_users
        )
        
        print(f"Found {conversation_count} conversations")
        
        # Write to CSV
        if conversations_data:
            df = pd.DataFrame(conversations_data)
            df.to_csv(args.output_file, index=False)
            print(f"Saved {len(conversations_data)} conversations to {args.output_file}")
        else:
            # Create empty CSV with headers
            columns = ["id", "workspace_id", "user_id", "user_email", "created_at", 
                    "last_active_at", "title", "message_count"]
            if args.include_messages:
                columns.extend(["first_user_message", "first_user_message_time", 
                            "first_assistant_message", "first_assistant_message_time"])
            
            pd.DataFrame(columns=columns).to_csv(args.output_file, index=False)
            print(f"No conversations found. Created empty CSV at {args.output_file}")
        
        return 0
    
    except Exception as e:
        print(f"Error: {str(e)}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
