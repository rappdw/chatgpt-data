"""ChatGPT Enterprise Compliance API client."""

import os
import sys
import time
import json
import logging
from datetime import datetime, timedelta
from chatgpt_data.utils.constants import DEFAULT_TIMEZONE, MESSAGE_TIMESTAMP_TOLERANCE, EARLIEST_EXPECTED_TIMESTAMP
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any, Iterator
import csv
import requests
from dataclasses import dataclass, field
import pandas as pd
from urllib.parse import urljoin
from tqdm import tqdm


# Configure loggers
# Logger for API consistency issues
api_consistency_logger = logging.getLogger('api_consistency_issues')
api_consistency_logger.setLevel(logging.WARNING)

# Create a file handler if not already added
if not api_consistency_logger.handlers:
    file_handler = logging.FileHandler('api_consistency_issues.log')
    file_handler.setLevel(logging.WARNING)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    api_consistency_logger.addHandler(file_handler)

# Logger for general API client operations
logger = logging.getLogger('compliance_api')
logger.setLevel(logging.WARNING)

# Create console handler if not already added
if not logger.handlers:
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.WARNING)
    formatter = logging.Formatter('%(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)


@dataclass
class User:
    """Represents a user in the ChatGPT Enterprise system."""
    user_id: str
    email: str
    name: str
    role: str
    created_at: float
    status: str
    conversations: List[str] = field(default_factory=list)

@dataclass
class GPT:
    """Represents a GPT model in the ChatGPT Enterprise system."""
    id: str
    name: str
    description: str
    created_at: float
    creator_id: str
    creator_email: str

@dataclass
class Project:
    """Represents a project in the ChatGPT Enterprise system."""
    id: str
    name: str
    description: str
    created_at: float
    creator_id: str
    creator_email: str





class EnterpriseComplianceAPI:
    """Client for the ChatGPT Enterprise Compliance API.
    
    This client implements the OpenAPI specification for the ChatGPT Enterprise Compliance API.
    It provides methods to interact with the API endpoints for retrieving and managing
    compliance data from a ChatGPT Enterprise workspace.
    """
    
    BASE_URL = "https://api.chatgpt.com/v1/"
    
    def __init__(
        self, 
        api_key: str, 
        org_id: str, 
        workspace_id: str,
        output_dir: Union[str, Path] = "./rawdata",
        allow_mock_data: bool = False,
        page_size: int = 200  # Default page size as per API spec
    ):
        """Initialize the API client.

        Args:
            api_key: Enterprise API key with compliance_export scope
            org_id: Organization ID
            workspace_id: Workspace ID to fetch data from
            output_dir: Directory to save downloaded reports
            allow_mock_data: If True, allow fallback to mock data when API calls fail
            page_size: Number of items to request per page (default: 200)
        """
        self.api_key = api_key
        self.org_id = org_id
        self.workspace_id = workspace_id
        self.allow_mock_data = allow_mock_data
        self.page_size = min(page_size, 200)  # Ensure page_size doesn't exceed API limit
        
        # Create output directory if it doesn't exist
        self.output_dir = Path(output_dir)
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Set up headers for API requests
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "OpenAI-Organization": self.org_id
        }
        
        logger.info(f"Initialized Enterprise Compliance API client for workspace: {self.workspace_id}")
        logger.info(f"Output directory: {self.output_dir}")
        logger.info(f"Page size: {self.page_size}")
        logger.info(f"Mock data fallback: {'Enabled' if self.allow_mock_data else 'Disabled'}")
        
        # Check if we're in test mode
        if api_key == "test_api_key" or api_key.startswith("test_"):
            logger.info("Running in test mode with mock data")
            self.allow_mock_data = True
    
    def _make_request(
        self, 
        endpoint: str, 
        method: str = "GET", 
        params: Optional[Dict[str, Any]] = None, 
        json_data: Optional[Dict[str, Any]] = None, 
        timeout: int = 30,
        _retry_count: int = 0  # Internal parameter to track retry attempts
    ) -> Dict[str, Any]:
        """Make a request to the Enterprise API.

        Args:
            endpoint: API endpoint to call (without base URL)
            method: HTTP method to use
            params: Query parameters
            json_data: JSON data for POST requests
            timeout: Request timeout in seconds
            _retry_count: Internal parameter to track retry attempts

        Returns:
            API response as a dictionary

        Raises:
            Exception: If the API request fails after retries
            SystemExit: If server errors persist after 3 retries
        """
        url = urljoin(self.BASE_URL, endpoint)
        max_retries = 3
        
        try:
            response = requests.request(
                method, 
                url, 
                headers=self.headers, 
                params=params, 
                json=json_data, 
                timeout=timeout
            )
            
            # Check for 404 errors on GPT and Project endpoints
            if response.status_code == 404:
                # Check if this is a GPT or project endpoint
                if '/gpts/' in endpoint or '/projects/' in endpoint:
                    resource_type = "GPT" if '/gpts/' in endpoint else "Project"
                    resource_id = endpoint.split('/')[-1].split('?')[0]  # Extract ID from the endpoint
                    # Log the missing resource instead of raising an exception
                    api_consistency_logger.warning(f"Missing {resource_type}: {resource_id} not found in workspace {self.workspace_id}")
                    # Return a standardized response for missing resources
                    return {"detail": f"{resource_type} not found"}
            
            # Raise an exception for other 4XX/5XX status codes
            response.raise_for_status()
            
            return response.json()
        except requests.exceptions.HTTPError as e:
            # Handle HTTP errors (4xx, 5xx)
            # We no longer need to handle simplified 404 errors here since we catch them earlier
                
            # Handle 400 Bad Request errors for invalid IDs
            if e.response.status_code == 400:
                response_text = e.response.text if hasattr(e.response, 'text') else ""
                
                # Check if this is an invalid GPT or project ID error
                if "Invalid gpt_id" in response_text and "/gpts/" in endpoint:
                    gpt_id = endpoint.split('/gpts/')[-1].split('/')[0]
                    logger.warning(f"Invalid GPT ID format: {gpt_id}")
                    raise requests.exceptions.HTTPError(f"GPT not found", response=e.response)
                    
                elif "Invalid project_id" in response_text and "/projects/" in endpoint:
                    project_id = endpoint.split('/projects/')[-1].split('/')[0]
                    logger.warning(f"Invalid project ID format: {project_id}")
                    raise requests.exceptions.HTTPError(f"Project not found", response=e.response)
            
            # Check if this is a server error (5xx) that should be retried
            is_server_error = e.response.status_code >= 500
            
            # For other errors, provide detailed information
            error_message = f"HTTP Error: {e}"
            
            # Try to get more detailed error information from the response
            try:
                error_content = e.response.json()
                if "error" in error_content and isinstance(error_content["error"], dict):
                    if "message" in error_content["error"]:
                        error_message += f"\nError message: {error_content['error']['message']}"
                    if "type" in error_content["error"]:
                        error_message += f"\nError type: {error_content['error']['type']}"
                else:
                    error_message += f"\nResponse: {error_content}"
            except:
                # If we can't parse the JSON, just include the text
                if hasattr(e.response, 'text') and e.response.text:
                    error_message += f"\nResponse text: {e.response.text[:200]}..."
            
            logger.error(error_message)
            
            # Include request details in the error message
            if params:
                logger.debug(f"Request parameters: {params}")
            if json_data:
                logger.debug(f"JSON Data: {json_data}")
            
            # Handle retry logic for server errors
            if is_server_error and _retry_count < max_retries:
                _retry_count += 1
                retry_delay = 2 ** _retry_count  # Exponential backoff: 2, 4, 8 seconds
                logger.warning(f"Server error encountered. Retrying in {retry_delay} seconds... (Attempt {_retry_count}/{max_retries})")
                import time
                time.sleep(retry_delay)
                return self._make_request(
                    endpoint=endpoint,
                    method=method,
                    params=params,
                    json_data=json_data,
                    timeout=timeout,
                    _retry_count=_retry_count
                )
            elif is_server_error and _retry_count >= max_retries:
                logger.error(f"Server error persisted after {max_retries} retry attempts. Exiting program.")
                import sys
                sys.exit(1)  # Exit with error code 1
                
            raise Exception(f"API request failed: {error_message}")
        except requests.exceptions.Timeout:
            logger.error(f"Request timed out after {timeout} seconds")
            logger.error(f"URL: {url}")
            logger.error(f"Method: {method}")
            
            # Handle retry logic for timeouts
            if _retry_count < max_retries:
                _retry_count += 1
                retry_delay = 2 ** _retry_count  # Exponential backoff
                logger.warning(f"Request timed out. Retrying in {retry_delay} seconds... (Attempt {_retry_count}/{max_retries})")
                import time
                time.sleep(retry_delay)
                return self._make_request(
                    endpoint=endpoint,
                    method=method,
                    params=params,
                    json_data=json_data,
                    timeout=timeout,
                    _retry_count=_retry_count
                )
            elif _retry_count >= max_retries:
                logger.error(f"Request timeout persisted after {max_retries} retry attempts. Exiting program.")
                import sys
                sys.exit(1)  # Exit with error code 1
                
            raise Exception(f"Request timed out after {timeout} seconds")
        except requests.exceptions.RequestException as e:
            # Handle other request exceptions (connection errors, etc.)
            logger.error(f"Request Error: {str(e)}")
            logger.error(f"URL: {url}")
            
            # Handle retry logic for general request exceptions
            if _retry_count < max_retries:
                _retry_count += 1
                retry_delay = 2 ** _retry_count  # Exponential backoff
                logger.warning(f"Request error encountered. Retrying in {retry_delay} seconds... (Attempt {_retry_count}/{max_retries})")
                import time
                time.sleep(retry_delay)
                return self._make_request(
                    endpoint=endpoint,
                    method=method,
                    params=params,
                    json_data=json_data,
                    timeout=timeout,
                    _retry_count=_retry_count
                )
            elif _retry_count >= max_retries:
                logger.error(f"Request error persisted after {max_retries} retry attempts. Exiting program.")
                import sys
                sys.exit(1)  # Exit with error code 1
                
            raise Exception(f"Request failed: {str(e)}")
    
    def get_users(self, after: Optional[str] = None, limit: Optional[int] = None) -> Dict[str, Any]:
        """Get users from the workspace.
        
        This method fetches users from the workspace according to the OpenAPI spec.
        It returns the raw API response, which includes pagination information.
        
        Args:
            after: The previous user ID from which to fetch the next page of users
            limit: The maximum number of users to list (default: self.page_size)
            
        Returns:
            Raw API response containing user data and pagination information
            
        Raises:
            Exception: If the API request fails and mock data is not allowed
        """
        try:
            
            # Prepare request parameters
            params = {
                "limit": limit or self.page_size
            }
            if after:
                params["after"] = after
            
            # Make the API request
            endpoint = f"compliance/workspaces/{self.workspace_id}/users"
            response = self._make_request(
                endpoint=endpoint,
                method="GET",
                params=params
            )
            
            return response
            
        except Exception as e:
            logger.error(f"API request failed when fetching users: {str(e)}")
            
            if self.allow_mock_data:
                logger.info("Using mock user data for testing...")
                
                # Create mock user data
                mock_users = []
                for i in range(10):
                    user_id = f"user-mock{i}"
                    mock_users.append({
                        "object": "compliance.workspace.user",
                        "id": user_id,
                        "email": f"user{i}@example.com",
                        "name": f"Test User {i}",
                        "role": "standard-user" if i > 0 else "account-owner",
                        "created_at": datetime.now(DEFAULT_TIMEZONE).timestamp(),
                        "status": "active"
                    })
                
                # Create mock response with pagination info
                mock_response = {
                    "object": "list",
                    "data": mock_users,
                    "has_more": False,
                    "last_id": mock_users[-1]["id"] if mock_users else None
                }
                
                logger.info(f"Generated {len(mock_users)} mock users")
                return mock_response
            else:
                # If mock data is not allowed, raise the exception
                raise Exception(f"Users endpoint failed: {str(e)}")
    
    def get_all_users(self, use_tqdm: bool = False) -> Dict[str, User]:
        """Get all users from the workspace with automatic pagination.
        
        This method handles pagination automatically and returns a dictionary of User objects
        keyed by user_id.
        
        Args:
            use_tqdm: If True, show a progress bar
        
        Returns:
            Dictionary of User objects keyed by user_id
            
        Raises:
            Exception: If the API request fails and mock data is not allowed
        """
        users = {}
        after = None
        has_more = True
        
        # First, get the first page to determine total count if possible
        first_response = self.get_users(after=after)
        total_users = first_response.get("total", 0)
        
        # Process users from the first page
        for user_data in first_response.get("data", []):
            user = User(
                user_id=user_data.get("id", ""),
                email=user_data.get("email", ""),
                name=user_data.get("name", ""),
                role=user_data.get("role", ""),
                created_at=user_data.get("created_at", 0),
                status=user_data.get("status", "")
            )
            users[user.user_id] = user
        
        # Check if we need to fetch more pages
        has_more = first_response.get("has_more", False)
        after = first_response.get("last_id")
        
        # Create progress bar if requested
        pbar = None
        if use_tqdm:
            pbar = tqdm(desc="Fetching users", unit="user")
            pbar.update(len(users))
        
        while has_more:
            response = self.get_users(after=after)
            
            # Process users from this page
            new_users = response.get("data", [])
            for user_data in new_users:
                user = User(
                    user_id=user_data.get("id", ""),
                    email=user_data.get("email", ""),
                    name=user_data.get("name", ""),
                    role=user_data.get("role", ""),
                    created_at=user_data.get("created_at", 0),
                    status=user_data.get("status", "")
                )
                users[user.user_id] = user
            
            # Update progress bar
            if pbar is not None:
                pbar.update(len(new_users))
            
            # Check if we need to fetch more pages
            has_more = response.get("has_more", False)
            after = response.get("last_id")
        
        # Close progress bar
        if pbar is not None:
            pbar.close()
        
        return users
    
    def _validate_conversation_timestamps(self, conversations: List[Dict[str, Any]]) -> None:
        """Validate and correct timestamps in conversation data.
        
        This ensures that message timestamps are always within the conversation's
        created_at and last_active_at boundaries. Any inconsistencies are logged to
        'api_consistency_issues.log' file.
        
        Args:
            conversations: List of conversation data from the API
        """
        for conversation in conversations:
            conv_id = conversation.get('id', 'unknown')
            
            # Skip if missing required timestamp fields
            if "created_at" not in conversation or "last_active_at" not in conversation:
                api_consistency_logger.warning(
                    f"Conversation {conv_id} missing required timestamp fields"
                )
                continue
                
            conv_created_at = conversation["created_at"]
            conv_last_active_at = conversation["last_active_at"]
            
            # Check if conversation predates the earliest expected timestamp
            if conv_created_at < EARLIEST_EXPECTED_TIMESTAMP:
                api_consistency_logger.warning(
                    f"Conversation {conv_id} has suspiciously early creation timestamp: "
                    f"{datetime.fromtimestamp(conv_created_at, DEFAULT_TIMEZONE)} "
                    f"(before {datetime.fromtimestamp(EARLIEST_EXPECTED_TIMESTAMP, DEFAULT_TIMEZONE)})"
                )
            
            # Check if conversation timestamps are invalid
            if conv_created_at > conv_last_active_at:
                # Check if the timestamp difference is within the tolerance window
                time_difference = conv_created_at - conv_last_active_at
                if time_difference > MESSAGE_TIMESTAMP_TOLERANCE:
                    api_consistency_logger.warning(
                        f"Conversation {conv_id} has invalid timestamps: "
                        f"created_at ({conv_created_at}) > last_active_at ({conv_last_active_at})"
                    )
                    continue
            
            # Validate message timestamps
            if "messages" in conversation and "data" in conversation["messages"]:
                for message in conversation["messages"]["data"]:
                    msg_id = message.get('id', 'unknown')

                    if "created_at" not in message or message["created_at"] is None:
                        api_consistency_logger.warning(
                            f"Conversation {conv_id}, Message {msg_id} missing required timestamp field"
                        )
                        continue
                    
                    if "created_at" in message and message["created_at"] is not None:
                        # Check if message predates the earliest expected timestamp
                        if message["created_at"] < EARLIEST_EXPECTED_TIMESTAMP:
                            api_consistency_logger.warning(
                                f"Conversation {conv_id}, Message {msg_id} has suspiciously early timestamp: "
                                f"{datetime.fromtimestamp(message['created_at'], DEFAULT_TIMEZONE)} "
                                f"(before {datetime.fromtimestamp(EARLIEST_EXPECTED_TIMESTAMP, DEFAULT_TIMEZONE)})"
                            )
                        
                        # Correct message timestamps that are outside the conversation boundaries
                        if message["created_at"] < conv_created_at:
                            # Check if the timestamp is within the tolerance window
                            time_difference = conv_created_at - message["created_at"]
                            if time_difference > MESSAGE_TIMESTAMP_TOLERANCE:
                                api_consistency_logger.warning(
                                    f"Conversation {conv_id}, Message {msg_id} has timestamp before "
                                    f"conversation creation: {message['created_at']} < {conv_created_at}. "
                                    f"Correcting timestamp."
                                )
                                message["created_at"] = conv_created_at
                            
                        if message["created_at"] > conv_last_active_at:
                            # Check if the timestamp is within the tolerance window
                            time_difference = message["created_at"] - conv_last_active_at
                            if time_difference > MESSAGE_TIMESTAMP_TOLERANCE:
                                api_consistency_logger.warning(
                                    f"Conversation {conv_id}, Message {msg_id} has timestamp after "
                                    f"conversation last activity: {message['created_at']} > {conv_last_active_at}. "
                                    f"Correcting timestamp."
                                )
                                message["created_at"] = conv_last_active_at
    
    def list_conversations(
        self, 
        since_timestamp: int = 0,
        after: Optional[str] = None,
        limit: Optional[int] = None,
        users: Optional[List[str]] = None,
        file_format: str = "url"
    ) -> Dict[str, Any]:
        """List conversations from the workspace.
        
        This method fetches conversations from the workspace according to the OpenAPI spec.
        It returns the raw API response, which includes pagination information.
        
        Args:
            since_timestamp: Unix timestamp to filter conversations updated after this time
            after: The previous conversation ID from which to fetch the next page
            limit: The maximum number of conversations to list (default: self.page_size)
            users: List of user IDs to filter conversations by
            file_format: Format for files in conversations ('url' or 'id')
            
        Returns:
            Raw API response containing conversation data and pagination information
            
        Raises:
            Exception: If the API request fails and mock data is not allowed
        """
        try:
            # Prepare request parameters
            params = {
                "limit": limit or self.page_size,
                "file_format": file_format
            }
            
            # Add optional parameters if provided
            if after:
                params["after"] = after
            elif since_timestamp:
                params["since_timestamp"] = since_timestamp
            else:
                params["since_timestamp"] = 0
                
            # Add users filter if provided
            if users:
                params["users"] = users
            
            # Make the API request
            endpoint = f"compliance/workspaces/{self.workspace_id}/conversations"
            response = self._make_request(
                endpoint=endpoint,
                method="GET",
                params=params
            )
            
            # Validate and correct timestamps in the response
            if response and "data" in response:
                self._validate_conversation_timestamps(response["data"])
            
            return response
            
        except Exception as e:
            logger.error(f"API request failed when fetching conversations: {str(e)}")
            
            if self.allow_mock_data:
                logger.info("Using mock conversation data for testing...")
                
                # Create mock conversation data
                mock_conversations = []
                # Use a single base timestamp for all calculations
                base_timestamp = int(datetime.now(DEFAULT_TIMEZONE).timestamp())
                
                for i in range(5):
                    conv_id = f"conv-mock{i}"
                    # Calculate conversation timestamps from the single base timestamp
                    conv_created_at = base_timestamp - 86400 * (i + 1)  # Created in the past
                    conv_last_active_at = base_timestamp - 3600 * i  # Active recently
                    
                    # Ensure message timestamps are within the conversation time range
                    msg1_created_at = conv_created_at  # First message at conversation creation time
                    msg2_created_at = min(conv_created_at + 60, conv_last_active_at)  # Response 60 seconds later, but not after last_active_at
                    
                    mock_conversations.append({
                        "object": "compliance.workspace.conversation",
                        "id": conv_id,
                        "workspace_id": self.workspace_id,
                        "user_id": f"user-mock{i % 3}",  # Distribute among 3 mock users
                        "user_email": f"user{i % 3}@example.com",
                        "created_at": conv_created_at,
                        "last_active_at": conv_last_active_at,
                        "title": f"Mock Conversation {i}",
                        "messages": {
                            "object": "list",
                            "data": [
                                {
                                    "id": f"msg-{conv_id}-1",
                                    "object": "compliance.workspace.message",
                                    "created_at": msg1_created_at,
                                    "content": {
                                        "content_type": "text",
                                        "parts": [f"This is a mock message in conversation {i}"]
                                    },
                                    "role": "user"
                                },
                                {
                                    "id": f"msg-{conv_id}-2",
                                    "object": "compliance.workspace.message",
                                    "created_at": msg2_created_at,
                                    "content": {
                                        "content_type": "text",
                                        "parts": [f"This is a mock response in conversation {i}"]
                                    },
                                    "role": "assistant"
                                }
                            ]
                        }
                    })
                
                # Create mock response with pagination info
                mock_response = {
                    "object": "list",
                    "data": mock_conversations,
                    "has_more": False,
                    "last_id": mock_conversations[-1]["id"] if mock_conversations else None
                }
                
                logger.info(f"Generated {len(mock_conversations)} mock conversations")
                return mock_response
            else:
                # If mock data is not allowed, raise the exception
                raise Exception(f"Conversations endpoint failed: {str(e)}")
    
    def process_all_conversations(
        self, 
        callback_fn: callable,
        since_timestamp: int = 0,
        users: Optional[List[str]] = None,
        file_format: str = "url",
        max_retries: int = 3,
        debug_logging: bool = False,
        use_tqdm: bool = False
    ) -> int:
        """Process all conversations from the workspace with automatic pagination.
        
        This method handles pagination automatically and calls the provided callback
        function for each conversation. The callback function should accept a single
        conversation object as its argument.
        
        Args:
            callback_fn: Function to call for each conversation
            since_timestamp: Unix timestamp to filter conversations updated after this time
            users: List of user IDs to filter conversations by
            file_format: Format for files in conversations ('url' or 'id')
            max_retries: Maximum number of retries for failed API requests
            debug_logging: Whether to print detailed debug logs
            
        Returns:
            Total number of conversations processed
            
        Raises:
            Exception: If the API request fails and mock data is not allowed
        """
        processed_count = 0
        after = None
        has_more = True
        
        # Keep track of conversation IDs we've already processed to avoid duplicates
        processed_ids = set()
        
        # Keep track of pagination attempts for debugging
        page_count = 0
        retry_count = 0
        
        # First, get the first page to determine total count if possible
        try:
            first_response = self.list_conversations(
                since_timestamp=since_timestamp,
                after=after,
                users=users,
                file_format=file_format
            )
            total_conversations = first_response.get("total", 0)
            
            # Process conversations from the first page
            first_page_conversations = first_response.get("data", [])
            
            # Create progress bar if requested
            pbar = None
            if use_tqdm:
                pbar = tqdm(desc="Processing conversations", unit="conv")
            
            # Process first page conversations
            for conversation in first_page_conversations:
                conversation_id = conversation.get("id")
                
                # Skip if we've already processed this conversation
                if conversation_id in processed_ids:
                    if debug_logging:
                        print(f"Skipping already processed conversation: {conversation_id}")
                    continue
                    
                # Call the callback function with the conversation
                try:
                    callback_fn(conversation)
                    
                    # Mark as processed
                    processed_ids.add(conversation_id)
                    processed_count += 1
                    
                    # Update progress bar
                    if pbar is not None:
                        pbar.update(1)
                except Exception as e:
                    # print stack trace
                    import traceback
                    logger.error(traceback.format_exc())
                    logger.error(f"Error processing conversation {conversation_id}: {str(e)}")
            
            # Check if we need to fetch more pages
            has_more = first_response.get("has_more", False)
            after = first_response.get("last_id")
            page_count += 1
            
        except Exception as e:
            logger.error(f"Error fetching first conversation page: {str(e)}")
            # Continue with the main loop which will handle retries
        
        while has_more:
            page_count += 1
            if debug_logging:
                logger.debug(f"Fetching conversation page {page_count} (after={after})")
                
            try:
                response = self.list_conversations(
                    since_timestamp=since_timestamp,
                    after=after,
                    users=users,
                    file_format=file_format
                )
                
                # Process conversations from this page
                conversations = response.get("data", [])
                if debug_logging:
                    logger.debug(f"Retrieved {len(conversations)} conversations on page {page_count}")
                
                for conversation in conversations:
                    conversation_id = conversation.get("id")
                    
                    # Skip if we've already processed this conversation
                    if conversation_id in processed_ids:
                        if debug_logging:
                            logger.debug(f"Skipping already processed conversation: {conversation_id}")
                        continue
                        
                    # Call the callback function with the conversation
                    try:
                        callback_fn(conversation)
                        
                        # Mark as processed
                        processed_ids.add(conversation_id)
                        processed_count += 1
                        
                        # Update progress bar
                        if 'pbar' in locals() and pbar is not None:
                            pbar.update(1)
                    except Exception as e:
                        # print stack trace
                        import traceback
                        logger.error(traceback.format_exc())
                        logger.error(f"Error processing conversation {conversation_id}: {str(e)}")
                
                # Check if we need to fetch more pages
                has_more = response.get("has_more", False)
                after = response.get("last_id")
                
                # Reset retry count on successful request
                retry_count = 0
                
            except Exception as e:
                retry_count += 1
                logger.error(f"Error fetching conversation page {page_count}: {str(e)}")
                
                if retry_count <= max_retries:
                    logger.warning(f"Retrying (attempt {retry_count}/{max_retries})...")
                    # Continue the loop without changing after, to retry the same page
                    continue
                else:
                    logger.warning(f"Max retries ({max_retries}) exceeded, skipping to next page")
                    # If we've reached max retries, try to continue with the next page if possible
                    if after:
                        logger.info(f"Continuing from last known ID: {after}")
                        retry_count = 0
                    else:
                        logger.warning("No pagination token available, stopping pagination")
                        has_more = False
        
        if debug_logging:
            logger.info(f"Processed {processed_count} conversations across {page_count} pages")
        
        # Close progress bar
        if 'pbar' in locals() and pbar is not None:
            pbar.close()
            
        return processed_count
    
    def get_gpt(self, gpt_id: str) -> GPT:
        """Get the builder name of a GPT by its ID.
        
        Args:
            gpt_id: The ID of the GPT
            
        Returns:
            The builder name of the GPT, or "External or Deleted" if not found
        """
        try:
            # Make the API request to get GPT configurations
            endpoint = f"compliance/workspaces/{self.workspace_id}/gpts/{gpt_id}"
            response = self._make_request(
                endpoint=endpoint,
                method="GET"
            )
            
            # Safely get the config data with proper error handling
            config_data = response.get("latest_config", {}).get("data", [])
            
            # Check if config_data is a non-empty list before accessing index 0
            if config_data and len(config_data) > 0:
                config = config_data[0]
            else:
                # Log the issue and use a default config
                api_consistency_logger.warning(f"Missing config data for GPT: {gpt_id} in workspace {self.workspace_id}")
                config = {}
                
            gpt = GPT(
                id=gpt_id,
                name=config.get("name", gpt_id),
                description=config.get("description", ""),
                created_at=response.get("created_at", 0),
                creator_id=response.get("owner_id", ""),
                creator_email=response.get("owner_email", ""),
            )
        
            return gpt            

        except requests.exceptions.HTTPError as e:
            gpt = GPT(
                id=gpt_id,
                name="External or Deleted",
                description="",
                created_at=0,
                creator_id="",
                creator_email="",
            )
            # If this is our simplified 404 error or a 400 with 'Invalid gpt_id', log and return default GPT
            if str(e).startswith("GPT not found") or (
                e.response.status_code == 400 and 
                "Invalid gpt_id" in e.response.text
            ):
                # Log the missing GPT to the consistency issues log
                api_consistency_logger.warning(f"Missing GPT: {gpt_id} not found in workspace {self.workspace_id}")
                return gpt
            
            # For other errors, log and handle as before
            error_msg = str(e)
            api_consistency_logger.error(f"API request failed when fetching GPT details for {gpt_id}: {error_msg}")
            
            if self.allow_mock_data:
                api_consistency_logger.info(f"Using mock GPT data for {gpt_id}")
                return GPT(
                    id=gpt_id,
                    name=f"Mock GPT {gpt_id}",
                    description="",
                    created_at=0,
                    creator_id="",
                    creator_email="",
                )
            else:
                # If mock data is not allowed, return "External or Deleted"
                return gpt
    
    def get_project(self, project_id: str) -> Project:
        """Get the builder name of a project by its ID.
        
        Args:
            project_id: The ID of the project
            
        Returns:
            The builder name of the project, or "External or Deleted" if not found
        """
        try:
            # Make the API request to get project configurations
            endpoint = f"compliance/workspaces/{self.workspace_id}/projects/{project_id}"
            response = self._make_request(
                endpoint=endpoint,
                method="GET",
            )

            # Safely get the config data with proper error handling
            config_data = response.get("latest_config", {}).get("data", [])
            
            # Check if config_data is a non-empty list before accessing index 0
            if config_data and len(config_data) > 0:
                config = config_data[0]
            else:
                # Log the issue and use a default config
                api_consistency_logger.warning(f"Missing config data for Project: {project_id} in workspace {self.workspace_id}")
                config = {}
                
            project = Project(
                id=project_id,
                name=config.get("name", project_id),
                description=config.get("description", ""),
                created_at=response.get("created_at", 0),
                creator_id=response.get("owner_id", ""),
                creator_email=response.get("owner_email", ""),
            )

            return project
            
        except requests.exceptions.HTTPError as e:
            project = Project(
                id=project_id,
                name="External or Deleted",
                description="",
                created_at=0,
                creator_id="",
                creator_email="",
            )
            
            # If this is our simplified 404 error or a 400 with 'Invalid project_id', log and return default project
            if str(e).startswith("Project not found") or (
                e.response.status_code == 400 and 
                "Invalid project_id" in e.response.text
            ):
                # Log the missing project to the consistency issues log
                api_consistency_logger.warning(f"Missing Project: {project_id} not found in workspace {self.workspace_id}")
                return project
            
            # For other errors, log and handle as before
            error_msg = str(e)
            api_consistency_logger.error(f"API request failed when fetching project details for {project_id}: {error_msg}")
            
            if self.allow_mock_data:
                api_consistency_logger.info(f"Using mock project data for {project_id}")
                return Project(
                    id=project_id,
                    name=f"Mock Project {project_id}",
                    description="",
                    created_at=0,
                    creator_id="",
                    creator_email="",
                )
            else:
                # If mock data is not allowed, return "External or Deleted"
                return project

    def _extract_message_text(self, message: Dict[str, Any]) -> str:
        """Extract text content from a message.
        
        Args:
            message: Message object from the API
            
        Returns:
            Text content of the message
        """
        content = message.get("content", {})
        
        # Handle text content
        if content.get("content_type") == "text":
            parts = content.get("parts", [])
            if parts:
                return parts[0]
        
        # For other content types or if text extraction fails
        return ""
        
    def delete_conversation(self, conversation_id: str) -> None:
        """Delete a conversation from the workspace.
        
        This method deletes a conversation from the workspace according to the OpenAPI spec.
        It deletes the conversation title, messages, files, and shared links from the workspace.
        
        Args:
            conversation_id: The ID of the conversation to delete
            
        Raises:
            Exception: If the API request fails and mock data is not allowed
        """
        try:
            # Make the API request to delete the conversation
            endpoint = f"compliance/workspaces/{self.workspace_id}/conversations/{conversation_id}"
            self._make_request(
                endpoint=endpoint,
                method="DELETE"
            )
            
            logger.info(f"Successfully deleted conversation: {conversation_id}")
            
        except Exception as e:
            logger.error(f"API request failed when deleting conversation: {str(e)}")
            
            if self.allow_mock_data:
                logger.info(f"Mock deletion of conversation: {conversation_id}")
            else:
                # If mock data is not allowed, raise the exception
                raise Exception(f"Delete conversation endpoint failed: {str(e)}")