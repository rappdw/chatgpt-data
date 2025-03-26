"""Tests for the Enterprise Compliance API client."""

import pytest
from unittest.mock import patch, MagicMock
from datetime import datetime
from chatgpt_data.utils.constants import DEFAULT_TIMEZONE

from chatgpt_data.api.compliance_api import EnterpriseComplianceAPI


def test_list_conversations():
    """Test retrieving conversations from the workspace."""
    # Create a mock API client with test mode enabled
    api = EnterpriseComplianceAPI(
        api_key="test-api-key",
        org_id="test-org-id",
        workspace_id="test-workspace", 
        allow_mock_data=True
    )
    
    # Get conversations
    response = api.list_conversations()
    
    # Verify the response structure
    assert response is not None
    assert "data" in response
    assert len(response["data"]) > 0
    
    # Verify conversation structure
    conversation = response["data"][0]
    assert "id" in conversation
    assert "messages" in conversation
    assert "data" in conversation["messages"]
    assert len(conversation["messages"]["data"]) > 0
    
    # Verify message structure
    message = conversation["messages"]["data"][0]
    assert "id" in message
    assert "role" in message
    assert "created_at" in message
    assert "content" in message


def test_extract_messages_from_conversations():
    """Test extracting messages from conversations response."""
    # Create a mock API client with test mode enabled
    api = EnterpriseComplianceAPI(
        api_key="test-api-key",
        org_id="test-org-id",
        workspace_id="test-workspace", 
        allow_mock_data=True
    )
    
    # Get conversations
    response = api.list_conversations()
    
    # Extract messages from the first conversation
    conversation = response["data"][0]
    messages = conversation["messages"]["data"]
    
    # Verify we got messages
    assert messages is not None
    assert len(messages) > 0
    
    # Verify message structure
    message = messages[0]
    assert "id" in message
    assert "role" in message
    assert "created_at" in message
    assert "content" in message


def test_list_conversations_with_pagination():
    """Test that pagination works correctly when listing conversations."""
    # Create a mock API client with test mode enabled
    api = EnterpriseComplianceAPI(
        api_key="test-api-key",
        org_id="test-org-id",
        workspace_id="test-workspace", 
        allow_mock_data=True
    )
    
    # Get first page of conversations
    response1 = api.list_conversations(limit=2)
    
    # Verify pagination info
    assert "has_more" in response1
    assert "last_id" in response1
    
    # If there are more conversations, get the next page
    if response1["has_more"] and response1["last_id"]:
        response2 = api.list_conversations(after=response1["last_id"], limit=2)
        
        # Verify we got different conversations
        conv_ids1 = [conv["id"] for conv in response1["data"]]
        conv_ids2 = [conv["id"] for conv in response2["data"]]
        
        # Check that the conversation IDs are different
        assert len(set(conv_ids1).intersection(set(conv_ids2))) == 0


@patch('chatgpt_data.api.compliance_api.EnterpriseComplianceAPI._make_request')
def test_list_conversations_real_api(mock_make_request):
    """Test listing conversations with mocked API responses."""
    # Use a single base timestamp for all calculations
    base_timestamp = int(datetime.now(DEFAULT_TIMEZONE).timestamp())
    
    # Calculate conversation timestamps
    conv_created_at = base_timestamp - 86400
    conv_last_active_at = base_timestamp - 3600
    
    # Calculate message timestamps within the conversation time range
    msg1_created_at = conv_created_at  # First message at conversation creation
    msg2_created_at = min(conv_created_at + 60, conv_last_active_at)  # Response 60 seconds later
    
    # Mock the API response
    mock_conversation = {
        "object": "compliance.workspace.conversation",
        "id": "real-conversation-id",
        "workspace_id": "real-workspace-id",
        "user_id": "real-user-id",
        "user_email": "user@example.com",
        "created_at": conv_created_at,
        "last_active_at": conv_last_active_at,
        "title": "Test Conversation",
        "messages": {
            "object": "list",
            "data": [
                {
                    "id": "msg-1",
                    "object": "compliance.workspace.message",
                    "created_at": msg1_created_at,
                    "content": {
                        "content_type": "text",
                        "parts": ["Test message 1"]
                    },
                    "role": "user"
                },
                {
                    "id": "msg-2",
                    "object": "compliance.workspace.message",
                    "created_at": msg2_created_at,
                    "content": {
                        "content_type": "text",
                        "parts": ["Test response 1"]
                    },
                    "role": "assistant"
                }
            ],
            "has_more": True,
            "next": "msg-2"
        }
    }
    
    mock_response = {
        "object": "list",
        "data": [mock_conversation],
        "has_more": False,
        "last_id": "real-conversation-id"
    }
    
    mock_make_request.return_value = mock_response
    
    # Create an API client
    api = EnterpriseComplianceAPI(
        api_key="test-api-key",
        org_id="test-org-id",
        workspace_id="real-workspace-id", 
        allow_mock_data=False
    )
    
    # List conversations
    conversations = api.list_conversations()
    
    # Verify the response structure
    assert conversations == mock_response
    mock_make_request.assert_called_once()


@patch('chatgpt_data.api.compliance_api.EnterpriseComplianceAPI._make_request')
def test_list_conversations_multiple_pages(mock_make_request):
    """Test retrieving multiple pages of conversations."""
    # Use a single base timestamp for all calculations
    base_timestamp = int(datetime.now(DEFAULT_TIMEZONE).timestamp())
    
    # Set up mock responses for pagination
    mock_make_request.side_effect = [
        # First page
        {
            "object": "list",
            "data": [
                {
                    "id": "conv-1",
                    "created_at": base_timestamp - 3700,
                    "last_active_at": base_timestamp - 3500,
                    "messages": {
                        "data": [
                            {"id": "msg-1", "role": "user", "created_at": base_timestamp - 3600},
                            {"id": "msg-2", "role": "assistant", "created_at": base_timestamp - 3540}
                        ]
                    }
                },
                {
                    "id": "conv-2",
                    "created_at": base_timestamp - 3500,
                    "last_active_at": base_timestamp - 3400,
                    "messages": {
                        "data": [
                            {"id": "msg-3", "role": "user", "created_at": base_timestamp - 3480},
                            {"id": "msg-4", "role": "assistant", "created_at": base_timestamp - 3420}
                        ]
                    }
                }
            ],
            "has_more": True,
            "last_id": "conv-2"
        },
        # Second page
        {
            "object": "list",
            "data": [
                {
                    "id": "conv-3",
                    "created_at": base_timestamp - 3400,
                    "last_active_at": base_timestamp - 3200,
                    "messages": {
                        "data": [
                            {"id": "msg-5", "role": "user", "created_at": base_timestamp - 3360},
                            {"id": "msg-6", "role": "assistant", "created_at": base_timestamp - 3300}
                        ]
                    }
                }
            ],
            "has_more": False,
            "last_id": "conv-3"
        }
    ]
    
    # Create an API client
    api = EnterpriseComplianceAPI(
        api_key="test-api-key",
        org_id="test-org-id",
        workspace_id="test-workspace", 
        allow_mock_data=False
    )
    
    # Get first page
    page1 = api.list_conversations(limit=2)
    
    # Verify first page
    assert len(page1["data"]) == 2
    assert page1["has_more"] is True
    assert page1["last_id"] == "conv-2"
    
    # Get second page
    page2 = api.list_conversations(after="conv-2", limit=2)
    
    # Verify second page
    assert len(page2["data"]) == 1
    assert page2["has_more"] is False
    assert page2["last_id"] == "conv-3"
    
    # Verify that _make_request was called twice with the correct parameters
    assert mock_make_request.call_count == 2
    mock_make_request.assert_any_call(
        endpoint="compliance/workspaces/test-workspace/conversations",
        method="GET",
        params={"limit": 2, "file_format": "url", "since_timestamp": 0}
    )
    mock_make_request.assert_any_call(
        endpoint="compliance/workspaces/test-workspace/conversations",
        method="GET",
        params={"limit": 2, "file_format": "url", "after": "conv-2"}
    )


@patch('chatgpt_data.api.compliance_api.EnterpriseComplianceAPI._make_request')
def test_delete_conversation(mock_make_request):
    """Test deleting a conversation."""
    # Create an API client
    api = EnterpriseComplianceAPI(
        api_key="test-api-key",
        org_id="test-org-id",
        workspace_id="test-workspace", 
        allow_mock_data=False
    )
    
    # Mock the API response (DELETE requests typically return None)
    mock_make_request.return_value = None
    
    # Delete a conversation
    api.delete_conversation("test-conversation-id")
    
    # Verify that _make_request was called with the correct parameters
    mock_make_request.assert_called_once_with(
        endpoint="compliance/workspaces/test-workspace/conversations/test-conversation-id",
        method="DELETE"
    )


@patch('chatgpt_data.api.compliance_api.EnterpriseComplianceAPI._make_request')
def test_delete_conversation_with_mock_data(mock_make_request):
    """Test deleting a conversation with mock data when API fails."""
    # Create an API client with mock data allowed
    api = EnterpriseComplianceAPI(
        api_key="test-api-key",
        org_id="test-org-id",
        workspace_id="test-workspace", 
        allow_mock_data=True
    )
    
    # Make _make_request raise an exception
    mock_make_request.side_effect = Exception("API error")
    
    # Delete a conversation - this should not raise an exception due to mock data fallback
    api.delete_conversation("test-conversation-id")
    
    # Verify that _make_request was called with the correct parameters
    mock_make_request.assert_called_once_with(
        endpoint="compliance/workspaces/test-workspace/conversations/test-conversation-id",
        method="DELETE"
    )


import requests
import sys
import time

@patch('requests.request')
@patch('time.sleep')  # Mock sleep to avoid waiting during tests
@patch('sys.exit')   # Mock sys.exit to prevent test from exiting
def test_retry_mechanism_for_server_errors(mock_exit, mock_sleep, mock_request):
    """Test that the retry mechanism works correctly for server errors."""
    # Create a mock response with a 500 status code
    mock_response = MagicMock()
    mock_response.status_code = 500
    mock_response.text = "Internal Server Error"
    mock_response.json.side_effect = ValueError("No JSON data")
    
    # Make the request method raise an HTTPError with the 500 response
    mock_request.return_value = mock_response
    mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError("500 Server Error", response=mock_response)
    
    # Create an API client
    api = EnterpriseComplianceAPI(
        api_key="test-api-key",
        org_id="test-org-id",
        workspace_id="test-workspace",
        allow_mock_data=False
    )
    
    # Call a method that uses _make_request
    exception_raised = False
    try:
        api.list_conversations()
    except Exception:
        exception_raised = True  # We now expect an exception instead of sys.exit
    
    # Verify an exception was raised after max retries
    assert exception_raised, "Expected an exception to be raised after max retries"
    
    # Verify that request was called 4 times (original + 3 retries)
    assert mock_request.call_count == 4
    
    # Verify that sleep was called three times (once after each of the first 3 failures)
    assert mock_sleep.call_count == 3
    
    # Verify that sleep was called with exponential backoff (2^1=2, 2^2=4, 2^3=8)
    mock_sleep.assert_any_call(2)  # First retry
    mock_sleep.assert_any_call(4)  # Second retry
    mock_sleep.assert_any_call(8)  # Third retry
    
    # Verify that sys.exit was NOT called (our new implementation raises an exception instead)
    mock_exit.assert_not_called()


@patch('requests.request')
@patch('time.sleep')  # Mock sleep to avoid waiting during tests
@patch('sys.exit')    # Mock sys.exit to prevent test from exiting
def test_retry_mechanism_for_timeout_errors(mock_exit, mock_sleep, mock_request):
    """Test that the retry mechanism works correctly for timeout errors."""
    # Make the request method raise a Timeout exception
    mock_request.side_effect = requests.exceptions.Timeout("Request timed out")
    
    # Create an API client
    api = EnterpriseComplianceAPI(
        api_key="test-api-key",
        org_id="test-org-id",
        workspace_id="test-workspace",
        allow_mock_data=False
    )
    
    # Call a method that uses _make_request
    exception_raised = False
    try:
        api.list_conversations()
    except Exception:
        exception_raised = True  # We now expect an exception instead of sys.exit
    
    # Verify an exception was raised after max retries
    assert exception_raised, "Expected an exception to be raised after max retries"
    
    # Verify that request was called 4 times (original + 3 retries)
    assert mock_request.call_count == 4
    
    # Verify that sleep was called three times (once after each of the first 3 failures)
    assert mock_sleep.call_count == 3
    
    # Verify that sleep was called with exponential backoff (2^1=2, 2^2=4, 2^3=8)
    mock_sleep.assert_any_call(2)  # First retry
    mock_sleep.assert_any_call(4)  # Second retry
    mock_sleep.assert_any_call(8)  # Third retry
    
    # Verify that sys.exit was NOT called (our new implementation raises an exception instead)
    mock_exit.assert_not_called()


@patch('requests.request')
@patch('time.sleep')  # Mock sleep to avoid waiting during tests
def test_retry_mechanism_success_after_retry(mock_sleep, mock_request):
    """Test that the retry mechanism succeeds after a retry."""
    # Create a failed response with a 500 status code
    failed_response = MagicMock()
    failed_response.status_code = 500
    failed_response.text = "Internal Server Error"
    failed_response.json.side_effect = ValueError("No JSON data")
    failed_response.raise_for_status.side_effect = requests.exceptions.HTTPError("500 Server Error", response=failed_response)
    
    # Use a single base timestamp for all calculations
    base_timestamp = int(datetime.now(DEFAULT_TIMEZONE).timestamp())
    
    # Calculate conversation and message timestamps
    conv_created_at = base_timestamp - 3700
    conv_last_active_at = base_timestamp - 3500
    msg_created_at = base_timestamp - 3600
    
    # Create a successful response for the second attempt
    success_data = {
        "object": "list",
        "data": [
            {
                "id": "conv-1",
                "created_at": conv_created_at,
                "last_active_at": conv_last_active_at,
                "messages": {
                    "data": [
                        {"id": "msg-1", "role": "user", "created_at": msg_created_at}
                    ]
                }
            }
        ],
        "has_more": False
    }
    success_response = MagicMock()
    success_response.status_code = 200
    success_response.json.return_value = success_data
    
    # Set up the mock to fail once then succeed
    mock_request.side_effect = [failed_response, success_response]
    
    # Create an API client
    api = EnterpriseComplianceAPI(
        api_key="test-api-key",
        org_id="test-org-id",
        workspace_id="test-workspace",
        allow_mock_data=False
    )
    
    # Call a method that uses _make_request
    response = api.list_conversations()
    
    # Verify that request was called twice (original + 1 retry)
    assert mock_request.call_count == 2
    
    # Verify that sleep was called once (after the first failure)
    mock_sleep.assert_called_once_with(2)  # First retry with 2^1=2 seconds delay
    
    # Verify we got the successful response
    assert response == success_data


def test_message_timestamps_within_conversation_range():
    """Test that message timestamps are always within the range of conversation timestamps."""
    # Create a mock API client with test mode enabled
    api = EnterpriseComplianceAPI(
        api_key="test-api-key",
        org_id="test-org-id",
        workspace_id="test-workspace", 
        allow_mock_data=True
    )
    
    # Get conversations
    response = api.list_conversations()
    
    # Verify the response structure
    assert response is not None
    assert "data" in response
    assert len(response["data"]) > 0
    
    # Check each conversation and its messages
    for conversation in response["data"]:
        # Verify conversation has required timestamp fields
        assert "created_at" in conversation, "Conversation missing created_at field"
        assert "last_active_at" in conversation, "Conversation missing last_active_at field"
        
        # Get conversation timestamp range
        conv_created_at = conversation["created_at"]
        conv_last_active_at = conversation["last_active_at"]
        
        # Verify conversation timestamps are valid
        assert conv_created_at <= conv_last_active_at, f"Conversation {conversation['id']} has created_at after last_active_at"
        
        # Check each message in the conversation
        assert "messages" in conversation
        assert "data" in conversation["messages"]
        
        for message in conversation["messages"]["data"]:
            # Verify message has a timestamp (if not nullable)
            if "created_at" in message and message["created_at"] is not None:
                msg_created_at = message["created_at"]
                
                # Verify message timestamp is within conversation range
                assert conv_created_at <= msg_created_at, \
                    f"Message {message['id']} has timestamp before conversation creation: {msg_created_at} < {conv_created_at}"
                assert msg_created_at <= conv_last_active_at, \
                    f"Message {message['id']} has timestamp after conversation last activity: {msg_created_at} > {conv_last_active_at}"
