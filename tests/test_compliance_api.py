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
    # Mock the API response
    mock_conversation = {
        "object": "compliance.workspace.conversation",
        "id": "real-conversation-id",
        "workspace_id": "real-workspace-id",
        "user_id": "real-user-id",
        "user_email": "user@example.com",
        "created_at": int(datetime.now(DEFAULT_TIMEZONE).timestamp()) - 86400,
        "last_active_at": int(datetime.now(DEFAULT_TIMEZONE).timestamp()) - 3600,
        "title": "Test Conversation",
        "messages": {
            "object": "list",
            "data": [
                {
                    "id": "msg-1",
                    "object": "compliance.workspace.message",
                    "created_at": int(datetime.now(DEFAULT_TIMEZONE).timestamp()) - 86400,
                    "content": {
                        "content_type": "text",
                        "parts": ["Test message 1"]
                    },
                    "role": "user"
                },
                {
                    "id": "msg-2",
                    "object": "compliance.workspace.message",
                    "created_at": int(datetime.now(DEFAULT_TIMEZONE).timestamp()) - 86400 + 60,
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
    # Set up mock responses for pagination
    mock_make_request.side_effect = [
        # First page
        {
            "object": "list",
            "data": [
                {
                    "id": "conv-1",
                    "messages": {
                        "data": [
                            {"id": "msg-1", "role": "user", "created_at": int(datetime.now(DEFAULT_TIMEZONE).timestamp()) - 3600},
                            {"id": "msg-2", "role": "assistant", "created_at": int(datetime.now(DEFAULT_TIMEZONE).timestamp()) - 3540}
                        ]
                    }
                },
                {
                    "id": "conv-2",
                    "messages": {
                        "data": [
                            {"id": "msg-3", "role": "user", "created_at": int(datetime.now(DEFAULT_TIMEZONE).timestamp()) - 3480},
                            {"id": "msg-4", "role": "assistant", "created_at": int(datetime.now(DEFAULT_TIMEZONE).timestamp()) - 3420}
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
                    "messages": {
                        "data": [
                            {"id": "msg-5", "role": "user", "created_at": int(datetime.now(DEFAULT_TIMEZONE).timestamp()) - 3360},
                            {"id": "msg-6", "role": "assistant", "created_at": int(datetime.now(DEFAULT_TIMEZONE).timestamp()) - 3300}
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
