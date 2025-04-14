"""Unit tests for incremental update functionality."""

import os
import unittest
import tempfile
import shutil
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock, call
import pickle
import json

from chatgpt_data.api.compliance_api import EnterpriseComplianceAPI, User
from chatgpt_data.cli.api_reports import (
    get_users_conversations,
    find_most_recent_data,
    RawData,
    Conversation,
    Message,
)
from chatgpt_data.utils.constants import DEFAULT_TIMEZONE


class TestIncrementalUpdate(unittest.TestCase):
    """Test suite for incremental update functionality."""

    def setUp(self):
        """Set up test environment."""
        # Create a temporary directory for test outputs
        self.test_dir = tempfile.mkdtemp()
        
        # Mock API client
        self.mock_api = MagicMock(spec=EnterpriseComplianceAPI)
        
        # Set up timestamps for testing
        now = datetime.now(DEFAULT_TIMEZONE)
        self.current_time = int(now.timestamp())
        self.week_ago = int((now - timedelta(days=7)).timestamp())
        self.two_weeks_ago = int((now - timedelta(days=14)).timestamp())
        
        # Create sample users
        self.user1 = User(
            user_id="user1",
            email="user1@example.com",
            name="User One",
            role="standard-user",
            created_at=self.two_weeks_ago,
            status="active",
            conversations=[]
        )
        
        self.user2 = User(
            user_id="user2",
            email="user2@example.com",
            name="User Two",
            role="standard-user",
            created_at=self.two_weeks_ago,
            status="active",
            conversations=[]
        )
        
        # Create sample conversations for testing
        self.old_conversation = {
            "id": "conv1",
            "user_id": "user1",
            "created_at": self.two_weeks_ago,
            "last_active_at": self.two_weeks_ago,
            "title": "Old Conversation",
            "messages": {
                "data": [
                    {
                        "id": "msg1",
                        "created_at": self.two_weeks_ago,
                        "author": {"role": "user"},
                        "content": {"content_type": "text", "parts": ["Hello"]},
                    },
                    {
                        "id": "msg2",
                        "created_at": self.two_weeks_ago,
                        "author": {"role": "assistant"},
                        "content": {"content_type": "text", "parts": ["Hi there"]},
                    }
                ],
                "has_more": False
            }
        }
        
        self.updated_conversation = {
            "id": "conv1",
            "user_id": "user1",
            "created_at": self.two_weeks_ago,
            "last_active_at": self.current_time,
            "title": "Old Conversation (Updated)",
            "messages": {
                "data": [
                    {
                        "id": "msg1",
                        "created_at": self.two_weeks_ago,
                        "author": {"role": "user"},
                        "content": {"content_type": "text", "parts": ["Hello"]},
                    },
                    {
                        "id": "msg2",
                        "created_at": self.two_weeks_ago,
                        "author": {"role": "assistant"},
                        "content": {"content_type": "text", "parts": ["Hi there"]},
                    },
                    {
                        "id": "msg3",
                        "created_at": self.current_time,
                        "author": {"role": "user"},
                        "content": {"content_type": "text", "parts": ["How are you?"]},
                    },
                    {
                        "id": "msg4",
                        "created_at": self.current_time,
                        "author": {"role": "assistant"},
                        "content": {"content_type": "text", "parts": ["I'm doing well!"]},
                    }
                ],
                "has_more": False
            }
        }
        
        self.new_conversation = {
            "id": "conv2",
            "user_id": "user2",
            "created_at": self.current_time,
            "last_active_at": self.current_time,
            "title": "New Conversation",
            "messages": {
                "data": [
                    {
                        "id": "msg5",
                        "created_at": self.current_time,
                        "author": {"role": "user"},
                        "content": {"content_type": "text", "parts": ["Brand new message"]},
                    },
                    {
                        "id": "msg6",
                        "created_at": self.current_time,
                        "author": {"role": "assistant"},
                        "content": {"content_type": "text", "parts": ["Welcome!"]},
                    }
                ],
                "has_more": False
            }
        }
    
    def tearDown(self):
        """Clean up after tests."""
        # Remove the temporary directory
        shutil.rmtree(self.test_dir)
    
    def create_previous_data(self):
        """Create a previous run dataset."""
        # Create old version of conversation 1
        conv1 = Conversation(
            id="conv1",
            created_at=self.two_weeks_ago,
            last_active_at=self.two_weeks_ago,
            title="Old Conversation"
        )
        
        # Add messages to conversation 1
        conv1.messages.append(Message(
            id="msg1",
            created_at=self.two_weeks_ago,
            role="user"
        ))
        
        conv1.messages.append(Message(
            id="msg2", 
            created_at=self.two_weeks_ago,
            role="assistant"
        ))
        
        # Add conversation to user
        self.user1.conversations.append(conv1)
        
        # Create raw data object
        prev_data = RawData(
            users={
                "user1": self.user1,
                "user2": self.user2
            },
            earliest_message_timestamp=self.two_weeks_ago,
            latest_message_timestamp=self.two_weeks_ago,
            project_map={},
            gpt_map={}
        )
        
        # Save pickle to the test directory
        timestamp = datetime.now(DEFAULT_TIMEZONE).strftime("%Y%m%d_%H%M%S")
        pickle_path = os.path.join(self.test_dir, f"raw_data_{timestamp}.pkl")
        with open(pickle_path, 'wb') as f:
            pickle.dump(prev_data, f)
        
        # Return the raw data and file path
        return prev_data, pickle_path
    
    @patch('chatgpt_data.cli.api_reports.find_most_recent_data')
    def test_new_conversations_added(self, mock_find_data):
        """Test that new conversations are correctly added during incremental update."""
        # Create and save previous data
        prev_data, _ = self.create_previous_data()
        
        # Set up mocks
        mock_find_data.return_value = (prev_data, self.two_weeks_ago)
        self.mock_api.get_all_users.return_value = {
            "user1": self.user1,
            "user2": self.user2
        }
        
        # Configure the process_all_conversations to return our test data
        def mock_process(callback_fn, since_timestamp, **kwargs):
            # This should only include the new conversation (since we're doing incremental update)
            # The old conversation has no new messages, so it shouldn't be returned
            self.assertEqual(since_timestamp, self.two_weeks_ago)
            callback_fn(self.new_conversation)
            return 1
            
        self.mock_api.process_all_conversations.side_effect = mock_process
        
        # Run the incremental update
        result = get_users_conversations(
            self.mock_api, 
            debug_logging=False, 
            incremental_update=True, 
            output_dir=self.test_dir
        )
        
        # Verify the new conversation was added
        self.assertEqual(len(result.users["user2"].conversations), 1)
        self.assertEqual(result.users["user2"].conversations[0].id, "conv2")
        self.assertEqual(len(result.users["user2"].conversations[0].messages), 2)
        
        # Verify the old conversation was preserved
        self.assertEqual(len(result.users["user1"].conversations), 1)
        self.assertEqual(result.users["user1"].conversations[0].id, "conv1")
        
        # Verify we called process_all_conversations with the correct timestamp
        self.mock_api.process_all_conversations.assert_called_once()
        args, kwargs = self.mock_api.process_all_conversations.call_args
        self.assertEqual(kwargs['since_timestamp'], self.two_weeks_ago)
    
    @patch('chatgpt_data.cli.api_reports.find_most_recent_data')
    def test_conversations_with_new_messages_updated(self, mock_find_data):
        """Test that conversations with new messages are updated correctly."""
        # Create and save previous data
        prev_data, _ = self.create_previous_data()
        
        # Set up mocks
        mock_find_data.return_value = (prev_data, self.two_weeks_ago)
        self.mock_api.get_all_users.return_value = {
            "user1": self.user1,
            "user2": self.user2
        }
        
        # Configure the process_all_conversations to return our test data
        def mock_process(callback_fn, since_timestamp, **kwargs):
            # This should include the updated conversation
            self.assertEqual(since_timestamp, self.two_weeks_ago)
            callback_fn(self.updated_conversation)
            return 1
            
        self.mock_api.process_all_conversations.side_effect = mock_process
        
        # Run the incremental update
        result = get_users_conversations(
            self.mock_api, 
            debug_logging=False, 
            incremental_update=True, 
            output_dir=self.test_dir
        )
        
        # Verify the conversation was updated with new messages
        self.assertEqual(len(result.users["user1"].conversations), 1)
        self.assertEqual(result.users["user1"].conversations[0].id, "conv1")
        self.assertEqual(len(result.users["user1"].conversations[0].messages), 4)
        
        # Verify message IDs to make sure all messages are present
        message_ids = [msg.id for msg in result.users["user1"].conversations[0].messages]
        self.assertIn("msg1", message_ids)
        self.assertIn("msg2", message_ids)
        self.assertIn("msg3", message_ids)
        self.assertIn("msg4", message_ids)
        
        # Verify the most recent message is correctly included
        new_messages = [msg for msg in result.users["user1"].conversations[0].messages if msg.id == "msg4"]
        self.assertEqual(len(new_messages), 1)
        self.assertEqual(new_messages[0].created_at, self.current_time)
    
    @patch('chatgpt_data.cli.api_reports.find_most_recent_data')
    def test_unchanged_conversations_not_reprocessed(self, mock_find_data):
        """Test that unchanged conversations are not reprocessed."""
        # Create and save previous data
        prev_data, _ = self.create_previous_data()
        
        # Set up mocks
        mock_find_data.return_value = (prev_data, self.two_weeks_ago)
        self.mock_api.get_all_users.return_value = {
            "user1": self.user1,
            "user2": self.user2
        }
        
        # Configure process_all_conversations to only return the new conversation
        # Simulating that the API doesn't return the old unchanged conversation
        def mock_process(callback_fn, since_timestamp, **kwargs):
            self.assertEqual(since_timestamp, self.two_weeks_ago)
            callback_fn(self.new_conversation)
            return 1
            
        self.mock_api.process_all_conversations.side_effect = mock_process
        
        # Run the incremental update
        result = get_users_conversations(
            self.mock_api, 
            debug_logging=False, 
            incremental_update=True, 
            output_dir=self.test_dir
        )
        
        # Verify the old conversation was preserved without reprocessing
        self.assertEqual(len(result.users["user1"].conversations), 1)
        self.assertEqual(result.users["user1"].conversations[0].id, "conv1")
        self.assertEqual(len(result.users["user1"].conversations[0].messages), 2)
        
        # Verify that the new conversation was added
        self.assertEqual(len(result.users["user2"].conversations), 1)
        self.assertEqual(result.users["user2"].conversations[0].id, "conv2")
        
        # Verify we called process_all_conversations with the correct timestamp
        self.mock_api.process_all_conversations.assert_called_once()
        args, kwargs = self.mock_api.process_all_conversations.call_args
        self.assertEqual(kwargs['since_timestamp'], self.two_weeks_ago)
    
    def test_find_most_recent_data(self):
        """Test finding the most recent data file."""
        # Create and save previous data
        prev_data, pickle_path = self.create_previous_data()
        
        # Call the function
        result_data, timestamp = find_most_recent_data(self.test_dir)
        
        # Verify we got the expected data
        self.assertIsNotNone(result_data)
        self.assertEqual(timestamp, self.two_weeks_ago)
        self.assertEqual(len(result_data.users), 2)
        self.assertIn("user1", result_data.users)
        self.assertIn("user2", result_data.users)


if __name__ == '__main__':
    unittest.main()
