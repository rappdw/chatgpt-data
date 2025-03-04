#!/usr/bin/env python3

import requests
import os
import pandas as pd
import time
import sys

# Try to import dotenv, install if not available
try:
    from dotenv import load_dotenv
except ImportError:
    print("python-dotenv package not found. Installing...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "python-dotenv"])
    from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("API_KEY")
WORKSPACE_ID = os.getenv("WORKSPACE_ID")
USER_ID = os.getenv("USER_ID")
START_TIMESTAMP = int(time.mktime(time.strptime(os.getenv("START_TIMESTAMP"), "%Y-%m-%d %H:%M:%S")))
END_TIMESTAMP = int(time.mktime(time.strptime(os.getenv("END_TIMESTAMP"), "%Y-%m-%d %H:%M:%S")))

headers = {"Authorization": f"Bearer {API_KEY}"}
base_url = f"https://api.chatgpt.com/v1/compliance/workspaces/{WORKSPACE_ID}/conversations"
messages = []
tool_messages = []
system_messages = []
assistant_messages = []
gpt_messages = []
project_messages = []
after_convo = None  # Used for conversation pagination
tool_debug = set()

while True:
    # Fetch a batch of conversations including messages
    params = {
        "users": USER_ID,
        "since_timestamp": START_TIMESTAMP,
        "limit": 500,
    }
    if after_convo:
        params["after"] = after_convo

    convo_response = requests.get(base_url, headers=headers, params=params)
    convo_data = convo_response.json()
    
    if "data" not in convo_data:
        print(f"Unexpected response: {convo_data}")
        break

    conversations = convo_data.get("data", [])
    convo_couunt = 0
    for convo in conversations:
        convo_messages = convo.get("messages", {}).get("data", [])

        message_count = 0        
        for msg in convo_messages:
            if msg["created_at"] is None:
                continue
            if not (START_TIMESTAMP <= msg["created_at"] <= END_TIMESTAMP):
                continue

            message_count += 1

            msg_type = "Standard Message"
            tool_type = None
            project_name = None

            append_here = None
            if msg.get("gpt_id"):
                append_here = gpt_messages
                msg_type = "Custom GPT Message"
            elif msg["author"]["role"] == "tool":
                append_here = tool_messages
                if msg['author']['tool_name'] not in tool_debug:
                    print(msg['author'])
                    tool_debug.add(msg['author']['tool_name'])
                msg_type = "Tool Message"
                tool_type = msg["author"].get("tool_name")
            elif msg['author']['role'] == "system":
                append_here = system_messages
            elif msg['author']['role'] == "assistant":
                append_here = assistant_messages
            elif msg.get("project_id"):
                append_here = project_messages
                msg_type = "Project Message"
                # Fetch project details
                project_url = f"https://api.chatgpt.com/v1/compliance/workspaces/{WORKSPACE_ID}/projects/{msg['project_id']}"
                project_resp = requests.get(project_url, headers=headers)
                project_name = project_resp.json().get("name", "Unknown Project")
            else:   
                append_here = messages

            append_here.append({
                "timestamp": msg["created_at"],
                "message_id": msg["id"],
                "type": msg_type,
                "tool_type": tool_type,
                "project_name": project_name,
                "gpt_id": msg.get("gpt_id"),
                "project_id": msg.get("project_id")
            })
        if message_count > 0:
            convo_couunt += 1

    # Handle conversation pagination
    if convo_data.get("has_more"):
        after_convo = convo_data["last_id"]
    else:
        break

# Convert to DataFrame
if messages:
    df = pd.DataFrame(messages)
else:
    print("No messages found. Creating an empty DataFrame.")
    df = pd.DataFrame(columns=["timestamp", "message_id", "type", "tool_type", "project_name", "gpt_id", "project_id"])

# Debugging columns
print("Columns in DataFrame:", df.columns.tolist())

# Compute required statistics
num_messages = len(df)
num_gpt_messages = len(df[df["type"] == "Custom GPT Message"]) if "type" in df.columns else 0
num_distinct_gpts = df["gpt_id"].nunique() if "gpt_id" in df.columns else 0
num_tool_messages = len(df[df["type"] == "Tool Message"]) if "type" in df.columns else 0
num_distinct_tools = df["tool_type"].nunique() if "tool_type" in df.columns else 0
num_project_messages = len(df[df["type"] == "Project Message"]) if "type" in df.columns else 0
num_distinct_projects = df["project_id"].nunique() if "project_id" in df.columns else 0

# Print summary statistics
print(f"Conversation Count: {convo_couunt}")
print(f"Total Messages: {len(messages)}")
print(f"GPT Messages: {len(gpt_messages)}")
print(f"Distinct GPTs Messaged: {num_distinct_gpts}")
print(f"Tool Messages: {len(tool_messages)}")
print(f"Distinct Tools Used: {num_distinct_tools}")
print(f"Project Messages: {len(project_messages)}")
print(f"Distinct Projects Messaged: {num_distinct_projects}")
print(f"System Messages: {len(system_messages)}")
print(f"Assistant Messages: {len(assistant_messages)}")
