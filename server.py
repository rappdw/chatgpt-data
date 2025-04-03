#!/usr/bin/env python3
"""
Simple HTTP server for the ChatGPT Enterprise Engagement Metrics dashboard.
Serves the HTML file and CSV data files for the dashboard.
"""

import http.server
import os
import socketserver
import boto3
from pathlib import Path

# Get the directory of this script
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "data"

# Default port
PORT = 8000


class RequestHandler(http.server.SimpleHTTPRequestHandler):
    """Custom request handler for the server."""

    def end_headers(self):
        """Add CORS headers to allow local file access."""
        self.send_header("Access-Control-Allow-Origin", "*")
        super().end_headers()


def sync_data_from_s3():
    """Sync data from S3 if needed."""
    if DATA_DIR.exists():
        print("Data directory already exists, skipping S3 sync")
        return

    s3_bucket = os.environ.get("CHATGPT_DATA_DEV_CHATGPT_DATA_BUCKET_NAME")
    if not s3_bucket:
        raise ValueError(
            "CHATGPT_DATA_DEV_CHATGPT_DATA_BUCKET_NAME environment variable must be set"
        )

    print(f"Data directory not found, syncing from S3 bucket: {s3_bucket}")
    s3 = boto3.client("s3")
    DATA_DIR.mkdir(exist_ok=True)

    try:
        # Use aws s3 sync through boto3
        paginator = s3.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=s3_bucket, Prefix="data/"):
            for obj in page.get("Contents", []):
                # Create the local file path
                relative_path = obj["Key"]
                local_path = DATA_DIR / relative_path

                # Ensure directory exists
                local_path.parent.mkdir(parents=True, exist_ok=True)

                # Download the file
                print(f"Downloading {relative_path}")
                s3.download_file(s3_bucket, obj["Key"], str(local_path))

        print("S3 sync completed successfully")
    except Exception as e:
        print(f"Error syncing from S3: {e}")
        raise


def main():
    """Start the server."""
    # Sync data from S3 if needed
    sync_data_from_s3()

    # Change to the script directory
    os.chdir(SCRIPT_DIR)

    # Create the server
    handler = RequestHandler
    httpd = socketserver.TCPServer(("", PORT), handler)

    print(f"Serving at http://localhost:{PORT}")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped.")
        httpd.server_close()


if __name__ == "__main__":
    main()
