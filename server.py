#!/usr/bin/env python3
"""
Simple HTTP server for the ChatGPT Enterprise Engagement Metrics dashboard.
Serves the HTML file and CSV data files for the dashboard.
"""

import http.server
import os
import socketserver
from pathlib import Path

# Get the directory of this script
SCRIPT_DIR = Path(__file__).resolve().parent

# Default port
PORT = 8000

class RequestHandler(http.server.SimpleHTTPRequestHandler):
    """Custom request handler for the server."""
    
    def end_headers(self):
        """Add CORS headers to allow local file access."""
        self.send_header('Access-Control-Allow-Origin', '*')
        super().end_headers()


def main():
    """Start the server."""
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
