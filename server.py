#!/usr/bin/env python3
"""
Simple HTTP server for the ChatGPT Enterprise Engagement Metrics dashboard.
Serves the HTML file and handles the command execution.
"""

import http.server
import json
import os
import socketserver
import subprocess
import sys
from pathlib import Path
from urllib.parse import parse_qs, urlparse

# Get the directory of this script
SCRIPT_DIR = Path(__file__).resolve().parent

# Default port
PORT = 8000

class RequestHandler(http.server.SimpleHTTPRequestHandler):
    """Custom request handler for the server."""
    
    def do_POST(self):
        """Handle POST requests."""
        if self.path == '/run-command':
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length).decode('utf-8')
            data = json.loads(post_data)
            
            if 'command' in data and data['command'] == 'all_trends':
                try:
                    # Run the all_trends command
                    result = self._run_all_trends()
                    
                    # Send response
                    self.send_response(200)
                    self.send_header('Content-type', 'application/json')
                    self.end_headers()
                    self.wfile.write(json.dumps({'success': True, 'result': result}).encode())
                except Exception as e:
                    self.send_response(500)
                    self.send_header('Content-type', 'application/json')
                    self.end_headers()
                    self.wfile.write(json.dumps({'success': False, 'error': str(e)}).encode())
            else:
                self.send_response(400)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({'success': False, 'error': 'Invalid command'}).encode())
        else:
            self.send_response(404)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({'success': False, 'error': 'Not found'}).encode())
    
    def _run_all_trends(self):
        """Run the all_trends command."""
        # Get the path to the all_trends.py script
        all_trends_script = SCRIPT_DIR / 'chatgpt_data' / 'cli' / 'all_trends.py'
        
        # Check if the script exists
        if not all_trends_script.exists():
            all_trends_script = SCRIPT_DIR / 'chatgpt_data' / 'cli' / 'all_trends.py'
            if not all_trends_script.exists():
                raise FileNotFoundError(f"Could not find all_trends.py script at {all_trends_script}")
        
        # Run the script
        result = subprocess.run(
            [sys.executable, str(all_trends_script)],
            capture_output=True,
            text=True,
            check=True
        )
        
        return {
            'stdout': result.stdout,
            'stderr': result.stderr,
            'returncode': result.returncode
        }


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
