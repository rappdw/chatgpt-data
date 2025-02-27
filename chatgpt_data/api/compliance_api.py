"""ChatGPT Enterprise Compliance API client."""

import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import requests
from requests.auth import HTTPBasicAuth


class EnterpriseComplianceAPI:
    """Client for the ChatGPT Enterprise Compliance API."""

    BASE_URL = "https://chatgpt.com/api/enterprise/v1"

    def __init__(
        self, 
        api_key: Optional[str] = None, 
        org_id: Optional[str] = None,
        output_dir: Union[str, Path] = "./rawdata"
    ):
        """Initialize the Enterprise Compliance API client.

        Args:
            api_key: Enterprise API key (defaults to OPENAI_ENTERPRISE_API_KEY env var)
            org_id: Organization ID (defaults to OPENAI_ORG_ID env var)
            output_dir: Directory to save downloaded reports
        """
        self.api_key = api_key or os.environ.get("OPENAI_ENTERPRISE_API_KEY")
        self.org_id = org_id or os.environ.get("OPENAI_ORG_ID")
        
        if not self.api_key:
            raise ValueError("API key is required. Set OPENAI_ENTERPRISE_API_KEY env var or pass api_key")
        
        if not self.org_id:
            raise ValueError("Organization ID is required. Set OPENAI_ORG_ID env var or pass org_id")
        
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
    def _get_headers(self) -> Dict[str, str]:
        """Get the headers required for API requests.

        Returns:
            Dictionary of HTTP headers
        """
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
    
    def _make_request(
        self, 
        endpoint: str, 
        method: str = "GET", 
        params: Optional[Dict] = None,
        json_data: Optional[Dict] = None
    ) -> Dict:
        """Make a request to the Enterprise Compliance API.

        Args:
            endpoint: API endpoint path
            method: HTTP method (GET, POST, etc.)
            params: URL parameters
            json_data: JSON request body

        Returns:
            JSON response data
        """
        url = f"{self.BASE_URL}/{endpoint}"
        headers = self._get_headers()
        
        response = requests.request(
            method=method,
            url=url,
            headers=headers,
            params=params,
            json=json_data,
            timeout=30
        )
        
        response.raise_for_status()
        return response.json()
    
    def get_user_engagement_report(
        self, 
        start_date: datetime,
        end_date: datetime
    ) -> Path:
        """Get user engagement report for the specified time period.

        Args:
            start_date: Start date for the report
            end_date: End date for the report

        Returns:
            Path to the saved report file
        """
        # Format dates as ISO strings
        start_str = start_date.strftime("%Y-%m-%d")
        end_str = end_date.strftime("%Y-%m-%d")
        
        # Make API request to fetch user engagement data
        data = self._make_request(
            endpoint="compliance/reports/user_engagement",
            method="POST",
            json_data={
                "org_id": self.org_id,
                "start_date": start_str,
                "end_date": end_str
            }
        )
        
        # Save data to CSV using the same format as existing files
        filename = f"proofpoint_user_engagement_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.csv"
        file_path = self.output_dir / filename
        
        # Process and save the data as CSV
        # (The exact format depends on the API response structure)
        with open(file_path, "w") as f:
            # Write header row
            if data.get("headers"):
                f.write(",".join(data["headers"]) + "\n")
            
            # Write data rows
            for row in data.get("rows", []):
                f.write(",".join(str(field) for field in row) + "\n")
        
        return file_path
    
    def get_gpt_engagement_report(
        self, 
        start_date: datetime,
        end_date: datetime
    ) -> Path:
        """Get GPT engagement report for the specified time period.

        Args:
            start_date: Start date for the report
            end_date: End date for the report

        Returns:
            Path to the saved report file
        """
        # Format dates as ISO strings
        start_str = start_date.strftime("%Y-%m-%d")
        end_str = end_date.strftime("%Y-%m-%d")
        
        # Make API request to fetch GPT engagement data
        data = self._make_request(
            endpoint="compliance/reports/gpt_engagement",
            method="POST",
            json_data={
                "org_id": self.org_id,
                "start_date": start_str,
                "end_date": end_str
            }
        )
        
        # Save data to CSV using the same format as existing files
        filename = f"proofpoint_gpt_engagement_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.csv"
        file_path = self.output_dir / filename
        
        # Process and save the data as CSV
        # (The exact format depends on the API response structure)
        with open(file_path, "w") as f:
            # Write header row
            if data.get("headers"):
                f.write(",".join(data["headers"]) + "\n")
            
            # Write data rows
            for row in data.get("rows", []):
                f.write(",".join(str(field) for field in row) + "\n")
        
        return file_path