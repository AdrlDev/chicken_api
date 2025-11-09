from typing import Optional
import os
import requests
from datetime import datetime, timezone
import jwt

class LabelStudioAuth:
    def __init__(self):
        self.url = os.getenv('LABEL_STUDIO_URL')
        self.refresh_token = os.getenv('LABEL_STUDIO_API_KEY')
        self._access_token: Optional[str] = None
        self._token_expiry: Optional[float] = None
    
    def get_access_token(self) -> str:
        """Get a valid access token, refreshing if necessary."""
        # Check if we have a valid token
        if self._access_token and self._token_expiry:
            # Add 30-second buffer to expiry check
            if datetime.now(timezone.utc).timestamp() < (self._token_expiry - 30):
                return self._access_token
        
        # Get new access token
        response = requests.post(
            f"{self.url}/api/token/refresh",
            json={"refresh": self.refresh_token},
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code != 200:
            raise Exception(f"Failed to refresh token: {response.text}")
        
        self._access_token = response.json()['access']
        
        # Decode token to get expiry
        decoded = jwt.decode(self._access_token, options={"verify_signature": False}) # type: ignore
        self._token_expiry = decoded.get('exp')
        
        return self._access_token # type: ignore
    
    def get_client(self):
        """Get a Label Studio SDK client with a fresh token."""
        from label_studio_sdk import Client
        return Client(url=self.url, api_key=self.get_access_token()) # type: ignore

# Global instance
label_studio = LabelStudioAuth()