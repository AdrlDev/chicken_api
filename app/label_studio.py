# app/label_studio.py
import os
from label_studio_sdk import AsyncLabelStudio

LS_URL = os.getenv("LABEL_STUDIO_URL")
LS_API_KEY = os.getenv("LABEL_STUDIO_API_KEY")

def get_client() -> AsyncLabelStudio:
    """
    Return a Label Studio SDK client
    """
    if not LS_URL or not LS_API_KEY:
        raise ValueError("LABEL_STUDIO_URL or LABEL_STUDIO_API_KEY not set in .env")
    client = AsyncLabelStudio(base_url=LS_URL, api_key=LS_API_KEY)
    return client