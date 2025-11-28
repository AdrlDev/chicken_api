## File: app/auth/config.py
# # app/auth/config.py
# # This module contains configuration settings for the authentication system.
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# --- Configuration Settings ---

# Get the SECRET_KEY from the environment variable. 
# The application will fail (stop) if the key is not set.
SECRET_KEY = os.getenv("SECRET_KEY")

if not SECRET_KEY:
    # This is a good practice to prevent the application from running without a key
    raise ValueError("SECRET_KEY environment variable not set. Check your .env file.")

ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30