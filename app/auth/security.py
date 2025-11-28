## File: app/auth/security.py
# app/auth/security.py
# This module handles JWT creation and user authentication in FastAPI.
from datetime import datetime, timedelta, timezone
from fastapi import HTTPException, status, Request
from jose import JWTError, jwt
from typing import Any, Union, cast

# Import configuration and models from your project structure
from .config import SECRET_KEY, ALGORITHM, ACCESS_TOKEN_EXPIRE_MINUTES
from .models import get_user_by_email
from .schemas import UserOut # 💡 NEW: Import the UserOut schema

# --- 1. JWT Creation Function ---

def create_access_token(data: dict) -> str:
    """Generates a signed JWT for the given data payload."""
    to_encode = data.copy()
    
    # Calculate expiration time
    expire = datetime.now(timezone.utc) + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire.timestamp()})
    
    # Encode the JWT using the secret key and algorithm
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

# ----------------------------------------------------------------------
## --- 2. Authentication Dependency (get_current_user) ---

async def get_current_user(request: Request) -> UserOut:
    """
    Dependency that decodes and validates the JWT, looks up the user, 
    and converts the DB row to a Pydantic UserOut model.
    """
    
    token = request.headers.get("authorization") 
    
    # Initialize exception object for cleaner error handling
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    if not token: 
        credentials_exception.detail = "Not authenticated: Missing Token"
        raise credentials_exception
    
    # Ensure token is stripped of 'Bearer ' prefix
    if token.lower().startswith("bearer "):
        token = token[7:] 
        
    email: str = "" # Initialize 'email' to satisfy Pylance
    
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email_value = payload.get("sub")
        
        if email_value is None:
            credentials_exception.detail = "Invalid token payload: Missing 'sub' claim"
            raise credentials_exception
        
        email = email_value
            
    except JWTError:
        credentials_exception.detail = "Invalid credentials: Token decoding failed or expired"
        raise credentials_exception
        
    # Look up the user in the database
    user_row = await get_user_by_email(email)
    
    if user_row is None:
        credentials_exception.detail = "User not found"
        raise credentials_exception
        
    # 💥 CRITICAL FIX: Explicitly convert the sqlite3.Row object to a dict,
    # then validate it with Pydantic. This resolves most 500 errors in this flow.
    try:
        user_data_dict = dict(user_row)
        return UserOut.model_validate(user_data_dict)
    except Exception as e:
        # This catches errors if the database columns don't match the Pydantic schema
        print(f"Pydantic validation failed during /auth/me: {e}")
        # Return a 401 instead of a 500, as the data is bad/invalid for this API
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User data format error. Contact admin.",
            headers={"WWW-Authenticate": "Bearer"},
        )