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

# CHANGE FUNCTION SIGNATURE to accept the Request object
async def get_current_user(request: Request):
    """
    Dependency that decodes and validates the JWT by manually reading 
    the Authorization header from the Request object.
    """
    # MANUALLY RETRIEVE the header, standardizing to lowercase 'authorization'
    token = request.headers.get("authorization") 
    
    # This is the line that was failing before:
    if not token: 
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated: Missing Token (Failed manual check)",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Handle the "Bearer " prefix (Now done on the retrieved string)
    if token.lower().startswith("bearer "):
        token = token[7:] 
    
    # The rest of your logic remains the same (try/except block)
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        
        email_value = payload.get("sub")
        
        if email_value is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token payload: Missing 'sub' claim",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        email: str = email_value
            
    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials: Token decoding failed or expired",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Look up the user in the database to ensure they still exist and are active
    user = await get_user_by_email(email)
    
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found",
            headers={"WWW-Authenticate": "Bearer"},
        )
        
    return dict(user)