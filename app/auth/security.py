## File: app/auth/security.py
# app/auth/security.py
# This module handles JWT creation and user authentication in FastAPI.
from datetime import datetime, timedelta, timezone
from typing import Annotated, Any

from fastapi import Header, HTTPException, status, Depends
from jose import JWTError, jwt

# Import configuration and models from your project structure
from .config import SECRET_KEY, ALGORITHM, ACCESS_TOKEN_EXPIRE_MINUTES
from .models import get_user_by_email

# --- 1. JWT Creation Function ---

def create_access_token(data: dict) -> str:
    """Generates a signed JWT for the given data payload."""
    to_encode = data.copy()
    
    # Calculate expiration time
    expire = datetime.now(timezone.utc) + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    
    # Encode the JWT using the secret key and algorithm
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

# ----------------------------------------------------------------------
## --- 2. Authentication Dependency (get_current_user) ---

async def get_current_user(
    # FastAPI/Starlette automatically looks for the Authorization header
    token: Annotated[str, Header()] = None  # type: ignore
):
    """
    Dependency that decodes and validates the JWT from the Authorization header.
    Returns the user dictionary if valid, raises 401 otherwise.
    """
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated: Missing Token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Handle the "Bearer " prefix if it's included in the header value
    if token.lower().startswith("bearer "):
        token = token[7:] 
        
    try:
        # Decode the token. This automatically checks the signature and the 'exp' (expiration) claim.
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        
        # The 'sub' claim (Subject) holds the user's email, which we set during login
        email_value = payload.get("sub")
        
        # 2. Check if the value is None (if 'sub' was missing from the token)
        if email_value is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token payload: Missing 'sub' claim",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        # 3. Since we checked it's not None, we can now safely assert the type
        email: str = email_value
            
    except JWTError:
        # This catches all validation failures (expired token, invalid signature, etc.)
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
        
    # The dependency successfully resolves, and the user data is passed to the endpoint
    return user