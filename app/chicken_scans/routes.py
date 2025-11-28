# app/api/chicken_scans/routes.py

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from .models import ScanResultIn, ScanResultOut, DISEASE_LABELS
from .db import insert_scan_result, get_scan_counts_by_diagnosis
from typing import Annotated
from app.auth.security import decode_access_token
# Assuming you have an auth system like this:
# from app.auth.dependencies import get_current_user 

router = APIRouter(
    prefix="/scans",
    tags=["chicken_scans"],
)

# 1. Define the OAuth2 scheme for token extraction (Looks for 'Bearer <token>' in the Authorization header)
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login") # Point this to your login endpoint

# 2. Updated Dependency function to extract and validate the User ID
async def get_current_user_id(token: Annotated[str, Depends(oauth2_scheme)]) -> int:
    """
    Extracts the user ID from the JWT token provided in the Authorization header.
    Raises 401 Unauthorized if the token is missing, expired, or invalid.
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    payload = decode_access_token(token)
    
    if payload is None:
        raise credentials_exception

    # **CRITICAL:** Ensure the user ID key used here matches the key you put into the token during login
    user_id_str = payload.get("sub") 
    
    if user_id_str is None:
        raise credentials_exception
    
    try:
        # The user ID is stored as a string in the JWT, so we convert it to an integer
        user_id = int(user_id_str)
    except ValueError:
        # Handles case where 'sub' is not a valid integer
        raise credentials_exception
    
    return user_id

@router.post(
    "/", 
    response_model=ScanResultOut,
    status_code=status.HTTP_201_CREATED
)
async def create_scan_result(
    scan_in: ScanResultIn, 
    user_id: int = Depends(get_current_user_id) # Use the authenticated user's ID
):
    """
    Saves a new chicken scan result to the database.
    """
    # 1. Input Validation
    if scan_in.diagnosis not in DISEASE_LABELS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid diagnosis label. Must be one of: {', '.join(DISEASE_LABELS)}"
        )
        
    # 2. Database Insertion
    try:
        new_scan = await insert_scan_result(scan_in, user_id)
        return new_scan
    except Exception as e:
        # Log the error (not shown here)
        print(f"Database insertion failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Could not save scan result."
        )
    
@router.get(
    "/counts", 
    # response_model=List[ScanDataOut] (implied list structure)
)
async def get_scan_breakdown(
    # 👈 Use the authenticated user's ID as the filter
    user_id: int = Depends(get_current_user_id) 
):
    """
    Retrieves the total number of scans grouped by their diagnosis for the authenticated user/farm.
    """
    try:
        # 👈 Pass the authenticated ID to the DB function
        data = await get_scan_counts_by_diagnosis(user_id) 
        return data
    except Exception as e:
        print(f"Database query failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Could not retrieve scan data."
        )