# app/api/chicken_scans/routes.py

from fastapi import APIRouter, Depends, HTTPException, status
from .models import ScanResultIn, ScanResultOut, DISEASE_LABELS
from ..database.db import insert_scan_result, get_scan_counts_by_diagnosis
from typing import Annotated
# Assuming you have an auth system like this:
# from app.auth.dependencies import get_current_user 
# 💡 IMPORT: Bring in the full user object validation function 
# and the User model from your auth system.
from app.auth.security import get_current_user 
from app.auth.schemas import UserOut # Assuming your ORM/DB model is named User and has an 'id' attribute

router = APIRouter(
    prefix="/scans",
    tags=["chicken_scans"],
)

# 2. Updated Dependency function to extract and validate the User ID
# ----------------------------------------------------------------------
# 💡 FIXED DEPENDENCY: Get the User ID from the validated User object
# ----------------------------------------------------------------------
async def get_current_user_id(current_user: Annotated[UserOut, Depends(get_current_user)]) -> int:
    """
    Retrieves the user's numerical ID from the fully validated User object.
    """
    
    # 💥 FINAL CLEANUP: Now that get_current_user returns a Pydantic model, 
    # we can use clean dot-access.
    return current_user.id # 👈 Use attribute access (.id)

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