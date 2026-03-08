# app/api/chicken_scans/routes.py

from fastapi import APIRouter, Depends, HTTPException, status, Query
# 💡 REQUIRED IMPORTS: Import date and timedelta for date calculations
from datetime import date, timedelta 
from .models import ScanResultIn, ScanResultOut, DISEASE_LABELS, TrendDataResponse
# 🚀 IMPORT FIX: Import the new async DB function 
# and remove unneeded SQLAlchemy imports (Session, func, case, Date, get_db)
from ..database.db import (
    insert_scan_result, 
    get_scan_counts_by_diagnosis, 
    get_health_trend_data # <-- NEW IMPORT
)
from typing import Annotated
from typing import List
# REMOVED: from sqlalchemy.orm import Session
# REMOVED: from sqlalchemy import func, extract, case, Date

# Assuming your auth system imports
from app.auth.security import get_current_user 
from app.auth.schemas import UserOut 
# REMOVED: from app.models.scan import Scan # Not needed for raw SQL function

router = APIRouter(
    prefix="/scans",
    tags=["chicken_scans"],
)

# 2. Updated Dependency function to extract and validate the User ID
# ----------------------------------------------------------------------
async def get_current_user_id(current_user: Annotated[UserOut, Depends(get_current_user)]) -> int:
    """
    Retrieves the user's numerical ID from the fully validated User object.
    """
    return current_user.id 

@router.post(
    "/", 
    response_model=ScanResultOut,
    status_code=status.HTTP_201_CREATED
)
async def create_scan_result(
    scan_in: ScanResultIn, 
    user_id: int = Depends(get_current_user_id)
):
    """
    Saves a new chicken scan result to the database.
    """
    scan_in.diagnosis = scan_in.diagnosis.replace("\u2019", "'").replace("\u2018", "'").lower()
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
    user_id: int = Depends(get_current_user_id) 
):
    """
    Retrieves the total number of scans grouped by their diagnosis for the authenticated user/farm.
    """
    try:
        data = await get_scan_counts_by_diagnosis(user_id) 
        return data
    except Exception as e:
        print(f"Database query failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Could not retrieve scan data."
        )
    
# ----------------------------------------------------------------------
# 🚀 TREND ENDPOINT: Now calls the raw SQL function in db.py
# ----------------------------------------------------------------------
@router.get(
    "/trend", 
    response_model=List[TrendDataResponse],
    summary="Get daily health scan trend for the authenticated user."
)
# 💡 FIX: This must now be an async function to await the DB call
async def get_health_trend(
    days: int = Query(7, ge=1, le=30, description="Number of past days to include in the trend."),
    # 💡 FIX: Pass the user_id instead of the full Session and User object
    user_id: int = Depends(get_current_user_id) 
):
    """
    Fetches the daily counts of 'Healthy Scans' and 'Detected Issues' 
    for the last N days for the logged-in user by calling the dedicated DB function.
    """
    try:
        # 🚀 LOGIC SWAP: Call the new async DB function which handles the SQL and date padding
        trend_data = await get_health_trend_data(user_id, days)
        return trend_data
    except Exception as e:
        print(f"Failed to fetch health trend data: {e}")
        # Raise a 500 error if the database operation fails
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Could not retrieve health trend data."
        )