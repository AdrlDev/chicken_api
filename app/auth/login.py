## app/auth/login.py
# app/auth/login.py
# This module defines the authentication routes for user registration and login.
from fastapi import APIRouter, HTTPException, Depends
from .schemas import UserCreate, UserLogin, UserOut, Token
from .models import create_user, get_user_by_email, verify_password
from typing import Annotated
from .security import get_current_user, create_access_token

router = APIRouter(prefix="/auth", tags=["Auth"])

@router.post("/register", response_model=UserOut)
async def register(data: UserCreate):
    existing = await get_user_by_email(data.email)
    if existing:
        raise HTTPException(status_code=400, detail="Email already registered")

    user, error = await create_user(data.email, data.password, accountType=data.accountType or "admin")

    if error:
        raise HTTPException(status_code=400, detail=error)

    # Now it's safe to access user["id"] etc.
    return UserOut(
        id=user["id"], # type: ignore
        email=user["email"],  # type: ignore
        accountType=user["accountType"],  # type: ignore
        createdAt=user["createdAt"]  # type: ignore
    )

@router.post("/login", response_model=Token) # <-- CHANGE RESPONSE MODEL TO Token
async def login(data: UserLogin):
    user = await get_user_by_email(data.email)

    if not user:
        raise HTTPException(status_code=400, detail="Invalid credentials")

    if not verify_password(data.password, user["password"]):
        raise HTTPException(status_code=400, detail="Invalid credentials")

    # --- ACTION REQUIRED: Generate JWT ---
    access_token = create_access_token(
        data={"sub": user["email"]} # Use email as the subject for the token
    )
    
    # --- ACTION REQUIRED: Return Token ---
    return Token(access_token=access_token, token_type="bearer") # type: ignore

# --- NEW PROTECTED ROUTE ---
@router.get("/me", response_model=UserOut)
async def read_users_me(
    # FastAPI automatically calls get_current_user. 
    # If get_current_user raises an HTTPException (e.g., 401), the function below is never called.
    current_user: UserOut = Depends(get_current_user)
):
    """
    Returns the currently authenticated user's details.
    """
    
    # 💥 FIX: Change from dictionary access (current_user["id"]) 
    #          to attribute access (current_user.id)
    return {
        "id": current_user.id,             # <-- Change required here
        "email": current_user.email,       # <-- Change required here
        "accountType": current_user.accountType, # <-- Change required here
        "createdAt": current_user.createdAt  # <-- Change required here
    }
    
    # Alternatively, if UserOut is correctly configured, you can just return the object:
    # return current_user
