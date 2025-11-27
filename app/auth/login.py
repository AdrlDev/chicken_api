from fastapi import APIRouter, HTTPException
from .schemas import UserCreate, UserLogin, UserOut
from .models import create_user, get_user_by_email, verify_password, MAX_BCRYPT_LENGTH

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

@router.post("/login", response_model=UserOut)
async def login(data: UserLogin):
    user = await get_user_by_email(data.email)

    if not user:
        raise HTTPException(status_code=400, detail="Invalid credentials")

    if not verify_password(data.password, user["password"]):
        raise HTTPException(status_code=400, detail="Invalid credentials")

    return UserOut(
        id=user["id"],
        email=user["email"],
        accountType=user["accountType"],
        createdAt=user["createdAt"]
    )
