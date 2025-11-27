from pydantic import BaseModel, EmailStr, field_validator
from typing import Optional

MAX_BCRYPT_LENGTH = 72

class UserCreate(BaseModel):
    email: EmailStr
    password: str
    accountType: Optional[str] = "admin"

    @field_validator("password")
    def max_bcrypt_length(cls, v: str) -> str:
        """Truncate password to 72 chars to avoid bcrypt errors"""
        if len(v) > MAX_BCRYPT_LENGTH:
            return v[:MAX_BCRYPT_LENGTH]
        return v

class UserLogin(BaseModel):
    email: EmailStr
    password: str

class UserOut(BaseModel):
    id: int
    email: EmailStr
    accountType: str
    createdAt: str

    class Config:
        from_attributes = True  # Pydantic v2 replaces orm_mode
