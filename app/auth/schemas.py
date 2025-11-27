from pydantic import BaseModel, EmailStr
from typing import Optional

class UserCreate(BaseModel):
    email: EmailStr
    password: str
    accountType: Optional[str] = "admin"  # optional, defaults to "user"

class UserLogin(BaseModel):
    email: EmailStr
    password: str

class UserOut(BaseModel):
    id: int
    email: EmailStr
    accountType: str
    createdAt: str

    class Config:
        orm_mode = True
