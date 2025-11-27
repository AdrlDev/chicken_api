# app/auth/models.py
from .database import get_db
from passlib.context import CryptContext
from datetime import datetime
from email_validator import validate_email, EmailNotValidError

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def validate_email_address(email: str) -> str:
    """Validates and normalizes an email. Raises ValueError if invalid."""
    try:
        valid = validate_email(email)
        return valid.email  # normalized email
    except EmailNotValidError as e:
        raise ValueError(str(e))


async def create_user(email: str, password: str, accountType: str = "admin"):
    # Validate email
    try:
        email = validate_email_address(email)
    except ValueError:
        return None  # Signal invalid email to router

    db = await get_db()
    hashed = pwd_context.hash(password)
    created_at = datetime.utcnow().isoformat()

    try:
        await db.execute(
            """
            INSERT INTO users (email, password, accountType, createdAt)
            VALUES (?, ?, ?, ?)
            """,
            (email, hashed, accountType, created_at)
        )
        await db.commit()
    except Exception:
        await db.close()
        return None  # email already exists or other error

    user = await db.execute(
        "SELECT id, email, accountType, createdAt FROM users WHERE email = ?",
        (email,)
    )
    row = await user.fetchone()
    await db.close()
    return row


async def get_user_by_email(email: str):
    try:
        email = validate_email_address(email)
    except ValueError:
        return None

    db = await get_db()
    result = await db.execute(
        "SELECT * FROM users WHERE email = ?", (email,)
    )
    row = await result.fetchone()
    await db.close()
    return row


def verify_password(plain, hashed):
    return pwd_context.verify(plain, hashed)
