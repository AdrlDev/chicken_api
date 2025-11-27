# app/auth/models.py
from .database import get_db
from passlib.context import CryptContext
from datetime import datetime
from email_validator import validate_email, EmailNotValidError

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
MAX_BCRYPT_LENGTH = 72


def validate_email_address(email: str) -> str:
    """Validates and normalizes an email. Raises ValueError if invalid."""
    try:
        valid = validate_email(email)
        return valid.email  # normalized email
    except EmailNotValidError as e:
        raise ValueError(str(e))


async def create_user(email: str, password: str, accountType: str = "admin"):
    """
    Creates a new user with hashed password, truncated to 72 bytes for bcrypt.
    Returns None if email is invalid or already exists.
    """
    # Validate email
    try:
        email = validate_email_address(email)
    except ValueError:
        return None  # invalid email

    # Truncate password to 72 characters to satisfy bcrypt
    password = password[:MAX_BCRYPT_LENGTH]

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
        return None  # email exists or other DB error

    user = await db.execute(
        "SELECT id, email, accountType, createdAt FROM users WHERE email = ?",
        (email,)
    )
    row = await user.fetchone()
    await db.close()
    return row


async def get_user_by_email(email: str):
    """Returns a user row by email, or None if invalid or not found."""
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


def verify_password(plain: str, hashed: str) -> bool:
    """Verifies a plain password against a hashed password."""
    # Truncate plain password to avoid bcrypt errors
    plain = plain[:MAX_BCRYPT_LENGTH]
    return pwd_context.verify(plain, hashed)
