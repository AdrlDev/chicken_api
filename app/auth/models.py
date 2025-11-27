# app/auth/models.py
from .database import get_db
from passlib.context import CryptContext
from datetime import datetime
from email_validator import validate_email, EmailNotValidError

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
MAX_BCRYPT_LENGTH = 72
MIN_PASSWORD_LENGTH = 6


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
        return None, "Invalid email"

    # Check password length
    if len(password) < MIN_PASSWORD_LENGTH:
        return None, f"Password too short, minimum {MIN_PASSWORD_LENGTH} characters"
    
    # Truncate password to MAX_BCRYPT_LENGTH bytes (UTF-8 safe)
    safe_password = truncate_password(password)
    hashed = pwd_context.hash(safe_password)
    created_at = datetime.utcnow().isoformat()

    db = await get_db()
    try:
        await db.execute(
            "INSERT INTO users (email, password, accountType, createdAt) VALUES (?, ?, ?, ?)",
            (email, hashed, accountType, created_at)
        )
        await db.commit()
    except Exception:
        await db.close()
        return None, "Email already exists or database error"

    user = await db.execute(
        "SELECT id, email, accountType, createdAt FROM users WHERE email = ?",
        (email,)
    )
    row = await user.fetchone()
    await db.close()
    return row, None


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

def truncate_password(password: str) -> str:
    # Truncate so that UTF-8 bytes are <= 72
    encoded = password.encode("utf-8")
    if len(encoded) <= MAX_BCRYPT_LENGTH:
        return password
    # Truncate character by character
    truncated = ""
    for char in password:
        if len((truncated + char).encode("utf-8")) > MAX_BCRYPT_LENGTH:
            break
        truncated += char
    return truncated
