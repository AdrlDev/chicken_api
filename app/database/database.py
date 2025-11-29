# app/auth/database.py
import aiosqlite
import os
from pathlib import Path

# --- FIX START ---
# Get the directory of the *current file* (database.py)
BASE_DIR = Path(__file__).resolve().parent

# Define the path to the database file:
# Go up one level (auth/), up one more level (app/), then into 'database' directory.
# This results in: /root/chicken_api/app/database/data.db
DB_PATH = BASE_DIR.parent.parent / "database" / "data.db"
# --- FIX END ---

async def init_db():
    # Ensure the directory exists before attempting to connect
    # Note: os.makedirs("app/auth", exist_ok=True) here is likely leftover
    # and unnecessary since the DB is in 'app/database'
    os.makedirs(DB_PATH.parent, exist_ok=True)

    async with aiosqlite.connect(DB_PATH.as_posix()) as db:
        await db.execute(
            """
             CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                email TEXT UNIQUE NOT NULL,
                password TEXT NOT NULL,
                accountType TEXT NOT NULL DEFAULT 'admin',
                createdAt TEXT NOT NULL
            );
            """
        )
        await db.execute(
            """
            CREATE TABLE IF NOT EXISTS scans (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                diagnosis TEXT NOT NULL,
                farm_id INTEGER NOT NULL,
                timestamp TEXT NOT NULL
            );
            """
        )
        await db.commit()

async def get_db():
    db = await aiosqlite.connect(DB_PATH.as_posix())
    db.row_factory = aiosqlite.Row
    return db
