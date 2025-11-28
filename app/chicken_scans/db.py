# app/api/chicken_scans/db.py

import aiosqlite
from datetime import datetime
from .models import ScanResultIn

DATABASE_URL = "app/chicken_scans/chicken_health_scans.db"

# --- Database Connection and Setup ---

async def get_db_connection():
    """Returns an asynchronous connection to the SQLite database with safer settings."""
    # 💡 FIX: Increase timeout for better concurrency handling
    db = await aiosqlite.connect(
        database=DATABASE_URL,
        timeout=10, # Wait up to 10 seconds for the database to unlock
    )
    # Ensure rows are returned as sqlite3.Row objects (which work with dict() conversion)
    # db.row_factory = aiosqlite.Row # (If you were not using dict(user_row))

    return db

async def setup_db():
    """Initializes the database table if it doesn't exist."""
    async with await get_db_connection() as db:
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

# --- Database Operations ---

async def insert_scan_result(scan_data: ScanResultIn, user_id: int) -> dict:
    """Inserts a new scan result into the database."""
    async with await get_db_connection() as db:
        
        timestamp = datetime.now().isoformat()
        
        cursor = await db.execute(
            """
            INSERT INTO scans (diagnosis, farm_id, timestamp) 
            VALUES (?, ?, ?)
            """,
            (scan_data.diagnosis, user_id, timestamp)
        )
        
        # Get the ID of the newly inserted row
        last_row_id = cursor.lastrowid
        
        await db.commit()
        
        return {
            "id": last_row_id,
            "diagnosis": scan_data.diagnosis,
            "farm_id": user_id,
            "timestamp": timestamp
        }
    
async def get_scan_counts_by_diagnosis(user_or_farm_id: int) -> list[dict]:
    """Retrieves the count of scans grouped by diagnosis, filtered by user/farm ID."""
    async with await get_db_connection() as db:
        
        cursor = await db.execute(
            """
            SELECT diagnosis, COUNT(*) AS count
            FROM scans
            WHERE farm_id = ?  -- 👈 ADDED FILTERING CONDITION
            GROUP BY diagnosis
            ORDER BY count DESC;
            """,
            (user_or_farm_id,) # 👈 PASS THE ID AS A PARAMETER
        )
        
        rows = await cursor.fetchall()
        
        results = [
            {"diagnosis": row[0], "value": row[1]} 
            for row in rows
        ]
        
        return results