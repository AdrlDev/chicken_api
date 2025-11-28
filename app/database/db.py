# app/api/chicken_scans/db.py

from datetime import datetime
from ..chicken_scans.models import ScanResultIn
from app.database.database import get_db

# --- Database Operations ---

async def insert_scan_result(scan_data: ScanResultIn, user_id: int) -> dict:
    """Inserts a new scan result into the database."""
    db = await get_db()

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
    db = await get_db()

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