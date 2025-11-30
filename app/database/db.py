# app/api/chicken_scans/db.py

from datetime import datetime, date, timedelta
from typing import List, Dict
from ..chicken_scans.models import ScanResultIn, TrendDataResponse
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

# ----------------------------------------------------------------------
# ✅ NEW FUNCTION: Get health trend data by day
# ----------------------------------------------------------------------

async def get_health_trend_data(user_id: int, days: int) -> List[TrendDataResponse]:
    """
    Fetches aggregated daily counts of healthy and issue scans 
    for the last N days for a specific user/farm, using raw SQL.
    """
    db = await get_db()
    
    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=days - 1)
    
    # Format the dates for use in the SQL query (e.g., '2025-11-25')
    start_date_str = start_date.isoformat()
    
    # The SQL uses conditional counting (SUM/COUNT with CASE) and date formatting.
    # Note: SQLite uses strftime('%Y-%m-%d', timestamp) to extract the date part.
    # If using PostgreSQL, you would use DATE(timestamp).
    sql = """
        SELECT
            strftime('%Y-%m-%d', timestamp) AS scan_date,
            SUM(CASE WHEN diagnosis = 'healthy' THEN 1 ELSE 0 END) AS healthy_count,
            SUM(CASE WHEN diagnosis != 'healthy' THEN 1 ELSE 0 END) AS issue_count
        FROM scans
        WHERE 
            farm_id = ? 
            AND strftime('%Y-%m-%d', timestamp) >= ?
        GROUP BY scan_date
        ORDER BY scan_date;
    """

    cursor = await db.execute(sql, (user_id, start_date_str))
    raw_results = await cursor.fetchall()
    
    # ------------------------------------------------------
    # 2. Fill in missing dates with zero counts (in Python)
    # ------------------------------------------------------
    
    # Map the results to a dictionary keyed by date for easy lookup
    db_data: Dict[date, TrendDataResponse] = {}
    for row in raw_results:
        scan_date = datetime.strptime(row[0], '%Y-%m-%d').date()
        db_data[scan_date] = TrendDataResponse(
            date=scan_date,
            healthy_count=row[1],
            issue_count=row[2]
        )

    # Generate the full range and fill missing dates
    trend_data: List[TrendDataResponse] = []
    for i in range(days):
        current_day = start_date + timedelta(days=i)
        
        # If the date is in the results, use the fetched data; otherwise, use zeros
        data_point = db_data.get(
            current_day,
            TrendDataResponse(date=current_day, healthy_count=0, issue_count=0)
        )
        trend_data.append(data_point)
            
    return trend_data