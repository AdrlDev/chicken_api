# app/auth/database.py
import aiosqlite
import os

DB_PATH = "app/auth/auth.db"

async def init_db():
    os.makedirs("app/auth", exist_ok=True)

    async with aiosqlite.connect(DB_PATH) as db:
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
        await db.commit()


async def get_db():
    db = await aiosqlite.connect(DB_PATH)
    db.row_factory = aiosqlite.Row
    return db
