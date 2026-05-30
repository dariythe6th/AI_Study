import logging
from datetime import datetime, timedelta
from typing import Optional

import jwt
from fastapi import HTTPException, status, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel

from src.config import settings
from src.service.schemas import UserRegister, UserLogin, Token

logger = logging.getLogger(__name__)

# Security
security = HTTPBearer()
SECRET_KEY = settings.SECRET_KEY
ALGORITHM = settings.ALGORITHM
ACCESS_TOKEN_EXPIRE_MINUTES = settings.ACCESS_TOKEN_EXPIRE_MINUTES


# ====================== DATABASE MODELS ======================
class User(BaseModel):
    id: int
    username: str
    email: str


# ====================== USER MANAGER ======================
class UserManager:
    def __init__(self):
        self._init_db()

    def _init_db(self):
        """Инициализация SQLite базы данных"""
        import sqlite3
        conn = sqlite3.connect('moodtune.db')
        cursor = conn.cursor()

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                email TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        # Favorites table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS favorites (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                track_id TEXT NOT NULL,
                song_name TEXT NOT NULL,
                artist_name TEXT NOT NULL,
                album TEXT,
                match_percentage INTEGER,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (id),
                UNIQUE(user_id, track_id)
            )
        ''')

        conn.commit()
        conn.close()

    def create_user(self, user_data: UserRegister) -> dict:
        import sqlite3
        import hashlib

        conn = sqlite3.connect('moodtune.db')
        cursor = conn.cursor()

        try:
            # Check if user exists
            cursor.execute(
                "SELECT id FROM users WHERE email = ? OR username = ?",
                (user_data.email, user_data.username)
            )
            if cursor.fetchone():
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Пользователь с таким email или username уже существует"
                )

            # Hash password
            password_hash = hashlib.sha256(user_data.password.encode()).hexdigest()

            cursor.execute(
                "INSERT INTO users (username, email, password_hash) VALUES (?, ?, ?)",
                (user_data.username, user_data.email, password_hash)
            )
            conn.commit()

            user_id = cursor.lastrowid
            logger.info(f"New user created: {user_data.username} ({user_data.email})")

            return {
                "message": "Пользователь успешно создан",
                "user_id": user_id,
                "username": user_data.username
            }

        except sqlite3.IntegrityError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Пользователь уже существует"
            )
        finally:
            conn.close()

    def authenticate_user(self, user_data: UserLogin) -> Token:
        import sqlite3
        import hashlib

        conn = sqlite3.connect('moodtune.db')
        cursor = conn.cursor()

        try:
            cursor.execute(
                "SELECT id, username, email, password_hash FROM users WHERE email = ?",
                (user_data.email,)
            )
            user = cursor.fetchone()

            if not user:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Неверный email или пароль"
                )

            password_hash = hashlib.sha256(user_data.password.encode()).hexdigest()

            if password_hash != user[3]:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Неверный email или пароль"
                )

            # Create JWT token
            access_token = self.create_access_token(data={"sub": str(user[0])})

            return Token(
                access_token=access_token,
                token_type="bearer"
            )

        finally:
            conn.close()

    def create_access_token(self, data: dict, expires_delta: Optional[timedelta] = None) -> str:
        to_encode = data.copy()
        expire = datetime.utcnow() + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
        to_encode.update({"exp": expire})
        encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
        return encoded_jwt

    def get_user_by_id(self, user_id: int) -> Optional[dict]:
        import sqlite3
        conn = sqlite3.connect('moodtune.db')
        cursor = conn.cursor()

        try:
            cursor.execute(
                "SELECT id, username, email FROM users WHERE id = ?",
                (user_id,)
            )
            user = cursor.fetchone()
            if user:
                return {
                    "id": user[0],
                    "username": user[1],
                    "email": user[2]
                }
            return None
        finally:
            conn.close()


# ====================== DEPENDENCY ======================
user_manager = UserManager()


async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    try:
        payload = jwt.decode(credentials.credentials, SECRET_KEY, algorithms=[ALGORITHM])
        user_id = int(payload.get("sub"))
        user = user_manager.get_user_by_id(user_id)

        if user is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Пользователь не найден"
            )
        return user

    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Срок действия токена истёк"
        )
    except jwt.InvalidTokenError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Недействительный токен"
        )
    except Exception as e:
        logger.error(f"Auth error: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Не удалось проверить авторизацию"
        )