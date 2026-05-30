from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional

class Settings(BaseSettings):
    SECRET_KEY: str = "super-secret-key-change-in-production"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 1440

    MODEL_NAME: str = "all-MiniLM-L6-v2"
    CHROMA_PERSIST_DIRECTORY: str = "./chroma_db"
    DATABASE_URL: str = "sqlite:///./moodtune.db"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

settings = Settings()