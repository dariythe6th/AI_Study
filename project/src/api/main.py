import logging
from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pathlib import Path
from contextlib import asynccontextmanager

from src.config import settings
from src.utils.logging_config import setup_logging
from src.service.schemas import RecommendRequest, RecommendResponse, UserRegister, UserLogin, Token, FavoriteTrack
from src.service.engine import MoodTuneEngine
from src.service.auth import UserManager, get_current_user

setup_logging()
logger = logging.getLogger(__name__)

engine = MoodTuneEngine()
user_manager = UserManager()

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀 Starting MoodTune API...")
    engine.initialize()
    logger.info("✅ MoodTune Engine initialized")
    yield
    logger.info("👋 Shutting down MoodTune")

app = FastAPI(title="MoodTune API", version="1.0.0", lifespan=lifespan)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],           # В продакшене лучше ограничить
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# === Статические файлы (фронтенд) ===
frontend_path = Path(__file__).parent.parent / "frontend"

# Монтируем папку frontend как /static
app.mount("/static", StaticFiles(directory=frontend_path), name="static")

@app.get("/")
async def serve_frontend():
    """Отдаём главную страницу"""
    index_file = frontend_path / "index.html"
    if index_file.exists():
        return FileResponse(index_file)
    return {"message": "Frontend not found. Check /static/index.html"}

# === API Endpoints ===
@app.get("/health")
async def health():
    return {"status": "ok", "message": "MoodTune is running"}

@app.post("/recommend", response_model=RecommendResponse)
async def recommend(req: RecommendRequest, user=Depends(get_current_user)):
    try:
        result = engine.get_recommendations(req.text, req.limit)
        return result
    except Exception as e:
        logger.error(f"Recommendation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Auth routes
@app.post("/auth/register")
async def register(user_data: UserRegister):
    return user_manager.create_user(user_data)

@app.post("/auth/login", response_model=Token)
async def login(user_data: UserLogin):
    return user_manager.authenticate_user(user_data)

@app.get("/auth/me")
async def get_current_user_profile(user=Depends(get_current_user)):
    """Получить информацию о текущем пользователе"""
    return user

# Favorites

@app.get("/favorites")
async def get_favorites(user=Depends(get_current_user)):
    """Получить избранные треки пользователя"""
    # Пока возвращаем пустой список (можно расширить позже)
    return {"favorites": []}

@app.post("/favorites")
async def add_favorite(track: FavoriteTrack, user=Depends(get_current_user)):
    return {"message": "Track added to favorites", "track": track.dict()}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api.main:app", host="0.0.0.0", port=8000, reload=True)