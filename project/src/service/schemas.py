from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

class RecommendRequest(BaseModel):
    text: str = Field(..., description="Описание настроения или ситуации")
    limit: int = Field(10, ge=1, le=50)

class TrackRecommendation(BaseModel):
    song_name: str
    artist_name: str
    album: Optional[str] = None
    mood: str
    popularity: Optional[int] = None
    match_percentage: int
    energy: Optional[float] = None
    danceability: Optional[float] = None
    why_it_matches: Optional[str] = None

class RecommendResponse(BaseModel):
    personalized_recommendations: List[TrackRecommendation]
    top_5_songs_for_this_vibe: List[Dict[str, Any]]
    detected_mood: str

class UserRegister(BaseModel):
    username: str
    email: str
    password: str

class UserLogin(BaseModel):
    email: str
    password: str

class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"

class FavoriteTrack(BaseModel):
    track_id: str
    song_name: str
    artist_name: str
    album: Optional[str] = None
    match_percentage: Optional[int] = None