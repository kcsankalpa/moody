from typing import Optional

from fastapi import FastAPI, HTTPException, Security, Depends, status, Request
from fastapi.responses import JSONResponse
from fastapi.security import APIKeyHeader
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http import models
import os
import uuid
from dotenv import load_dotenv
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '../../.env'))

app = FastAPI()

# Rate limiting setup
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter

@app.exception_handler(RateLimitExceeded)
async def custom_rate_limit_handler(request: Request, exc: RateLimitExceeded):
    return JSONResponse(
        status_code=status.HTTP_429_TOO_MANY_REQUESTS,
        content={"detail": "Too many requests. Please try again later."},
    )

# API Key Auth setup
API_KEY = os.getenv("API_KEY")
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

async def verify_api_key(api_key: str = Depends(api_key_header)):
    if not api_key or api_key != API_KEY:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Forbidden: Invalid or missing API key",
        )
    return api_key

# Load model
model_name = "sankalpakc/NepaliKD-SentenceTransformers-paraphrase-multilingual-MiniLM-L12-v2"
model = SentenceTransformer(model_name)

# Qdrant client
qdrant_url = os.getenv("QDRANT_URL")
qdrant_api_key = os.getenv("QDRANT_API_KEY")

if qdrant_api_key:
    client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
else:
    client = QdrantClient(url=qdrant_url)

collection_name = "moody_songs"

# Ensure collection exists
try:
    client.get_collection(collection_name=collection_name)
except Exception:
    client.create_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(distance=models.Distance.COSINE, size=384),
    )


class SongIngest(BaseModel):
    title: str
    artist: str
    lyrics: str


class SearchQuery(BaseModel):
    mood: str
    artist: Optional[str] = None


@app.post("/ingest", dependencies=[Depends(verify_api_key)])
@limiter.limit("30/minute")
async def ingest_song(request: Request, song: SongIngest):
    vector = model.encode(song.lyrics).tolist()
    point_id = str(uuid.uuid4())

    client.upsert(
        collection_name=collection_name,
        points=[
            models.PointStruct(
                id=point_id,
                vector=vector,
                payload={
                    "title": song.title,
                    "artist": song.artist,
                    "lyrics": song.lyrics,
                },
            )
        ],
    )
    return {"status": "success", "id": point_id}


@app.post("/search", dependencies=[Depends(verify_api_key)])
@limiter.limit("60/minute")
async def search_songs(request: Request, query: SearchQuery):
    vector = model.encode(query.mood).tolist()

    query_filter = None
    if query.artist:
        query_filter = models.Filter(
            must=[
                models.FieldCondition(
                    key="artist",
                    match=models.MatchText(text=query.artist),
                )
            ]
        )

    search_result = client.query_points(
        collection_name=collection_name,
        query=vector,
        query_filter=query_filter,
        limit=5,
    )

    if not search_result.points:
        return {"results": [], "message": "No songs found"}

    results = []
    for point in search_result.points:
        results.append(
            {
                "title": point.payload["title"],
                "artist": point.payload["artist"],
                "lyrics": point.payload["lyrics"],
                "score": point.score,
            }
        )

    return {"results": results}

