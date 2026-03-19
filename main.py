from typing import Optional

from fastapi import FastAPI, HTTPException, Security, Depends, status, Request
from fastapi.responses import JSONResponse
from fastapi.security import APIKeyHeader
from pydantic import BaseModel
import onnxruntime as ort
from tokenizers import Tokenizer
import numpy as np
import httpx
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

# Load ONNX model and tokenizer
MODEL_DIR = os.getenv("MODEL_DIR", "/app/model")
MAX_LENGTH = 128
sess_options = ort.SessionOptions()
sess_options.intra_op_num_threads = 1
sess_options.inter_op_num_threads = 1
session = ort.InferenceSession(os.path.join(MODEL_DIR, "model_quantized.onnx"), sess_options)
tokenizer = Tokenizer.from_file(os.path.join(MODEL_DIR, "tokenizer.json"))
tokenizer.enable_padding(pad_id=0, pad_token="[PAD]", length=MAX_LENGTH)
tokenizer.enable_truncation(max_length=MAX_LENGTH)


def encode(text: str) -> list[float]:
    """Encode text to a 384-dim embedding using ONNX model with mean pooling."""
    encoded = tokenizer.encode(text)
    input_ids = np.array([encoded.ids], dtype=np.int64)
    attention_mask = np.array([encoded.attention_mask], dtype=np.int64)
    token_type_ids = np.array([encoded.type_ids], dtype=np.int64)

    output = session.run(
        None,
        {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
        },
    )[0]
    # Mean pooling
    mask = np.broadcast_to(
        np.expand_dims(attention_mask, -1), output.shape
    )
    sum_embeddings = np.sum(output * mask, axis=1)
    sum_mask = np.clip(mask.sum(axis=1), a_min=1e-9, a_max=None)
    embedding = (sum_embeddings / sum_mask)[0]
    return embedding.tolist()

# Qdrant REST client
QDRANT_URL = os.getenv("QDRANT_URL", "").rstrip("/")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION = "moody_songs"

qdrant_headers = {"Content-Type": "application/json"}
if QDRANT_API_KEY:
    qdrant_headers["api-key"] = QDRANT_API_KEY

qdrant = httpx.Client(base_url=QDRANT_URL, headers=qdrant_headers, timeout=30)

# Ensure collection exists
resp = qdrant.get(f"/collections/{COLLECTION}")
if resp.status_code == 404:
    qdrant.put(f"/collections/{COLLECTION}", json={
        "vectors": {"size": 384, "distance": "Cosine"}
    })


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
    vector = encode(song.lyrics)
    point_id = str(uuid.uuid4())

    qdrant.put(f"/collections/{COLLECTION}/points", json={
        "points": [{
            "id": point_id,
            "vector": vector,
            "payload": {
                "title": song.title,
                "artist": song.artist,
                "lyrics": song.lyrics,
            },
        }]
    })
    return {"status": "success", "id": point_id}


@app.post("/search", dependencies=[Depends(verify_api_key)])
@limiter.limit("60/minute")
async def search_songs(request: Request, query: SearchQuery):
    vector = encode(query.mood)

    search_body: dict = {"vector": vector, "limit": 5, "with_payload": True}
    if query.artist:
        search_body["filter"] = {
            "must": [{"key": "artist", "match": {"text": query.artist}}]
        }

    resp = qdrant.post(f"/collections/{COLLECTION}/points/search", json=search_body)
    data = resp.json().get("result", [])

    if not data:
        return {"results": [], "message": "No songs found"}

    results = [
        {
            "title": point["payload"]["title"],
            "artist": point["payload"]["artist"],
            "lyrics": point["payload"]["lyrics"],
            "score": point["score"],
        }
        for point in data
    ]

    return {"results": results}

