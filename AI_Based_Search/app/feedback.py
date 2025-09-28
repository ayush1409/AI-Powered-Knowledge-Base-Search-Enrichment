# backend/app/feedback.py
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from .config import FEEDBACK_DIR
import json, time, uuid
from pathlib import Path

router = APIRouter()

class FlagPayload(BaseModel):
    query_id: str
    note: str = ""

class RatingPayload(BaseModel):
    query_id: str
    rating: int  # 1-5
    comment: str = ""

@router.post("/flag")
def flag(payload: FlagPayload):
    fname = FEEDBACK_DIR / f"flag_{payload.query_id}_{int(time.time())}.json"
    obj = {"id": str(uuid.uuid4()), "query_id": payload.query_id, "note": payload.note, "timestamp": int(time.time())}
    with open(fname, "w") as f:
        json.dump(obj, f, indent=2)
    return {"status": "ok", "saved": str(fname)}

@router.post("/rate")
def rate(payload: RatingPayload):
    if payload.rating < 1 or payload.rating > 5:
        raise HTTPException(status_code=400, detail="rating must be 1-5")
    fname = FEEDBACK_DIR / f"rating_{payload.query_id}_{int(time.time())}.json"
    obj = {"id": str(uuid.uuid4()), "query_id": payload.query_id, "rating": payload.rating, "comment": payload.comment, "timestamp": int(time.time())}
    with open(fname, "w") as f:
        json.dump(obj, f, indent=2)
    return {"status": "ok", "saved": str(fname)}
