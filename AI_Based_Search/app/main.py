# backend/app/main.py
from fastapi import FastAPI
# from ingest import router as ingest_router
from .ingest import router as ingest_router
from .search import router as search_router
from .feedback import router as feedback_router
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="KB-RAG - LangChain + FAISS + FastAPI")

# allow local UI (Streamlit) to talk
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(ingest_router, prefix="/ingest", tags=["ingest"])
app.include_router(search_router, prefix="/search", tags=["search"])
app.include_router(feedback_router, prefix="/feedback", tags=["feedback"])

@app.get("/")
def index():
    return {"status": "ok", "message": "KB-RAG backend running"}
