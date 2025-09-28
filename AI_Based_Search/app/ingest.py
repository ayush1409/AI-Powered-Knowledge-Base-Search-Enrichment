import shutil
from fastapi import APIRouter, UploadFile, File, HTTPException
from typing import List
from pathlib import Path
from .config import UPLOAD_DIR, CHUNK_SIZE, CHUNK_OVERLAP
from .store import VectorStoreManager
from langchain.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
import uuid
import requests
from bs4 import BeautifulSoup
import time
import re
from .config import OPENAI_EMBEDDING_MODEL
from langchain.embeddings import OpenAIEmbeddings
import os

router = APIRouter()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
embeddings = OpenAIEmbeddings(
    model=OPENAI_EMBEDDING_MODEL, 
    openai_api_key=None if not os.getenv("OPENAI_API_KEY") else os.getenv("OPENAI_API_KEY")
)

def load_file_to_docs(local_path: str):
    ext = Path(local_path).suffix.lower()
    if ext == ".pdf":
        loader = PyPDFLoader(local_path)
    elif ext in [".txt", ".md"]:
        loader = TextLoader(local_path, encoding="utf-8")
    elif ext in [".docx"]:
        loader = Docx2txtLoader(local_path)
    else:
        loader = TextLoader(local_path, encoding="utf-8")
    docs = loader.load()
    if not docs:
        return []
    chunks = text_splitter.split_documents(docs)
    return chunks

@router.post("/upload")
async def upload_documents(files: List[UploadFile] = File(...)):
    """
    Accepts multiple files and ingests them into FAISS vectorstore.
    """
    store_manager = VectorStoreManager()
    saved = []
    for f in files:
        file_id = str(uuid.uuid4())
        filename = f.filename or f"{file_id}"
        dest = UPLOAD_DIR / f"{file_id}_{filename}"

        with open(dest, "wb") as out_f:
            shutil.copyfileobj(f.file, out_f)

        try:
            chunks = load_file_to_docs(str(dest))
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"failed to parse {filename}: {e}")
        for i, d in enumerate(chunks):
            metadata = dict(d.metadata or {})
            metadata.update({"source_file": filename, "source_file_id": file_id, "chunk_index": i})
            d.metadata = metadata
        store_manager.add_documents(chunks)
        saved.append({"file_id": file_id, "filename": filename, "chunks_ingested": len(chunks)})
    return {"status": "ok", "ingested": saved}

def clean_text(text: str) -> str:
    """Remove extra spaces, newlines, and normalize text before chunking."""
    text = re.sub(r"\s+", " ", text)  # collapse multiple spaces/newlines into one space
    return text.strip()

@router.post("/url")
def ingest_url(payload: dict):
    """
    Fetch a URL, extract text and ingest as a single document.
    """
    url = payload.get("url")
    print(url)
    if not url:
        raise HTTPException(status_code=400, detail="url missing")
    r = requests.get(url, timeout=10)
    if r.status_code != 200:
        raise HTTPException(status_code=400, detail=f"failed to fetch url: {r.status_code}")
    soup = BeautifulSoup(r.text, "html.parser")
    # extract visible text
    for s in soup(["script", "style", "noscript"]):
        s.decompose()
    text = soup.get_text(separator="\n")
    text = clean_text(text)

    # write to temp file and reuse loader logic
    tmpfile = UPLOAD_DIR / f"url_ingest_{int(time.time())}.txt"
    with open(tmpfile, "w", encoding="utf-8") as f:
        f.write(text)
    # now chunk and ingest
    chunks = load_file_to_docs(str(tmpfile))
    for i, d in enumerate(chunks):
        metadata = dict(d.metadata or {})
        metadata.update({"source_file": url, "source_file_id": url, "chunk_index": i})
        d.metadata = metadata
    store_manager = VectorStoreManager()
    store_manager.add_documents(chunks)

    # debugging
    results = store_manager.store.similarity_search_with_score("self-attention", k=2)
    for doc, score in results:
        print(score, doc.page_content[:200])

    print("Docs in FAISS:", len(store_manager.store.index_to_docstore_id))

    return {"status": "ok", "url": url, "chunks_ingested": len(chunks)}
