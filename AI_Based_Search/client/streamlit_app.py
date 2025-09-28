import streamlit as st
import requests
import json
from pathlib import Path

# API_BASE = st.secrets.get("API_BASE", "http://localhost:8000")
API_BASE = "http://localhost:8000"

st.set_page_config(page_title="AI Search", layout="wide")

st.title("AI-Powered Knowledge Base Search & Enrichment")

with st.sidebar:
    st.header("Actions")
    uploaded_files = st.file_uploader("Upload documents (pdf, txt, docx)", accept_multiple_files=True)
    if st.button("Ingest uploaded files"):
        if not uploaded_files:
            st.warning("Select files first")
        else:
            files_payload = []
            for f in uploaded_files:
                files_payload.append(("files", (f.name, f.getvalue(), f.type or "application/octet-stream")))
            resp = requests.post(f"{API_BASE}/ingest/upload", files=files_payload)
            st.write(resp.json())
    st.markdown("---")
    st.header("Auto-enrichment (URL)")
    url = st.text_input("Provide a URL to fetch text and ingest (simple crawler)")
    if st.button("Ingest URL"):
        if not url:
            st.warning("provide url")
        else:
            resp = requests.post(f"{API_BASE}/ingest/url", json={"url": url})
            st.write(resp.json())

st.header("Ask a question")
question = st.text_area("Your question", height=120)
k_val = st.number_input("Top-K (retrieval)", min_value=1, max_value=10, value=4)
if st.button("Ask"):
    if not question.strip():
        st.warning("Type a question")
    else:
        payload = {"question": question, "k": k_val}
        resp = requests.post(f"{API_BASE}/search/", json=payload)
        if resp.status_code != 200:
            st.error(f"Error: {resp.status_code} {resp.text}")
        else:
            data = resp.json()
            st.write("Query time (ms):", data.get("query_time_ms"))
            res = data.get("result", {})
            st.json(res)
            st.session_state["last_query_result"] = data

# flag & rate
st.markdown("---")
st.header("Feedback")
if st.button("Flag last answer as incomplete"):
    last = st.session_state.get("last_query_result")
    if not last:
        st.warning("No last query to flag")
    else:
        qid = last.get("query_time_ms")  # replace with unique id in prod
        note = st.text_input("Flag note", value="Answer incomplete")
        resp = requests.post(f"{API_BASE}/feedback/flag", json={"query_id": str(qid), "note": note})
        st.write(resp.json())

rating = st.slider("Rate last answer (1-5)", 1, 5, 3)
if st.button("Submit rating"):
    last = st.session_state.get("last_query_result")
    if not last:
        st.warning("No last query to rate")
    else:
        qid = last.get("query_time_ms")
        resp = requests.post(f"{API_BASE}/feedback/rate", json={"query_id": str(qid), "rating": rating, "comment": ""})
        st.write(resp.json())

st.markdown("---")
st.write("Sources preview (from last answer)")
if "last_query_result" in st.session_state:
    srcs = st.session_state["last_query_result"]["result"].get("sources", [])
    for s in srcs:
        st.write(f"File: {s.get('source_file')} chunk: {s.get('chunk_index')}")
        st.write(s.get("text_preview")[:1000])
