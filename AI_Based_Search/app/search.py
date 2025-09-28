# backend/app/search.py
from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from .store import VectorStoreManager
from .config import TOP_K, OPENAI_LLM_MODEL
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain_core.prompts import PromptTemplate
# from langchain import LLMChain
from langchain_openai import OpenAI
import json
import math
import time
from pydantic import BaseModel

router = APIRouter()

class QueryRequest(BaseModel):
    question: str
    k: Optional[int] = TOP_K
    auto_enrich: Optional[bool] = False

# Build a robust prompt template instructing strict JSON output
PROMPT_TEMPLATE = """
You are a precise assistant. Use ONLY the provided SOURCES to answer the user question. Do NOT hallucinate or add facts not in the sources.
SOURCES: 
{sources_block}

QUESTION:
{question}

INSTRUCTIONS:
- Provide an output that is STRICT JSON (no additional commentary).
- The JSON must match this schema exactly:
{
  "answer": string,
  "confidence": number,         // 0.0 - 1.0; numeric confidence of the answer
  "missing_info": [            // array of objects, empty if fully answered
     {"field": string, "why_missing": string, "suggestion": string}
  ],
  "sources": [                 // array with source references used
     {"source_file_id": string, "source_file": string, "chunk_index": number, "text_preview": string}
  ],
  "enrichment_suggestions": [  // actionable suggestions to enrich KB
     {"type": "document|api|csv|crawl|contact", "detail": string, "priority": "high|medium|low"}
  ]
}

If the question cannot be fully answered, be explicit in "missing_info" and provide concrete enrichment suggestions.
Respond ONLY with valid JSON.
"""

# Initialize LLM
llm = ChatOpenAI(model_name=OPENAI_LLM_MODEL, temperature=0)

def extract_sources_block(retrieved):
    """
    retrieved: list of tuples (Document, score)
    returns a string block of numbered sources with truncated previews
    """
    lines = []
    for i, (doc, score) in enumerate(retrieved):
        meta = doc.metadata or {}
        preview = doc.page_content[:1200].replace("\n", " ")
        lines.append(f"SOURCE_{i} | score: {score:.4f} | file: {meta.get('source_file')} | chunk: {meta.get('chunk_index')} \n{preview}")
    return "\n\n---\n\n".join(lines)

def parse_json_from_text(text: str) -> Optional[Dict[str, Any]]:
    # Extract JSON substring robustly
    try:
        # find first { and last }
        st = text.find("{")
        ed = text.rfind("}")
        if st == -1 or ed == -1:
            return None
        raw = text[st:ed+1]
        return json.loads(raw)
    except Exception:
        # try single quotes -> double quotes
        try:
            fixed = text.replace("'", '"')
            st = fixed.find("{"); ed = fixed.rfind("}")
            if st == -1 or ed == -1:
                return None
            return json.loads(fixed[st:ed+1])
        except Exception:
            return None

def scores_to_heuristic_conf(scores):
    """
    Map FAISS scores to a heuristic confidence in [0,1].
    FAISS returns distances (L2) by default; lower is better. For a quick mapping we use 1/(1+avg_score).
    """
    if not scores:
        return 0.0
    avg = sum(scores) / len(scores)
    val = 1.0 / (1.0 + avg)
    # clamp
    return max(0.0, min(1.0, val))

@router.post("/")
def query(q: QueryRequest):
    if not q.question.strip():
        raise HTTPException(status_code=400, detail="question cannot be empty")
    start = time.time()
    store = VectorStoreManager()
    retrieved = store.similarity_search_with_score(q.question, k=q.k)
    if not retrieved:
        print("KB is empty")
        # nothing in KB
        empty_response = {
            "answer": "",
            "confidence": 0.0,
            "missing_info": [{"field": "all", "why_missing": "No documents found in knowledge base", "suggestion": "Upload relevant documents or use auto-enrich"}],
            "sources": [],
            "enrichment_suggestions": [{"type": "document", "detail": "Upload an authoritative doc containing the requested information (e.g. product manual, policy PDF)", "priority": "high"}]
        }
        return JSONResponse(content=jsonable_encoder({"query_time_ms": int((time.time()-start)*1000), "result": empty_response}))
        # return {"query_time_ms": int((time.time()-start)*1000), "result": empty_response}

    # Build sources block
    sources_block = extract_sources_block(retrieved)
    # prompt = PROMPT_TEMPLATE.format(sources_block=sources_block, question=q.question)


    # prompt = PromptTemplate(
    #     input_variables=['sources_block', 'question'],
    #     template=PROMPT_TEMPLATE
    # )

    prompt = ChatPromptTemplate.from_template("""
        You are a knowledge base assistant. Use the provided context to answer the question.

        Context:
        {context}

        Question: {question}

        If the context does not fully answer the question, flag missing information and suggest enrichment.

        {format_instructions}
        """)

    response_schemas = [
        ResponseSchema(name="answer", description="The answer to the user's question."),
        ResponseSchema(name="confidence", description="Confidence score between 0 and 1."),
        ResponseSchema(name="missing_info", description="What information is missing, if any."),
        ResponseSchema(name="enrichment_suggestion", description="Suggestions to enrich the knowledge base.")
    ]

    parser = StructuredOutputParser.from_response_schemas(response_schemas)

    format_instructions = parser.get_format_instructions()

    # Call LLM via LangChain LLMChain for deterministic output
    chain = prompt | llm | parser
    # chain = prompt | llm | StructuredOutputParser()
    llm_out = chain.invoke({
        "context": sources_block, 
        "question": q.question, 
        "format_instructions": format_instructions})
    
    if llm_out is None:
        # fallback safe response if parsing fails
        llm_out = {
            "answer": "",
            "confidence": 0.0,
            "missing_info": [{"field": "parsing_error", "why_missing": "LLM returned unparsable output", "suggestion": "Increase model output size or adjust prompt"}],
            "sources": [],
            "enrichment_suggestions": []
        }
    # Compute heuristic from scores
    # scores = [s for (_, s) in retrieved]
    # heuristic_conf = scores_to_heuristic_conf(scores)
    # # LLM may provide a numeric confidence; blend if present
    # llm_conf = float(parsed.get("confidence", 0.0)) if isinstance(parsed.get("confidence", 0.0), (int, float)) else 0.0
    # final_conf = 0.5 * llm_conf + 0.5 * heuristic_conf
    # parsed["confidence"] = round(max(0.0, min(1.0, final_conf)), 3)

    # map sources to expected schema with minimal preview
    mapped_sources = []
    for doc, score in retrieved:
        meta = doc.metadata or {}
        mapped_sources.append({
            "source_file_id": meta.get("source_file_id", ""),
            "source_file": meta.get("source_file", ""),
            "chunk_index": meta.get("chunk_index", -1),
            "text_preview": doc.page_content[:600]
        })
    llm_out["sources"] = mapped_sources

    # # Basic safety: ensure missing_info is a list of objects
    # if not isinstance(parsed.get("missing_info", []), list):
    #     parsed["missing_info"] = []

    # done
    return JSONResponse(content=jsonable_encoder({"query_time_ms": int((time.time()-start)*1000), "result": llm_out}))
    # return {"query_time_ms": int((time.time()-start)*1000), "result": parsed}