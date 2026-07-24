import os
import io
import time
import json
import logging
import uuid
from typing import List, Dict, Any

from fastapi import FastAPI, UploadFile, File, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

if __package__:
    from .utils import extract_text_from_file, chunk_text
    from .rag_pipeline import RAGPipeline
    from .agent import AIAgent
else:
    from backend.utils import extract_text_from_file, chunk_text
    from backend.rag_pipeline import RAGPipeline
    from backend.agent import AIAgent

# ---------------- LOGGING ----------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------- LIMITS ----------------
MAX_FILE_SIZE_BYTES = int(os.getenv("MAX_FILE_SIZE_BYTES", str(5 * 1024 * 1024)))
MAX_CHUNKS_PER_UPLOAD = int(os.getenv("MAX_CHUNKS_PER_UPLOAD", "200"))
MAX_QUERY_LENGTH = int(os.getenv("MAX_QUERY_LENGTH", "500"))
MAX_RAG_RESULTS = int(os.getenv("MAX_RAG_RESULTS", "5"))

# ---------------- APP ----------------
app = FastAPI(title="NotebookLM Clone API")

# ---------------- RATE LIMITER ----------------
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter

@app.exception_handler(RateLimitExceeded)
async def rate_limit_handler(request: Request, exc: RateLimitExceeded):
    return JSONResponse(
        status_code=429,
        content={"status": "error", "message": "Too many requests"},
    )

# ---------------- GLOBAL ERROR HANDLER ----------------
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled error")
    return JSONResponse(
        status_code=500,
        content={"status": "error", "message": "Internal server error"},
    )

# ---------------- CORS ----------------
def _parse_cors_origins() -> List[str]:
    raw = os.getenv("CORS_ALLOWED_ORIGINS", "").strip()
    if not raw:
        return []
    return [origin.strip() for origin in raw.split(",") if origin.strip()]


default_cors_origins = [
    "http://localhost:5500",
    "http://127.0.0.1:5500",
    "https://4chuck.github.io",
    "http://127.0.0.1:8000",
    "http://localhost:8000",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "https://your-buddy-phi.vercel.app",
    "https://your-buddy-frontend.onrender.com",
]

configured_cors_origins = _parse_cors_origins()
allowed_cors_origins = configured_cors_origins or default_cors_origins

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_cors_origins,
    allow_origin_regex=r"https://.*\.vercel\.app",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------- INIT ----------------
rag = RAGPipeline()
agent = AIAgent()

# ---------------- MEMORY ----------------
chat_memory: Dict[str, List[Dict[str, str]]] = {}

# ---------------- SPAM PROTECTION ----------------
last_request_time: Dict[str, float] = {}

def is_spamming(ip: str) -> bool:
    now = time.time()
    if ip in last_request_time and now - last_request_time[ip] < 1:
        return True
    last_request_time[ip] = now
    return False

# ---------------- REQUEST MODEL ----------------
class QueryRequest(BaseModel):
    query: str
    mode: str = "qa"
    options: Dict[str, Any] = Field(default_factory=dict)
    selected_document_ids: List[str] = Field(default_factory=list)

# ---------------- HEALTH ----------------
@app.get("/")
def root():
    return {"status": "ok", "message": "Backend running"}

@app.get("/health")
def health():
    return {"status": "ok"}

# ---------------- FILE VALIDATION ----------------
ALLOWED_EXTENSIONS = {".pdf", ".docx", ".txt", ".pptx", ".md", ".csv", ".json", ".log", ".html", ".xml"}

def is_valid_file(filename: str) -> bool:
    return os.path.splitext(filename)[1].lower() in ALLOWED_EXTENSIONS

# ---------------- SESSION ID ----------------
def get_session_id(request: Request) -> str:
    user_id = (request.headers.get("x-user-id") or "").strip()
    if user_id:
        return user_id
    if request.client and request.client.host:
        return request.client.host
    return ""


def get_bearer_api_key(request: Request) -> str | None:
    auth_header = (request.headers.get("authorization") or "").strip()
    if not auth_header:
        return None

    parts = auth_header.split(" ", 1)
    if len(parts) != 2:
        return None

    scheme, token = parts[0].strip().lower(), parts[1].strip()
    if scheme != "bearer" or not token:
        return None
    return token


def is_agent_auth_error(response_payload: Any) -> bool:
    if response_payload == getattr(agent, "AUTH_ERROR_SENTINEL", "__AUTH_ERROR__"):
        return True

    if not isinstance(response_payload, str):
        return False

    text = response_payload.strip()
    if not text:
        return False

    try:
        parsed = json.loads(text)
    except Exception:
        return False

    if not isinstance(parsed, dict):
        return False

    return str(parsed.get("code") or "").lower() == "auth_error"

# ---------------- UPLOAD ----------------
@app.post("/upload")
@limiter.limit("10/minute")
async def upload_files(
    request: Request,
    files: List[UploadFile] = File(default=[]),
    file: UploadFile | None = File(default=None),
):
    try:
        ip = request.client.host if request.client else "unknown"

        if is_spamming(ip):
            return {"status": "error", "message": "Too many requests"}

        if (not files) and file is not None:
            files = [file]
        if not files:
            return JSONResponse(
                status_code=400,
                content={"status": "error", "message": "No files provided"},
            )

        session_id = get_session_id(request)
        if not session_id:
            return JSONResponse(
                status_code=400,
                content={"status": "error", "message": "Missing session id (send x-user-id header)"},
            )

        total_chunks = 0
        uploaded_documents: List[Dict[str, Any]] = []

        for file in files:
            if not file.filename or not is_valid_file(file.filename):
                return JSONResponse(
                    status_code=400,
                    content={"status": "error", "message": f"Invalid file: {file.filename}"},
                )

            file_bytes = await file.read()

            if len(file_bytes) > MAX_FILE_SIZE_BYTES:
                return JSONResponse(
                    status_code=400,
                    content={"status": "error", "message": "File too large"},
                )

            document_id = str(uuid.uuid4())
            try:
                text = extract_text_from_file(file.filename, io.BytesIO(file_bytes))
            except Exception as e:
                logger.exception("Text extraction failed")
                return JSONResponse(
                    status_code=500,
                    content={"status": "error", "message": f"Failed to extract text from {file.filename}: {str(e)}"},
                )

            if not text:
                return JSONResponse(
                    status_code=400,
                    content={"status": "error", "message": f"No text could be extracted from {file.filename}"},
                )

            remaining = max(0, MAX_CHUNKS_PER_UPLOAD - total_chunks)
            if remaining == 0:
                return JSONResponse(
                    status_code=400,
                    content={"status": "error", "message": "Too many chunks"},
                )

            try:
                chunks = chunk_text(text, max_chunks=remaining)
            except Exception as e:
                logger.exception("Chunking failed")
                return JSONResponse(
                    status_code=500,
                    content={"status": "error", "message": f"Failed to chunk text from {file.filename}: {str(e)}"},
                )

            if isinstance(chunks, list):
                total_chunks += len(chunks)

                if total_chunks > MAX_CHUNKS_PER_UPLOAD:
                    return JSONResponse(
                        status_code=400,
                        content={"status": "error", "message": "Too many chunks"},
                    )

                if chunks:
                    try:
                        rag.add_documents(
                            chunks,
                            user_id=session_id,
                            session_id=session_id,
                            source_file=file.filename or "upload",
                            document_id=document_id,
                        )
                        uploaded_documents.append(
                            {
                                "id": document_id,
                                "name": file.filename or "upload",
                                "chunk_count": len(chunks),
                                "size_bytes": len(file_bytes),
                            }
                        )
                    except Exception as e:
                        logger.exception("Document store write failed")
                        return JSONResponse(
                            status_code=500,
                            content={
                                "status": "error",
                                "message": f"Failed to store documents: {str(e) or 'unknown error'}",
                            },
                        )

        logger.info(f"{ip} uploaded {len(files)} files -> {total_chunks} chunks")

        return {
            "status": "success",
            "files_received": len(files),
            "chunks_created": total_chunks,
            "documents": uploaded_documents,
        }

    except Exception:
        logger.exception("Upload failed")
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": "Upload failed"},
        )

# ---------------- QUERY ----------------
@app.post("/query")
@limiter.limit("5/minute")
async def query(req: QueryRequest, request: Request):
    try:
        ip = request.client.host if request.client else "unknown"

        if is_spamming(ip):
            return {"status": "error", "message": "Too fast"}

        if not req.query.strip():
            return {"status": "error", "message": "Empty query"}

        if len(req.query) > MAX_QUERY_LENGTH:
            return {"status": "error", "message": "Query too long"}

        session_id = get_session_id(request)
        if not session_id:
            return JSONResponse(
                status_code=400,
                content={"status": "error", "message": "Missing session id (send x-user-id header)"},
            )
        history = chat_memory.get(session_id, [])
        api_key_override = get_bearer_api_key(request)

        selected_document_ids = [str(doc_id).strip() for doc_id in (req.selected_document_ids or []) if str(doc_id).strip()]
        if not selected_document_ids:
            return {
                "status": "error",
                "message": "Please select at least one document to generate an answer.",
            }

        logger.info(f"{ip} -> {req.mode} -> {req.query}")

        # ---------------- RAG ----------------
        try:
            results = rag.query(
                req.query,
                n_results=MAX_RAG_RESULTS,
                user_id=session_id,
                session_id=session_id,
                selected_document_ids=selected_document_ids,
            )
        except Exception as e:
            logger.exception("RAG query failed")
            return JSONResponse(
                status_code=500,
                content={"status": "error", "message": f"RAG query failed: {str(e) or 'unknown error'}"},
            )

        documents = results.get("documents") or [[]]
        docs = documents[0] if isinstance(documents, list) and documents else []
        if not isinstance(docs, list):
            docs = []

        context = "\n\n".join([str(d).strip() for d in docs if str(d).strip()])

        if not context:
            return {
                "status": "error",
                "message": "No relevant content found in the selected documents.",
            }

        # ---------------- AGENT ----------------
        if req.mode == "quiz":
            response = agent.generate_quiz(
                context,
                int(req.options.get("num_questions", 5) or 5),
                user_id=session_id,
                api_key_override=api_key_override,
            )

        elif req.mode == "simplify":
            response = agent.explain_simply(
                context,
                user_id=session_id,
                api_key_override=api_key_override,
            )

        elif req.mode == "agent":
            response = agent.handle_agent_task(
                req.query,
                context,
                user_id=session_id,
                api_key_override=api_key_override,
            )

        else:
            response = agent.ask_question(
                req.query,
                context,
                user_id=session_id,
                api_key_override=api_key_override,
            )

        if is_agent_auth_error(response):
            return JSONResponse(
                status_code=401,
                content={
                    "status": "error",
                    "code": "auth_error",
                    "message": "API key authentication failed",
                },
            )

        # ---------------- MEMORY ----------------
        history.append({"role": "user", "content": req.query})
        history.append({"role": "assistant", "content": response})
        chat_memory[session_id] = history[-10:]

        return {
            "status": "success",
            "response": response,
        }

    except Exception:
        logger.exception("Query failed")
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": "Query failed"},
        )


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)


@app.get("/documents")
@limiter.limit("30/minute")
async def list_documents(request: Request):
    session_id = get_session_id(request)
    if not session_id:
        return JSONResponse(
            status_code=400,
            content={"status": "error", "message": "Missing session id (send x-user-id header)"},
        )

    documents = rag.list_documents(session_id=session_id)
    return {"status": "success", "documents": documents}


@app.delete("/documents/{document_id}")
@limiter.limit("20/minute")
async def delete_document(document_id: str, request: Request):
    session_id = get_session_id(request)
    if not session_id:
        return JSONResponse(
            status_code=400,
            content={"status": "error", "message": "Missing session id (send x-user-id header)"},
        )

    deleted = rag.delete_document(session_id=session_id, document_id=document_id)
    return {"status": "success", "deleted_chunks": deleted}
