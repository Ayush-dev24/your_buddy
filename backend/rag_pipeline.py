import os
import uuid
import logging
import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import chromadb
from sentence_transformers import CrossEncoder

try:
    from backend.firebase_config import db
except ImportError:
    from firebase_config import db

from backend.agent import AIAgent

logger = logging.getLogger(__name__)


class RAGPipeline:
    def __init__(self, collection_name: str = "document_chunks"):
        self.collection_name = collection_name
        self.max_chunk_chars = int(os.getenv("RAG_MAX_CHUNK_CHARS", "4000"))
        self.max_candidates = int(os.getenv("RAG_MAX_CANDIDATES", "200"))
        self.firestore_timeout_s = float(os.getenv("FIRESTORE_TIMEOUT_SECONDS", "10"))
        self.force_local = (os.getenv("RAG_FORCE_LOCAL", "0") or "").strip().lower() in {"1", "true", "yes"}

        self.firestore_enabled = (db is not None) and (not self.force_local)

        self.agent = AIAgent()
        
        # Load Sentence-Transformer for Re-ranking
        logger.info("Loading CrossEncoder model...")
        self.re_ranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        
        # Initialize ChromaDB
        chroma_db_path = os.getenv(
            "LOCAL_RAG_DB_PATH",
            os.path.join(os.path.dirname(__file__), "chroma_db")
        )
        os.makedirs(chroma_db_path, exist_ok=True)
        self.chroma_client = chromadb.PersistentClient(path=chroma_db_path)
        self.collection = self.chroma_client.get_or_create_collection(name=self.collection_name)

    def _safe_content(self, doc: Any) -> str:
        if isinstance(doc, dict):
            return str(doc.get("content") or "").strip()
        return str(doc).strip()

    def add_documents(
        self,
        chunks: List[Any],
        user_id: str,
        session_id: str,
        source_file: Optional[str] = None,
        document_id: Optional[str] = None,
    ) -> None:
        if not chunks:
            return
        if not session_id or not str(session_id).strip():
            return

        if self.firestore_enabled:
            try:
                self._add_documents_firestore(chunks, user_id, session_id, source_file, document_id)
            except Exception as e:
                logger.warning("Firestore add_documents failed; falling back to local: %s", e)
                self.firestore_enabled = False

        self._add_documents_chroma(chunks, user_id, session_id, source_file, document_id)

    def _add_documents_firestore(self, chunks, user_id, session_id, source_file, document_id):
        if db is None:
            raise RuntimeError("Firestore client not initialized")
        collection = db.collection(self.collection_name)
        batch = db.batch()
        pending = 0

        for i, chunk in enumerate(chunks):
            content = self._safe_content(chunk)
            if not content:
                continue
            metadata = chunk.get("metadata") if isinstance(chunk, dict) and isinstance(chunk.get("metadata"), dict) else None

            doc_id_val = str(uuid.uuid4())
            doc = {
                "chunk_id": doc_id_val,
                "document_id": str(document_id or ""),
                "user_id": str(user_id or session_id),
                "session_id": str(session_id),
                "content": content[: self.max_chunk_chars],
                "chunk_index": i,
                "source_file": source_file or "upload",
                "created_at": datetime.now(timezone.utc),
            }
            if metadata:
                doc["metadata"] = metadata

            doc_ref = collection.document(doc_id_val)
            batch.set(doc_ref, doc)
            pending += 1

            if pending >= 400:
                batch.commit(timeout=self.firestore_timeout_s)
                batch = db.batch()
                pending = 0

        if pending:
            batch.commit(timeout=self.firestore_timeout_s)

    def _add_documents_chroma(self, chunks, user_id, session_id, source_file, document_id):
        now = datetime.now(timezone.utc).timestamp()
        
        ids = []
        documents = []
        metadatas = []

        for i, chunk in enumerate(chunks):
            content = self._safe_content(chunk)
            if not content:
                continue
                
            chunk_metadata = chunk.get("metadata") if isinstance(chunk, dict) and isinstance(chunk.get("metadata"), dict) else {}

            doc_id_val = str(uuid.uuid4())
            ids.append(doc_id_val)
            documents.append(content[: self.max_chunk_chars])
            
            # Chroma metadata must be flat dictionaries of str, int, float, or bool
            meta = {
                "document_id": str(document_id or ""),
                "user_id": str(user_id or session_id),
                "session_id": str(session_id),
                "chunk_index": i,
                "source_file": str(source_file or "upload"),
                "created_at": float(now),
            }
            # Optional extra flat metadata
            for k, v in chunk_metadata.items():
                if isinstance(v, (str, int, float, bool)):
                    meta[f"meta_{k}"] = v
            metadatas.append(meta)

        if ids:
            self.collection.add(ids=ids, documents=documents, metadatas=metadatas)

    def _expand_query(self, query: str) -> List[str]:
        try:
            count = int(os.getenv("QUERY_EXPANSION_COUNT", "3"))
        except Exception:
            count = 3
            
        prompt = f"""
Generate {count} alternative phrasings of the following user query, each on its own line.
Make them semantically equivalent but phrased differently to improve search retrieval.
Query: {query}
"""
        try:
            response = self.agent._generate(prompt)
            alternatives = [line.strip() for line in response.splitlines() if line.strip()]
            result = list(dict.fromkeys([query] + alternatives))
            return result[: count + 1]
        except Exception as e:
            logger.warning(f"Query expansion failed: {e}")
            return [query]

    def query(
        self,
        query_text: str,
        n_results: int = 5,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        selected_document_ids: Optional[List[str]] = None,
    ) -> Dict[str, List[List[str]]]:
        scope_session_id = str(session_id or "").strip() or str(user_id or "").strip()
        if not scope_session_id:
            return {"documents": [[]]}

        where_clause = {"session_id": scope_session_id}
        if selected_document_ids and len(selected_document_ids) > 0:
            valid_docs = [str(doc_id).strip() for doc_id in selected_document_ids if str(doc_id).strip()]
            if len(valid_docs) == 1:
                where_clause = {"$and": [{"session_id": scope_session_id}, {"document_id": valid_docs[0]}]}
            elif len(valid_docs) > 1:
                where_clause = {"$and": [{"session_id": scope_session_id}, {"document_id": {"$in": valid_docs}}]}

        # If no query text, return recent documents
        if not query_text or not str(query_text).strip():
            try:
                results = self.collection.get(where=where_clause, limit=n_results)
                docs = results.get("documents", [])
                return {"documents": [docs[:n_results] if docs else []]}
            except Exception as e:
                logger.error("Failed to get recent documents: %s", e)
                return {"documents": [[]]}

        # 1. Expand Query
        expanded_queries = self._expand_query(query_text)
        
        # 2. Retrieve candidates from ChromaDB
        all_candidates_dict = {}
        try:
            results = self.collection.query(
                query_texts=expanded_queries,
                n_results=min(n_results * 2, self.max_candidates),
                where=where_clause
            )
            
            # Flatten results and deduplicate by ID
            for i in range(len(expanded_queries)):
                if not results.get("ids") or len(results["ids"]) <= i:
                    continue
                for j in range(len(results["ids"][i])):
                    doc_id = results["ids"][i][j]
                    doc_content = results["documents"][i][j]
                    if doc_id not in all_candidates_dict:
                        all_candidates_dict[doc_id] = doc_content
        except Exception as e:
            logger.error("ChromaDB query failed: %s", e)
            return {"documents": [[]]}

        candidates = list(all_candidates_dict.values())
        if not candidates:
            return {"documents": [[]]}

        # 3. Re-rank with CrossEncoder
        try:
            rerank_k = int(os.getenv("RERANK_TOP_K", str(n_results)))
        except Exception:
            rerank_k = n_results

        pairs = [(query_text, content) for content in candidates]
        try:
            scores = self.re_ranker.predict(pairs)
            ranked_pairs = sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)
            top_documents = [content for _, content in ranked_pairs[:rerank_k]]
        except Exception as e:
            logger.error("Re-ranking failed: %s", e)
            top_documents = candidates[:rerank_k]

        return {"documents": [top_documents]}

    def list_documents(self, session_id: str) -> List[Dict[str, Any]]:
        scope = str(session_id or "").strip()
        if not scope:
            return []

        docs_by_id: Dict[str, Dict[str, Any]] = {}
        
        try:
            results = self.collection.get(
                where={"session_id": scope},
                include=["metadatas"]
            )
            metadatas = results.get("metadatas", [])
            
            for meta in metadatas:
                doc_id = meta.get("document_id")
                if not doc_id:
                    continue
                
                source = meta.get("source_file", "upload")
                created = meta.get("created_at", 0.0)
                
                if doc_id not in docs_by_id:
                    docs_by_id[doc_id] = {
                        "id": doc_id,
                        "name": source,
                        "chunk_count": 1,
                        "last_seen": float(created),
                    }
                else:
                    docs_by_id[doc_id]["chunk_count"] += 1
                    if float(created) > docs_by_id[doc_id]["last_seen"]:
                        docs_by_id[doc_id]["last_seen"] = float(created)
                        
        except Exception as e:
            logger.error("ChromaDB list_documents failed: %s", e)

        return sorted(docs_by_id.values(), key=lambda x: float(x.get("last_seen") or 0), reverse=True)

    def delete_document(self, session_id: str, document_id: str) -> int:
        scope = str(session_id or "").strip()
        doc_id = str(document_id or "").strip()
        if not scope or not doc_id:
            return 0

        count = 0
        try:
            results = self.collection.get(
                where={"$and": [{"session_id": scope}, {"document_id": doc_id}]}
            )
            ids_to_delete = results.get("ids", [])
            if ids_to_delete:
                self.collection.delete(ids=ids_to_delete)
                count = len(ids_to_delete)
        except Exception as e:
            logger.error("ChromaDB delete_document failed: %s", e)

        return count

    def clear_collection(self, session_id: Optional[str] = None) -> int:
        count = 0
        try:
            if session_id:
                results = self.collection.get(where={"session_id": str(session_id)})
                ids_to_delete = results.get("ids", [])
                if ids_to_delete:
                    self.collection.delete(ids=ids_to_delete)
                    count = len(ids_to_delete)
            else:
                count = self.collection.count()
                self.chroma_client.delete_collection(self.collection_name)
                self.collection = self.chroma_client.create_collection(name=self.collection_name)
        except Exception as e:
            logger.error("ChromaDB clear_collection failed: %s", e)

        return count
