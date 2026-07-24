import json
import logging
import os
import time
import traceback
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from dotenv import load_dotenv
from google import genai
from google.genai import types

# Optional Firebase support:
# If backend/firebase_config.py exists and exposes `db`, the agent will store
# chats/quizzes there. If not, the app still runs normally.
try:
    if __package__:
        from .firebase_config import db as firestore_db  # type: ignore
    else:
        from backend.firebase_config import db as firestore_db  # type: ignore
except Exception:
    firestore_db = None

# Load local .env; Render environment variables still take priority.
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _first_env(*names: str, default: Optional[str] = None) -> Optional[str]:
    for name in names:
        value = os.getenv(name)
        if value is not None and value.strip() != "":
            return value.strip()
    return default


def _env_int(*names: str, default: int) -> int:
    value = _first_env(*names)
    try:
        return int(value) if value is not None else default
    except ValueError:
        return default


def _env_float(*names: str, default: float) -> float:
    value = _first_env(*names)
    try:
        return float(value) if value is not None else default
    except ValueError:
        return default


def _dedupe_keep_order(items: list[Optional[str]]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        if item and item not in seen:
            seen.add(item)
            result.append(item)
    return result


def _strip_json_fences(text: str) -> str:
    cleaned = text.strip()
    if "```" in cleaned:
        cleaned = cleaned.replace("```json", "").replace("```", "").strip()
    return cleaned


def _truncate(text: str, max_chars: int) -> str:
    return text if len(text) <= max_chars else text[:max_chars]


class AIAgent:
    AUTH_ERROR_SENTINEL = "__AUTH_ERROR__"

    def __init__(self):
        self.api_key = _first_env("GEMINI_API_KEY", "GOOGLE_API_KEY")
        self.internal_fallback_key = _first_env("DEFAULT_GEMINI_API_KEY")
        self.missing_key_message = _first_env(
            "GEMINI_MISSING_KEY_MESSAGE",
            default="AI service is unavailable. Please provide a valid API key.",
        )
        self._client_cache: dict[str, genai.Client] = {}

        if self.api_key:
            self._client_cache[self.api_key] = genai.Client(api_key=self.api_key)

        if self.internal_fallback_key and self.internal_fallback_key not in self._client_cache:
            self._client_cache[self.internal_fallback_key] = genai.Client(
                api_key=self.internal_fallback_key
            )

        self.max_retries = _env_int("GEMINI_MAX_RETRIES", default=2)
        self.retry_delay = _env_float("GEMINI_RETRY_DELAY", default=1.0)

        self.temperature = _env_float("GEMINI_TEMPERATURE", default=0.4)
        self.top_p = _env_float("GEMINI_TOP_P", default=0.9)
        self.top_k = _env_int("GEMINI_TOP_K", default=40)

        self.max_context_chars = _env_int("GEMINI_MAX_CONTEXT_CHARS", default=12000)
        self.max_saved_chars = _env_int("GEMINI_MAX_SAVED_CHARS", default=6000)
        self.firestore_ttl_days = _env_int("FIRESTORE_TTL_DAYS", default=7)

        self.persist_enabled = _first_env("ENABLE_FIRESTORE_STORAGE", default="1") != "0"
        self.model_candidates = _dedupe_keep_order(
            [
                "gemini-2.5-flash",
                _first_env("GEMINI_MODEL"),
            ]
        )

    def _is_auth_error_text(self, message: str) -> bool:
        text = str(message or "").lower()
        return any(
            marker in text
            for marker in (
                "unauthorized",
                "forbidden",
                "api key",
                "permission denied",
                "authentication",
                "invalid argument",
                "401",
                "403",
            )
        )

    def _resolve_api_key(self, api_key_override: Optional[str]) -> Optional[str]:
        override = str(api_key_override or "").strip()
        if override:
            return override
        if self.api_key:
            return self.api_key
        return self.internal_fallback_key

    def _get_client(self, api_key_override: Optional[str]) -> Optional[genai.Client]:
        key = self._resolve_api_key(api_key_override)
        if not key:
            return None

        cached = self._client_cache.get(key)
        if cached is not None:
            return cached

        try:
            client = genai.Client(api_key=key)
            self._client_cache[key] = client
            return client
        except Exception as e:
            logger.warning("Gemini client init failed: %s", str(e))
            return None

    def _build_config(
        self,
        *,
        system_instruction: Optional[str] = None,
        response_json_schema: Optional[dict[str, Any]] = None,
    ) -> types.GenerateContentConfig:
        kwargs: dict[str, Any] = {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
        }

        if system_instruction:
            kwargs["system_instruction"] = system_instruction

        if response_json_schema is not None:
            kwargs["response_mime_type"] = "application/json"
            kwargs["response_json_schema"] = response_json_schema

        return types.GenerateContentConfig(**kwargs)

    def _firestore_save(self, collection: str, payload: dict[str, Any]) -> None:
        if not self.persist_enabled or firestore_db is None:
            return

        try:
            now = datetime.now(timezone.utc)
            doc = dict(payload)
            doc["created_at"] = now
            if self.firestore_ttl_days > 0:
                doc["expires_at"] = now + timedelta(days=self.firestore_ttl_days)

            # Keep document size bounded.
            for key in ("query", "response", "context", "prompt", "raw_output"):
                if key in doc and isinstance(doc[key], str):
                    doc[key] = _truncate(doc[key], self.max_saved_chars)

            firestore_db.collection(collection).add(doc)
        except Exception as e:
            logger.warning("Firestore save failed for %s: %s", collection, str(e))

    def _generate(
        self,
        prompt: str,
        *,
        system_instruction: Optional[str] = None,
        response_json_schema: Optional[dict[str, Any]] = None,
        expect_json: bool = False,
        api_key_override: Optional[str] = None,
    ) -> str:
        client = self._get_client(api_key_override)
        if client is None:
            if expect_json:
                return json.dumps(
                    {
                        "status": "error",
                        "code": "ai_unavailable",
                        "message": self.missing_key_message,
                    },
                    ensure_ascii=False,
                )
            return self.missing_key_message

        last_error: Optional[Exception] = None
        saw_auth_error = False

        for model_name in self.model_candidates:
            for attempt in range(self.max_retries):
                try:
                    config = self._build_config(
                        system_instruction=system_instruction,
                        response_json_schema=response_json_schema,
                    )

                    response = client.models.generate_content(
                        model=model_name,
                        contents=prompt,
                        config=config,
                    )

                    text = getattr(response, "text", None)
                    if text and text.strip():
                        return text.strip()

                    if expect_json:
                        return json.dumps(
                            {
                                "status": "error",
                                "message": "Empty response from AI",
                            },
                            ensure_ascii=False,
                        )

                    return "AI returned an empty response. Try a different prompt or model."

                except Exception as e:
                    last_error = e
                    if self._is_auth_error_text(str(e)):
                        saw_auth_error = True
                    logger.warning(
                        "Gemini attempt failed (model=%s, attempt=%s/%s): %s",
                        model_name,
                        attempt + 1,
                        self.max_retries,
                        str(e),
                    )

                    if attempt < self.max_retries - 1:
                        time.sleep(self.retry_delay)

        if last_error:
            logger.error("Gemini generation failed: %s", str(last_error))
            traceback.print_exc()
            if saw_auth_error:
                if expect_json:
                    return json.dumps(
                        {
                            "status": "error",
                            "code": "auth_error",
                            "message": "API key authentication failed",
                        },
                        ensure_ascii=False,
                    )
                return self.AUTH_ERROR_SENTINEL

            if expect_json:
                return json.dumps(
                    {
                        "status": "error",
                        "message": str(last_error) or "Gemini generation failed",
                    },
                    ensure_ascii=False,
                )

            return f"Gemini generation failed: {str(last_error) or 'unknown error'}"

        if expect_json:
            return json.dumps(
                {
                    "status": "error",
                    "message": "Unknown error",
                },
                ensure_ascii=False,
            )

        return "AI returned an empty response. Try a different prompt or model."

    def ask_question(
        self,
        query: str,
        context: str,
        user_id: str = "anonymous",
        api_key_override: Optional[str] = None,
    ) -> str:
        context = _truncate(context, self.max_context_chars)

        prompt = f"""
You are a strict document-based AI assistant.

RULES:
- Use ONLY the provided context
- If not found, say: Not found in document
- Be concise

CONTEXT:
{context}

QUESTION:
{query}

ANSWER:
"""
        response = self._generate(prompt, api_key_override=api_key_override)

        self._firestore_save(
            "chat_logs",
            {
                "user_id": user_id,
                "mode": "qa",
                "query": query,
                "context": context,
                "response": response,
            },
        )
        return response

    def generate_quiz(
        self,
        context: str,
        num_questions: int = 5,
        user_id: str = "anonymous",
        api_key_override: Optional[str] = None,
    ) -> str:
        context = _truncate(context, self.max_context_chars)

        schema = {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "question": {"type": "string"},
                    "options": {
                        "type": "array",
                        "items": {"type": "string"},
                        "minItems": 4,
                        "maxItems": 4,
                    },
                    "answer_index": {"type": "integer", "minimum": 0, "maximum": 3},
                },
                "required": ["question", "options", "answer_index"],
                "additionalProperties": False,
            },
        }

        prompt = f"""
Create {num_questions} MCQs from the context.

Return ONLY valid JSON. No markdown. No explanation.

Context:
{context}
"""
        result = self._generate(
            prompt,
            response_json_schema=schema,
            expect_json=True,
            api_key_override=api_key_override,
        )

        try:
            cleaned = _strip_json_fences(result)
            parsed = json.loads(cleaned)

            if isinstance(parsed, dict) and parsed.get("status") == "error":
                self._firestore_save(
                    "quiz_logs",
                    {
                        "user_id": user_id,
                        "mode": "quiz",
                        "context": context,
                        "response": json.dumps(parsed, ensure_ascii=False),
                    },
                )
                return json.dumps(parsed, ensure_ascii=False)

            if not isinstance(parsed, list):
                error_obj = {
                    "status": "error",
                    "message": "Quiz response was not a JSON array",
                }
                self._firestore_save(
                    "quiz_logs",
                    {
                        "user_id": user_id,
                        "mode": "quiz",
                        "context": context,
                        "response": json.dumps(error_obj, ensure_ascii=False),
                    },
                )
                return json.dumps(error_obj, ensure_ascii=False)

            quiz_json = json.dumps(parsed, indent=2, ensure_ascii=False)

            self._firestore_save(
                "quiz_logs",
                {
                    "user_id": user_id,
                    "mode": "quiz",
                    "num_questions": num_questions,
                    "context": context,
                    "response": quiz_json,
                },
            )
            return quiz_json

        except Exception as e:
            logger.error("Quiz JSON parse failed: %s", str(e))
            error_obj = {
                "status": "error",
                "message": "Invalid quiz JSON from AI",
            }

            self._firestore_save(
                "quiz_logs",
                {
                    "user_id": user_id,
                    "mode": "quiz",
                    "context": context,
                    "response": json.dumps(error_obj, ensure_ascii=False),
                },
            )
            return json.dumps(error_obj, ensure_ascii=False)

    def explain_simply(
        self,
        context: str,
        user_id: str = "anonymous",
        api_key_override: Optional[str] = None,
    ) -> str:
        context = _truncate(context, self.max_context_chars)

        prompt = f"""
Explain the following in simple bullet points.

Use short sentences and simple language.

Context:
{context}
"""
        response = self._generate(prompt, api_key_override=api_key_override)

        self._firestore_save(
            "chat_logs",
            {
                "user_id": user_id,
                "mode": "simplify",
                "context": context,
                "response": response,
            },
        )
        return response

    def handle_agent_task(
        self,
        task: str,
        context: str,
        user_id: str = "anonymous",
        api_key_override: Optional[str] = None,
    ) -> str:
        context = _truncate(context, self.max_context_chars)

        prompt = f"""
Task: {task}

Context:
{context}

Return a structured step-by-step response.
"""
        response = self._generate(prompt, api_key_override=api_key_override)

        self._firestore_save(
            "chat_logs",
            {
                "user_id": user_id,
                "mode": "agent",
                "query": task,
                "context": context,
                "response": response,
            },
        )
        return response
