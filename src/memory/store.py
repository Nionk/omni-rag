import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Optional

import psycopg
import redis

from src.core.config import POSTGRES_DSN, REDIS_URL, SESSION_MEMORY_TTL_SECONDS


logger = logging.getLogger(__name__)


CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS conversation_summaries (
    id BIGSERIAL PRIMARY KEY,
    user_id TEXT NOT NULL,
    session_id TEXT NOT NULL,
    summary TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (user_id, session_id)
)
"""
CREATE_INDEX_SQL = """
CREATE INDEX IF NOT EXISTS idx_conversation_summaries_user_updated
    ON conversation_summaries (user_id, updated_at DESC)
"""


@dataclass(frozen=True)
class MemoryRecord:
    user_id: str
    session_id: str
    summary: str
    updated_at: str
    source: str


class ConversationMemoryStore:
    """
    Redis хранит активное summary с TTL, PostgreSQL — долговечный архив.

    Ошибка одной из баз не останавливает основной RAG: методы возвращают
    безопасный результат и оставляют диагностическое сообщение в логах.
    """

    def __init__(
        self,
        redis_url: str = REDIS_URL,
        postgres_dsn: str = POSTGRES_DSN,
        ttl_seconds: int = SESSION_MEMORY_TTL_SECONDS,
        redis_client: Any = None,
        connection_factory: Optional[Callable[[], Any]] = None,
    ):
        self.ttl_seconds = max(60, ttl_seconds)
        self.redis = redis_client or redis.Redis.from_url(
            redis_url,
            decode_responses=True,
            socket_connect_timeout=2,
            socket_timeout=2,
        )
        self._connect = connection_factory or (
            lambda: psycopg.connect(postgres_dsn, connect_timeout=3)
        )
        self._schema_ready = False

    @staticmethod
    def _key(session_id: str) -> str:
        return f"omni-rag:memory:{session_id}"

    def _ensure_schema(self) -> bool:
        if self._schema_ready:
            return True
        try:
            with self._connect() as connection:
                with connection.cursor() as cursor:
                    cursor.execute(CREATE_TABLE_SQL)
                    cursor.execute(CREATE_INDEX_SQL)
            self._schema_ready = True
            return True
        except Exception as exc:
            logger.warning("PostgreSQL memory schema is unavailable: %s", exc)
            return False

    def _load_hot(self, user_id: str, session_id: str) -> Optional[MemoryRecord]:
        try:
            raw = self.redis.get(self._key(session_id))
            if not raw:
                return None
            payload = json.loads(raw)
            if payload.get("user_id") != user_id:
                return None
            self.redis.expire(self._key(session_id), self.ttl_seconds)
            return MemoryRecord(
                user_id=user_id,
                session_id=session_id,
                summary=payload.get("summary", ""),
                updated_at=payload.get("updated_at", ""),
                source="redis",
            )
        except Exception as exc:
            logger.warning("Redis hot memory is unavailable: %s", exc)
            return None

    def _load_cold(self, user_id: str) -> Optional[MemoryRecord]:
        if not self._ensure_schema():
            return None
        try:
            with self._connect() as connection:
                with connection.cursor() as cursor:
                    cursor.execute(
                        """
                        SELECT session_id, summary, updated_at
                        FROM conversation_summaries
                        WHERE user_id = %s
                        ORDER BY updated_at DESC
                        LIMIT 1
                        """,
                        (user_id,),
                    )
                    row = cursor.fetchone()
            if not row:
                return None
            return MemoryRecord(
                user_id=user_id,
                session_id=str(row[0]),
                summary=row[1],
                updated_at=row[2].isoformat(),
                source="postgres",
            )
        except Exception as exc:
            logger.warning("PostgreSQL cold memory read failed: %s", exc)
            return None

    def _save_hot(self, record: MemoryRecord) -> bool:
        try:
            payload = json.dumps(
                {
                    "user_id": record.user_id,
                    "session_id": record.session_id,
                    "summary": record.summary,
                    "updated_at": record.updated_at,
                },
                ensure_ascii=False,
            )
            self.redis.setex(
                self._key(record.session_id), self.ttl_seconds, payload
            )
            return True
        except Exception as exc:
            logger.warning("Redis hot memory write failed: %s", exc)
            return False

    def _save_cold(self, record: MemoryRecord) -> bool:
        if not self._ensure_schema():
            return False
        try:
            with self._connect() as connection:
                with connection.cursor() as cursor:
                    cursor.execute(
                        """
                        INSERT INTO conversation_summaries
                            (user_id, session_id, summary, updated_at)
                        VALUES (%s, %s, %s, %s)
                        ON CONFLICT (user_id, session_id)
                        DO UPDATE SET
                            summary = EXCLUDED.summary,
                            updated_at = EXCLUDED.updated_at
                        """,
                        (
                            record.user_id,
                            record.session_id,
                            record.summary,
                            record.updated_at,
                        ),
                    )
            return True
        except Exception as exc:
            logger.warning("PostgreSQL cold memory write failed: %s", exc)
            return False

    def load_summary(self, user_id: str, session_id: str) -> Optional[MemoryRecord]:
        """Читает hot summary, а при промахе восстанавливает его из архива."""
        hot = self._load_hot(user_id, session_id)
        if hot:
            return hot

        cold = self._load_cold(user_id)
        if not cold:
            return None

        rehydrated = MemoryRecord(
            user_id=user_id,
            session_id=session_id,
            summary=cold.summary,
            updated_at=cold.updated_at,
            source="postgres",
        )
        self._save_hot(rehydrated)
        return rehydrated

    def save_summary(
        self, user_id: str, session_id: str, summary: str
    ) -> str:
        now = datetime.now(timezone.utc).isoformat()
        record = MemoryRecord(
            user_id=user_id,
            session_id=session_id,
            summary=summary,
            updated_at=now,
            source="application",
        )
        cold_saved = self._save_cold(record)
        hot_saved = self._save_hot(record)
        if hot_saved:
            return "redis"
        if cold_saved:
            return "postgres"
        return ""

    def end_session(self, session_id: str) -> bool:
        """Удаляет только горячую копию; архив в PostgreSQL сохраняется."""
        try:
            self.redis.delete(self._key(session_id))
            return True
        except Exception as exc:
            logger.warning("Redis hot memory delete failed: %s", exc)
            return False
