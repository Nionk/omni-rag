import os
import json
import uuid
import hashlib
from pathlib import Path
import concurrent.futures

from src.indexing.router import DocumentRouter
from src.indexing.indexer import VectorIndexer
from src.core.config import (
    DATA_DIR,
    DB_DIR,
    DLQ_FILE,
    EMBEDDING_MODEL,
    INDEX_SCHEMA_VERSION,
    QDRANT_COLLECTION,
    QDRANT_URL,
    STATE_FILE,
)
from src.core.logger import setup_logger
from langchain_community.embeddings import HuggingFaceEmbeddings

logger = setup_logger(__name__)



def get_file_hash(filepath: Path) -> str:
    hasher = hashlib.md5()
    try:
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hasher.update(chunk)
        return hasher.hexdigest()
    except Exception as e:
        logger.error(f"Не удалось вычислить хэш для {filepath}: {e}")
        return ""

def load_json(filepath: Path, default: dict) -> dict:
    if filepath.exists():
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return default
    return default

def save_json(filepath: Path, data: dict):
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def delete_qdrant_documents(doc_ids: list[str]) -> bool:
    """Удаляет все чанки документов по их метаданным."""
    if not doc_ids:
        return True

    from qdrant_client import QdrantClient
    from qdrant_client.http import models

    try:
        client = QdrantClient(url=QDRANT_URL)
        if client.collection_exists(QDRANT_COLLECTION):
            client.delete(
                collection_name=QDRANT_COLLECTION,
                points_selector=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="metadata.doc_id",
                            match=models.MatchAny(any=doc_ids),
                        )
                    ]
                ),
                wait=True,
            )
        return True
    except Exception as exc:
        logger.error(f"Ошибка при удалении документов из Qdrant: {exc}")
        return False

def garbage_collection(state: dict) -> dict:
    """
    Удаляет из базы Qdrant и doc_store документы, которых больше нет на диске в папке data/.
    Возвращает обновленный state.
    """
    logger.info("=== Этап 0: Сборка мусора (Garbage Collection) ===")
    deleted_keys = []
    ids_to_delete = []

    for filepath_str, info in state.items():
        if not os.path.exists(os.path.join(DATA_DIR, filepath_str)):
            logger.info(f"Файл {filepath_str} удален с диска. Подготовка к удалению из Qdrant.")
            doc_id = info.get("doc_id")
            if doc_id:
                ids_to_delete.append(doc_id)
            deleted_keys.append(filepath_str)

    if ids_to_delete:
        if delete_qdrant_documents(ids_to_delete):
            logger.info(f"Успешно удалено {len(ids_to_delete)} устаревших документов из Qdrant.")
        else:
            return state # В случае ошибки отменяем удаление из state

    for key in deleted_keys:
        del state[key]
        
    if not deleted_keys:
        logger.info("Удаленных файлов не найдено.")
        
    return state

def process_file_task(filepath_str: str) -> dict:
    router = DocumentRouter()
    return router.process(filepath_str)

def run_pipeline():
    DATA_DIR.mkdir(exist_ok=True)
    DB_DIR.mkdir(exist_ok=True)

    state = load_json(STATE_FILE, {})
    dlq = load_json(DLQ_FILE, {})
    
    # 0. Сборка мусора
    state = garbage_collection(state)
    
    router = DocumentRouter()
    docs_to_index = []
    
    # Мы больше не пересоздаем state с нуля, мы его обновляем,
    # чтобы не потерять файлы, которые не изменились.
    updated_state = state.copy()

    logger.info("=== Этап 1: Обход директории data/ и парсинг ===")
    # Собираем список файлов для обработки
    files_to_process = []
    for filepath in DATA_DIR.rglob("*"):
        if not filepath.is_file():
            continue
            
        file_hash = get_file_hash(filepath)
        if not file_hash:
            continue
            
        filename = str(filepath.relative_to(DATA_DIR))
        
        # Проверка идемпотентности
        previous_info = state.get(filename, {})
        if (
            filename in state
            and previous_info.get("hash") == file_hash
            and previous_info.get("index_schema_version") == INDEX_SCHEMA_VERSION
            and not previous_info.get("superseded_doc_ids")
        ):
            logger.info(f"Пропуск файла (не изменился): {filename}")
            continue
            
        logger.info(f"Обнаружен новый/измененный файл: {filename}")
        previous_doc_ids = list(previous_info.get("superseded_doc_ids", []))
        if previous_info.get("doc_id"):
            previous_doc_ids.append(previous_info["doc_id"])
        files_to_process.append((filepath, filename, file_hash, previous_doc_ids))

    if files_to_process:
        logger.info(f"Запуск параллельного парсинга для {len(files_to_process)} файлов...")
        with concurrent.futures.ProcessPoolExecutor() as executor:
            # Маппинг futures к данным файлов
            future_to_file = {
                executor.submit(process_file_task, str(filepath)): (
                    filename,
                    file_hash,
                    previous_doc_ids,
                )
                for filepath, filename, file_hash, previous_doc_ids in files_to_process
            }
            
            for future in concurrent.futures.as_completed(future_to_file):
                filename, file_hash, previous_doc_ids = future_to_file[future]
                try:
                    result = future.result()
                    if result:
                        doc_id = str(uuid.uuid4())
                        result["id"] = doc_id
                        docs_to_index.append(
                            (filename, file_hash, doc_id, previous_doc_ids, result)
                        )
                    else:
                        logger.warning(f"Файл {filename} отброшен в DLQ.")
                        dlq[filename] = {"reason": "router_returned_none", "hash": file_hash}
                except Exception as exc:
                    logger.error(f"Файл {filename} вызвал ошибку: {exc}")
                    dlq[filename] = {"reason": f"exception: {exc}", "hash": file_hash}
            
    if not docs_to_index:
        logger.info("✅ Нет новых файлов для векторизации. Пайплайн завершен.")
        save_json(STATE_FILE, updated_state)
        save_json(DLQ_FILE, dlq)
        return

    logger.info(f"=== Этап 2: Векторизация ({len(docs_to_index)} новых документов) ===")
    try:
        indexer = VectorIndexer()
        raw_docs = [item[4] for item in docs_to_index]
        
        doc_chunk_counts = indexer.build_and_save_index(raw_docs)
        
        if doc_chunk_counts:
            superseded_doc_ids = [
                old_doc_id
                for item in docs_to_index
                for old_doc_id in item[3]
                if old_doc_id and old_doc_id != item[2]
            ]
            old_chunks_deleted = delete_qdrant_documents(superseded_doc_ids)
            if not old_chunks_deleted:
                logger.warning(
                    "Новые чанки записаны, но старые версии удалить не удалось. "
                    "Повторная индексация устранит дубликаты после восстановления Qdrant."
                )

            for filename, file_hash, doc_id, previous_doc_ids, _ in docs_to_index:
                chunk_count = doc_chunk_counts.get(doc_id, 0)
                updated_state[filename] = {
                    "hash": file_hash,
                    "doc_id": doc_id,
                    "chunk_count": chunk_count,
                    "index_schema_version": INDEX_SCHEMA_VERSION,
                }
                if not old_chunks_deleted and previous_doc_ids:
                    updated_state[filename]["superseded_doc_ids"] = previous_doc_ids
            save_json(STATE_FILE, updated_state)
            logger.info("✅ Пайплайн успешно завершен!")
        else:
            logger.error("❌ Векторизация завершилась неудачно. State не обновлен.")
            
    except Exception as e:
        logger.error(f"❌ Критическая ошибка пайплайна: {e}")
        
    finally:
        save_json(DLQ_FILE, dlq)

if __name__ == "__main__":
    run_pipeline()
