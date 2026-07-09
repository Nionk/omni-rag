import os
import sys
import logging
import torch
from typing import Set

from src.core.config import FAISS_INDEX_PATH, EMBEDDING_MODEL
from src.core.rag import build_rag_chain
from src.core.logger import setup_logger

try:
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_community.vectorstores import FAISS
except ImportError:
    print("Ошибка импорта. Установите зависимости.")
    sys.exit(1)

logger = setup_logger(__name__)
logging.getLogger("faiss").setLevel(logging.ERROR)

def main():
    print("=== Инициализация RAG-агента (CLI) ===")
    
    if not os.path.exists(FAISS_INDEX_PATH):
        print(f"Ошибка: Индекс FAISS не найден по пути '{FAISS_INDEX_PATH}'")
        sys.exit(1)
        
    print("Загрузка эмбеддингов (HuggingFace)...")
    try:
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={'device': device},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        print("Подключение к индексу FAISS...")
        vector_store = FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
    except Exception as e:
        logger.error(f"Критическая ошибка при загрузке базы данных: {e}")
        sys.exit(1)
        
    # В CLI версии можно использовать просто FAISS без BM25 для простоты
    # Но для качества мы соберем базовую цепочку
    rag_chain, _ = build_rag_chain(vector_store, bm25_retriever=None, top_k=10, temperature=0.0)
        
    print("\n✅ Агент готов к работе!")
    print("Введите ваш вопрос (или 'exit' / 'quit' для выхода).\n")
    
    # Интерактивный цикл (CLI) с поддержкой Streaming
    chat_history = []
    
    while True:
        try:
            user_query = input("Вы: ").strip()
            if not user_query:
                continue
                
            if user_query.lower() in ['exit', 'quit']:
                print("Завершение работы агента. До свидания!")
                break
                
            print("Агент: ", end="", flush=True)
            
            sources: Set[str] = set()
            full_response = ""
            
            # Streaming генерация ответа
            for chunk in rag_chain.stream({"input": user_query, "chat_history": chat_history}):
                if "answer" in chunk:
                    print(chunk["answer"], end="", flush=True)
                    full_response += chunk["answer"]
                    
                if "context" in chunk:
                    for doc in chunk["context"]:
                        source_name = doc.metadata.get("source", "Неизвестный источник")
                        sources.add(source_name)
                        
            print("\n")
            if sources:
                sources_str = ", ".join(sorted(list(sources)))
                print(f"Источники: {sources_str}\n")
            else:
                print("Источники: Не найдено\n")
                
            chat_history.append(("human", user_query))
            chat_history.append(("ai", full_response))
                
        except KeyboardInterrupt:
            print("\nПрервано пользователем. Для выхода введите 'exit'.")
        except Exception as e:
            logger.error(f"\n[Ошибка при генерации ответа]: {e}\n")

if __name__ == "__main__":
    main()
