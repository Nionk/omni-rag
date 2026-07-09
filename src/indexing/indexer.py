from src.core.logger import setup_logger
import os
import time
import re
from typing import List, Dict, Any, Tuple, Optional
import torch

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_core.documents import Document
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_community.vectorstores import FAISS
    from transformers import AutoTokenizer
except ImportError:
    RecursiveCharacterTextSplitter = None
    Document = None
    HuggingFaceEmbeddings = None
    FAISS = None
    AutoTokenizer = None

logger = setup_logger(__name__)

class VectorIndexer:
    """
    Класс для токен-чанкинга нормализованных документов и их векторизации 
    с сохранением в локальную базу данных FAISS.
    """
    def __init__(self):
        if HuggingFaceEmbeddings is None or AutoTokenizer is None:
            logger.error("Библиотеки LangChain, FAISS или Transformers не установлены.")
            raise ImportError("Выполните установку: pip install langchain langchain-community sentence-transformers faiss-cpu transformers")

        self.model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        logger.info(f"Инициализация HuggingFaceEmbeddings ({self.model_name})...")
        try:
            device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
            self.embeddings = HuggingFaceEmbeddings(
                model_name=self.model_name,
                model_kwargs={'device': device},
                encode_kwargs={'normalize_embeddings': True}
            )
            # Загружаем токенизатор напрямую для точного подсчета лимитов контекста
            tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        except Exception as e:
            logger.error(f"Не удалось инициализировать эмбеддинги или токенизатор: {e}")
            raise

        # Настройка токен-сплиттера с жестким лимитом 250 токенов (окно модели 256)
        self.text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
            tokenizer=tokenizer,
            separators=["\n## ", "\n### ", "\n\n", "\n", ".", " "],
            chunk_size=250,
            chunk_overlap=70
        )

    def _protect_tables(self, text: str) -> str:
        """
        Предварительная обработка Markdown таблиц. 
        Разбивает длинные таблицы на несколько коротких, дублируя шапку.
        Это предотвращает отрыв данных от заголовков колонок при сплиттинге.
        """
        import re
        
        table_pattern = re.compile(r'((?:\|.*\|\n?)+)')
        
        def chunk_table(match):
            lines = match.group(1).strip().split('\n')
            if len(lines) <= 5: 
                return match.group(1)
                
            header = lines[0]
            separator = lines[1]
            
            # Проверка, что это действительно таблица (вторая строка содержит только |-:)
            if not set(separator.replace('|', '').replace('-', '').replace(':', '').replace(' ', '')) == set():
                return match.group(1)
                
            data_rows = lines[2:]
            chunked_tables = []
            chunk_size = 5 # по 5 строк на под-таблицу
            for i in range(0, len(data_rows), chunk_size):
                chunk = [header, separator] + data_rows[i:i+chunk_size]
                chunked_tables.append('\n'.join(chunk))
                
            return '\n\n'.join(chunked_tables) + '\n'
            
        return table_pattern.sub(chunk_table, text)

    def _split_documents(self, docs: List[Dict[str, Any]]) -> Tuple[List[Document], List[str], Dict[str, int]]:
        """
        Конвертирует словари в Document, нарезает на чанки и генерирует уникальные ID чанков.
        Возвращает: чанки, ID чанков, словарь (doc_id -> количество чанков)
        """
        langchain_docs = []
        for d in docs:
            text = self._protect_tables(d.get("text", ""))
            metadata = d.get("metadata", {})
            doc_id = d.get("id", "unknown_id")
            
            # Сохраняем привязку чанка к исходному документу
            metadata["doc_id"] = doc_id
            
            if text:
                langchain_docs.append(Document(page_content=text, metadata=metadata))
        
        logger.info(f"Начало токен-чанкинга {len(langchain_docs)} документов...")
        chunks = self.text_splitter.split_documents(langchain_docs)
        
        # Генерируем ID для каждого чанка (формат: UUID_документа_chunk_N)
        chunk_ids = []
        counts = {}
        
        for chunk in chunks:
            doc_id = chunk.metadata.get("doc_id", "unknown")
            counts[doc_id] = counts.get(doc_id, 0) + 1
            chunk_ids.append(f"{doc_id}_chunk_{counts[doc_id]}")
            
        logger.info(f"Документы нарезаны на {len(chunks)} токен-чанков.")
        return chunks, chunk_ids, counts

    def build_and_save_index(self, docs: List[Dict[str, Any]], save_path: str = "faiss_index", batch_size: int = 1000) -> Optional[Dict[str, int]]:
        """
        Сохраняет векторы в базу и возвращает маппинг: doc_id -> количество сгенерированных чанков.
        Возвращает None в случае ошибки.
        """
        if not docs:
            logger.warning("Пустой список документов.")
            return None

        chunks, chunk_ids, doc_chunk_counts = self._split_documents(docs)
        if not chunks:
            return None

        total_chunks = len(chunks)
        vector_store = None

        if os.path.exists(save_path) and os.path.isdir(save_path) and os.path.exists(os.path.join(save_path, "index.faiss")):
            try:
                logger.info(f"Найден существующий индекс '{save_path}'. Загрузка...")
                vector_store = FAISS.load_local(save_path, self.embeddings, allow_dangerous_deserialization=True)
            except Exception as e:
                logger.error(f"Ошибка при загрузке индекса: {e}")
                vector_store = None

        logger.info(f"Начинается векторизация батчами по {batch_size} шт. (всего {total_chunks} чанков).")
        
        try:
            for i in range(0, total_chunks, batch_size):
                batch_chunks = chunks[i:i + batch_size]
                batch_ids = chunk_ids[i:i + batch_size]
                
                start_time = time.time()
                
                if vector_store is None:
                    vector_store = FAISS.from_documents(batch_chunks, self.embeddings, ids=batch_ids)
                else:
                    vector_store.add_documents(batch_chunks, ids=batch_ids)
                    
                elapsed = time.time() - start_time
                batch_num = i // batch_size + 1
                total_batches = (total_chunks + batch_size - 1) // batch_size
                logger.info(f"Батч {batch_num}/{total_batches} векторизован за {elapsed:.2f} сек.")
            
            if vector_store:
                vector_store.save_local(save_path)
                logger.info(f"Индекс успешно сохранен в директории: {save_path}")
                return doc_chunk_counts
                
        except Exception as e:
            logger.error(f"Ошибка во время векторизации: {e}")
            return None
            
        return None
