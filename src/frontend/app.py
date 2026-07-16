import streamlit as st
import os
import time
import torch
import tempfile
import uuid

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore

try:
    from langchain_qdrant import QdrantVectorStore, FastEmbedSparse, RetrievalMode
    from qdrant_client import QdrantClient
except ImportError:
    QdrantVectorStore = None

from src.core.config import (
    DB_DIR,
    EMBEDDING_MODEL,
    GRAPH_TRAVERSAL_DEPTH,
    GRAPH_TRAVERSAL_ENABLED,
    GRAPH_TRAVERSAL_MAX_NODES,
    MEMORY_ENABLED,
    QDRANT_COLLECTION,
    QDRANT_GRAPH_COLLECTION,
    QDRANT_URL,
)
from src.core.rag import build_rag_chain
from src.graph.traversal import GraphTraversalRetriever, QdrantGraphTraverser
from src.indexing.router import DocumentRouter
from src.memory import ConversationMemoryStore, DialogueSummarizer

# 3.1 Caching
from langchain_core.globals import set_llm_cache
from langchain_community.cache import SQLiteCache
os.makedirs(DB_DIR, exist_ok=True)
set_llm_cache(SQLiteCache(database_path=os.path.join(DB_DIR, "langchain_cache.db")))

st.set_page_config(page_title="OmniRAG", layout="wide")

@st.cache_resource(show_spinner=False)
def get_embeddings():
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={'device': device},
        encode_kwargs={'normalize_embeddings': True}
    )

@st.cache_resource(show_spinner=False)
def load_db():
    """Загрузка векторной базы Qdrant с Hybrid Search (Dense + Sparse) и ParentDocumentRetriever"""
    if QdrantVectorStore is None:
        return None, "Библиотека langchain-qdrant не установлена."
        
    try:
        embeddings = get_embeddings()
        sparse_embeddings = FastEmbedSparse(model_name="Qdrant/bm25")
        
        client = QdrantClient(url=QDRANT_URL)
        if not client.collection_exists(QDRANT_COLLECTION):
            return None, "База данных не найдена или коллекция пуста. Загрузите документы."
            
        vector_store = QdrantVectorStore(
            client=client,
            collection_name=QDRANT_COLLECTION,
            embedding=embeddings,
            sparse_embedding=sparse_embeddings,
            retrieval_mode=RetrievalMode.HYBRID
        )
        
        retriever = vector_store.as_retriever(search_kwargs={"k": 15})
        
        return retriever, "Success"
    except Exception as e:
        return None, f"Критическая ошибка инициализации: {e}"

@st.cache_resource(show_spinner=False)
def load_graph_traverser():
    if not GRAPH_TRAVERSAL_ENABLED:
        return None
    try:
        client = QdrantClient(url=QDRANT_URL)
        if not client.collection_exists(QDRANT_GRAPH_COLLECTION):
            return None
    except Exception as graph_error:
        print(f"Graph traversal initialization failed: {graph_error}")
        return None
    return QdrantGraphTraverser(
        client=client,
        collection_name=QDRANT_GRAPH_COLLECTION,
        max_depth=GRAPH_TRAVERSAL_DEPTH,
        max_nodes=GRAPH_TRAVERSAL_MAX_NODES,
    )

@st.cache_resource(show_spinner=False)
def load_memory_store():
    if not MEMORY_ENABLED:
        return None
    return ConversationMemoryStore()

def build_memory_context(summary, messages):
    if summary:
        return f"Краткое summary предыдущего диалога: {summary}"

    recent_messages = messages[:-1][-2:]
    if not recent_messages:
        return "История отсутствует."
    return "\n".join(
        f"{'Пользователь' if message['role'] == 'user' else 'Ассистент'}: "
        f"{message['content']}"
        for message in recent_messages
    )

def process_temp_files(uploaded_files):
    """Обработка временных файлов для InMemory VectorStore"""
    if "temp_vectorstore" not in st.session_state:
        st.session_state.temp_vectorstore = InMemoryVectorStore(embedding=get_embeddings())
        st.session_state.temp_files_names = set()

    router = DocumentRouter()
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

    for uf in uploaded_files:
        if uf.name not in st.session_state.temp_files_names:
            # Сохраняем во временный файл для парсинга
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uf.name)[1]) as tmp:
                tmp.write(uf.getbuffer())
                tmp_path = tmp.name
            
            # Парсим
            parsed_data = router.process(tmp_path)
            os.remove(tmp_path)
            
            if parsed_data and "text" in parsed_data:
                doc = Document(
                    page_content=parsed_data["text"], 
                    metadata={"source": uf.name}
                )
                chunks = splitter.split_documents([doc])
                st.session_state.temp_vectorstore.add_documents(chunks)
                st.session_state.temp_files_names.add(uf.name)

# --- Инициализация UI ---
st.title("OmniRAG")
st.caption("Задайте вопрос, и получите ответ, основанный на реальных данных!")

top_k = 4  # Резко снижаем количество кусков, чтобы маленькая модель не теряла фокус
fetch_k = 15 # Количество кусков для извлечения из векторной базы перед реренкингом
with st.spinner("Подключение к базе данных..."):
    main_retriever, status_msg = load_db()

if not main_retriever:
    st.error(status_msg)
    st.stop()

graph_traverser = load_graph_traverser()
memory_store = load_memory_store()

if "messages" not in st.session_state:
    st.session_state.messages = []

default_user_id = os.getenv("DEFAULT_USER_ID", "local-user")
user_id = st.sidebar.text_input(
    "Идентификатор пользователя",
    value=default_user_id,
    help="По нему PostgreSQL восстанавливает последнее summary при новом входе.",
).strip() or default_user_id

if st.session_state.get("memory_user_id") != user_id:
    st.session_state.memory_user_id = user_id
    st.session_state.conversation_session_id = str(uuid.uuid4())
    st.session_state.messages = []
    st.session_state.memory_summary = ""
    st.session_state.memory_source = ""
    if memory_store:
        memory_record = memory_store.load_summary(
            user_id, st.session_state.conversation_session_id
        )
        if memory_record:
            st.session_state.memory_summary = memory_record.summary
            st.session_state.memory_source = memory_record.source

if memory_store:
    source_label = {
        "redis": "Redis (активный сеанс)",
        "postgres": "PostgreSQL (архив)",
    }.get(st.session_state.get("memory_source"), "новый диалог")
    st.sidebar.caption(f"Память: {source_label}")
    if st.session_state.get("memory_summary"):
        with st.sidebar.expander("Текущее summary"):
            st.write(st.session_state.memory_summary)
    if st.sidebar.button("Завершить сеанс", use_container_width=True):
        memory_store.end_session(st.session_state.conversation_session_id)
        st.session_state.messages = []
        st.session_state.memory_summary = ""
        st.session_state.memory_source = ""
        st.session_state.conversation_session_id = str(uuid.uuid4())
        st.rerun()
else:
    st.sidebar.caption("Память отключена")

# 3.2 Временные файлы для чата
with st.expander("Прикрепить временный файл"):
    temp_files = st.file_uploader("Файлы только для текущего сеанса", accept_multiple_files=True, key="temp_uploader")
    if temp_files:
        process_temp_files(temp_files)
        st.success(f"Загружено файлов: {len(st.session_state.temp_files_names)}. Они будут автоматически учтены при ответе.")

for msg_idx, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
if prompt := st.chat_input("Например: Какие побочные эффекты у тренболона?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        with st.spinner("Анализ базы данных..."):
            try:
                # Устанавливаем search_kwargs для глобального ретривера
                if not hasattr(main_retriever, "search_kwargs") or main_retriever.search_kwargs is None:
                    main_retriever.search_kwargs = {}
                main_retriever.search_kwargs.update({'k': fetch_k})

                if "temp_vectorstore" in st.session_state and st.session_state.temp_files_names:
                    from langchain_classic.retrievers import MergerRetriever
                    temp_retriever = st.session_state.temp_vectorstore.as_retriever(
                        search_kwargs={'k': fetch_k}
                    )
                    active_retriever = MergerRetriever(retrievers=[main_retriever, temp_retriever])
                else:
                    active_retriever = main_retriever

                if graph_traverser:
                    active_retriever = GraphTraversalRetriever(
                        base_retriever=active_retriever,
                        traverser=graph_traverser,
                    )

                # Собираем chain на лету с активным ретривером
                rag_chain, llm = build_rag_chain(active_retriever, top_k)

                chat_history = build_memory_context(
                    st.session_state.get("memory_summary", ""),
                    st.session_state.messages,
                )
                        
                last_update_time = time.time()
                UPDATE_INTERVAL = 0.1
                
                for chunk in rag_chain.stream({"input": prompt, "chat_history": chat_history}):
                    if "answer" in chunk:
                        full_response += chunk["answer"]
                        if time.time() - last_update_time > UPDATE_INTERVAL:
                            message_placeholder.markdown(full_response + "▌")
                            last_update_time = time.time()
                            
                message_placeholder.markdown(full_response)
                
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": full_response
                })

                if memory_store:
                    try:
                        updated_summary = DialogueSummarizer(llm).summarize(
                            st.session_state.get("memory_summary", ""),
                            prompt,
                            full_response,
                        )
                        st.session_state.memory_summary = updated_summary
                        memory_source = memory_store.save_summary(
                            user_id,
                            st.session_state.conversation_session_id,
                            updated_summary,
                        )
                        if memory_source:
                            st.session_state.memory_source = memory_source
                    except Exception as memory_error:
                        print(f"Memory summary update failed: {memory_error}")
                st.rerun()
                
            except Exception as e:
                import traceback
                traceback.print_exc()
                st.error(f"Произошла ошибка при генерации ответа: {e}")

