import streamlit as st
import os
import time
import torch

from langchain_ollama import ChatOllama
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from src.core.config import FAISS_INDEX_PATH, EMBEDDING_MODEL, OLLAMA_HOST, DATA_DIR
from src.core.rag import build_rag_chain
from src.cli.ingest import run_pipeline

DB_PATH = FAISS_INDEX_PATH

st.set_page_config(page_title="OmniRAG", page_icon="🧠", layout="wide")

def get_db_mtime():
    faiss_file = os.path.join(DB_PATH, "index.faiss")
    if os.path.exists(faiss_file):
        return os.path.getmtime(faiss_file)
    return 0

@st.cache_resource(show_spinner=False)
def load_db(db_mtime):
    """Загрузка векторной базы и BM25 (кэшируется в памяти)"""
    if not os.path.exists(DB_PATH):
        return None, None, "База данных не найдена. Загрузите документы."
        
    try:
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={'device': device},
            encode_kwargs={'normalize_embeddings': True}
        )
        vector_store = FAISS.load_local(DB_PATH, embeddings, allow_dangerous_deserialization=True)
        all_docs = list(vector_store.docstore._dict.values())
        bm25_retriever = BM25Retriever.from_documents(all_docs)
        
        return vector_store, bm25_retriever, "Success"
    except Exception as e:
        return None, None, f"Критическая ошибка инициализации: {e}"




# --- Инициализация UI ---
st.title("🧠 OmniRAG")
st.caption("Задайте вопрос, и получите ответ, основанный на реальных данных!")

if "selected_citation" not in st.session_state:
    st.session_state.selected_citation = None

# Двухколоночный макет
col_chat, col_viewer = st.columns([6, 4])

# --- Сайдбар (Настройки и Загрузка) ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2836/2836856.png", width=60)
    st.title("Управление")
    
    st.header("⚙️ Настройки AI")
    temperature = st.slider("Креативность", 0.0, 1.0, 0.0, 0.1, help="0.0 - строгие факты из базы. 1.0 - больше свободы для нейросети.")
    top_k = st.slider("Источники (Top K)", 1, 30, 10, 1, help="Сколько фрагментов текста находить в базе.")
    
    st.divider()
    st.header("📄 База знаний")
    uploaded_files = st.file_uploader("Загрузить документы", accept_multiple_files=True, type=["pdf", "docx", "txt", "xml", "html"])
    if uploaded_files:
        if st.button("Индексировать в базу", type="primary"):
            # Создаем папку data если нет
            os.makedirs(DATA_DIR, exist_ok=True)
            for uf in uploaded_files:
                with open(os.path.join(DATA_DIR, uf.name), "wb") as f:
                    f.write(uf.getbuffer())
            with st.spinner("Идет индексация документов... Это может занять несколько минут."):
                run_pipeline()
            st.success("База успешно обновлена!")
            time.sleep(2)
            st.rerun()

with st.spinner("Инициализация нейросетей..."):
    db_mtime = get_db_mtime()
    vector_store, bm25_retriever, status_msg = load_db(db_mtime)

if not vector_store:
    st.error(status_msg)
    st.stop()

# Собираем chain на лету с новыми настройками
rag_chain, _ = build_rag_chain(vector_store, bm25_retriever, top_k, temperature)

with col_chat:
    # История сообщений
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Вывод истории
    for msg_idx, msg in enumerate(st.session_state.messages):
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            
            # Вывод интерактивных кнопок-цитат
            if "citations" in msg and msg["citations"]:
                st.markdown("**🔍 Источники:**")
                # Разбиваем кнопки на колонки для вывода в строку (по 4 в ряд)
                cols = st.columns(4)
                for cite_idx, citation in enumerate(msg["citations"]):
                    with cols[cite_idx % 4]:
                        if st.button(f"📄 {citation['source']}", key=f"btn_{msg_idx}_{cite_idx}", use_container_width=True):
                            st.session_state.selected_citation = citation

    # Ввод пользователя
    if prompt := st.chat_input("Например: Какие побочные эффекты у тренболона?"):
        # 1. Показываем вопрос пользователя
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # 2. Генерируем ответ агента со Streaming
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            citations = []
            seen_sources = set()
            
            with st.spinner("Анализ базы данных..."):
                try:
                    # Преобразуем историю чата в формат LangChain
                    chat_history = []
                    # Берем все сообщения кроме последнего (так как последнее это текущий prompt)
                    for msg in st.session_state.messages[:-1]: 
                        if msg["role"] == "user":
                            chat_history.append(("human", msg["content"]))
                        else:
                            chat_history.append(("ai", msg["content"]))
                            
                    for chunk in rag_chain.stream({"input": prompt, "chat_history": chat_history}):
                        if "answer" in chunk:
                            full_response += chunk["answer"]
                            # Динамическое обновление UI (Streaming)
                            message_placeholder.markdown(full_response + "▌")
                            
                        if "context" in chunk:
                            for doc in chunk["context"]:
                                source = doc.metadata.get("source", "Неизвестный источник")
                                # Чтобы не дублировать кнопки для одного и того же файла (если не нужно), 
                                # но здесь мы хотим сохранить уникальные куски текста.
                                # Поэтому используем хэш текста или просто добавляем все.
                                content_hash = hash(doc.page_content)
                                if content_hash not in seen_sources:
                                    citations.append({
                                        "source": source,
                                        "content": doc.page_content
                                    })
                                    seen_sources.add(content_hash)
                                
                    # Финальный вывод текста без курсора
                    message_placeholder.markdown(full_response)
                    
                    # Сохраняем в историю
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": full_response,
                        "citations": citations
                    })
                    
                    st.rerun() # Обновляем UI, чтобы появились кнопки цитат
                    
                except Exception as e:
                    st.error(f"Произошла ошибка при генерации ответа: {e}")

with col_viewer:
    st.subheader("📖 Просмотр источника")
    if st.session_state.selected_citation:
        st.markdown(f"**Источник:** `{st.session_state.selected_citation['source']}`")
        st.info(st.session_state.selected_citation['content'])
        # Здесь в будущем будет реализован предпросмотр PDF (iframe)
    else:
        st.info("👈 Нажмите на любой источник в чате, чтобы прочитать точную цитату, из которой нейросеть взяла информацию.")
