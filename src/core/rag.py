import os
from langchain_community.retrievers import BM25Retriever
from langchain_ollama import ChatOllama
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from typing import List, Any
from pydantic import Field

from src.core.config import OLLAMA_HOST, OLLAMA_MODEL

class CustomEnsembleRetriever(BaseRetriever):
    retrievers: List[BaseRetriever]
    weights: List[float]
    
    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        docs1 = self.retrievers[0].invoke(query)
        docs2 = self.retrievers[1].invoke(query)
        
        unique_docs = []
        seen_content = set()
        
        # Interleave
        for d1, d2 in zip(docs1, docs2):
            if d1.page_content not in seen_content:
                unique_docs.append(d1)
                seen_content.add(d1.page_content)
            if d2.page_content not in seen_content:
                unique_docs.append(d2)
                seen_content.add(d2.page_content)
                
        # Add remaining
        for d in docs1[len(docs2):] + docs2[len(docs1):]:
            if d.page_content not in seen_content:
                unique_docs.append(d)
                seen_content.add(d.page_content)
                
        return unique_docs

def build_rag_chain(vector_store, bm25_retriever=None, top_k=10, temperature=0.0):
    """
    Универсальная функция для создания RAG-цепочки.
    Используется и в консольном чате, и в веб-интерфейсе.
    """
    faiss_retriever = vector_store.as_retriever(search_kwargs={"k": top_k})
    
    if bm25_retriever:
        bm25_retriever.k = top_k
        # Объединяем оба поиска (Hybrid Search)
        retriever = CustomEnsembleRetriever(
            retrievers=[bm25_retriever, faiss_retriever], weights=[0.5, 0.5]
        )
    else:
        retriever = faiss_retriever
    
    llm = ChatOllama(
        model=OLLAMA_MODEL, 
        temperature=temperature,
        base_url=OLLAMA_HOST,
        keep_alive="24h"
    )
    
    system_prompt = (
        "Ты — эксперт по спортивной фармакологии. Твоя задача — давать развернутые и точные ответы "
        "на вопросы, основываясь на предоставленном контексте. Ты можешь делать аналитические выводы, "
        "опираясь на тексты. Если информации для ответа в контексте совершенно нет, честно скажи: "
        "'В моей базе знаний нет достаточной информации об этом'. Не выдумывай факты и дозировки, "
        "которых нет в источниках. "
        "Отвечай СТРОГО на русском языке. Использование китайского или других языков запрещено."
        "\n\n"
        "Контекст:\n"
        "{context}"
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
    ])
    
    document_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, document_chain)
    
    return rag_chain, llm
