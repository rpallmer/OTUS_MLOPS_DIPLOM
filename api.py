import uuid
from contextlib import asynccontextmanager
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.runnables import RunnableConfig
from langfuse.langchain import CallbackHandler

from llm.mlx_backend import MLXBackend
from graph.builder import build_workflow
from config.config import settings

from src.retrieval_QA import Retriever_QA  
from src.retrieval_KB import Retriever_KB 

# ─────────────────────────────────────────────────────────────
# Глобальное состояние (кешируется на время жизни приложения)
# ─────────────────────────────────────────────────────────────
llm_instance = None
workflow_app = None
langfuse_handler: Optional[BaseCallbackHandler] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Инициализация при старте и очистка при завершении сервера."""
    global llm_instance, workflow_app, langfuse_handler
    
    print("🚀 Инициализация LLM и сборка графа...")
    try:
        langfuse_handler = CallbackHandler()
        print("✅ Langfuse мониторинг подключен")
    except Exception as e:
        print(f"⚠️ Ошибка подключения Langfuse: {e}")
        langfuse_handler = None

    llm_instance = MLXBackend(model_path=settings.llm_model)
    workflow_app = build_workflow(llm_instance)
    print("✅ Граф успешно собран и готов к обработке запросов")
    
    yield
    
    # Очистка при остановке сервера
    if langfuse_handler and hasattr(langfuse_handler, "langfuse"):
        langfuse_handler.langfuse.shutdown()
        print("📤 Langfuse сессия завершена.")

app = FastAPI(title="EnSB+ Support AI Demo", version="0.2.0", lifespan=lifespan)

# ─────────────────────────────────────────────────────────────
# Pydantic модели
# ─────────────────────────────────────────────────────────────
class IngestRequest(BaseModel):
    corpus_path: Optional[str] = None

class RetrieveRequest(BaseModel):
    query: str
    col_name: str  # 🆕 Обязательно: имя коллекции в Qdrant
    mode: str = Field(
        default="hybrid",
        pattern="^(semantic|hybrid|semantic_rerank|hybrid_rerank)$"
    )
    top_k: int = 10

class RespondRequest(BaseModel):
    query: str
    mode: str = Field(default="hybrid_rerank", pattern="^(semantic|hybrid|semantic_rerank|hybrid_rerank)$")
    top_k: int = 5

class WorkflowRequest(BaseModel):
    """Запрос для вызова основного графа из run.py"""
    query: str
    crm_ticket: Optional[str] = None
    trace_id: Optional[str] = None

class WorkflowResponse(BaseModel):
    status: str
    final_response: str
    errors: List[str]
    thread_id: str
    decomposed_count: int

# ─────────────────────────────────────────────────────────────
# Эндпоинты
# ─────────────────────────────────────────────────────────────
@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}

@app.post("/retrieve_QA")
def retrieve(req: RetrieveRequest) -> dict:
    # Инициализация нового ретривера
    retriever = Retriever_QA()
    
    # Поиск с передачей имени коллекции
    docs = retriever.search(
        query=req.query, 
        col_name=req.col_name, 
        mode=req.mode, 
        top_k=req.top_k
    )
    
    return {
        "query": req.query,
        "col_name": req.col_name,
        "mode": req.mode,
        "results": [
            {
                "id": d.point_id,
                "score": d.score,
                "question": d.question,
                "answer": d.answer,
                "source": d.source,      
                "category": d.category,  
                "dense_rank": d.dense_rank,
                "sparse_rank": d.sparse_rank,
                "final_rank": d.final_rank,
            }
            for d in docs
        ],
    }

# ЭНДПОИНТ: Вызов графа из run.py
@app.post("/workflow", response_model=WorkflowResponse)
def run_workflow(req: WorkflowRequest):
    if workflow_app is None:
        raise HTTPException(status_code=503, detail="Сервис ещё инициализируется. Повторите запрос через несколько секунд.")

    thread_id = req.crm_ticket or f"req-{uuid.uuid4().hex[:8]}"
    trace_id = req.trace_id or f"trace-{uuid.uuid4().hex[:8]}"

    callbacks: List[BaseCallbackHandler] = [langfuse_handler] if langfuse_handler else []

    # Формируем входной словарь в точном соответствии с run.py
    graph_input = {
        "original_query": req.query,
        "metadata": {"trace_id": trace_id, "crm_ticket": thread_id},
        "status": "init",
        "decomposed_items": [],
        "partial_results": {},
        "final_response": "",
        "errors": [],
        "current_index": 0,
        "total_items": 0
    }

    base_config: RunnableConfig = {
        "configurable": {"thread_id": thread_id},
        "callbacks": callbacks,
        "run_name": f"RAG-Workflow-{thread_id}",
        "recursion_limit": 10
    }

    try:
        # Синхронный вызов графа (FastAPI автоматически пулит его в тредпул)
        final_state = workflow_app.invoke(graph_input, config=base_config)

        # Принудительная отправка трейсов в Langfuse после каждого запроса
        if langfuse_handler and hasattr(langfuse_handler, "langfuse"):
            langfuse_handler.langfuse.flush()

        return WorkflowResponse(
            status=final_state.get("status", "unknown"),
            final_response=final_state.get("final_response", "Ответ не сгенерирован"),
            errors=final_state.get("errors", []),
            thread_id=thread_id,
            decomposed_count=len(final_state.get("decomposed_items", []))
        )
    except Exception as e:
        # Логируем ошибку и пробрасываем в API
        print(f"❌ Ошибка выполнения графа: {e}")
        raise HTTPException(status_code=500, detail=f"Ошибка выполнения пайплайна: {str(e)}")
    
