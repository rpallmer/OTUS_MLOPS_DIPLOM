# run.py
import os
import logging  # 🔑 Добавлен импорт
from typing import List
from dotenv import load_dotenv
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.runnables import RunnableConfig
from langfuse.langchain import CallbackHandler
from llm.mlx_backend import MLXBackend
from graph.builder import build_workflow
from config.config import settings

# ─────────────────────────────────────────────────────────────
# 0. ЗАГРУЗКА .ENV & НАСТРОЙКА ЛОГИРОВАНИЯ
# ─────────────────────────────────────────────────────────────
load_dotenv()

# 🔑 Базовая настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
    force=True  # Перезаписывает существующие конфиги, если есть
)
logger = logging.getLogger(__name__)  # Создаём именованный логгер для модуля

def main():
    langfuse_handler = None
    try:
        langfuse_handler = CallbackHandler()
        logger.info("✅ Langfuse мониторинг подключен")  # 🔁 print → logger.info
    except Exception as e:
        logger.warning(f"⚠️ Ошибка подключения Langfuse: {e}")  # 🔁 print → logger.warning
        
    callbacks: List[BaseCallbackHandler] = [langfuse_handler] if langfuse_handler else []

    # 1. Инициализация LLM
    llm = MLXBackend(model_path=settings.llm_model)

    # 2. Сборка графа
    app = build_workflow(llm)

    # 3. Тестовый запрос
    test_input = {
        "original_query": "Я продала квартиру, хочу переоформить лицевой счёт на нового собственника. Что для этого нужно сделать и какие документы нужны?",
        "metadata": {"trace_id": "test-001", "crm_ticket": "CRM-12345"},
        "status": "init",
        "decomposed_items": [],
        "partial_results": {},
        "final_response": "",
        "errors": [],
        "current_index": 0,
        "total_items": 0
    }

    thread_id = test_input["metadata"]["crm_ticket"]
    base_config: RunnableConfig = {
        "configurable": {"thread_id": thread_id},
        "callbacks": callbacks,
        "run_name": f"RAG-Workflow-{thread_id}",
        "recursion_limit": 10
    }

    logger.info(f"🚀 Запуск графа (thread_id={thread_id})...")  # 🔁
    final_state = app.invoke(test_input, config=base_config)

    logger.info(f"\n📊 Статус: {final_state.get('status')}")  # 🔁
    if final_state.get("errors"):
        logger.warning(f"⚠️ Ошибки пайплайна: {final_state['errors']}")  # 🔁
    logger.info(f"\n📝 Декомпозировано объектов: {len(final_state.get('decomposed_items', []))}")  # 🔁
    logger.info(f"\n💬 Финальный ответ:\n{final_state.get('final_response', 'Ответ не сгенерирован')}")  # 🔁

    if langfuse_handler and hasattr(langfuse_handler, "langfuse"):
        langfuse_handler.langfuse.flush()
        logger.info("📤 Трейсы Langfuse успешно отправлены.")  # 🔁

# ─────────────────────────────────────────────────────────────
# ▶️ ЗАПУСК
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()