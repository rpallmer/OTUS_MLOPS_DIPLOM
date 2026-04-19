import re
import json
import ast
from typing import Dict
from langchain_core.messages import HumanMessage, SystemMessage
from models.schemas import WorkflowState, SubQuery, ItemResult
from llm.base import BaseLLMBackend
from src.prompts import DECOMPOSER_SYSTEM_PROMPT, AGGREGATOR_SYSTEM_PROMPT
from src.retrieval_QA import run_hybrid_search_QA
from src.retrieval_KB import run_hybrid_search_KB
from config.config import settings
from models.schemas import RetrievedDoc
from src.api_system import UtilityClient


def create_nodes(llm: BaseLLMBackend):
  
    def decomposer_node(state: WorkflowState) -> dict:
        try:
            messages = [
                SystemMessage(content=DECOMPOSER_SYSTEM_PROMPT),
                HumanMessage(content=state["original_query"])
            ]
            raw_output = llm.invoke(messages)
            
        
            raw_text = raw_output.content if hasattr(raw_output, 'content') else str(raw_output)
            cleaned_text = re.sub(r'^```(?:json)?\s*|\s*```$', '', raw_text.strip(), flags=re.MULTILINE)
                     
            start_idx = cleaned_text.find('/think>\n\n')+9
            end_idx = cleaned_text.rfind('\n}') + 2
            if start_idx == -1 or end_idx == 0:
                raise ValueError("JSON-объект не найден в ответе LLM")
                
            json_str = cleaned_text[start_idx:end_idx]
            parsed = json.loads(json_str)
            
            raw_items = parsed.get("decomposed_items", [])
            if not isinstance(raw_items, list):
                raw_items = []
                
            items = []
            for item in raw_items[:4]:  # Ограничение не более декомпозированных подвопросов отправляем в обработку
                try:
                    items.append(SubQuery(**item))
                except Exception as val_err:
                    # Логируем ошибку валидации, но не роняем весь пайплайн
                    print(f"⚠️ Пропущен невалидный sub_query: {val_err} | Данные: {item}")
                    
            return {
                "decomposed_items": items,
                "current_index": 0,
                "total_items": len(items),
                "partial_results": {},
                "status": "routing"
            }
            
        except Exception as e:
            print(f"❌ DECOMPOSER CRASH: {type(e).__name__} | {e}")
            return {
                "decomposed_items": [],
                "current_index": 0,
                "total_items": 0,
                "partial_results": {},
                "final_response": "",
                "status": "failed",
                "errors": [f"Decomposer failed: {str(e)}"]
            }
    
     
    def process_item_node(state: WorkflowState) -> dict:
        """
        Этап 2: Обработка ОДНОГО элемента по индексу.
        🔀 Маршрутизация внутри функции через item.routing_target
        """
        idx = state["current_index"]
        item = state["decomposed_items"][idx]  # 👈 Извлекаем текущий item из состояния
        
        try:
            # 🔀 ВНУТРЕННИЙ ДИСПЕТЧЕР: выбираем функцию по routing_target
            match item.routing_target:
                case "legal_templates_collection":
                    result = search_legal_templates(item)
                case "reference_info_collection":
                    result = search_knowledge_base(item)
                case "external_api_tools":
                    result = call_billing_api(item)
                case "operator_queue":
                    result = create_operator_ticket(item)
                case _:
                    result = fallback_handler(item)
                    
            status = "processing"
            
        except Exception as e:
            print(f"⚠️ PROCESS_ITEM ERROR (id={item.id}): {e}")
            # При ошибке не прерываем цикл, а помечаем результат как failed
            result = ItemResult(
                id=item.id, 
                answer="", 
                status="failed", 
                source=item.routing_target, 
                error=str(e)
            )
            status = "processing"
        
        # 🔑 Возвращаем: результат + сдвиг индекса
        return {
            "partial_results": {item.id: result},  # редьюсер добавит в общий dict
            "current_index": idx + 1,              # 🔄 Переход к следующему элементу
            "status": status
        }

    # 🔧 Заглушки функций обработки 
    def search_legal_templates_old(item: SubQuery) -> ItemResult:
        # Пример: использование item.search_keywords + item.entities.contract_id
        return ItemResult(
            id=item.id,
            answer=f"Найден шаблон по запросу: {item.search_keywords}",
            status="processed",
            source="legal_templates_collection",
            is_official=True
        )

    def search_legal_templates(item: SubQuery) -> ItemResult:
        """Поиск в юридической коллекции через гибридный ретривер"""
        try:
            # 1️⃣ Запуск поиска
            docs: list[RetrievedDoc] = run_hybrid_search_QA(
                query=item.sub_query,      
                col_name=settings.collection_name_QA,    # "legal_templates_collection"
                top_k=1,                         # Нужен только лучший документ для ItemResult
                mode="hybrid_rerank"             # Reranking повышает точность для юридических шаблонов
            )

            # 2️⃣ Обработка пустого ответа
            if not docs:
                return ItemResult(
                    id=item.id,
                    answer="Релевантный юридический шаблон не найден. Требуется ручная проверка.",
                    status="unofficial",
                    source=item.routing_target,
                    is_official=False
                )

            best_doc = docs[0]

            # 3️⃣ Формирование результата согласно требованиям Aggregator
            # В промпте указано: score > 0.9 → официальный ответ
            is_official = best_doc.score > 0.9

            return ItemResult(
                id=item.id,
                answer=f"{best_doc.answer}\n\n📎 Источник: {best_doc.question}",
                status="processed",
                source=best_doc.source,
                is_official=is_official,
                error=None
            )
        except Exception as e:
            print(f"❌ Ошибка поиска в {item.routing_target} (id={item.id}): {e}")
            return ItemResult(
                id=item.id,
                answer="",
                status="failed",
                source=item.routing_target,
                error=str(e)
            )

    def search_knowledge_base(item: SubQuery) -> ItemResult:
        return ItemResult(
            id=item.id,
            answer=f"Ответ из KB: {item.sub_query}",
            status="processed", 
            source="reference_info_collection",
            is_official=False
        )

    def search_knowledge_base_real(item: SubQuery) -> ItemResult:
        """Поиск в юридической коллекции через гибридный ретривер"""
        try:
            # 1️⃣ Запуск поиска
            docs: list[RetrievedDoc] = run_hybrid_search_KB(
                query=item.sub_query,      
                col_name=settings.collection_name_KB,    # "legal_templates_collection"
                top_k=3,                         # пока оставим  вывод трех документов с высоким рангом для последующего использования 
                mode="hybrid_rerank"             # Reranking повышает точность 
            )

            # 2️⃣ Обработка пустого ответа
            if not docs:
                return ItemResult(
                    id=item.id,
                    answer="Релевантный ответ из Баз знаний не найден. Требуется ручная проверка.",
                    status="unofficial",
                    source=item.routing_target,
                    is_official=False
                )

            best_doc = docs[0]+docs[1]+docs[2]# объединяем три документа

            # 3️⃣ Формирование результата согласно требованиям Aggregator
        
            is_official = best_doc.score > 0.9

            return ItemResult(
                id=item.id,
                answer=f"{best_doc.answer}\n\n📎 Источник: {best_doc.question}",
                status="processed",
                source=best_doc.source,
                is_official=is_official,
                error=None
            )
        except Exception as e:
            print(f"❌ Ошибка поиска в {item.routing_target} (id={item.id}): {e}")
            return ItemResult(
                id=item.id,
                answer="",
                status="failed",
                source=item.routing_target,
                error=str(e)
            )

    def call_billing_api(item: SubQuery) -> ItemResult:
        """  функция определиться после определения формата взаимодейтвия с внешней системой   
        client = UtilityClient(
            base_url="https://api.crm.com",
            api_key="secret_token_here",
            timeout=15
        )

        try:
            # 1️⃣ Получаем показания
            readings = client.get_readings(
                account_id="LS-100293",
                period="2024-01",
                resource="electricity"
            )
            print("📊 Показания:", readings)

            # 2️⃣ Получаем начисления
            charges = client.get_charges(
                account_id="LS-100293",
                billing_month="2024-01"
            )
            print("💰 Начисления:", charges)

        except Exception as e:
            print("❌ Ошибка при работе с API:", e)
        finally:
            client.close()  # освобождаем TCP-соединения
        """
        return ItemResult(
            id=item.id,
            answer=f"Данные из биллинга: {item.entities.account_number}. Начисления за период 234,78 рублей",
            status="processed",
            source="external_api_tools",
            is_official=False
        )

    def create_operator_ticket(item: SubQuery) -> ItemResult:
        return ItemResult(
            id=item.id,
            answer="Запрос передан оператору",
            status="operator_pending",
            source="operator_queue",
            is_official=False
        )

    def fallback_handler(item: SubQuery) -> ItemResult:
        return ItemResult(
            id=item.id,
            answer="Обработка завершена",
            status="processed",
            source="fallback",
            is_official=False
        )

    def aggregator_node(state: WorkflowState) -> dict:
        parts = state.get("partial_results", {})
        if not parts:
            return {
                "final_response": "Не удалось сформировать ответ: результаты обработки отсутствуют.",
                "status": "failed"
            }
        queries_map = {sq.id: sq.sub_query for sq in state.get("decomposed_items", [])}
        parts_text = "\n\n".join([
                f"🔹 Под-вопрос #{r.id}: {queries_map.get(r.id, 'Неизвестный запрос')}\n"
                f"   ✅ Ответ: {r.answer}\n"
                f"   📊 Статус: {r.status} | 📜 Официально: {'Да' if r.is_official else 'Нет'}"
                + (f"\n   ⚠️ Ошибка: {r.error}" if r.error else "")
                for r in sorted(parts.values(), key=lambda x: x.id)
            ])

        prompt_text = (
                f"Исходный запрос клиента: {state['original_query']}\n\n"
                f"Результаты покомпонентной обработки:\n{parts_text}\n\n"
                f"Задача: Сформируй связный, вежливый и точный итоговый ответ для клиента. "
                f"Учти статус обработки каждого аспекта. Если что-то передано оператору — "
                f"честно сообщи об этом и укажи сроки/дальнейшие шаги."
            )
        messages = [
            SystemMessage(content=AGGREGATOR_SYSTEM_PROMPT),
            HumanMessage(content=prompt_text)
        ]
        final_text = llm.invoke(messages)
        return {"final_response": final_text, "status": "completed"}

    return decomposer_node, process_item_node, aggregator_node