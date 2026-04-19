import json
import time
import re
import logging
from pathlib import Path
from typing import List, Optional, Dict, Any
from mlx_lm import load, generate

# ─────────────────────────────────────────────────────────────
# 🔧 КОНФИГУРАЦИЯ
# ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S"
)

MODEL_ID = "mlx-community/Qwen3.5-9B-MLX-4bit"  
TEMPERATURE = 0.3
MAX_TOKENS = 3072
MAX_RETRIES = 3
RETRY_DELAY = 2

SYSTEM_PROMPT = """Ты — эксперт по аугментации пользовательских запросов для RAG.
[РЕЖИМ ВЫПОЛНЕНИЯ]
• НЕ генерируй рассуждения, анализ, план или "Thinking Process"
• НЕ объясняй свой выбор стилей или формулировок
• Твой ответ — это ТОЛЬКО валидный JSON-массив, начинающийся с "["
• Если ты начнёшь с любого другого текста — задача будет считаться невыполненной

[ЗАДАЧА]
Сгенерируй ровно 4 варианта вопроса на русском языке на основе входного запроса.

[СТИЛИ]
1. Разговорный (естественная речь)
2. Формальный (деловой тон)
3. Краткий (5–12 слов, суть)
4. Детализированный (с контекстом)

[ЖЁСТКИЕ ПРАВИЛА]
• Сохраняй смысл и интент оригинала.
• Только русский язык, без англицизмов.
• НЕ добавляй выдуманные факты или процедуры.

[ФОРМАТ ВЫВОДА — СТРОГО]
• Верни ТОЛЬКО валидный JSON-массив.
• Начни ответ сразу с символа "[".
• Не используй markdown-разметку (никаких ```json).
• Не пиши пояснений, мыслей или текста до/после JSON.

[ПРИМЕР]
Input: "Как оплатить квитанцию?"
Output:
[
  {"Variant": "Подскажите, как лучше оплатить квитанцию?"},
  {"Variant": "Прошу разъяснить порядок оплаты квитанции."},
  {"Variant": "Как оплатить квитанцию?"},
  {"Variant": "Какие способы оплаты квитанции доступны в личном кабинете или через банк?"}
]

[ВХОДНОЙ ВОПРОС]
"{original_question}"

[НАЧНИ ОТВЕТ С ЭТОЙ СТРОКИ]
[
"""

USER_PROMPT = 'Исходный вопрос: "{question}"\n\nСгенерируй варианты по инструкции выше.'

# ─────────────────────────────────────────────────────────────
# 🔧 ФУНКЦИИ
# ─────────────────────────────────────────────────────────────
def _call_mlx(model, tokenizer, prompt: str) -> Optional[str]:
    """Вызов генерации через mlx-lm"""
    try:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ]
        chat_prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        response = generate(
            model=model,
            tokenizer=tokenizer,
            prompt=chat_prompt,
            max_tokens=MAX_TOKENS,
            verbose=False 
        )
        return response
    except Exception as e:
        logging.warning(f"Ошибка MLX-LM: {e}")
        return None

def _parse_variants(content: str) -> List[str]:
        
    content = content.strip()
    match = re.search(r'\[\s*\{[^}]*"Variant"[^}]*\}(?:\s*,\s*\{[^}]*"Variant"[^}]*\}){3}\s*\]', content, re.DOTALL)
    json_str = match.group(0) if match else content

    try:
        data = json.loads(json_str)
        
        if not isinstance(data, list):
            return []
            
        results = []
        for item in data:
       
            if isinstance(item, dict) and "Variant" in item:
                val = item["Variant"]
                if isinstance(val, str) and val.strip():
                    results.append(val.strip())
            
            elif isinstance(item, str) and item.strip():
                results.append(item.strip())
                
        return results
        
    except json.JSONDecodeError as e:
        print(f"⚠️ Ошибка парсинга JSON: {e}")
        return []


def generate_question_variants(
    input_path: str,
    output_path: str,
    start_index: int = 0,
    end_index: Optional[int] = None
) -> Dict[str, Any]:

    logging.info(f"🔄 Загрузка модели {MODEL_ID} (ожидание 1-2 мин при первом запуске)...")
    model, tokenizer = load(MODEL_ID)
    logging.info("✅ Модель успешно загружена в Unified Memory.\n")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    stats = {"processed": 0, "skipped": 0, "errors": 0, "total_variants": 0}

    with open(input_path, "r", encoding="utf-8") as f_in, \
         open(output_path, "w", encoding="utf-8") as f_out:
        
        for line_idx, line in enumerate(f_in, 1):
            line = line.strip()
            if not line:
                continue
                
            # Фильтрация по диапазону строк
            if line_idx <= start_index:
                continue
            if end_index is not None and line_idx > end_index:
                break

            # Парсинг строки JSONL
            try:
                doc = json.loads(line)
            except json.JSONDecodeError:
                logging.warning(f"Строка {line_idx}: битый JSON, пропуск.")
                stats["skipped"] += 1
                continue

            q_id = doc.get("id")
            question = doc.get("question", "").strip()
            
            if not question:
                logging.warning(f"Строка {line_idx}: поле question пустое, пропуск.")
                stats["skipped"] += 1
                continue

            logging.info(f"📦 [{line_idx}] ID={q_id} | Вопрос: {question[:60]}...")
            prompt = USER_PROMPT.format(question=question)
            
            variants = []
            success = False
            for attempt in range(MAX_RETRIES):
                response = _call_mlx(model, tokenizer, prompt)
                if response:
                    variants = _parse_variants(response)
                    if variants:
                        success = True
                        break
                logging.warning(f"  🔄 Попытка {attempt+1}/{MAX_RETRIES} не удалась.")
                time.sleep(RETRY_DELAY)
                
            if not success:
                logging.error(f"  ❌ Не удалось сгенерировать варианты для ID={q_id}")
                stats["errors"] += 1
                continue

            # Запись результатов в JSONL
            for var in variants:
                record = {"query": var, "relevant_ids": [q_id]}
                f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
                
            stats["processed"] += 1
            stats["total_variants"] += len(variants)
            logging.info(f"  ✅ Записано {len(variants)} вариантов\n")

    logging.info("🎉 Готово! Статистика: %s", json.dumps(stats, ensure_ascii=False, indent=2))
    return stats

# ─────────────────────────────────────────────────────────────
# ▶️ ЗАПУСК
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    generate_question_variants(
        input_path="data/processed/corpus_test.jsonl",   # Укажите путь к вашему файлу
        output_path="data/processed/eval.jsonl",
        start_index=0,
        end_index=None  # None = обработать до конца файла
    )