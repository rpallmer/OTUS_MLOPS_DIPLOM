import csv
import json
import re
from pathlib import Path

def parse_query_text(text: str) -> str:
    """
    Извлекает чистый текст обращения.
    
    Форматы:
    - "опиок -направить принципалу///Текст..." -> возвращает "Текст..."
    - "Просто текст" -> возвращает как есть
    """
    text = text.strip()
    if '///' in text:
        parts = text.split('///', 1)
        return parts[1].strip() if len(parts) > 1 else text
    return text

def csv_to_jsonl(csv_path: str, jsonl_path: str, encoding: str = 'utf-8'):
    """
    Конвертирует CSV в JSONL.
    
    На выходе: одна JSON-строка на запись, поля: id (int), query (str)
    """
    records = []
    
    with open(csv_path, 'r', encoding=encoding) as csvfile:
        reader = csv.reader(csvfile, delimiter=';', quotechar='"')
        next(reader, None)  # пропускаем заголовок
        
        for row in reader:
            if not row or not row[0].strip():
                continue
            
            raw_text = row[0].strip()
            query = parse_query_text(raw_text)
            
            if not query:  # пропускаем пустые обращения
                continue
                
            records.append({'id': len(records) + 1, 'query': query})
    
    # Запись в JSONL (каждая запись — отдельная строка)
    with open(jsonl_path, 'w', encoding='utf-8') as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
    
    print(f"✅ Обработано записей: {len(records)}")
    print(f"📄 Результат: {jsonl_path}")
    return records

# ==================== ЗАПУСК ====================
if __name__ == '__main__':
    # Пути к файлам — измените под ваши нужды
    INPUT_CSV = 'data/raw/query216.csv'
    OUTPUT_JSON = 'data/processed/appeals216.jsonl'
    
    # Проверка наличия входного файла
    if not Path(INPUT_CSV).exists():
        print(f"❌ Файл {INPUT_CSV} не найден!")
        # Создадим пример входного файла для демонстрации
        with open(INPUT_CSV, 'w', encoding='utf-8') as f:
            f.write('Суть обращения;;;;;;\n')
            f.write('"опиок -направить принципалу///Я нахожусь в рабочей командировке, в связи с этим по месту прописки не проживал в течении длительного времени.\nИ вот я приезжаю в отпуск 27.09.2025 года, и что я вижу...\nВ моей комнате, да и на этаже в целом, невозможно находится, не то, чтобы проживать!\nПотолок в комнате частично обвалился, спальное место, теле-, IT оборудование можно выносить на свалку...\nВ коридоре уборка работниками обслуживающей организации не производится.\nНа жалобы жильцов, моих соседей, никакой реакции!!!\nА счёт за содержание жилья выставляется в полном объёме.\nСразу вопрос...\nЗаранее благодарю за вашу соответствующую реакцию, и ваш вразумительный ответ.\nНу а потом, конечно же по ситуации, решим вопрос с оплатой.";;;;;;\n')
            f.write('Замена счетчиков воды, как зарегистрировать счетчики?;;;;;;\n')
        print(f"📝 Создан демонстрационный файл {INPUT_CSV}")
    
    # Запуск конвертации
    data = csv_to_jsonl(INPUT_CSV, OUTPUT_JSON)
    
    # Вывод примера результата
    print("\n🔍 Пример вывода:")
    for item in data[:2]:
        print(json.dumps(item, ensure_ascii=False, indent=2))