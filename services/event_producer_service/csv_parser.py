import csv
import io
from typing import AsyncGenerator, Dict


async def parse_csv_stream(request_iterator) -> AsyncGenerator[Dict[str, str], None]:
    """
    Асинхронно парсит CSV из потока gRPC чанков.
    Каждый чанк — объект с полем .content (bytes).
    Возвращает словари для каждой строки CSV (без заголовка или с заголовком — зависит от вашей логики).
    """
    remainder = b''
    header = None  # если нужен заголовок, обработайте первую строку отдельно

    async for chunk in request_iterator:
        # chunk — это CSVChunk (protobuf), извлекаем байты
        data = remainder + chunk.content
        lines = data.split(b'\n')
        remainder = lines.pop()  # последний кусок может быть неполным

        for line in lines:
            if not line.strip():
                continue  # пропускаем пустые строки
            # Декодируем строку в текст
            try:
                line_str = line.decode('utf-8')
            except UnicodeDecodeError:
                # Если другая кодировка — замените на нужную, например cp1251
                line_str = line.decode('cp1251')

            # Парсим CSV
            csv_reader = csv.reader(io.StringIO(line_str))
            try:
                row = next(csv_reader)
            except StopIteration:
                continue  # пустая строка после декодирования

            # Если у вас есть заголовок, обработайте его здесь
            if header is None:
                header = row
                continue  # пропускаем строку заголовка, не возвращаем её

            # Превращаем список значений в словарь (если нужен)
            # Предполагаем, что количество колонок совпадает с заголовком
            if len(row) != len(header):
                # Можете выбросить ошибку или пропустить
                continue
            row_dict = dict(zip(header, row))
            yield row_dict

    # Обрабатываем остаток (последняя строка без \n в конце файла)
    if remainder:
        line_str = remainder.decode('utf-8')
        if line_str.strip():
            csv_reader = csv.reader(io.StringIO(line_str))
            try:
                row = next(csv_reader)
                if header and len(row) == len(header):
                    row_dict = dict(zip(header, row))
                    yield row_dict
            except StopIteration:
                pass