import logging
import os
import tempfile
import mimetypes
import requests
from pathlib import Path
from typing import Optional, Dict, Any
from urllib.parse import urlparse

from src.indexing.extractors import parse_pdf, parse_docx, parse_xml, parse_html
from src.core.config import MAX_FILE_SIZE
from src.core.logger import setup_logger

logger = setup_logger(__name__)



class DocumentRouter:
    """
    Маршрутизатор документов.
    Единая точка входа, определяющая тип источника (файл/URL) 
    и направляющая его в соответствующий парсер.
    """

    def process(self, source: str) -> Optional[Dict[str, Any]]:
        """
        Главный метод для обработки источника данных.
        
        :param source: Локальный путь к файлу или URL (http/https).
        :return: Словарь с извлеченным текстом и метаданными, либо None в случае ошибки или неподдерживаемого формата.
        """
        parsed_url = urlparse(source)
        if parsed_url.scheme in ("http", "https"):
            return self._handle_url(source)

        return self._handle_local_file(source)

    def _handle_url(self, url: str) -> Optional[Dict[str, Any]]:
        try:
            logger.info(f"Обработка URL: {url}")
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36"
            }
            # Используем stream=True для чтения только заголовков без скачивания тела
            response = requests.get(url, headers=headers, stream=True, timeout=15)
            response.raise_for_status()
            
            # Проверка размера файла (если сервер отдает Content-Length)
            content_length = response.headers.get("Content-Length")
            if content_length and int(content_length) > MAX_FILE_SIZE:
                logger.error(f"URL {url} отклонен: размер ({int(content_length)/1024/1024:.2f} MB) превышает лимит в 50 MB.")
                return None
                
            content_type = response.headers.get("Content-Type", "").lower()
            
            # Если это напрямую ссылка на PDF
            if "application/pdf" in content_type or url.lower().endswith(".pdf"):
                logger.info(f"URL ведет на PDF-файл. Начинаю загрузку во временный файл...")
                # Скачиваем во временный файл
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            tmp.write(chunk)
                    tmp_path = tmp.name
                
                try:
                    result = self._parse_pdf(tmp_path)
                    # Подменяем локальный временный путь обратно на оригинальный URL в метаданных
                    if result and "metadata" in result:
                        result["metadata"]["source"] = url
                    return result
                finally:
                    # Обязательно удаляем временный файл
                    os.unlink(tmp_path)
            
            # Иначе обрабатываем как обычную веб-страницу
            logger.info("URL обрабатывается как HTML-страница.")
            return self._parse_html_url(url)
            
        except Exception as e:
            logger.error(f"Ошибка при обработке URL {url}: {e}")
            return None

    def _handle_local_file(self, source: str) -> Optional[Dict[str, Any]]:
        file_path = Path(source)
        if not file_path.exists() or not file_path.is_file():
            logger.error(f"Файл не найден или не является файлом: {source}")
            return None

        # Проверка размера
        file_size = os.path.getsize(file_path)
        if file_size > MAX_FILE_SIZE:
            logger.error(f"Локальный файл {source} отклонен: размер ({file_size/1024/1024:.2f} MB) превышает лимит.")
            return None

        extension = file_path.suffix.lower()
        
        # Если расширения нет или оно неизвестно, пытаемся угадать по "магическим байтам"
        if not extension or extension not in [".pdf", ".docx", ".xml", ".html", ".htm"]:
            extension = self._guess_extension_by_signature(file_path)

        logger.info(f"Обрабатывается локальный файл: {source} (определенный формат: {extension})")

        match extension:
            case ".pdf":
                return self._parse_pdf(source)
            case ".docx":
                return self._parse_docx(source)
            case ".xml":
                return self._parse_xml(source)
            case ".html" | ".htm":
                return self._parse_html_file(source)
            case _:
                logger.warning(f"Unsupported format: {extension} для файла {source}")
                return None

    def _guess_extension_by_signature(self, file_path: Path) -> str:
        """Попытка определить тип файла по первым байтам или mimetypes."""
        try:
            with open(file_path, "rb") as f:
                header = f.read(4)
                if header.startswith(b"%PDF"):
                    return ".pdf"
                elif header.startswith(b"PK\x03\x04"):
                    # Это ZIP-архив, DOCX тоже является ZIP-архивом
                    return ".docx"
                elif header.startswith(b"<?xm"):
                    return ".xml"
        except Exception as e:
            logger.debug(f"Не удалось прочитать сигнатуру {file_path}: {e}")

        # Fallback на встроенный модуль mimetypes
        mime_type, _ = mimetypes.guess_type(str(file_path))
        if mime_type == "application/pdf":
            return ".pdf"
        elif mime_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            return ".docx"
        elif mime_type in ["text/xml", "application/xml"]:
            return ".xml"
            
        return file_path.suffix.lower() # Возвращаем оригинальное, если ничего не подошло

    # --- Подключение парсеров ---

    def _parse_pdf(self, file_path: str) -> Optional[Dict[str, Any]]:
        return parse_pdf(file_path)

    def _parse_docx(self, file_path: str) -> Optional[Dict[str, Any]]:
        return parse_docx(file_path)

    def _parse_xml(self, file_path: str) -> Optional[Dict[str, Any]]:
        return parse_xml(file_path)

    def _parse_html_file(self, file_path: str) -> Optional[Dict[str, Any]]:
        return parse_html(file_path)

    def _parse_html_url(self, url: str) -> Optional[Dict[str, Any]]:
        return parse_html(url)


if __name__ == "__main__":
    router = DocumentRouter()

    # Моковые тестовые данные для проверки маршрутизатора
    test_inputs = [
        "http://example.com/article", # HTML
        "https://www.w3.org/WAI/ER/tests/xhtml/testfiles/resources/pdf/dummy.pdf", # Прямая ссылка на PDF
        "test.pdf",
        "no_extension_file", # Проверим работу магических байт
        "too_large.pdf"
    ]

    # Создаем фиктивные файлы для тестов
    Path("test.pdf").write_bytes(b"%PDF-1.4 mock content") # Правильная сигнатура
    Path("no_extension_file").write_bytes(b"%PDF-1.4 hidden pdf content") # Правильная сигнатура, нет расширения
    
    # Создаем большой файл (>50MB), забивая его нулями (sparse файл для скорости)
    large_file = Path("too_large.pdf")
    with open(large_file, "wb") as f:
        f.seek((55 * 1024 * 1024) - 1)
        f.write(b'\0')

    print("=== Начало тестирования маршрутизатора ===")
    for test_input in test_inputs:
        print(f"\n--- Входные данные: {test_input} ---")
        result = router.process(test_input)
        if result:
            print(f"Успех! Извлечено символов: {len(result.get('text', ''))}")
            print(f"Метаданные: {result.get('metadata')}")
        else:
            print("Результат: None")
    
    print("\n=== Завершение тестирования ===")

    # Очистка
    Path("test.pdf").unlink(missing_ok=True)
    Path("no_extension_file").unlink(missing_ok=True)
    Path("too_large.pdf").unlink(missing_ok=True)
