import logging
import re
from pathlib import Path
from typing import Optional, Dict, Any

try:
    import fitz  # PyMuPDF
except ImportError:
    fitz = None

try:
    import docx
    from docx import Document
except ImportError:
    docx = None
    Document = None

try:
    from bs4 import BeautifulSoup
except ImportError:
    BeautifulSoup = None

try:
    import requests
    from readability import Document as ReadabilityDocument
except ImportError:
    requests = None
    ReadabilityDocument = None

logger = logging.getLogger(__name__)

# ==========================================
# 1. PDF Parser (Улучшенный)
# ==========================================
def parse_pdf(file_path: str) -> Optional[Dict[str, Any]]:
    if fitz is None:
        logger.error("PyMuPDF (fitz) is not installed.")
        return None
    try:
        doc = fitz.open(file_path)
    except Exception as e:
        logger.error(f"Error opening PDF {file_path}: {e}")
        return None

    full_text = []
    stop_parsing = False
    
    page_num_pattern = re.compile(r'^\s*\d+\s*$')
    # Улучшенный паттерн References: ловит "10. References", "VII. Bibliography"
    references_pattern = re.compile(
        r'^\s*(?:\d+[\.\)]?\s*|[IVX]+\.?\s*)?(references|bibliography|literature cited)\.?\s*$', 
        re.IGNORECASE
    )

    for page_num in range(len(doc)):
        if stop_parsing:
            break
            
        page = doc[page_num]
        
        # --- Извлечение таблиц ---
        tables = page.find_tables()
        table_bboxes = []
        for table in tables:
            table_bboxes.append(fitz.Rect(table.bbox))
            try:
                md = table.to_markdown()
                full_text.append(md)
            except AttributeError:
                rows = table.extract()
                for row in rows:
                    clean_row = [str(cell).strip().replace('\n', ' ') if cell else "" for cell in row]
                    full_text.append("| " + " | ".join(clean_row) + " |")

        # --- Извлечение текста с умной сортировкой (sort=True) ---
        blocks = page.get_text("blocks", sort=True)
        page_height = page.rect.height
        
        # Берем только текстовые блоки
        text_blocks = [b for b in blocks if b[6] == 0]
        
        for b in text_blocks:
            x0, y0, x1, y1, text, block_no, block_type = b
            b_rect = fitz.Rect(x0, y0, x1, y1)
            
            # Пропускаем блок, если он пересекается с таблицей
            if any(b_rect.intersects(tb) for tb in table_bboxes):
                continue
                
            text_clean = text.strip()
            if not text_clean:
                continue
                
            # Дефисация: склеиваем разорванные слова (напр. "фарма-\nкология")
            text_clean = re.sub(r'-\s*\n\s*', '', text_clean)
            # Убираем лишние переносы строк внутри одного блока
            text_clean = text_clean.replace('\n', ' ').strip()
                
            if references_pattern.match(text_clean):
                logger.info(f"Найден раздел References в {file_path}. Остановка парсинга.")
                stop_parsing = True
                break
                
            # Фильтрация колонтитулов (верхние/нижние 5% страницы)
            if y0 < page_height * 0.05 or y1 > page_height * 0.95:
                continue
                
            # Фильтрация одиночных номеров страниц
            if page_num_pattern.match(text_clean):
                continue
                
            full_text.append(text_clean)

    doc.close()
    return {
        "text": "\n".join(full_text),
        "metadata": {
            "source": Path(file_path).name,
            "format": "pdf"
        }
    }

# ==========================================
# 2. DOCX Parser (Улучшенный порядок)
# ==========================================
def parse_docx(file_path: str) -> Optional[Dict[str, Any]]:
    if Document is None or docx is None:
        logger.error("python-docx is not installed.")
        return None
    try:
        doc = Document(file_path)
    except Exception as e:
        logger.error(f"Error opening DOCX {file_path}: {e}")
        return None
        
    full_text = []
    
    # Обход XML-дерева body для сохранения строгого порядка элементов
    for child in doc.element.body:
        if child.tag.endswith('p'):
            para = docx.text.paragraph.Paragraph(child, doc)
            text = para.text.strip()
            if text:
                full_text.append(text)
        elif child.tag.endswith('tbl'):
            table = docx.table.Table(child, doc)
            for row in table.rows:
                row_data = [cell.text.strip().replace('\n', ' ') for cell in row.cells]
                full_text.append("| " + " | ".join(row_data) + " |")
            
    return {
        "text": "\n\n".join(full_text),
        "metadata": {
            "source": Path(file_path).name,
            "format": "docx"
        }
    }

# ==========================================
# 3. XML Parser (JATS) с Markdown
# ==========================================
def parse_xml(file_path: str) -> Optional[Dict[str, Any]]:
    if BeautifulSoup is None:
        logger.error("BeautifulSoup4 (bs4) is not installed.")
        return None
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            soup = BeautifulSoup(f, "xml")
    except Exception as e:
        logger.error(f"Error reading XML {file_path}: {e}")
        return None
        
    # Удаляем нежелательные блоки
    for tag in soup(["ref-list", "back"]):
        tag.decompose()
        
    full_text = []
    
    title = soup.find("article-title")
    if title:
        full_text.append(f"# {title.get_text(separator=' ', strip=True)}")
        
    abstract = soup.find("abstract")
    if abstract:
        full_text.append("## Abstract")
        full_text.append(abstract.get_text(separator=' ', strip=True))
        
    body = soup.find("body")
    if body:
        # Внедряем Markdown для внутренних заголовков секций
        for sec_title in body.find_all("title"):
            sec_title.string = f"\n### {sec_title.get_text(strip=True)}\n"
            
        full_text.append(body.get_text(separator=' ', strip=True))
        
    return {
        "text": "\n\n".join(full_text),
        "metadata": {
            "source": Path(file_path).name,
            "format": "xml"
        }
    }

# ==========================================
# 4. HTML/URL Parser
# ==========================================
def parse_html(source: str) -> Optional[Dict[str, Any]]:
    if requests is None or ReadabilityDocument is None or BeautifulSoup is None:
        logger.error("requests, readability-lxml or bs4 is not installed.")
        return None
        
    html_content = ""
    is_url = source.startswith("http://") or source.startswith("https://")
    
    try:
        if is_url:
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36"
            }
            response = requests.get(source, headers=headers, timeout=15)
            response.raise_for_status()
            html_content = response.text
        else:
            with open(source, "r", encoding="utf-8") as f:
                html_content = f.read()
    except Exception as e:
        logger.error(f"Error fetching HTML from {source}: {e}")
        return None
        
    try:
        doc = ReadabilityDocument(html_content)
        article_html = doc.summary()
        
        soup = BeautifulSoup(article_html, "html.parser")
        clean_text = soup.get_text(separator='\n', strip=True)
    except Exception as e:
        logger.error(f"Error parsing HTML {source}: {e}")
        return None
        
    return {
        "text": clean_text,
        "metadata": {
            "source": source if is_url else Path(source).name,
            "format": "url" if is_url else "html"
        }
    }
