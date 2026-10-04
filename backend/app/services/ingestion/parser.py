"""
Document parser service.

Dispatches parsing strategy based on file type:
- PDF: text extraction + table extraction + optional vision page understanding
- DOCX: text + table extraction
- XLSX/CSV: spreadsheet-aware cell and sheet extraction
- HTML: clean text extraction with structure preservation
"""

import logging
from pathlib import Path
from typing import Optional, Protocol

logger = logging.getLogger(__name__)

# Prompt sent with each rendered PDF page in the text_table_vision strategy.
VISION_PROMPT = (
    "Analyze this document page. Extract ALL information including:\n"
    "1. Any tables — output each as a markdown table\n"
    "2. Any charts, diagrams, or figures — describe them in detail "
    "with all visible data points, labels, and values\n"
    "3. Any visual layouts (flowcharts, org charts) — describe structure and content\n"
    "If the page is just plain text, reply with: PLAIN_TEXT_ONLY\n"
    "Be exhaustive. Include every number, label, and data point visible."
)


class ParsedDocument(Protocol):
    """Parsed document interface returned by all parsers."""
    text_chunks: list[dict]
    table_chunks: list[dict]
    image_chunks: list[dict]
    metadata: dict


def parse_document(file_path: Path, strategy: str = "text_only", report: Optional[dict] = None) -> list[dict]:
    """
    Parse a document into structured chunks.

    Strategies:
        text_only           Plain text extraction
        text_table          Text + structured table extraction
        text_table_vision   Text + tables + vision page understanding (PDF only;
                            calls settings.VISION_MODEL for up to
                            settings.VISION_MAX_PAGES pages)
        spreadsheet_aware   Workbook-level cell and sheet extraction

    If ``report`` is given, parsers that do costly or lossy work record what
    happened in it (the vision path adds a "vision" entry with pages sent,
    skipped and failed).

    Returns:
        List of chunk dicts, each with keys:
            content, chunk_type, page, table_id, metadata
    """
    suffix = file_path.suffix.lower()

    if suffix == ".pdf":
        return _parse_pdf(file_path, strategy, report)
    elif suffix in (".docx", ".doc"):
        return _parse_docx(file_path, strategy)
    elif suffix in (".xlsx", ".csv"):
        return _parse_spreadsheet(file_path)
    elif suffix in (".html", ".htm"):
        return _parse_html(file_path)

    raise ValueError(f"Unsupported file type: {suffix}")


def _parse_pdf(path: Path, strategy: str, report: Optional[dict] = None) -> list[dict]:
    if strategy == "text_table_vision":
        return _parse_pdf_with_vision(path, report)

    import fitz  # PyMuPDF

    doc = fitz.open(str(path))
    chunks = []
    filename = path.name
    page_count = len(doc)

    if strategy in ("text_only", "text_table", "spreadsheet_aware"):
        # Extract text from every page
        for page_num, page in enumerate(doc):
            text = page.get_text("text").strip()
            if text:
                chunks.append({
                    "content": text,
                    "chunk_type": "text",
                    "page": page_num + 1,
                    "table_id": None,
                    "metadata": {
                        "filename": filename,
                        "page_count": page_count,
                        "strategy": strategy,
                    },
                })

    if strategy == "text_table":
        # Also extract tables using pdfplumber
        chunks.extend(_extract_pdf_tables(path, filename, page_count, strategy))

    doc.close()
    return chunks


def _extract_pdf_tables(path: Path, filename: str, page_count: int, strategy: str) -> list[dict]:
    """Extract PDF tables with pdfplumber as markdown table chunks."""
    chunks = []
    try:
        import pdfplumber
        with pdfplumber.open(str(path)) as pdf_doc:
            for page_num, page in enumerate(pdf_doc.pages):
                tables = page.extract_tables()
                for table_idx, table in enumerate(tables):
                    if not table:
                        continue
                    # Serialize table as markdown
                    md_rows = []
                    for row in table:
                        cells = [str(cell or "").strip() for cell in row]
                        md_rows.append("| " + " | ".join(cells) + " |")
                    if len(md_rows) > 1:
                        # Insert separator after header
                        separator = "| " + " | ".join(["---"] * len(table[0])) + " |"
                        md_rows.insert(1, separator)
                    md_content = "\n".join(md_rows)
                    chunks.append({
                        "content": md_content,
                        "chunk_type": "table",
                        "page": page_num + 1,
                        "table_id": f"table_{page_num + 1}_{table_idx}",
                        "metadata": {
                            "filename": filename,
                            "page_count": page_count,
                            "strategy": strategy,
                            "table_index": table_idx,
                        },
                    })
    except Exception as exc:
        # Non-fatal: text chunks are still returned, but say so.
        logger.warning("pdfplumber table extraction failed for %s: %s", filename, exc)
    return chunks


def _make_vision_client(api_key: str):
    """OpenAI client for page vision calls (a function so tests can replace it)."""
    from openai import OpenAI
    return OpenAI(api_key=api_key)


def _describe_page(client, model: str, png_bytes: bytes) -> str:
    """Send one rendered page to the vision model and return its description."""
    import base64

    b64_img = base64.b64encode(png_bytes).decode("utf-8")
    response = client.chat.completions.create(
        model=model,
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": VISION_PROMPT},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_img}"}},
            ],
        }],
        max_tokens=2000,
        temperature=0,
    )
    return response.choices[0].message.content or ""


def _parse_pdf_with_vision(path: Path, report: Optional[dict] = None) -> list[dict]:
    """
    Text + table + vision page understanding.

    For each page: extract text (like text_table); for the first
    settings.VISION_MAX_PAGES pages, also render the page as an image and send
    it to settings.VISION_MODEL to describe charts, diagrams, tables and visual
    layouts that text extraction misses. Then extract tables with pdfplumber.

    Every vision chunk has chunk_type "image" and metadata.source
    "gpt4o_vision" (what diagnostics look for); metadata.vision_has_table
    marks descriptions that contain a markdown table. A failed vision call is
    logged and recorded in ``report``; the other pages still parse.
    """
    import fitz
    from app.core.config import settings

    if not settings.OPENAI_API_KEY:
        raise RuntimeError("text_table_vision parsing requires OPENAI_API_KEY")

    doc = fitz.open(str(path))
    chunks = []
    filename = path.name
    page_count = len(doc)
    max_pages = max(0, settings.VISION_MAX_PAGES)
    model = settings.VISION_MODEL
    stats = {
        "model": model,
        "max_pages": max_pages,
        "pages_sent": 0,
        "pages_skipped": 0,
        "plain_text_pages": 0,
        "vision_chunks": 0,
        "errors": [],
    }

    client = _make_vision_client(settings.OPENAI_API_KEY) if page_count and max_pages else None

    try:
        for page_num, page in enumerate(doc):
            # 1. Text extraction (same as text_table)
            text = page.get_text("text").strip()
            if text:
                chunks.append({
                    "content": text,
                    "chunk_type": "text",
                    "page": page_num + 1,
                    "table_id": None,
                    "metadata": {
                        "filename": filename,
                        "page_count": page_count,
                        "strategy": "text_table_vision",
                    },
                })

            # 2. Render page as image and send to the vision model (bounded)
            if page_num >= max_pages:
                stats["pages_skipped"] += 1
                continue
            stats["pages_sent"] += 1
            try:
                png_bytes = page.get_pixmap(dpi=150).tobytes("png")
                vision_content = _describe_page(client, model, png_bytes)
            except Exception as exc:
                logger.warning("Vision parsing failed for %s page %d: %s", filename, page_num + 1, exc)
                stats["errors"].append({"page": page_num + 1, "error": f"{type(exc).__name__}: {exc}"[:300]})
                continue

            if not vision_content.strip() or "PLAIN_TEXT_ONLY" in vision_content:
                stats["plain_text_pages"] += 1
                continue
            has_table = "|" in vision_content and "---" in vision_content
            chunks.append({
                "content": vision_content,
                "chunk_type": "image",
                "page": page_num + 1,
                "table_id": f"vision_{page_num + 1}",
                "metadata": {
                    "filename": filename,
                    "page_count": page_count,
                    "strategy": "text_table_vision",
                    "source": "gpt4o_vision",
                    "vision_model": model,
                    "vision_has_table": has_table,
                },
            })
            stats["vision_chunks"] += 1
    finally:
        doc.close()

    if stats["pages_skipped"]:
        logger.info(
            "Vision parsing of %s: sent %d of %d pages (VISION_MAX_PAGES=%d), skipped %d",
            filename, stats["pages_sent"], page_count, max_pages, stats["pages_skipped"],
        )
    if stats["errors"]:
        logger.error(
            "Vision parsing of %s: %d of %d pages failed",
            filename, len(stats["errors"]), stats["pages_sent"],
        )

    # 3. Also extract tables via pdfplumber (same as text_table)
    chunks.extend(_extract_pdf_tables(path, filename, page_count, "text_table_vision"))

    if report is not None:
        report["vision"] = stats
    return chunks


def _parse_docx(path: Path, strategy: str) -> list[dict]:
    from docx import Document

    doc = Document(str(path))
    filename = path.name
    chunks = []

    # Extract paragraphs
    for para_idx, para in enumerate(doc.paragraphs):
        text = para.text.strip()
        if not text:
            continue
        meta: dict = {"filename": filename, "strategy": strategy, "para_index": para_idx}
        if para.style and para.style.name and para.style.name.startswith("Heading"):
            meta["heading_level"] = para.style.name
        chunks.append({
            "content": text,
            "chunk_type": "text",
            "page": None,
            "table_id": None,
            "metadata": meta,
        })

    if strategy == "text_table":
        for table_idx, table in enumerate(doc.tables):
            md_rows = []
            for row in table.rows:
                cells = [cell.text.strip() for cell in row.cells]
                md_rows.append("| " + " | ".join(cells) + " |")
            if len(md_rows) > 1:
                separator = "| " + " | ".join(["---"] * len(table.rows[0].cells)) + " |"
                md_rows.insert(1, separator)
            md_content = "\n".join(md_rows)
            if md_content.strip():
                chunks.append({
                    "content": md_content,
                    "chunk_type": "table",
                    "page": None,
                    "table_id": f"table_{table_idx}",
                    "metadata": {"filename": filename, "strategy": strategy},
                })

    return chunks


def _parse_spreadsheet(path: Path) -> list[dict]:
    filename = path.name
    chunks = []

    if path.suffix.lower() == ".csv":
        import csv
        with open(str(path), newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            headers = reader.fieldnames or []
            for row_idx, row in enumerate(reader):
                content = "Row {}: {}".format(
                    row_idx + 1,
                    ", ".join(f"{k}={v}" for k, v in row.items() if v),
                )
                chunks.append({
                    "content": content,
                    "chunk_type": "spreadsheet_cell",
                    "page": 0,
                    "table_id": "sheet_0",
                    "metadata": {"filename": filename, "headers": headers},
                })
        return chunks

    # XLSX
    import openpyxl
    wb = openpyxl.load_workbook(str(path), read_only=True, data_only=True)
    for sheet_idx, sheet_name in enumerate(wb.sheetnames):
        ws = wb[sheet_name]
        rows = list(ws.iter_rows(values_only=True))
        if not rows:
            continue

        headers = [str(h or "").strip() for h in rows[0]]
        # Sheet summary chunk
        chunks.append({
            "content": f"Sheet: {sheet_name} | Columns: {', '.join(h for h in headers if h)}",
            "chunk_type": "text",
            "page": sheet_idx,
            "table_id": sheet_name,
            "metadata": {"filename": filename, "sheet_name": sheet_name},
        })

        # Row-level chunks
        for row_idx, row in enumerate(rows[1:], start=2):
            parts = []
            for header, val in zip(headers, row):
                if val is not None and str(val).strip():
                    parts.append(f"{header}={val}")
            if parts:
                content = f"{sheet_name} | row {row_idx}: " + ", ".join(parts)
                chunks.append({
                    "content": content,
                    "chunk_type": "spreadsheet_cell",
                    "page": sheet_idx,
                    "table_id": sheet_name,
                    "metadata": {"filename": filename, "sheet_name": sheet_name, "row": row_idx},
                })

    wb.close()
    return chunks


def _parse_html(path: Path) -> list[dict]:
    filename = path.name
    chunks = []

    try:
        from unstructured.partition.html import partition_html
        elements = partition_html(filename=str(path))
        for elem_idx, elem in enumerate(elements):
            text = str(elem).strip()
            if text:
                chunk_type = "table" if "Table" in type(elem).__name__ else "text"
                chunks.append({
                    "content": text,
                    "chunk_type": chunk_type,
                    "page": None,
                    "table_id": f"table_{elem_idx}" if chunk_type == "table" else None,
                    "metadata": {"filename": filename, "element_type": type(elem).__name__},
                })
    except Exception as exc:
        logger.info("unstructured HTML parsing unavailable for %s (%s); using basic strip", filename, exc)
        # Fallback: basic HTML strip
        from html.parser import HTMLParser

        class _Stripper(HTMLParser):
            def __init__(self):
                super().__init__()
                self._parts = []

            def handle_data(self, data):
                stripped = data.strip()
                if stripped:
                    self._parts.append(stripped)

        with open(str(path), encoding="utf-8", errors="replace") as f:
            content = f.read()

        stripper = _Stripper()
        stripper.feed(content)
        full_text = "\n".join(stripper._parts)
        if full_text:
            chunks.append({
                "content": full_text,
                "chunk_type": "text",
                "page": None,
                "table_id": None,
                "metadata": {"filename": filename},
            })

    return chunks
