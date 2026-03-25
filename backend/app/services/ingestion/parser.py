"""
Document parser service.

Dispatches parsing strategy based on file type:
- PDF: text extraction + table extraction + optional vision page understanding
- DOCX: text + table extraction
- XLSX/CSV: spreadsheet-aware cell and sheet extraction
- HTML: clean text extraction with structure preservation
"""

from pathlib import Path
from typing import Protocol


class ParsedDocument(Protocol):
    """Parsed document interface returned by all parsers."""
    text_chunks: list[dict]
    table_chunks: list[dict]
    image_chunks: list[dict]
    metadata: dict


def parse_document(file_path: Path, strategy: str = "text_only") -> list[dict]:
    """
    Parse a document into structured chunks.

    Strategies:
        text_only           Plain text extraction
        text_table          Text + structured table extraction
        text_table_vision   Text + tables + vision page understanding (Phase 3)
        spreadsheet_aware   Workbook-level cell and sheet extraction

    Returns:
        List of chunk dicts, each with keys:
            content, chunk_type, page, table_id, metadata
    """
    suffix = file_path.suffix.lower()

    if suffix == ".pdf":
        return _parse_pdf(file_path, strategy)
    elif suffix in (".docx", ".doc"):
        return _parse_docx(file_path, strategy)
    elif suffix in (".xlsx", ".csv"):
        return _parse_spreadsheet(file_path)
    elif suffix in (".html", ".htm"):
        return _parse_html(file_path)

    raise ValueError(f"Unsupported file type: {suffix}")


def _parse_pdf(path: Path, strategy: str) -> list[dict]:
    if strategy == "text_table_vision":
        return _parse_pdf_with_vision(path)

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
                        table_id = f"table_{page_num + 1}_{table_idx}"
                        chunks.append({
                            "content": md_content,
                            "chunk_type": "table",
                            "page": page_num + 1,
                            "table_id": table_id,
                            "metadata": {
                                "filename": filename,
                                "page_count": page_count,
                                "strategy": strategy,
                                "table_index": table_idx,
                            },
                        })
        except Exception:
            pass  # pdfplumber failure is non-fatal; text chunks still returned

    doc.close()
    return chunks


def _parse_pdf_with_vision(path: Path) -> list[dict]:
    """
    Text + table + vision page understanding.

    For each page: extract text and tables (like text_table), then render
    the page as an image and send it to GPT-4o vision to describe any
    charts, diagrams, or visual layouts that text extraction misses.
    Falls back to text_table if vision calls fail.
    """
    import fitz
    import base64

    from openai import OpenAI
    from app.core.config import settings

    doc = fitz.open(str(path))
    chunks = []
    filename = path.name
    page_count = len(doc)

    client = OpenAI(api_key=settings.OPENAI_API_KEY)

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

        # 2. Render page as image and send to GPT-4o vision
        try:
            pix = page.get_pixmap(dpi=150)
            img_bytes = pix.tobytes("png")
            b64_img = base64.b64encode(img_bytes).decode("utf-8")

            response = client.chat.completions.create(
                model=settings.OPENAI_MODEL,
                messages=[{
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "Analyze this document page. Extract ALL information including:\n"
                                "1. Any tables — output each as a markdown table\n"
                                "2. Any charts, diagrams, or figures — describe them in detail "
                                "with all visible data points, labels, and values\n"
                                "3. Any visual layouts (flowcharts, org charts) — describe structure and content\n"
                                "If the page is just plain text, reply with: PLAIN_TEXT_ONLY\n"
                                "Be exhaustive. Include every number, label, and data point visible."
                            ),
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{b64_img}"},
                        },
                    ],
                }],
                max_tokens=2000,
                temperature=0,
            )

            vision_content = response.choices[0].message.content or ""
            if vision_content.strip() and "PLAIN_TEXT_ONLY" not in vision_content:
                # Determine chunk type from content
                has_table = "|" in vision_content and "---" in vision_content
                chunk_type = "table" if has_table else "image"
                chunks.append({
                    "content": vision_content,
                    "chunk_type": chunk_type,
                    "page": page_num + 1,
                    "table_id": f"vision_{page_num + 1}" if has_table else None,
                    "metadata": {
                        "filename": filename,
                        "page_count": page_count,
                        "strategy": "text_table_vision",
                        "source": "gpt4o_vision",
                    },
                })
        except Exception:
            pass  # Vision failure is non-fatal; text chunks still returned

    # 3. Also extract tables via pdfplumber (same as text_table)
    try:
        import pdfplumber
        with pdfplumber.open(str(path)) as pdf_doc:
            for page_num, page in enumerate(pdf_doc.pages):
                tables = page.extract_tables()
                for table_idx, table in enumerate(tables):
                    if not table:
                        continue
                    md_rows = []
                    for row in table:
                        cells = [str(cell or "").strip() for cell in row]
                        md_rows.append("| " + " | ".join(cells) + " |")
                    if len(md_rows) > 1:
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
                            "strategy": "text_table_vision",
                            "table_index": table_idx,
                        },
                    })
    except Exception:
        pass

    doc.close()
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
    except Exception:
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
