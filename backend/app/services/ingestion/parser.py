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
        text_table_vision   Text + tables + vision page understanding
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
    raise NotImplementedError


def _parse_docx(path: Path, strategy: str) -> list[dict]:
    raise NotImplementedError


def _parse_spreadsheet(path: Path) -> list[dict]:
    raise NotImplementedError


def _parse_html(path: Path) -> list[dict]:
    raise NotImplementedError
