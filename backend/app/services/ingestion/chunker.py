"""
Chunking strategy module.

Implements multiple text splitting strategies used across pipeline variants.
Note: Table and spreadsheet chunks from the parser are already atomic units
and should bypass the chunker entirely — only apply to text-type chunks.
"""


def chunk(text: str, strategy: str = "recursive", **kwargs) -> list[dict]:
    """
    Split text into chunks using the specified strategy.

    Strategies:
        fixed_size          Fixed character/token window with overlap
        recursive           LangChain RecursiveCharacterTextSplitter
        semantic            Embedding-similarity boundary detection (Phase 2)
        structure_aware     Heading- and layout-aware splitting (Phase 2)

    Returns:
        List of chunk dicts with keys: content, start_index, end_index, metadata
    """
    strategies = {
        "fixed_size": _fixed_size_chunk,
        "recursive": _recursive_chunk,
        "semantic": _semantic_chunk,
        "structure_aware": _structure_aware_chunk,
    }

    if strategy not in strategies:
        raise ValueError(f"Unknown chunking strategy: {strategy}")

    return strategies[strategy](text, **kwargs)


def _fixed_size_chunk(text: str, chunk_size: int = 512, overlap: int = 64) -> list[dict]:
    """Simple character-window split with fixed size and overlap."""
    if not text:
        return []

    chunks = []
    start = 0
    text_len = len(text)

    while start < text_len:
        end = min(start + chunk_size, text_len)
        chunk_text = text[start:end]
        if chunk_text.strip():
            chunks.append({
                "content": chunk_text,
                "start_index": start,
                "end_index": end,
                "metadata": {"strategy": "fixed_size", "chunk_size": chunk_size, "overlap": overlap},
            })
        next_start = start + chunk_size - overlap
        if next_start <= start:
            next_start = start + 1
        start = next_start

    return chunks


def _recursive_chunk(text: str, chunk_size: int = 512, overlap: int = 64) -> list[dict]:
    """LangChain RecursiveCharacterTextSplitter-based chunking."""
    if not text:
        return []

    from langchain.text_splitter import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    splits = splitter.split_text(text)

    chunks = []
    cursor = 0
    for split_text in splits:
        if not split_text.strip():
            continue
        start = text.find(split_text, cursor)
        if start == -1:
            start = cursor
        end = start + len(split_text)
        chunks.append({
            "content": split_text,
            "start_index": start,
            "end_index": end,
            "metadata": {"strategy": "recursive", "chunk_size": chunk_size, "overlap": overlap},
        })
        cursor = max(cursor, start)

    return chunks


def _semantic_chunk(text: str, threshold: float = 0.85) -> list[dict]:
    raise NotImplementedError("Semantic chunking deferred to Phase 2")


def _structure_aware_chunk(text: str) -> list[dict]:
    """
    Heading-aware chunking for Phase 1: split at major heading boundaries,
    then apply recursive chunking within each section.
    """
    if not text:
        return []

    import re

    # Match markdown headings (# H1, ## H2, etc.) or ALL-CAPS heading lines
    heading_pattern = re.compile(r"^(#{1,4}\s.+|[A-Z][A-Z\s]{3,})$", re.MULTILINE)
    matches = list(heading_pattern.finditer(text))

    if not matches:
        # Fall back to recursive chunking
        return _recursive_chunk(text)

    sections = []
    for i, match in enumerate(matches):
        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        section_title = match.group(0).strip()
        section_text = text[start:end].strip()
        sections.append((section_title, section_text, start))

    chunks = []
    for section_title, section_text, section_start in sections:
        sub_chunks = _recursive_chunk(section_text)
        for sc in sub_chunks:
            sc["metadata"]["section_title"] = section_title
            sc["metadata"]["strategy"] = "structure_aware"
            sc["start_index"] += section_start
            sc["end_index"] += section_start
            chunks.append(sc)

    return chunks
