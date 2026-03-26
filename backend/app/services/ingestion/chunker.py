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

    from langchain_text_splitters import RecursiveCharacterTextSplitter

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
    """
    Embedding-similarity boundary detection chunking.

    Splits text into sentences, embeds each one, then places chunk boundaries
    where cosine similarity between consecutive sentence groups drops below
    the threshold. Groups of semantically similar sentences stay together.
    """
    if not text:
        return []

    import re
    import numpy as np
    from langchain_openai import OpenAIEmbeddings
    from app.core.config import settings

    # Split text into sentences
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    sentences = [s.strip() for s in sentences if s.strip()]

    if len(sentences) <= 3:
        # Too few sentences — return as a single chunk
        return [{
            "content": text,
            "start_index": 0,
            "end_index": len(text),
            "metadata": {"strategy": "semantic", "threshold": threshold},
        }]

    # Embed all sentences
    embeddings_model = OpenAIEmbeddings(
        model=settings.EMBEDDING_MODEL,
        openai_api_key=settings.OPENAI_API_KEY,
    )
    vectors = embeddings_model.embed_documents(sentences)
    vectors_np = np.array(vectors)

    # Compute cosine similarity between consecutive sentences
    def _cosine_sim(a, b):
        dot = np.dot(a, b)
        norm = np.linalg.norm(a) * np.linalg.norm(b)
        return dot / norm if norm > 0 else 0.0

    similarities = [
        _cosine_sim(vectors_np[i], vectors_np[i + 1])
        for i in range(len(vectors_np) - 1)
    ]

    # Find chunk boundaries where similarity drops below threshold
    boundaries = [0]
    for i, sim in enumerate(similarities):
        if sim < threshold:
            boundaries.append(i + 1)
    boundaries.append(len(sentences))

    # Build chunks from sentence groups
    chunks = []
    cursor = 0
    for i in range(len(boundaries) - 1):
        start_sent = boundaries[i]
        end_sent = boundaries[i + 1]
        chunk_text = " ".join(sentences[start_sent:end_sent])
        if not chunk_text.strip():
            continue

        start_idx = text.find(sentences[start_sent], cursor)
        if start_idx == -1:
            start_idx = cursor
        last_sent = sentences[end_sent - 1]
        end_idx = text.find(last_sent, start_idx)
        if end_idx != -1:
            end_idx += len(last_sent)
        else:
            end_idx = start_idx + len(chunk_text)

        chunks.append({
            "content": chunk_text,
            "start_index": start_idx,
            "end_index": end_idx,
            "metadata": {
                "strategy": "semantic",
                "threshold": threshold,
                "sentence_count": end_sent - start_sent,
            },
        })
        cursor = max(cursor, start_idx)

    return chunks


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
