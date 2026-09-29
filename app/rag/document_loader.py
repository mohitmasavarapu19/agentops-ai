"""Document loading and text chunking utilities for AgentOps AI RAG."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Sequence


@dataclass
class Document:
    """Represents a text document or chunk with associated metadata.

    Attributes:
        page_content: The text content of the document or chunk.
        metadata: Associated metadata dictionary (e.g. source filename, chunk index).
    """

    page_content: str
    metadata: Dict[str, Any] = field(default_factory=dict)


def load_document(file_path: Path | str) -> Document:
    """Load a single text or Markdown file into a Document object.

    Args:
        file_path: Path to the target file.

    Returns:
        Document instance containing the file contents and metadata.

    Raises:
        FileNotFoundError: If the specified file does not exist.
    """
    path = Path(file_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Document file not found: {path}")

    content = path.read_text(encoding="utf-8")
    metadata: Dict[str, Any] = {
        "source": path.name,
        "file_path": str(path),
    }
    return Document(page_content=content, metadata=metadata)


def load_directory(
    dir_path: Path | str,
    extensions: Sequence[str] = (".md", ".txt"),
) -> List[Document]:
    """Discover and load all supported documents from a directory.

    Args:
        dir_path: Path to the directory containing knowledge files.
        extensions: File extensions to load (defaults to .md and .txt).

    Returns:
        List of loaded Document objects.
    """
    path = Path(dir_path).resolve()
    if not path.is_dir():
        return []

    documents: List[Document] = []
    for ext in extensions:
        for file in sorted(path.glob(f"*{ext}")):
            if file.is_file():
                documents.append(load_document(file))

    return documents


def split_text(
    text: str,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> List[str]:
    """Split text recursively into chunks by paragraph, line, and word boundaries.

    Args:
        text: Source text to split.
        chunk_size: Maximum character count per chunk.
        chunk_overlap: Character overlap between consecutive chunks.

    Returns:
        List of text chunks.
    """
    clean_text = text.strip()
    if not clean_text:
        return []

    separators = ["\n\n", "\n", ". ", " "]

    def _split(txt: str, sep_idx: int) -> List[str]:
        if len(txt) <= chunk_size or sep_idx >= len(separators):
            return [txt.strip()] if txt.strip() else []

        sep = separators[sep_idx]
        parts = txt.split(sep)
        result: List[str] = []
        current = ""

        for part in parts:
            candidate = f"{current}{sep}{part}" if current else part
            if len(candidate) <= chunk_size:
                current = candidate
            else:
                if current:
                    result.append(current.strip())
                    overlap_seed = current[-chunk_overlap:].strip() if chunk_overlap > 0 else ""
                    current = f"{overlap_seed}{sep}{part}" if overlap_seed else part
                else:
                    # Single section exceeds chunk_size, split with finer separator
                    result.extend(_split(part, sep_idx + 1))

        if current and current.strip():
            result.append(current.strip())

        return result

    return [c for c in _split(clean_text, 0) if c]


def chunk_document(
    doc: Document,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> List[Document]:
    """Split a Document into multiple chunk Documents, preserving and enriching metadata.

    Args:
        doc: The source Document to partition.
        chunk_size: Maximum character count per chunk.
        chunk_overlap: Character overlap between consecutive chunks.

    Returns:
        List of chunk Document instances.
    """
    raw_chunks = split_text(doc.page_content, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunked_docs: List[Document] = []
    total = len(raw_chunks)
    source_name = doc.metadata.get("source", "doc")

    for idx, chunk in enumerate(raw_chunks):
        chunk_metadata = dict(doc.metadata)
        chunk_metadata["chunk_index"] = idx
        chunk_metadata["total_chunks"] = total
        chunk_metadata["chunk_id"] = f"{source_name}_chunk_{idx}"
        chunked_docs.append(Document(page_content=chunk, metadata=chunk_metadata))

    return chunked_docs
