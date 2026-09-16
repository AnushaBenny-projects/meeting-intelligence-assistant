from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

from rag.embeddings import EmbeddingModel, get_embedding_model

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class DocumentChunk:
    id: str
    text: str
    metadata: dict[str, str]


SECTION_HEADERS = {
    "Discussion",
    "Decisions",
    "Action Items",
    "Follow-up",
    "Risks",
}


def parse_transcript(path: Path) -> list[DocumentChunk]:
    text = path.read_text(encoding="utf-8")
    metadata = _parse_header(text, path.name)
    chunks: list[DocumentChunk] = []
    section_blocks = _split_sections(text)
    counter = 1
    for section, body in section_blocks:
        for paragraph in _paragraph_chunks(body):
            chunk_id = f"{metadata['meeting_id']}-C{counter:02d}"
            chunk_metadata = {**metadata, "section": section, "chunk_id": chunk_id}
            chunks.append(DocumentChunk(chunk_id, paragraph.strip(), chunk_metadata))
            counter += 1
    return chunks


def load_transcripts(data_dir: str | Path = "data") -> list[DocumentChunk]:
    root = Path(data_dir)
    chunks: list[DocumentChunk] = []
    for path in sorted(root.glob("M*.txt")):
        chunks.extend(parse_transcript(path))
    return chunks


def ingest_transcripts(
    data_dir: str | Path = "data",
    persist_dir: str | Path = ".chroma",
    collection_name: str = "meeting_transcripts",
    embedding_model: EmbeddingModel | None = None,
):
    from rag.retrieval import MeetingRetriever

    LOGGER.info("[RAG] Ingesting transcripts from %s", data_dir)
    return MeetingRetriever.from_chunks(
        load_transcripts(data_dir),
        persist_dir=persist_dir,
        collection_name=collection_name,
        embedding_model=embedding_model or get_embedding_model(),
    )


def _parse_header(text: str, source_file: str) -> dict[str, str]:
    fields = {}
    for key in ["Meeting ID", "Meeting Title", "Date", "Participants"]:
        match = re.search(rf"^{re.escape(key)}:\s*(.+)$", text, flags=re.MULTILINE)
        if not match:
            raise ValueError(f"Missing {key} in {source_file}")
        fields[key] = match.group(1).strip()
    return {
        "meeting_id": fields["Meeting ID"],
        "meeting_title": fields["Meeting Title"],
        "date": fields["Date"],
        "participants": fields["Participants"],
        "source_file": source_file,
    }


def _split_sections(text: str) -> list[tuple[str, str]]:
    matches = list(re.finditer(r"^(Discussion|Decisions|Action Items|Follow-up|Risks):\s*$", text, flags=re.MULTILINE))
    blocks: list[tuple[str, str]] = []
    for index, match in enumerate(matches):
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        blocks.append((match.group(1), text[start:end].strip()))
    return blocks


def _paragraph_chunks(body: str, max_chars: int = 900) -> Iterable[str]:
    paragraphs = [line.strip("- ").strip() for line in body.splitlines() if line.strip()]
    current: list[str] = []
    size = 0
    for paragraph in paragraphs:
        if current and size + len(paragraph) > max_chars:
            yield "\n".join(current)
            current = []
            size = 0
        current.append(paragraph)
        size += len(paragraph)
    if current:
        yield "\n".join(current)
