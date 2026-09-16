from __future__ import annotations

from rag.retrieval import RetrievedDocument


def format_citations(docs: list[RetrievedDocument]) -> str:
    unique: list[RetrievedDocument] = []
    seen = set()
    for doc in docs:
        key = (doc.metadata["meeting_id"], doc.metadata["section"], doc.metadata["source_file"])
        if key not in seen:
            unique.append(doc)
            seen.add(key)
    lines = []
    for index, doc in enumerate(unique, start=1):
        meta = doc.metadata
        lines.append(
            f"[{index}] {meta['meeting_id']} - {meta['meeting_title']}\n"
            f"Date: {meta['date']}\n"
            f"Section: {meta['section']}\n"
            f"Source: {meta['source_file']}"
        )
    return "\n\n".join(lines)


def sources_payload(docs: list[RetrievedDocument]) -> list[dict[str, str]]:
    return [{**doc.metadata, "score": f"{doc.score:.3f}", "text": doc.text} for doc in docs]
