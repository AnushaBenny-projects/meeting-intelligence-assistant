from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from rag.embeddings import EmbeddingModel, get_embedding_model
from rag.ingestion import DocumentChunk

LOGGER = logging.getLogger(__name__)


@dataclass
class RetrievedDocument:
    id: str
    text: str
    metadata: dict[str, str]
    score: float


class MeetingRetriever:
    def __init__(
        self,
        chunks: list[DocumentChunk],
        embedding_model: EmbeddingModel | None = None,
        persist_dir: str | Path = "chroma_db",
        collection_name: str = "meeting_transcripts",
    ) -> None:
        self.chunks = chunks
        self.embedding_model = embedding_model or get_embedding_model()
        self.persist_dir = Path(persist_dir)
        self.collection_name = collection_name
        self.collection = self._build_chroma_collection()

    @classmethod
    def from_chunks(
        cls,
        chunks: list[DocumentChunk],
        persist_dir: str | Path = "chroma_db",
        collection_name: str = "meeting_transcripts",
        embedding_model: EmbeddingModel | None = None,
    ) -> "MeetingRetriever":
        return cls(chunks, embedding_model or get_embedding_model(), persist_dir, collection_name)

    def search(
        self,
        query: str,
        top_k: int = 5,
        meeting_id: str | None = None,
        threshold: float = 0.16,
    ) -> list[RetrievedDocument]:
        threshold = float(os.getenv("RAG_RELEVANCE_THRESHOLD", str(threshold)))
        LOGGER.info("[RAG] Query: %s", query)
        query_vector = self.embedding_model.embed([query])[0]
        query_terms = set(_terms(query))
        where = {"meeting_id": meeting_id} if meeting_id else None
        raw = self.collection.query(
            query_embeddings=[query_vector],
            n_results=max(top_k * 4, top_k),
            where=where,
            include=["documents", "metadatas", "distances"],
        )
        results: list[RetrievedDocument] = []
        for chunk_id, text, metadata, distance in _iter_chroma_results(raw):
            semantic = max(0.0, 1.0 - float(distance))
            lexical = _lexical_score(query_terms, text)
            score = (0.78 * semantic) + (0.17 * lexical) + _domain_bonus(query, _metadata_chunk(chunk_id, text, metadata), lexical)
            if query_terms and lexical == 0 and score < max(threshold + 0.25, 0.62):
                continue
            if score >= threshold:
                results.append(RetrievedDocument(chunk_id, text, dict(metadata), score))
        deduped = _dedupe(results)
        deduped.sort(key=lambda item: item.score, reverse=True)
        selected = deduped[:top_k]
        _log_retrieval_debug(query, selected)
        return selected

    def matching_meetings(self, query: str, top_k: int = 4, threshold: float = 0.14) -> list[dict[str, str]]:
        docs = self.search(query, top_k=12, threshold=threshold)
        scores: dict[str, float] = {}
        metadata_by_id: dict[str, dict[str, str]] = {}
        for doc in docs:
            meeting_id = doc.metadata["meeting_id"]
            scores[meeting_id] = max(scores.get(meeting_id, 0.0), doc.score)
            metadata_by_id[meeting_id] = doc.metadata
        ordered = sorted(scores, key=scores.get, reverse=True)[:top_k]
        return [metadata_by_id[meeting_id] for meeting_id in ordered]

    def get_meeting_documents(self, meeting_id: str) -> list[RetrievedDocument]:
        raw = self.collection.get(where={"meeting_id": meeting_id}, include=["documents", "metadatas"])
        ids = raw.get("ids", [])
        docs = []
        for chunk_id, text, metadata in zip(ids, raw.get("documents", []), raw.get("metadatas", [])):
            docs.append(RetrievedDocument(chunk_id, text, dict(metadata), 1.0))
        docs.sort(key=lambda doc: doc.metadata.get("chunk_id", ""))
        return docs

    def answer_question(self, question: str, meeting_id: str | None = None) -> dict:
        docs = self.search(question, meeting_id=meeting_id)
        if not validate_evidence(docs):
            return {
                "answer": "I couldn't find sufficient evidence in the meeting transcripts to answer this question.",
                "sources": [],
                "retrieved_documents": docs,
            }
        answer = _extract_answer(question, docs)
        support = _supporting_docs(question, answer, docs)
        return {"answer": answer, "sources": support, "retrieved_documents": docs}

    def _build_chroma_collection(self) -> Any:
        import chromadb

        self.persist_dir.mkdir(parents=True, exist_ok=True)
        client = chromadb.PersistentClient(path=str(self.persist_dir))
        embedding_backend = self.embedding_model.__class__.__name__
        collection = client.get_or_create_collection(
            self.collection_name,
            metadata={"hnsw:space": "cosine", "embedding_backend": embedding_backend},
        )
        stored = collection.get(include=["metadatas"])
        stored_ids = set(stored.get("ids", []))
        expected_ids = {chunk.id for chunk in self.chunks}
        stored_backend = (collection.metadata or {}).get("embedding_backend")
        if stored_ids != expected_ids or stored_backend != embedding_backend:
            LOGGER.info("[RAG] Rebuilding Chroma collection %s", self.collection_name)
            client.delete_collection(self.collection_name)
            collection = client.get_or_create_collection(
                self.collection_name,
                metadata={"hnsw:space": "cosine", "embedding_backend": embedding_backend},
            )
            embeddings = self.embedding_model.embed(chunk.text for chunk in self.chunks)
            collection.add(
                ids=[chunk.id for chunk in self.chunks],
                documents=[chunk.text for chunk in self.chunks],
                metadatas=[chunk.metadata for chunk in self.chunks],
                embeddings=embeddings,
            )
        return collection


def _iter_chroma_results(raw: dict[str, Any]) -> Iterable[tuple[str, str, dict[str, Any], float]]:
    ids = raw.get("ids", [[]])[0]
    documents = raw.get("documents", [[]])[0]
    metadatas = raw.get("metadatas", [[]])[0]
    distances = raw.get("distances", [[]])[0]
    for chunk_id, text, metadata, distance in zip(ids, documents, metadatas, distances):
        yield chunk_id, text, metadata or {}, float(distance)


def _metadata_chunk(chunk_id: str, text: str, metadata: dict[str, Any]) -> DocumentChunk:
    return DocumentChunk(id=chunk_id, text=text, metadata={key: str(value) for key, value in metadata.items()})


def _log_retrieval_debug(query: str, docs: list[RetrievedDocument]) -> None:
    LOGGER.info("[RAG] Retrieved document IDs: %s", [doc.id for doc in docs])
    for doc in docs:
        LOGGER.info(
            "[RAG] id=%s score=%.3f meeting=%s section=%s source=%s text=%s",
            doc.id,
            doc.score,
            doc.metadata.get("meeting_id"),
            doc.metadata.get("section"),
            doc.metadata.get("source_file"),
            doc.text,
        )


def validate_evidence(docs: list[RetrievedDocument], min_score: float = 0.18) -> bool:
    return bool(docs and docs[0].score >= min_score)


def _extract_answer(question: str, docs: list[RetrievedDocument]) -> str:
    candidates = _rank_evidence_sentences(question, docs)
    if not candidates:
        return "The retrieved meeting evidence contains relevant information, but no concise answer could be extracted."

    q_type = _question_type(question)
    selected = [candidates[0]]
    if q_type in {"list", "summary"}:
        selected.extend(candidate for candidate in candidates[1:3] if candidate[2] >= candidates[0][2] * 0.72)

    answer = " ".join(sentence for sentence, _, _ in selected)
    if q_type == "yes_no":
        answer = _format_yes_no_answer(question, selected[0][0])
    return answer.strip()


def _supporting_docs(question: str, answer: str, docs: list[RetrievedDocument]) -> list[RetrievedDocument]:
    exact = [doc for doc in docs if answer.lower() in doc.text.lower()]
    if exact:
        return exact
    answer_terms = set(_terms(answer))
    support: list[RetrievedDocument] = []
    for doc in docs:
        text_terms = set(_terms(doc.text))
        overlap = len(answer_terms & text_terms)
        if overlap >= max(2, min(6, len(answer_terms))):
            support.append(doc)
    if support:
        primary_meeting = support[0].metadata["meeting_id"]
        return [doc for doc in support if doc.metadata["meeting_id"] == primary_meeting]
    return docs[:1]


def _terms(text: str) -> list[str]:
    stop = {
        "the",
        "a",
        "an",
        "of",
        "in",
        "on",
        "at",
        "by",
        "from",
        "with",
        "as",
        "to",
        "for",
        "about",
        "did",
        "what",
        "who",
        "is",
        "are",
        "was",
        "were",
        "send",
        "email",
        "follow",
        "up",
        "meeting",
        "first",
        "release",
        "tell",
        "me",
    }
    return [term for term in re.findall(r"[a-z0-9]+", text.lower()) if term not in stop]


def _lexical_score(query_terms: set[str], text: str) -> float:
    if not query_terms:
        return 0.0
    doc_terms = set(_terms(text))
    return len(query_terms & doc_terms) / len(query_terms)


def _domain_bonus(query: str, chunk: DocumentChunk, lexical: float) -> float:
    q = query.lower()
    text = chunk.text.lower()
    bonus = 0.0
    q_type = _question_type(query)
    section = chunk.metadata.get("section", "").lower()
    if q_type == "decision" and section == "decisions":
        bonus += 0.14
        if any(term in q for term in ["selected", "choose", "chosen", "method"]) and any(signal in text for signal in ["decided", "agreed", "recommended", " use ", " uses "]):
            bonus += 0.18
        if any(term in q for term in ["selected", "choose", "chosen", "method"]) and any(signal in text for signal in ["decided to use", "selected", "chose", "chosen"]):
            bonus += 0.16
    if q_type in {"owner", "deadline"} and section == "action items":
        bonus += 0.14
    if q_type == "yes_no" and any(term in text for term in ["not", "out of scope", "cannot", "will not", "no "]):
        bonus += 0.08
    if q_type == "deadline" and _DATE_PATTERN.search(text):
        bonus += 0.08
    if q_type == "owner" and any(signal in text for signal in ["owner:", " will ", "responsible", "lead"]):
        bonus += 0.08
    if lexical > 0.35:
        bonus += 0.06
    return bonus


_DATE_PATTERN = re.compile(r"\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2}(?:\s+to\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2})?,\s+\d{4}\b|\b\d{4}-\d{2}-\d{2}\b")


def _question_type(question: str) -> str:
    q = question.lower()
    if any(term in q for term in ["who", "owner", "owns", "responsible"]):
        return "owner"
    if any(term in q for term in ["when", "deadline", "date", "window"]):
        return "deadline"
    if any(term in q for term in ["decide", "decided", "decision", "selected", "choose", "chosen", "method"]):
        return "decision"
    if any(term in q for term in ["list", "summarize", "summary", "action items", "tests", "risks"]):
        return "list"
    if any(term in q for term in ["what", "why", "how"]):
        return "summary"
    if re.search(r"\b(is|are|was|were|do|does|did|can|could|should|will)\b", q):
        return "yes_no"
    return "summary"


def _rank_evidence_sentences(question: str, docs: list[RetrievedDocument]) -> list[tuple[str, RetrievedDocument, float]]:
    query_terms = set(_terms(question))
    q_type = _question_type(question)
    candidates: list[tuple[str, RetrievedDocument, float]] = []
    for doc_index, doc in enumerate(docs):
        for sentence in _split_evidence_sentences(doc.text):
            score = _sentence_score(sentence, query_terms, q_type, doc, doc_index)
            if score > 0:
                candidates.append((sentence, doc, score))
    candidates.sort(key=lambda item: item[2], reverse=True)
    return candidates


def _split_evidence_sentences(text: str) -> list[str]:
    cleaned_lines = [line.strip("- ").strip() for line in text.splitlines() if line.strip()]
    sentences: list[str] = []
    for line in cleaned_lines:
        if any(marker in line.lower() for marker in ["owner:", "deadline:"]):
            sentences.append(line)
            continue
        parts = re.split(r"(?<=[.!?])\s+", line)
        sentences.extend(part.strip() for part in parts if part.strip())
    return sentences


def _sentence_score(sentence: str, query_terms: set[str], q_type: str, doc: RetrievedDocument, doc_index: int) -> float:
    sentence_terms = set(_terms(sentence))
    overlap = len(query_terms & sentence_terms)
    causal_question = any(term in _terms(" ".join(query_terms)) for term in ["slowed", "blocked", "risk", "concern"])
    causal_sentence = any(term in sentence.lower() for term in ["because", "due to", "blocked", "slowed", "later than expected", "risk", "concern"])
    if query_terms and overlap == 0 and not (causal_question and causal_sentence):
        return 0.0

    score = overlap / max(len(query_terms), 1)
    lowered = sentence.lower()
    section = doc.metadata.get("section", "").lower()
    if q_type == "decision" and section == "decisions":
        score += 0.35
    if q_type in {"owner", "deadline"} and section == "action items":
        score += 0.35
    if q_type == "deadline" and _DATE_PATTERN.search(sentence):
        score += 0.3
    if q_type == "owner" and any(signal in lowered for signal in ["owner:", " will ", "responsible", "lead"]):
        score += 0.3
    if q_type == "yes_no" and any(term in lowered for term in ["not", "out of scope", "cannot", "will not", "no "]):
        score += 0.24
    if any(term in lowered for term in ["deadline:", "owner:", "decided", "agreed", "confirmed"]):
        score += 0.12
    if q_type == "decision" and any(signal in lowered for signal in ["decided to use", "selected", "chose", "chosen"]):
        score += 0.22
    if "tests" in query_terms and any(signal in lowered for signal in ["test", "tests", "regression"]):
        score += 0.35
    if "slowed" in query_terms and any(signal in lowered for signal in ["later than expected", "because", "due to", "blocked"]):
        score += 0.65
    if "slowed" in query_terms and lowered.startswith("the team discussed"):
        score -= 0.7
    if any(term in lowered for term in ["because", "due to", "blocked", "slowed", "later than expected", "risk", "concern"]):
        score += 0.08
    score += max(0.0, doc.score - 0.16) * 0.2
    score -= doc_index * 0.015
    return score


def _format_yes_no_answer(question: str, evidence_sentence: str) -> str:
    lowered = evidence_sentence.lower()
    if any(term in lowered for term in ["not", "out of scope", "cannot", "will not", "no "]):
        return f"No. {evidence_sentence}"
    if any(term in lowered for term in ["yes", "will", "decided", "agreed", "confirmed", "included"]):
        return f"Yes. {evidence_sentence}"
    return evidence_sentence


def _dedupe(results: Iterable[RetrievedDocument]) -> list[RetrievedDocument]:
    by_id: dict[str, RetrievedDocument] = {}
    for item in results:
        if item.id not in by_id or item.score > by_id[item.id].score:
            by_id[item.id] = item
    return list(by_id.values())
