from rag.citations import format_citations
from rag.ingestion import ingest_transcripts, load_transcripts


def test_correct_document_retrieval():
    retriever = ingest_transcripts()
    docs = retriever.search("What authentication method was selected?")
    assert docs
    assert docs[0].metadata["meeting_id"] == "M001"


def test_chromadb_query_is_used(monkeypatch):
    retriever = ingest_transcripts()
    original_query = retriever.collection.query
    calls = {"count": 0}

    def spy_query(*args, **kwargs):
        calls["count"] += 1
        return original_query(*args, **kwargs)

    monkeypatch.setattr(retriever.collection, "query", spy_query)
    docs = retriever.search("Who will update the API documentation with token lifetime and refresh behavior?")
    assert calls["count"] == 1
    assert docs


def test_correct_meeting_retrieval():
    chunks = load_transcripts()
    assert len({chunk.metadata["meeting_id"] for chunk in chunks}) == 6
    assert all(chunk.metadata["chunk_id"] for chunk in chunks)


def test_citation_generation():
    retriever = ingest_transcripts()
    docs = retriever.search("JWT authentication")
    citations = format_citations(docs)
    assert "Source:" in citations
    assert "M001" in citations


def test_metadata_preserved_from_chroma():
    retriever = ingest_transcripts()
    docs = retriever.search("Who will update the API documentation with token lifetime and refresh behavior?")
    assert docs
    meta = docs[0].metadata
    for key in ["meeting_id", "meeting_title", "date", "participants", "section", "source_file", "chunk_id"]:
        assert meta[key]


def test_task_owner_deadline_relationship_preserved():
    retriever = ingest_transcripts()
    result = retriever.answer_question("Who will update the API documentation with token lifetime and refresh behavior?")
    assert "John Miller" in result["answer"]
    assert "September 16, 2026" in result["answer"]
    assert "Sarah Patel" not in result["answer"]
    assert any(doc.metadata["meeting_id"] == "M001" and doc.metadata["section"] == "Action Items" for doc in result["sources"])


def test_unknown_information_handling():
    retriever = ingest_transcripts()
    answer = retriever.answer_question("What database vendor was selected?")
    assert "couldn't find sufficient evidence" in answer["answer"].lower()
    assert answer["sources"] == []
