from rag.citations import format_citations
from rag.ingestion import ingest_transcripts, load_transcripts


def test_correct_document_retrieval():
    retriever = ingest_transcripts()
    docs = retriever.search("What authentication method was selected?")
    assert docs
    assert docs[0].metadata["meeting_id"] == "M001"


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


def test_unknown_information_handling():
    retriever = ingest_transcripts()
    answer = retriever.answer_question("What database vendor was selected?")
    assert "couldn't find sufficient evidence" in answer["answer"].lower()
    assert answer["sources"] == []
