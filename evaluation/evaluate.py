from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent.graph import MeetingAgent
from rag.ingestion import ingest_transcripts


def main() -> None:
    questions = json.loads(Path("evaluation/questions.json").read_text(encoding="utf-8"))
    agent = MeetingAgent.create()
    retriever = ingest_transcripts()
    rows = []
    for item in questions:
        question = item["question"]
        if question.lower().startswith(("send", "email")):
            state = agent.run(question)
            ok = item["expected_answer"].lower() in state["response"].lower() or bool(state.get("clarification_needed"))
            rows.append((question, ok, "agent", state["response"][:100]))
        else:
            result = retriever.answer_question(question)
            source_ok = item["expected_source"] is None or any(doc.metadata["meeting_id"] == item["expected_source"] for doc in result["sources"])
            if item["expected_answer"] == "not found":
                answer_ok = not result["sources"] and "sufficient evidence" in result["answer"].lower()
            else:
                answer_ok = item["expected_answer"].lower() in result["answer"].lower()
            rows.append((question, source_ok and answer_ok, item["expected_source"], result["answer"][:100]))
    passed = sum(1 for _, ok, _, _ in rows if ok)
    print(f"Evaluation passed {passed}/{len(rows)} checks")
    for question, ok, expected, observed in rows:
        print(f"{'PASS' if ok else 'FAIL'} | expected={expected} | {question} | {observed}")


if __name__ == "__main__":
    main()
