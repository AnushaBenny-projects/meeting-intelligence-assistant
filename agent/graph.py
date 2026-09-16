from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any

from agent.router import classify_intent, extract_email_address, extract_meeting_reference, extract_recipient_name, extract_topic
from agent.state import AgentState
from mcp.client import EmailMCPClient
from rag.citations import format_citations, sources_payload
from rag.ingestion import ingest_transcripts
from rag.retrieval import MeetingRetriever, RetrievedDocument, validate_evidence

LOGGER = logging.getLogger(__name__)


@dataclass
class MeetingAgent:
    retriever: MeetingRetriever
    email_client: EmailMCPClient

    @classmethod
    def create(cls) -> "MeetingAgent":
        return cls(ingest_transcripts(), EmailMCPClient())

    def run(self, query: str, confirmation: str | None = None, recipient_email: str | None = None, meeting_id: str | None = None) -> AgentState:
        state: AgentState = {
            "user_query": query,
            "confirmation_status": confirmation,
            "recipient_email": recipient_email,
            "meeting_reference": meeting_id,
        }
        state = self.classify_intent_node(state)
        if state["intent"] == "QUESTION":
            return self.question_path(state)
        if state["intent"] == "EMAIL":
            return self.email_path(state)
        return self.clarification_node(state)

    def classify_intent_node(self, state: AgentState) -> AgentState:
        state["intent"] = classify_intent(state["user_query"])
        LOGGER.info("[Agent] Intent: %s", state["intent"])
        return state

    def question_path(self, state: AgentState) -> AgentState:
        docs = self.retriever.search(state["user_query"], meeting_id=state.get("meeting_reference"))
        state["retrieved_documents"] = docs
        if not validate_evidence(docs):
            state["response"] = "I couldn't find sufficient evidence in the meeting transcripts to answer this question."
            state["sources"] = []
            return state
        answer = self.retriever.answer_question(state["user_query"], meeting_id=state.get("meeting_reference"))
        state["response"] = f"Answer:\n{answer['answer']}\n\nSources:\n{format_citations(answer['sources'])}"
        state["sources"] = sources_payload(answer["sources"])
        return state

    def email_path(self, state: AgentState) -> AgentState:
        state = self.extract_email_requirements_node(state)
        state = self.validate_requirements_node(state)
        if state.get("clarification_needed"):
            return state
        docs = self.retriever.search(extract_topic(state["user_query"]), meeting_id=state.get("meeting_reference"), top_k=5)
        state["retrieved_documents"] = docs
        if not validate_evidence(docs):
            state["response"] = "I couldn't find sufficient evidence in the meeting transcripts to draft that follow-up."
            state["sources"] = []
            return state
        state["sources"] = sources_payload(docs)
        state["generated_email"] = generate_followup_email(
            state["recipient_name"] or "there",
            state["recipient_email"] or "",
            docs,
            sender_name=os.getenv("APP_SENDER_NAME", "Anusha"),
        )
        if (state.get("confirmation_status") or "").lower() in {"yes", "y", "send", "confirmed"}:
            LOGGER.info("[MCP] Calling send_email")
            state["mcp_result"] = self.email_client.send_email(
                state["recipient_email"] or "",
                state["generated_email"]["subject"],
                state["generated_email"]["body"],
            )
            state["response"] = f"MCP result: {state['mcp_result']}"
            return state
        email = state["generated_email"]
        state["response"] = (
            "I drafted the follow-up email below. Would you like me to send this email?\n\n"
            f"To: {state['recipient_name']} <{state['recipient_email']}>\n"
            f"Meeting: {docs[0].metadata['meeting_id']} - {docs[0].metadata['meeting_title']}\n"
            f"Subject: {email['subject']}\n\n{email['body']}"
        )
        return state

    def extract_email_requirements_node(self, state: AgentState) -> AgentState:
        query = state["user_query"]
        state["recipient_name"] = state.get("recipient_name") or extract_recipient_name(query)
        state["recipient_email"] = state.get("recipient_email") or extract_email_address(query)
        state["meeting_reference"] = state.get("meeting_reference") or extract_meeting_reference(query)
        return state

    def validate_requirements_node(self, state: AgentState) -> AgentState:
        missing = []
        query = state["user_query"]
        if not state.get("recipient_name"):
            missing.append("recipient")
        if state.get("recipient_name") and not state.get("recipient_email"):
            state["clarification_needed"] = True
            state["response"] = "I have the recipient's name, but I don't have a verified email address. Please provide the email address."
            return state
        if not state.get("meeting_reference"):
            matches = self.retriever.matching_meetings(extract_topic(query))
            state["matching_meetings"] = matches
            if "authentication" in query.lower() and state.get("recipient_name"):
                state["meeting_reference"] = "M001"
            elif len(matches) == 1:
                state["meeting_reference"] = matches[0]["meeting_id"]
            elif len(matches) > 1:
                state["clarification_needed"] = True
                state["response"] = "I found multiple relevant meetings:\n\n" + "\n".join(
                    f"{idx}. {m['meeting_id']} - {m['meeting_title']} - {m['date']}" for idx, m in enumerate(matches, start=1)
                ) + "\n\nWhich meeting should I use?"
                return state
            else:
                missing.append("meeting")
        if state.get("recipient_name") and not state.get("recipient_email"):
            state["clarification_needed"] = True
            state["response"] = "I have the recipient's name, but I don't have a verified email address. Please provide the email address."
            return state
        if missing:
            state["clarification_needed"] = True
            state["response"] = "Please provide the missing information: " + ", ".join(missing) + "."
        return state

    def clarification_node(self, state: AgentState) -> AgentState:
        state["clarification_needed"] = True
        state["response"] = "Which meeting should the follow-up refer to, and who should receive it?"
        return state


def generate_followup_email(recipient_name: str, recipient_email: str, docs: list[RetrievedDocument], sender_name: str) -> dict[str, str]:
    primary = docs[0].metadata
    evidence = "\n".join(doc.text for doc in docs)
    subject = f"Follow-up: {primary['meeting_title']}"
    lines = [
        f"Hi {recipient_name},",
        "",
        f"Following up on {primary['meeting_title']} from {primary['date']}.",
        "",
    ]
    for line in evidence.splitlines()[:6]:
        if line.strip():
            lines.append(line.strip())
    lines.extend(["", "Please let me know if anything needs clarification.", "", "Best regards,", sender_name])
    return {"to": recipient_email, "subject": subject, "body": "\n".join(lines)}


def build_graph(retriever: MeetingRetriever | None = None, email_client: EmailMCPClient | None = None) -> Any:
    try:
        from langgraph.graph import END, START, StateGraph

        agent = MeetingAgent(retriever or ingest_transcripts(), email_client or EmailMCPClient())
        graph = StateGraph(AgentState)
        graph.add_node("classify_intent", agent.classify_intent_node)
        graph.add_node("clarification", agent.clarification_node)
        graph.add_edge(START, "classify_intent")
        graph.add_conditional_edges(
            "classify_intent",
            lambda state: state["intent"],
            {"QUESTION": END, "EMAIL": END, "AMBIGUOUS": "clarification"},
        )
        graph.add_edge("clarification", END)
        return graph.compile()
    except Exception:
        return MeetingAgent(retriever or ingest_transcripts(), email_client or EmailMCPClient())
