from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass
from typing import Any

from agent.llm import generate_llm_email
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

    def send_generated_email(self, generated_email: dict[str, str]) -> dict:
        LOGGER.info("[MCP] Calling send_email")
        return self.email_client.send_email(
            generated_email.get("to", ""),
            generated_email.get("subject", ""),
            generated_email.get("body", ""),
        )

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
            user_request=state["user_query"],
        )
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
            if len(matches) == 1:
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


def generate_followup_email(
    recipient_name: str,
    recipient_email: str,
    docs: list[RetrievedDocument],
    sender_name: str,
    user_request: str = "",
) -> dict[str, str]:
    primary = docs[0].metadata
    topic = _clean_email_topic(user_request)
    llm_email = generate_llm_email(recipient_name, docs, sender_name, user_request)
    if llm_email:
        return {"to": recipient_email, "subject": llm_email["subject"], "body": llm_email["body"]}

    points = _select_email_points(recipient_name, topic, docs)
    subject = f"Follow-up: {topic.title() if topic else primary['meeting_title']}"
    lines = [
        f"Hi {recipient_name},",
        "",
        f"Following up on {primary['meeting_title']} from {primary['date']}"
        + (f" regarding {topic}." if topic else "."),
        "",
    ]
    if points["decision"]:
        lines.append(points["decision"][0])
        lines.append("")
    if points["recipient_action"]:
        lines.append("Your action item:")
        lines.extend(f"- {line}" for line in points["recipient_action"])
        lines.append("")
    elif points["topic_action"]:
        lines.append("Relevant action item:")
        lines.extend(f"- {line}" for line in points["topic_action"])
        lines.append("")
    if points["context"]:
        lines.append(points["context"][0])
        lines.append("")
    if points["follow_up"]:
        lines.append(points["follow_up"][0])
        lines.append("")
    lines.extend(["Please let me know if anything needs clarification or if there are any blockers.", "", "Best regards,", sender_name])
    return {"to": recipient_email, "subject": subject, "body": "\n".join(lines)}


def _select_email_points(recipient_name: str, topic: str, docs: list[RetrievedDocument]) -> dict[str, list[str]]:
    recipient = recipient_name.lower().split()[0]
    topic_terms = _email_terms(topic)
    buckets: dict[str, list[tuple[float, str]]] = {
        "decision": [],
        "recipient_action": [],
        "topic_action": [],
        "context": [],
        "follow_up": [],
    }
    for doc in docs:
        section = doc.metadata.get("section", "")
        for raw_line in doc.text.splitlines():
            line = raw_line.strip("- ").strip()
            if not line:
                continue
            score = _email_line_score(line, section, topic_terms, recipient)
            if score <= 0:
                continue
            if section == "Decisions":
                buckets["decision"].append((score, line))
            elif section == "Action Items" and _action_owned_by_recipient(line, recipient):
                buckets["recipient_action"].append((score, line))
            elif section == "Action Items":
                buckets["topic_action"].append((score, line))
            elif section == "Follow-up":
                buckets["follow_up"].append((score, line))
            else:
                buckets["context"].append((score, line))
    return {
        "decision": _top_unique(buckets["decision"], 1),
        "recipient_action": _top_unique(buckets["recipient_action"], 2),
        "topic_action": _top_unique(buckets["topic_action"], 1),
        "context": _top_unique(buckets["context"], 1),
        "follow_up": _top_unique(buckets["follow_up"], 1),
    }


def _email_line_score(line: str, section: str, topic_terms: set[str], recipient: str) -> float:
    lowered = line.lower()
    line_terms = _email_terms(line)
    overlap = len(topic_terms & line_terms)
    if topic_terms and overlap == 0 and recipient not in lowered:
        return 0.0
    score = float(overlap)
    if recipient and recipient in lowered:
        score += 4.0
    if section == "Action Items":
        score += 3.0
    elif section == "Decisions":
        score += 2.0
    elif section == "Follow-up":
        score += 1.0
    if any(signal in lowered for signal in ["owner:", "deadline:", "will ", "decided", "agreed", "confirmed", "recommended"]):
        score += 1.0
    return score


def _clean_email_topic(user_request: str) -> str:
    cleaned = user_request.strip().strip(". ")
    cleaned = re.sub(
        r"^\s*(?:please\s+)?(?:send|email|mail)\s+(?:a\s+)?follow[- ]?up\s+(?:mail|email)?\s+to\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?\s*",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    if " about " in cleaned.lower():
        cleaned = re.split(r"\s+about\s+", cleaned, maxsplit=1, flags=re.IGNORECASE)[1]
    cleaned = re.sub(r"^\s*for\s+", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"^\s*share\s+", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\bfor\s+M\d{3}\b", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\bM\d{3}\b", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\bher email is\b.*$", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\bhis email is\b.*$", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\bconducted\s+(?:at|on)\s+.+$", "", cleaned, flags=re.IGNORECASE)
    return cleaned.strip().strip(". ")


def _email_terms(text: str) -> set[str]:
    stop = {"send", "email", "follow", "up", "follow-up", "to", "about", "the", "a", "an", "and", "for", "meeting", "her", "his", "is"}
    return {term for term in re.findall(r"[a-z0-9]+", text.lower()) if term not in stop}


def _action_owned_by_recipient(line: str, recipient: str) -> bool:
    lowered = line.lower()
    owner_match = re.search(r"owner:\s*([A-Za-z]+)", line)
    if owner_match:
        return owner_match.group(1).lower() == recipient
    return lowered.startswith(recipient) and " will " in lowered


def _top_unique(scored_lines: list[tuple[float, str]], limit: int) -> list[str]:
    scored_lines.sort(key=lambda item: item[0], reverse=True)
    selected: list[str] = []
    seen: set[str] = set()
    for _, line in scored_lines:
        if line not in seen:
            selected.append(line)
            seen.add(line)
        if len(selected) >= limit:
            break
    return selected


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
