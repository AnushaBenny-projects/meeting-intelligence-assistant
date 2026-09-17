from __future__ import annotations

from typing import Any, Literal, TypedDict


Intent = Literal["QUESTION", "EMAIL", "MEETING_LOOKUP", "AMBIGUOUS"]


class AgentState(TypedDict, total=False):
    user_query: str
    intent: Intent
    meeting_reference: str | None
    recipient_name: str | None
    recipient_email: str | None
    retrieved_documents: list[Any]
    sources: list[dict[str, str]]
    generated_email: dict[str, str] | None
    confirmation_status: str | None
    response: str
    mcp_result: dict[str, Any] | None
    clarification_needed: bool
    matching_meetings: list[dict[str, str]]
