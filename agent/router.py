from __future__ import annotations

import re

from agent.state import Intent


AMBIGUOUS_EMAIL_PATTERNS = [
    r"^\s*send\s+it\s*$",
    r"^\s*send\s+the\s+follow[- ]?up\s*(email)?\s*$",
    r"^\s*email\s+them\s*$",
]


def classify_intent(query: str) -> Intent:
    raw_text = query.strip().lower()
    text = raw_text.strip(".!? ")
    if not text:
        return "AMBIGUOUS"
    if any(re.search(pattern, text) for pattern in AMBIGUOUS_EMAIL_PATTERNS):
        return "AMBIGUOUS"
    if re.search(r"\b(send|email|mail)\b", text) and re.search(r"\b(follow|summary|action|items|decision|about|to)\b", text):
        return "EMAIL"
    if extract_meeting_reference(query) and re.search(r"\b(show|open|display|summarize|summary|details|meeting)\b", text):
        return "MEETING_LOOKUP"
    if "?" in raw_text or re.search(r"\b(what|who|when|where|which|how|is|are|do|does|did|was|were|decided|owns|responsible)\b", text):
        return "QUESTION"
    return "AMBIGUOUS"


def extract_recipient_name(query: str) -> str | None:
    match = re.search(r"\b(?:to|email|send)\s+([A-Z][a-z]+)(?:\s+[A-Z][a-z]+)?", query)
    if match:
        name = match.group(1)
        if name.lower() not in {"a", "the", "follow"}:
            return name
    return None


def extract_email_address(query: str) -> str | None:
    match = re.search(r"[\w.+-]+@[\w.-]+\.[a-zA-Z]{2,}", query)
    return match.group(0) if match else None


def extract_meeting_reference(query: str) -> str | None:
    match = re.search(r"\b(M\d{3})\b", query, flags=re.IGNORECASE)
    return match.group(1).upper() if match else None


def extract_topic(query: str) -> str:
    lowered = query.lower()
    if "about " in lowered:
        return lowered.split("about ", 1)[1].strip(". ")
    return query
