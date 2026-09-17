from __future__ import annotations

import logging
import re

import streamlit as st

from agent.graph import MeetingAgent

logging.basicConfig(level=logging.INFO, format="%(message)s")


def _extract_email(text: str) -> str | None:
    match = re.search(r"[\w.+-]+@[\w.-]+\.[a-zA-Z]{2,}", text)
    return match.group(0) if match else None


def _resolve_meeting_choice(text: str, matches: list[dict[str, str]]) -> str | None:
    lowered = text.strip().lower()
    id_match = re.search(r"\b(M\d{3})\b", text, flags=re.IGNORECASE)
    if id_match:
        meeting_id = id_match.group(1).upper()
        if any(match.get("meeting_id") == meeting_id for match in matches):
            return meeting_id

    for match in matches:
        title = match.get("meeting_title", "").lower()
        date = match.get("date", "").lower()
        meeting_id = match.get("meeting_id", "")
        if meeting_id and title and title in lowered:
            return meeting_id
        if meeting_id and title and date and title in lowered and date in lowered:
            return meeting_id
    return None


st.set_page_config(page_title="Meeting Intelligence Assistant", layout="wide")
st.title("Meeting Intelligence & Follow-up Assistant")

if "agent" not in st.session_state:
    with st.spinner("Loading transcripts and retrieval index..."):
        st.session_state.agent = MeetingAgent.create()
if "history" not in st.session_state:
    st.session_state.history = []
if "pending_email" not in st.session_state:
    st.session_state.pending_email = None
if "pending_email_request" not in st.session_state:
    st.session_state.pending_email_request = None
if "pending_email_context" not in st.session_state:
    st.session_state.pending_email_context = None

with st.sidebar:
    st.subheader("Email confirmation")
    recipient_email = st.text_input("Recipient email", value="")
    if st.session_state.pending_email and st.button("Send previewed email", type="primary"):
        result = st.session_state.agent.send_generated_email(st.session_state.pending_email["generated_email"])
        content = f"MCP result: {result}"
        st.session_state.history.append({"role": "assistant", "content": content, "sources": []})
        st.session_state.pending_email = None
        st.rerun()

for message in st.session_state.history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            with st.expander("Sources"):
                for source in message["sources"]:
                    st.write(source)

query = st.chat_input("Ask about a meeting or request a follow-up email")
if query:
    with st.chat_message("user"):
        st.markdown(query)

    email_from_reply = _extract_email(query)
    request = query
    effective_recipient_email = recipient_email or email_from_reply
    meeting_id = None
    pending_context = st.session_state.pending_email_context or {}
    if pending_context:
        meeting_choice = _resolve_meeting_choice(query, pending_context.get("matching_meetings", []))
        if meeting_choice:
            request = pending_context["request"]
            meeting_id = meeting_choice
            effective_recipient_email = pending_context.get("recipient_email") or effective_recipient_email
        elif email_from_reply:
            request = pending_context["request"]
            meeting_id = pending_context.get("meeting_id")
            effective_recipient_email = email_from_reply
    elif st.session_state.pending_email_request and email_from_reply:
        request = st.session_state.pending_email_request

    state = st.session_state.agent.run(request, recipient_email=effective_recipient_email or None, meeting_id=meeting_id)
    st.session_state.history.append({"role": "user", "content": query})
    st.session_state.history.append({"role": "assistant", "content": state["response"], "sources": state.get("sources", [])})
    with st.chat_message("assistant"):
        st.markdown(state["response"])
        if state.get("sources"):
            with st.expander("Sources"):
                for source in state["sources"]:
                    st.write(source)
        if state.get("generated_email"):
            st.session_state.pending_email = state
            st.session_state.pending_email_request = None
            st.session_state.pending_email_context = None
            st.info("Email preview generated. Use the sidebar button to send it through MCP.")
        elif state.get("intent") == "EMAIL" and state.get("clarification_needed"):
            st.session_state.pending_email_request = request
            st.session_state.pending_email_context = {
                "request": request,
                "recipient_email": effective_recipient_email,
                "meeting_id": meeting_id or state.get("meeting_reference"),
                "matching_meetings": state.get("matching_meetings", []),
            }
