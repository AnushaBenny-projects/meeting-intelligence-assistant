from __future__ import annotations

import logging

import streamlit as st

from agent.graph import MeetingAgent

logging.basicConfig(level=logging.INFO, format="%(message)s")

st.set_page_config(page_title="Meeting Intelligence Assistant", layout="wide")
st.title("Meeting Intelligence & Follow-up Assistant")

if "agent" not in st.session_state:
    with st.spinner("Loading transcripts and retrieval index..."):
        st.session_state.agent = MeetingAgent.create()
if "history" not in st.session_state:
    st.session_state.history = []
if "pending_email" not in st.session_state:
    st.session_state.pending_email = None

with st.sidebar:
    st.subheader("Email confirmation")
    recipient_email = st.text_input("Recipient email", value="")
    confirm_send = st.checkbox("Send generated email after preview")

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
    confirmation = "yes" if confirm_send else None
    state = st.session_state.agent.run(query, confirmation=confirmation, recipient_email=recipient_email or None)
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
            st.info("Email preview generated. Check the confirmation box and submit the same request again to send via MCP.")
