from agent.graph import MeetingAgent


def test_ambiguous_request_asks_for_clarification():
    agent = MeetingAgent.create()
    state = agent.run("Send the follow-up email.")
    assert state["intent"] == "AMBIGUOUS"
    assert state["clarification_needed"]


def test_multiple_matching_meetings_trigger_clarification():
    agent = MeetingAgent.create()
    state = agent.run("Send a follow-up about the API.")
    assert state["clarification_needed"]
    assert "multiple relevant meetings" in state["response"].lower()


def test_missing_recipient_does_not_send():
    agent = MeetingAgent.create()
    state = agent.run("Send a follow-up about authentication.")
    assert state["clarification_needed"]
    assert "recipient" in state["response"].lower() or "multiple relevant meetings" in state["response"].lower()


def test_missing_email_address_does_not_send():
    agent = MeetingAgent.create()
    state = agent.run("Send a follow-up email to Sarah about authentication.")
    assert state["clarification_needed"]
    assert "email address" in state["response"].lower()
