from agent.graph import MeetingAgent


def test_question_routes_to_rag():
    agent = MeetingAgent.create()
    state = agent.run("What authentication method was selected?")
    assert state["intent"] == "QUESTION"
    assert "JWT" in state["response"]
    assert "Sources:" in state["response"]


def test_email_workflow_preview_then_send():
    agent = MeetingAgent.create()
    preview = agent.run("Send a follow-up email to Sarah about authentication.", recipient_email="sarah@example.com")
    assert preview["intent"] == "EMAIL"
    assert preview["generated_email"]
    assert "Would you like me to send" in preview["response"]
    sent = agent.run("Send a follow-up email to Sarah about authentication.", recipient_email="sarah@example.com", confirmation="yes")
    assert sent["mcp_result"]["status"] == "sent"
