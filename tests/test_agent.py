from agent.graph import MeetingAgent


class RecordingEmailClient:
    def __init__(self):
        self.calls = []

    def send_email(self, to: str, subject: str, body: str) -> dict:
        self.calls.append((to, subject, body))
        return {"status": "sent", "recipient": to}


def test_question_routes_to_rag():
    agent = MeetingAgent.create()
    state = agent.run("What authentication method was selected?")
    assert state["intent"] == "QUESTION"
    assert "JWT" in state["response"]
    assert "Sources:" in state["response"]


def test_email_workflow_preview_then_send():
    email_client = RecordingEmailClient()
    agent = MeetingAgent(MeetingAgent.create().retriever, email_client)
    preview = agent.run("Send a follow-up email to Sarah about authentication for M001.", recipient_email="sarah@example.com")
    assert preview["intent"] == "EMAIL"
    assert preview["generated_email"]
    assert "Would you like me to send" in preview["response"]
    assert email_client.calls == []
    result = agent.send_generated_email(preview["generated_email"])
    assert result["status"] == "sent"
    assert email_client.calls == [(
        "sarah@example.com",
        preview["generated_email"]["subject"],
        preview["generated_email"]["body"],
    )]


def test_email_preview_is_topic_and_recipient_focused():
    agent = MeetingAgent.create()
    state = agent.run("Send a follow-up email to Sarah about authentication for M001.", recipient_email="sarah@example.com")
    body = state["generated_email"]["body"]
    assert "Hi Sarah" in body
    assert "Sarah Patel will implement the authentication API" in body
    assert "Deadline: September 15, 2026" in body
    assert "John Miller will update" not in body
    assert "Maya Chen will prepare" not in body


def test_email_preview_uses_llm_draft_when_available(monkeypatch):
    def fake_llm_email(recipient_name, docs, sender_name, user_request):
        return {
            "subject": "Follow-up: Authentication API",
            "body": f"Hi {recipient_name},\n\nLLM draft from meeting evidence.\n\nBest regards,\n{sender_name}",
        }

    monkeypatch.setattr("agent.graph.generate_llm_email", fake_llm_email)
    agent = MeetingAgent.create()
    state = agent.run("Send a follow-up email to Sarah about authentication for M001.", recipient_email="sarah@example.com")
    assert state["generated_email"]["subject"] == "Follow-up: Authentication API"
    assert "LLM draft from meeting evidence" in state["generated_email"]["body"]


def test_email_request_asks_when_topic_matches_multiple_meetings():
    agent = MeetingAgent.create()
    state = agent.run("Send a follow-up email to Sarah about authentication.", recipient_email="sarah@example.com")
    assert state["clarification_needed"]
    assert "multiple relevant meetings" in state["response"]
