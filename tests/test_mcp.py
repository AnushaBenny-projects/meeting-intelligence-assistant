from mcp.client import EmailMCPClient
from mcp.server import send_email_tool


def test_send_email_tool_exists():
    assert callable(send_email_tool)


def test_valid_email_succeeds():
    result = EmailMCPClient().send_email("verified@example.com", "Subject", "Body")
    assert result["status"] == "sent"


def test_invalid_email_rejected():
    result = EmailMCPClient().send_email("not-an-email", "Subject", "Body")
    assert result["status"] == "error"


def test_mcp_errors_are_handled():
    result = EmailMCPClient().send_email("verified@example.com", "", "Body")
    assert result["status"] == "error"
