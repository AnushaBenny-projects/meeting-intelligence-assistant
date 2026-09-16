from __future__ import annotations

from mcp.server import send_email_tool


class EmailMCPClient:
    """Client facade that keeps the agent on the MCP boundary.

    In mock mode the same validated tool function is called locally for fast tests.
    The server module exposes this function through MCP FastMCP for real stdio use.
    """

    def send_email(self, to: str, subject: str, body: str) -> dict:
        return send_email_tool(to=to, subject=subject, body=body)
