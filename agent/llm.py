from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass
from typing import Any

from dotenv import load_dotenv

from agent.prompts import EMAIL_SYSTEM_PROMPT
from rag.retrieval import RetrievedDocument

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class LLMSettings:
    provider: str
    model: str
    api_key: str
    base_url: str | None


def get_llm_settings() -> LLMSettings | None:
    load_dotenv()
    provider = os.getenv("LLM_PROVIDER", "").strip().lower()
    if provider not in {"groq", "openai"}:
        return None

    if provider == "groq":
        api_key = os.getenv("GROQ_API_KEY", "").strip()
        model = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b").strip()
        base_url = os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1").strip()
    else:
        api_key = os.getenv("OPENAI_API_KEY", "").strip()
        model = os.getenv("OPENAI_MODEL", "gpt-4o-mini").strip()
        base_url = os.getenv("OPENAI_BASE_URL", "").strip() or None

    if not api_key or not model:
        return None
    return LLMSettings(provider=provider, model=model, api_key=api_key, base_url=base_url)


def generate_llm_email(
    recipient_name: str,
    docs: list[RetrievedDocument],
    sender_name: str,
    user_request: str,
) -> dict[str, str] | None:
    settings = get_llm_settings()
    if not settings:
        return None

    prompt = EMAIL_SYSTEM_PROMPT.format(
        context=_format_evidence(docs),
        question=user_request,
        recipient_name=recipient_name,
        sender_name=sender_name,
    )
    try:
        raw = _chat_completion(settings, prompt)
        parsed = _parse_json_object(raw)
    except Exception as exc:
        LOGGER.warning("[LLM] Email generation failed; using deterministic fallback: %s", exc)
        return None

    if parsed.get("insufficient_evidence") is True:
        return None

    subject = str(parsed.get("subject", "")).strip()
    body = str(parsed.get("body", "")).strip()
    if not subject or not body:
        return None
    if _looks_unsupported(body, docs):
        LOGGER.warning("[LLM] Email draft contained unsupported-looking content; using deterministic fallback")
        return None
    return {"subject": subject, "body": body}


def _chat_completion(settings: LLMSettings, prompt: str) -> str:
    from openai import OpenAI

    client = OpenAI(api_key=settings.api_key, base_url=settings.base_url)
    response = client.chat.completions.create(
        model=settings.model,
        messages=[
            {
                "role": "system",
                "content": "Return only valid JSON for an evidence-grounded meeting follow-up email.",
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0,
        response_format={"type": "json_object"},
    )
    content = response.choices[0].message.content
    if not content:
        raise ValueError("empty LLM response")
    return content


def _parse_json_object(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    fence = re.match(r"^```(?:json)?\s*(.*?)\s*```$", cleaned, flags=re.DOTALL)
    if fence:
        cleaned = fence.group(1).strip()
    parsed = json.loads(cleaned)
    if not isinstance(parsed, dict):
        raise ValueError("LLM response was not a JSON object")
    return parsed


def _format_evidence(docs: list[RetrievedDocument]) -> str:
    blocks = []
    for index, doc in enumerate(docs, start=1):
        metadata = doc.metadata
        blocks.append(
            "\n".join(
                [
                    f"[{index}] meeting_id: {metadata.get('meeting_id', '')}",
                    f"title: {metadata.get('meeting_title', '')}",
                    f"date: {metadata.get('date', '')}",
                    f"participants: {metadata.get('participants', '')}",
                    f"section: {metadata.get('section', '')}",
                    f"source_file: {metadata.get('source_file', '')}",
                    "text:",
                    doc.text.strip(),
                ]
            )
        )
    return "\n\n".join(blocks)


def _looks_unsupported(body: str, docs: list[RetrievedDocument]) -> bool:
    evidence = "\n".join(doc.text for doc in docs).lower()
    risky_terms = re.findall(r"\b(?:january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{1,2},\s+\d{4}\b", body.lower())
    return any(term not in evidence for term in risky_terms)
