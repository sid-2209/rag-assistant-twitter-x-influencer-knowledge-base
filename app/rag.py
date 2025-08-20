from __future__ import annotations

import os
from typing import Any, Dict, List

try:
    from openai import OpenAI  # type: ignore
except Exception:  # pragma: no cover - optional
    OpenAI = None  # type: ignore


# 🔹 Centralized prompt template
SYSTEM_PROMPT = (
    "You are a helpful assistant for exploring Twitter/X influencers.\n"
    "Your job is to answer questions concisely using ONLY the provided influencer context.\n"
    "Always mention influencer names and handles when relevant.\n"
    "If the answer is not in the context, say you don’t know."
)


def _format_context(docs: List[Dict[str, Any]]) -> str:
    """Turn influencer documents into readable context for the LLM."""
    lines: List[str] = []
    for d in docs:
        name = d.get("name") or d.get("metadata", {}).get("name", "Unknown")
        handle = d.get("handle") or d.get("metadata", {}).get("handle", "")
        niche = d.get("niche") or d.get("metadata", {}).get("niche", "")
        post = d.get("sample_post") or d.get("metadata", {}).get("sample_post", "")
        lines.append(f"- {name} ({handle}) | Niche: {niche} | Post: {post}")
    return "\n".join(lines)


def _fallback_answer(query: str, docs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Deterministic offline response if API is unavailable or fails."""
    citations = []
    mentions: List[str] = []
    for d in docs:
        name = d.get("name") or d.get("metadata", {}).get("name", "Unknown")
        handle = d.get("handle") or d.get("metadata", {}).get("handle", "")
        citations.append({"name": name, "handle": handle})
        mentions.append(f"{name} ({handle})")

    answer = (
        f"Based on the provided context for your question '{query}', "
        f"relevant influencers include: {', '.join(mentions)}."
    )
    return {"answer": answer, "citations": citations}


def generate_answer(query: str, docs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Generate an answer using OpenAI GPT with provided docs as context.

    Returns:
        dict: { 'answer': str, 'citations': List[Dict] }
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or OpenAI is None:
        return _fallback_answer(query, docs)

    client = OpenAI(api_key=api_key)
    context = _format_context(docs)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"Question: {query}\n\n"
                f"Influencer Context:\n{context}\n\n"
                "Respond concisely and ground your answer in the above context."
            ),
        },
    ]

    try:
        chat = client.chat.completions.create(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),  # 🔹 override with env var
            messages=messages,
            temperature=0.2,
            max_tokens=300,
        )
        output = chat.choices[0].message.content or ""
    except Exception:
        return _fallback_answer(query, docs)

    citations = [
        {
            "name": d.get("name") or d.get("metadata", {}).get("name", "Unknown"),
            "handle": d.get("handle") or d.get("metadata", {}).get("handle", ""),
        }
        for d in docs
    ]

    return {"answer": output.strip(), "citations": citations}