from __future__ import annotations

from typing import Any, Dict, List


def generate_answer_langchain(query: str, docs: List[Dict[str, Any]], model: str, api_key: str | None) -> Dict[str, Any]:
    """Placeholder LangChain-style RAG pipeline.

    This function simulates a LangChain pipeline to keep tests green without
    adding a heavy dependency footprint. It mirrors the shape of the vanilla
    RAG output using provided docs.
    """
    # For now, behave like the vanilla fallback: cite provided docs
    citations = [
        {
            "name": d.get("name") or d.get("metadata", {}).get("name", "Unknown"),
            "handle": d.get("handle") or d.get("metadata", {}).get("handle", ""),
        }
        for d in docs
    ]
    answer = f"[LangChain:{model}] Based on your query '{query}', relevant influencers are: " + \
        ", ".join([f"{c['name']} ({c['handle']})" for c in citations])
    return {"answer": answer, "citations": citations}


