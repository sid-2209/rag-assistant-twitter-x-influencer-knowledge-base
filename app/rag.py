from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

try:
    from openai import OpenAI  # type: ignore
except Exception:  # pragma: no cover - optional
    OpenAI = None  # type: ignore

from .embeddings import get_embedding
from .config import DEFAULT_GENERATION_MODEL_NAME
import openai
from .hallucination_detector import hallucination_detector


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


def generate_answer(
    query: str,
    citations: List[Dict[str, Any]],
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Generate an answer using RAG with hallucination detection.
    
    Args:
        query: User query
        citations: Retrieved citations
        model: Model name (optional)
        api_key: API key (optional)
        base_url: Base URL (optional)
    
    Returns:
        Dictionary containing answer, citations, and hallucination analysis
    """
    # Use provided parameters or fall back to config
    model = model or DEFAULT_GENERATION_MODEL_NAME
    api_key = api_key or os.getenv("OPENAI_API_KEY")
    base_url = base_url or os.getenv("OPENAI_BASE_URL")
    
    # Generate the answer using existing logic
    answer = _generate_answer_internal(query, citations, model, api_key, base_url)
    
    # Ensure answer is not None
    if answer is None:
        answer = "I apologize, but I couldn't generate a response. Please try again."
    
    # Perform hallucination detection
    hallucination_analysis = hallucination_detector.detect_hallucination(
        query=query,
        answer=answer,
        citations=citations
    )
    
    return {
        "answer": answer,
        "citations": citations,
        "hallucination_analysis": hallucination_analysis
    }


def _generate_answer_internal(
    query: str,
    citations: List[Dict[str, Any]],
    model: str,
    api_key: str,
    base_url: str,
) -> str:
    """Internal function to generate answer (existing logic)."""
    if not citations:
        return "I don't have enough information to answer this question. Please upload relevant data first."
    
    # Debug: print citation structure
    print(f"Citation structure: {citations[0].keys() if citations else 'No citations'}")
    
    # Try to use OpenAI API first
    if api_key and OpenAI:
        try:
            print(f"Attempting API call with model: {model}")
            
            # Format citations for the prompt
            citation_texts = []
            for i, citation in enumerate(citations, 1):
                # Handle different possible citation structures
                text = citation.get('text') or citation.get('content') or citation.get('sample_post', '')
                metadata = citation.get("metadata", {})
                name = metadata.get("name", citation.get("name", "Unknown"))
                handle = metadata.get("handle", citation.get("handle", ""))
                niche = metadata.get("niche", citation.get("niche", ""))
                
                citation_texts.append(f"{i}. {text} (Source: {name} @{handle}, {niche})")
            
            citations_str = "\n".join(citation_texts)
            
            # Create the prompt
            prompt = f"""You are a helpful AI assistant analyzing Twitter/X influencer data. Answer the user's question based ONLY on the provided citations.

IMPORTANT GUIDELINES:
- When asked for posts about "X and Y", show posts that mention either X OR Y (not necessarily both)
- When asked for posts about "X", show all posts that mention X
- Always be consistent in your interpretation
- Cite specific examples from the provided citations
- If no relevant information is found, clearly state what was not found

Citations:
{citations_str}

Question: {query}

Answer:"""
            
            client = OpenAI(api_key=api_key, base_url=base_url)
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=500,
                temperature=0.3,
            )
            result = response.choices[0].message.content
            if result and result.strip():
                print("API call successful")
                return result.strip()
            else:
                print("API returned empty response, falling back to deterministic answer")
                
        except Exception as e:
            print(f"OpenAI API error: {e}")
            print("Falling back to deterministic answer generation")
    
    # Fallback: deterministic answer based on citations
    print("Using fallback answer generation")
    return _generate_fallback_answer(query, citations)


def _generate_fallback_answer(query: str, citations: List[Dict[str, Any]]) -> str:
    """Generate a fallback answer when LLM is not available."""
    if not citations:
        return "I don't have enough information to answer this question. Please upload relevant data first."
    
    # Extract key information from citations
    influencers = []
    niches = set()
    posts = []
    
    for citation in citations:
        # Handle different citation structures
        metadata = citation.get("metadata", {})
        name = metadata.get("name", citation.get("name", ""))
        handle = metadata.get("handle", citation.get("handle", ""))
        niche = metadata.get("niche", citation.get("niche", ""))
        post = metadata.get("sample_post", citation.get("sample_post", ""))
        
        if name and handle:
            influencers.append(f"{name} (@{handle})")
        if niche:
            niches.add(niche)
        if post:
            posts.append(post[:100] + "..." if len(post) > 100 else post)
    
    # Analyze the query to provide targeted answers
    query_lower = query.lower()
    
    # Handle "who is" queries specifically
    if "who" in query_lower and "@" in query:
        import re
        handle_match = re.search(r'@(\w+)', query)
        if handle_match:
            target_handle = handle_match.group(1)
            found_match = False
            for citation in citations:
                metadata = citation.get("metadata", {})
                handle = metadata.get("handle", citation.get("handle", ""))
                if handle and target_handle.lower() in handle.lower():
                    name = metadata.get("name", citation.get("name", "Unknown"))
                    niche = metadata.get("niche", citation.get("niche", ""))
                    post = metadata.get("sample_post", citation.get("sample_post", ""))
                    followers = metadata.get("followers", citation.get("followers", 0))
                    found_match = True
                    
                    # Build a comprehensive answer for "who is" queries
                    if name and name != "Unknown":
                        answer_parts = [f"@{handle} is {name}"]
                    else:
                        answer_parts = [f"@{handle} is an influencer in the dataset"]
                    
                    additional_info = []
                    if niche and niche.strip():
                        additional_info.append(f"in the {niche} niche")
                    if followers:
                        additional_info.append(f"with {followers:,} followers")
                    
                    if additional_info:
                        answer_parts.append(", ".join(additional_info))
                    
                    if post and post.strip():
                        answer_parts.append(f"Sample post: \"{post[:150]}{'...' if len(post) > 150 else ''}\"")
                    else:
                        answer_parts.append("No sample posts are available in the dataset")
                    
                    return ". ".join(answer_parts) + "."
            
            # If handle not found, provide helpful information
            if not found_match:
                available_handles = []
                for citation in citations:
                    metadata = citation.get("metadata", {})
                    handle = metadata.get("handle", citation.get("handle", ""))
                    if handle:
                        available_handles.append(handle)
                
                if available_handles:
                    sample_handles = available_handles[:5]
                    return f"@{target_handle} was not found in the dataset. Available handles include: {', '.join(sample_handles)}. Try asking about one of these handles instead."
                else:
                    return f"@{target_handle} was not found in the dataset and no other handles are available."
    
    # Handle other handle-specific queries
    elif "handle" in query_lower and "@" in query:
        # Extract handle from query
        import re
        handle_match = re.search(r'@(\w+)', query)
        if handle_match:
            target_handle = handle_match.group(1)
            found_match = False
            for citation in citations:
                metadata = citation.get("metadata", {})
                handle = metadata.get("handle", citation.get("handle", ""))
                if handle and target_handle.lower() in handle.lower():
                    name = metadata.get("name", citation.get("name", "Unknown"))
                    niche = metadata.get("niche", citation.get("niche", "Unknown"))
                    post = metadata.get("sample_post", citation.get("sample_post", ""))
                    followers = metadata.get("followers", citation.get("followers", 0))
                    found_match = True
                    
                    # Build a comprehensive answer
                    answer_parts = [f"Based on the data, {name} (@{handle})"]
                    
                    if niche and niche != "Unknown":
                        answer_parts.append(f"is in the {niche} niche")
                    else:
                        answer_parts.append("has no specific niche listed")
                    
                    if followers:
                        answer_parts.append(f"with {followers:,} followers")
                    
                    if post:
                        answer_parts.append(f"Sample post: \"{post[:200]}{'...' if len(post) > 200 else ''}\"")
                    
                    return ". ".join(answer_parts) + "."
            
            # If handle not found, provide helpful information
            if not found_match:
                available_handles = []
                for citation in citations:
                    metadata = citation.get("metadata", {})
                    handle = metadata.get("handle", citation.get("handle", ""))
                    if handle:
                        available_handles.append(handle)
                
                if available_handles:
                    sample_handles = available_handles[:5]
                    return f"The handle @{target_handle} was not found in the dataset. Available handles include: {', '.join(sample_handles)}. Please try searching for one of these handles or ask a different question."
                else:
                    return f"The handle @{target_handle} was not found in the dataset. No handles are available in the current data."
    
    # Handle niche queries
    if "niche" in query_lower:
        if niches:
            niche_list = list(niches)[:5]  # Limit to 5
            return f"Based on the dataset, I found these niches: {', '.join(niche_list)}. The influencers in these niches include: {', '.join(influencers[:3])}."
    
    # Handle "who talks about" queries
    if "who" in query_lower and ("talks" in query_lower or "discuss" in query_lower):
        if influencers:
            unique_influencers = list(set(influencers))[:5]
            return f"Based on the data, these influencers discuss relevant topics: {', '.join(unique_influencers)}."
    
    # Handle general content queries
    if "content" in query_lower or "post" in query_lower or "tweet" in query_lower:
        if posts:
            sample_posts = posts[:3]
            return f"Based on the dataset, here are some sample posts: {' '.join(sample_posts)}"
    
    # Handle follower count queries
    if "follower" in query_lower or "followers" in query_lower:
        follower_counts = []
        for citation in citations:
            metadata = citation.get("metadata", {})
            followers = metadata.get("followers", citation.get("followers", 0))
            name = metadata.get("name", citation.get("name", "Unknown"))
            if followers:
                follower_counts.append(f"{name}: {followers:,} followers")
        
        if follower_counts:
            return f"Follower counts from the dataset: {'; '.join(follower_counts[:5])}"
    
    # Generic but informative answer
    unique_influencers = list(set(influencers))[:3]
    unique_niches = list(niches)[:3]
    
    answer_parts = []
    if unique_influencers:
        answer_parts.append(f"Found {len(influencers)} influencers including {', '.join(unique_influencers)}")
    if unique_niches:
        answer_parts.append(f"covering niches like {', '.join(unique_niches)}")
    if posts:
        answer_parts.append(f"with sample content available")
    
    if answer_parts:
        return f"Based on the available data: {'; '.join(answer_parts)}."
    else:
        return f"Based on the available data, I found {len(citations)} relevant citations. Please ask a more specific question about influencers, niches, or content."