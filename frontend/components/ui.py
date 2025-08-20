from __future__ import annotations
import streamlit as st
from typing import List, Dict


def inject_css(path: str = "frontend/theme.css"):
    with open(path, "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


def app_header():
    st.markdown(
        """
        <div class="hero">
          <h1>🐦 Twitter/X Influencer RAG Assistant</h1>
          <p style="opacity:0.8;margin-top:-6px">
            Compare <b>Vanilla RAG</b> vs <b>LangChain RAG</b>, pick your model, and query your own dataset.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown('<div class="hr"></div>', unsafe_allow_html=True)


def info_card(title: str, body_md: str):
    st.markdown(f'<div class="card"><h3 style="margin-top:0">{title}</h3>{body_md}</div>', unsafe_allow_html=True)


def citation_chips(citations: List[Dict]):
    if not citations:
        st.caption("No citations")
        return
    html = "".join([f'<span class="chip">{c.get("name","Unknown")} {c.get("handle","")}</span>' for c in citations])
    st.markdown(html, unsafe_allow_html=True)
