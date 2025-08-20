import os, json, io, time, requests, streamlit as st
import sys
from typing import Dict, Any, List
# Ensure project root on sys.path so 'frontend' package is importable when
# Streamlit runs this file from the frontend directory
_CURRENT_DIR = os.path.dirname(__file__)
_PROJECT_ROOT = os.path.abspath(os.path.join(_CURRENT_DIR, ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from frontend.components.ui import inject_css, app_header, info_card, citation_chips

st.set_page_config(page_title="Twitter/X Influencer RAG", page_icon="🐦", layout="wide")
inject_css()
app_header()

# --- Sidebar (Settings)

with st.sidebar:
	st.caption("Service")
	backend_url = st.text_input("Backend URL", value=os.getenv("BACKEND_URL", "http://127.0.0.1:8000"), key="sb_backend_url")
	api_key = st.text_input("API Key (optional)", type="password", key="sb_api_key")
	st.caption("RAG Options")
	impl = st.radio("Implementation", options=["vanilla","langchain"], horizontal=False, index=0, key="sb_impl")
	model = st.text_input("Model name", value="gpt-4o-mini", key="sb_model")
	top_k = st.slider("Top-K", 1, 10, 4, key="sb_top_k")
	temperature = st.slider("Temperature", 0.0, 1.0, 0.2, key="sb_temp")
	st.caption("")

if "dataset_id" not in st.session_state: st.session_state.dataset_id = None
if "history" not in st.session_state: st.session_state.history = []

tabs = st.tabs(["📤 Upload Dataset", "❓ Ask", "🕘 History", "ℹ️ About"])

# --- Helpers
def post_json(url, payload, headers=None, timeout=90):
	t0 = time.time()
	try:
		r = requests.post(url, json=payload, headers=headers or {}, timeout=timeout)
		r.raise_for_status()
		return r.json(), int((time.time()-t0)*1000)
	except requests.RequestException as e:
		st.error(f"Request failed: {e}")
		return None, None

# --- Upload tab
with tabs[0]:
	st.subheader("Upload Dataset")
	f = st.file_uploader("Upload JSON/CSV dataset", type=["json","csv"], key="upload_dataset")
	if f is not None:
		size_mb = len(f.getvalue())/1_000_000
		st.caption(f"File: **{f.name}** · {size_mb:.2f} MB")
		if size_mb > 50:
			st.error("File too large (>50MB).")
		else:
			if st.button("Ingest"):
				try:
					files = {"file": (f.name, f.getvalue())}
					data = {"implementation": impl, "model": model, "api_key": api_key}
					resp = requests.post(f"{backend_url}/upload_dataset", data=data, files=files, timeout=180)
					resp.raise_for_status()
					out = resp.json()
					st.session_state.dataset_id = out.get("dataset_id") or f.name
					st.success(f"Ingested ✅ Dataset ID: {st.session_state.dataset_id}")
				except requests.RequestException as e:
					st.error(f"Upload failed: {e}")

# --- Ask tab
with tabs[1]:
	st.subheader("Ask a question")
	q = st.text_input("Your question", placeholder="Who talks about AI startups?", key="ask_question")
	if st.button("Query", type="primary", use_container_width=False):
		if not st.session_state.dataset_id:
			st.warning("No dataset ingested yet.")
		elif not q.strip():
			st.warning("Please enter a question.")
		else:
			payload = {
				"query": q.strip(),  # backend expects 'query'
				"implementation": impl,
				"model": model,
				"api_key": api_key,
				"dataset_id": st.session_state.dataset_id,
				"top_k": top_k,
				"temperature": temperature
			}
			out, t_ms = post_json(f"{backend_url}/query", payload)
			if out:
				answer = out.get("answer","")
				citations = out.get("citations",[])
				st.markdown(f'<div class="card"><h3 style="margin-top:0">Answer</h3><div>{answer}</div><p style="opacity:.6;margin-top:8px">Latency: {t_ms} ms</p></div>', unsafe_allow_html=True)
				st.caption("Citations")
				citation_chips(citations)
				st.session_state.history.append({
					"q": q.strip(), "a": answer, "citations": citations, "t_ms": t_ms,
					"impl": impl, "model": model, "ts": time.time()
				})

# --- History tab
with tabs[2]:
	st.subheader("History")
	if not st.session_state.history:
		st.caption("No queries yet.")
	else:
		for i, h in enumerate(reversed(st.session_state.history)):
			st.markdown(f'**Q:** {h["q"]}  \n**A:** {h["a"][:220]}{"..." if len(h["a"])>220 else ""}  \n*{h["impl"]} · {h["model"]} · {h["t_ms"]} ms*')
			citation_chips(h.get("citations",[]))
			st.markdown('<div class="hr"></div>', unsafe_allow_html=True)
		# Export
		st.download_button("Export JSON", data=json.dumps(st.session_state.history, indent=2), file_name="history.json", mime="application/json")

# --- About tab
with tabs[3]:
	info_card("What is this?",
			  "<p>A demo to compare <b>Vanilla RAG</b> vs <b>LangChain RAG</b> on influencer datasets. "
			  "Upload your data, choose a model, and get grounded answers with citations.</p>")
	info_card("How it works",
			  "<ol><li>Upload JSON/CSV → we chunk & embed it into a vector store.</li>"
			  "<li>Ask a question → we retrieve top chunks and call your chosen LLM.</li>"
			  "<li>We always show citations so you can audit the answer.</li></ol>")
import io
import json
import requests
import time
import streamlit as st
from frontend.components.ui import inject_css, app_header, citation_chips, info_card
from streamlit_lottie import st_lottie
import streamlit.components.v1 as components
from streamlit_extras.stylable_container import stylable_container


st.set_page_config(page_title="RAG Assistant", page_icon="🐦", layout="centered")
# Inject custom CSS theme and header
inject_css()
app_header()

# ---- HTTP Client ----
def post_json(url, payload, headers=None, timeout=60):
    t0 = time.time()
    try:
        r = requests.post(url, json=payload, headers=headers or {}, timeout=timeout)
        latency = (time.time() - t0) * 1000
        r.raise_for_status()
        return r.json(), latency
    except requests.RequestException as e:
        st.error(f"Request failed: {e}")
        return None, None

# ---- Session State Defaults ----
defaults = {
    "backend_url": "http://127.0.0.1:8000",
    "api_key": "",
    "impl": "vanilla",
    "model": "gpt-4o-mini",
    "top_k": 3,
    "temperature": 0.2,
    "dataset_id": None,
    "history": [],
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v
st.title("🐦 Twitter/X Influencer RAG Assistant")

st.sidebar.subheader("Settings")
st.session_state.backend_url = st.sidebar.text_input("Backend URL", value=st.session_state.backend_url)

show_key = st.sidebar.checkbox("Show API key", value=False)
st.session_state.api_key = st.sidebar.text_input(
    "API Key (optional)", value=st.session_state.api_key, type=("text" if show_key else "password")
)

st.session_state.impl = st.sidebar.radio("Implementation", ["vanilla", "langchain"], index=(0 if st.session_state.impl=="vanilla" else 1))
st.session_state.model = st.sidebar.text_input("Model name", value=st.session_state.model)
st.session_state.top_k = st.sidebar.number_input("Top-K", min_value=1, max_value=20, value=int(st.session_state.top_k), step=1)
st.session_state.temperature = st.sidebar.slider("Temperature", min_value=0.0, max_value=1.0, value=float(st.session_state.temperature), step=0.05)

tabs = st.tabs(["Upload Dataset", "Ask", "History", "About"])

with tabs[0]:
    st.header("Upload Dataset")
    uploaded = st.file_uploader("Upload JSON/CSV dataset", type=["json", "csv"]) 
    if uploaded is not None:
        # Details card
        size_bytes = len(uploaded.getvalue())
        size_mb = size_bytes / (1024 * 1024)
        ext_ok = uploaded.name.lower().endswith((".json", ".csv"))
        info_card(
            "Selected file",
            f"<ul><li><b>Name:</b> {uploaded.name}</li><li><b>Size:</b> {size_mb:.2f} MB</li><li><b>MIME:</b> {uploaded.type}</li></ul>"
        )

        # Validation
        if not ext_ok:
            st.error("Only .json or .csv files are allowed.")
        elif size_mb > 50:
            st.error("File too large. Max size is 50 MB.")
        else:
            # Progress indicator
            progress = st.progress(0)
            try:
                files = {
                    "file": (uploaded.name, uploaded.getvalue(), uploaded.type or "application/octet-stream"),
                }
                data = {
                    "impl": st.session_state.impl,
                    "model": st.session_state.model,
                    "api_key": st.session_state.api_key or "",
                }
                progress.progress(30)
                r = requests.post(
                    f"{st.session_state.backend_url}/upload_dataset",
                    files=files,
                    data=data,
                    timeout=120,
                )
                progress.progress(90)
                if r.ok:
                    st.session_state.dataset_id = uploaded.name
                    st.success(f"Dataset ingested: ID {st.session_state.dataset_id}")
                else:
                    st.error(f"Upload failed: {r.status_code} {r.text}")
            except Exception as e:
                st.error(f"Upload error: {e}")
            finally:
                progress.progress(100)
    else:
        try:
            with open("assets/empty.json", "r", encoding="utf-8") as f:
                anim = json.load(f)
            st_lottie(anim, height=180, key="empty_upload")
        except Exception:
            pass

with tabs[1]:
    st.header("Ask a question")
    # Brand gradient bar below the header
    components.html(
        """
        <div style="
          height:56px;
          background: linear-gradient(90deg, #7C3AED, #3B82F6, #06B6D4);
          border-radius:14px;
          margin: 6px 0 18px 0;
          opacity:0.9;">
        </div>
        """,
        height=64,
    )
    question = st.text_input("Your question", value="Who talks about AI startups?", key="q_input")
    if st.button("Query", key="query_btn") and question:
        payload = {
            "query": question,
            "implementation": st.session_state.impl,
            "model": st.session_state.model,
            "api_key": (st.session_state.api_key or None),
            "dataset_id": st.session_state.dataset_id,
            "top_k": st.session_state.top_k,
            "temperature": st.session_state.temperature,
        }
        try:
            data, latency_ms = post_json(f"{st.session_state.backend_url}/query", payload, timeout=60)
            if data is not None:
                # User bubble
                st.markdown(
                    f"""
                    <div style="display:inline-block; padding:8px 12px; border-radius:12px; border:1px solid rgba(255,255,255,0.12); background:rgba(255,255,255,0.04); margin-bottom:8px;">
                        <b>You</b>: {question}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                # Model bubble using .card depth
                answer_md = data.get('answer','')
                with stylable_container(key="answer_card", css_styles="""
                  {transition: transform .15s ease}
                  &:hover {transform: translateY(-2px)}
                """):
                    st.markdown(
                        f'<div class="card"><div style="opacity:0.75; font-size:12px; margin-bottom:6px;">{st.session_state.impl} · {st.session_state.model} · {latency_ms:.0f} ms</div><h3>Answer</h3>{answer_md}</div>',
                        unsafe_allow_html=True,
                    )

                st.subheader("Citations")
                citation_chips(data.get("citations", []))

                # Save to history with latency
                st.session_state.history.insert(0, {
                    "q": question,
                    "a": data.get("answer", ""),
                    "citations": data.get("citations", []),
                    "t_ms": latency_ms,
                })
                st.success("Query successful")
        except Exception as e:
            st.error(f"Request error: {e}")
    else:
        try:
            with open("assets/empty.json", "r", encoding="utf-8") as f:
                anim = json.load(f)
            st_lottie(anim, height=180, key="empty_query")
        except Exception:
            pass

with tabs[2]:
    st.header("History")
    if not st.session_state.history:
        st.caption("No history yet.")
    for idx, item in enumerate(st.session_state.history):
        q = item.get("q", "")
        a = item.get("a", "")
        preview = (a[:140] + "…") if len(a) > 140 else a
        cols = st.columns([6, 1])
        with cols[0]:
            st.markdown(f"**Q:** {q}")
            st.markdown(f"**A:** {preview}")
            if item.get("t_ms") is not None:
                st.caption(f"Latency: {item.get('t_ms'):.0f} ms")
        with cols[1]:
            if st.button("Re-run", key=f"rerun_{idx}"):
                st.session_state.q_input = q
                st.experimental_rerun()
        citation_chips(item.get("citations", []))
        st.markdown('<div class="hr"></div>', unsafe_allow_html=True)

    if st.session_state.history:
        st.download_button(
            label="Export JSON",
            data=json.dumps(st.session_state.history, ensure_ascii=False, indent=2),
            file_name="history.json",
            mime="application/json",
        )

with tabs[3]:
    st.header("About")
    st.markdown(
        """
        This demo compares Vanilla RAG and a LangChain-like RAG pipeline over influencer datasets.
        
        1. Upload a dataset (JSON/CSV)
        2. Choose your implementation and model
        3. Ask a question and review citations
        
        Backend is FastAPI; Vector store can be native (FAISS/NumPy) or Chroma.
        """
    )


