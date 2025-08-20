import io
import json
import requests
import streamlit as st


st.set_page_config(page_title="RAG Assistant", page_icon="🐦", layout="centered")
# Inject custom CSS theme
with open("frontend/theme.css", "r", encoding="utf-8") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
st.title("🐦 Twitter/X Influencer RAG Assistant")

backend_url = st.sidebar.text_input("Backend URL", value="http://127.0.0.1:8000")
api_key = st.sidebar.text_input("API Key (optional)", type="password")
implementation = st.sidebar.radio("Implementation", ["vanilla", "langchain"], index=0)
model = st.sidebar.text_input("Model name", value="gpt-4o-mini")

st.header("Upload Dataset")
uploaded = st.file_uploader("Upload JSON/CSV dataset", type=["json", "csv"]) 
if uploaded is not None:
    files = {"file": (uploaded.name, uploaded.getvalue())}
    r = requests.post(f"{backend_url}/upload_dataset", files=files)
    if r.ok:
        st.success(f"Uploaded and ingested: {uploaded.name}")
    else:
        st.error(f"Upload failed: {r.status_code} {r.text}")

st.header("Ask a question")
question = st.text_input("Your question", value="Who talks about AI startups?")
if st.button("Query") and question:
    payload = {
        "query": question,
        "implementation": implementation,
        "model": model,
        "api_key": api_key or None,
    }
    r = requests.post(f"{backend_url}/query", json=payload)
    if r.ok:
        data = r.json()
        st.subheader("Answer")
        st.write(data.get("answer", ""))
        st.subheader("Citations")
        st.json(data.get("citations", []))
    else:
        st.error(f"Query failed: {r.status_code} {r.text}")


