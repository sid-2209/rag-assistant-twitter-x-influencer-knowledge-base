
# 🐦 Twitter/X Influencer Knowledge Base (RAG Assistant)

## 🎯 Overview
The **Twitter/X Influencer Knowledge Base Assistant** is an AI-powered **Retrieval-Augmented Generation (RAG)** system that helps users discover and query information about influencers across domains such as **AI, fashion, gaming, fitness, and memes**.  

It ingests influencer data, builds a vector database, and answers questions with citations, ensuring that responses are both **accurate** and **attributable** to real influencers.  

---

## ✅ Features
- 🔍 **Data Ingestion Pipeline**: Collects and processes influencer data (profiles, niches, posts).  
- 🧠 **Vector Search**: Embeds influencer data using OpenAI embeddings & stores in FAISS/Chroma.  
- 🤖 **RAG Engine**: Combines vector retrieval with LLMs (GPT-4o-mini/GPT-5).  
- 📜 **Citations**: Always references influencer names & handles.  
- 🌐 **API**: FastAPI-powered REST endpoints for queries.  
- 🧪 **Testing**: Pytest coverage for RAG, vector search, and API.  
- 📦 **Infra**: Docker + docker-compose for easy deployment.  
- 🚀 **Deployable**: Works locally and on cloud (AWS/GCP/Render).  

---

## 🏗️ Project Structure
```
twitter-influencer-assistant/
├── app/
│   ├── __init__.py
│   ├── api.py          # FastAPI entrypoint
│   ├── routes.py       # Mock routes mounted under /mock
│   ├── config.py       # settings management
│   ├── embeddings.py   # embedding logic
│   ├── pipeline.py     # ETL pipeline (raw → processed → vector DB)
│   ├── rag.py          # RAG orchestration
├── data/
│   ├── raw/            # raw influencer data
│   ├── processed/      # cleaned dataset
│   └── sample.json
├── models/             # vector DB storage
├── tests/
│   ├── test_api.py
│   ├── test_rag.py
│   ├── test_vector.py
│   └── conftest.py
├── infra/
│   ├── Dockerfile
│   ├── docker-compose.yml
├── infra/requirements.txt
├── pyproject.toml
├── README.md
```

---

## ⚙️ Setup & Installation

### 1️⃣ Clone Repository
```bash
git clone https://github.com/yourusername/twitter-influencer-assistant.git
cd twitter-influencer-assistant
```

### 2️⃣ Create Virtual Environment
```bash
python -m venv .venv
source .venv/bin/activate   # macOS/Linux
.venv\Scripts\activate    # Windows
```

### 3️⃣ Install Dependencies
```bash
pip install -r infra/requirements.txt
```

### 4️⃣ Setup Environment Variables
Create a `.env` file:
```ini
OPENAI_API_KEY=your_openai_api_key_here
```

### 5️⃣ Run Pipeline
```bash
python app/pipeline.py
```

### 6️⃣ Start API Server
```bash
uvicorn app.api:app --reload
```

API will be live at: `http://127.0.0.1:8000`  

---

## 🌐 API Endpoints

### **1. Health Check**
```http
GET /healthz
```
✅ Returns service status.  

### **2. Query Influencers**
```http
POST /query
Content-Type: application/json

{
  "query": "Who are top AI influencers on Twitter?"
}
```
🔄 Response:
```json
{
  "answer": "Some of the top AI influencers include Jane Doe (@janedoe)...",
  "citations": ["Jane Doe (@janedoe)", "John AI (@john_ai)"]
}
```

---

### Mock/demo routes

```http
POST /mock/query
```

Demo endpoint using static mock docs from `app/routes.py` for quick testing. This does not use the vector store and is mounted under `/mock` to avoid conflicts with the primary `/query` endpoint.

---

## 🧪 Testing
Run all tests:
```bash
pytest -q
```

Test files:
- `test_rag.py`: Validates RAG context building & fallback.  
- `test_vector.py`: Ensures vector retrieval works correctly.  
- `test_api.py`: End-to-end API tests.  

---

## 📦 Deployment

### Docker
```bash
docker build -t twitter-assistant .
docker run -p 8000:8000 twitter-assistant
```

### Docker Compose
```bash
docker-compose up --build
```

### Cloud Deployment
- ✅ **Render / GCP Cloud Run / AWS ECS** (lightweight FastAPI service).  
- Use secrets manager for `OPENAI_API_KEY`.  

---

## 🎨 (Optional) UI Layer
- Build a **Streamlit app** or **Next.js frontend**.  
- Allow users to type queries & display influencer profile cards with citations.  

---

## 📈 Extensions (Future Work)
- 🔢 Analytics: track most-cited influencers.  
- 📷 Multimodal: include profile pics with CLIP embeddings.  
- 🏷️ Fine-tuning: adapt LLM for influencer Q&A style.  
- 📊 Evaluation: Compare RAG vs vanilla GPT responses.  

---

## 📜 Deliverables
- ✅ Codebase with clean structure.  
- ✅ REST API (FastAPI).  
- ✅ Tests with pytest.  
- ✅ Dockerized app, cloud-ready.  
- ✅ Documentation (this README).  

---

## 👨‍💻 Author
Built with ❤️ by [Your Name]  
For questions: open an issue or reach out on [LinkedIn/GitHub].
