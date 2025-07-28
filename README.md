
# 🧠 Changi Airport AI Chatbot (Ollama + FastAPI)

This project is a smart chatbot that answers queries about **Changi Airport** and **Jewel Changi** using an open-source LLM (e.g., `llama3`) powered by **Ollama**, deployed on **RunPod**, and served via **FastAPI** on **Render**. It uses RAG (Retrieval-Augmented Generation) with local PDF data.

---

## 📦 Features

- 💬 Conversational chatbot interface (FastAPI/HTML UI)
- 🧠 RAG-enabled using FAISS vector store
- 📚 Custom-trained on Changi + Jewel Changi Airport PDFs
- ⚡ Ollama-powered LLM (e.g., llama3.2:1b)
- ☁️ Ollama hosted on **RunPod GPU**
- 🌐 FastAPI backend hosted on **Render**
- 🔍 Semantic search with `all-MiniLM-L6-v2` embeddings

---

## 🛠️ Project Structure

```
chatbot/
├── data/
│   ├── src/
│   │   ├── api_server.py         # FastAPI backend entrypoint
│   │   ├── chat_ui.py            # HTML + JS-based frontend
│   │   ├── final_chatbot1.py     # Core RAG chatbot logic
│   │   ├── vector_store.py       # FAISS index creation
│   │   ├── scrapers.py           # Website scraper (optional)
│   │   └── text_processor.py     # PDF parsing & chunking
│   ├── vector_store_*.json       # Vector index + stats
│   └── scraped_data_*.json       # Raw scraped text
```

---

## 🚀 Setup Instructions

### Step 1: 🧠 Deploy Ollama on RunPod

1. Go to [RunPod.io](https://console.runpod.io/)
2. Click `Deploy a Pod` → Choose **GPU** tab (e.g., NVIDIA T4)
3. Search & select **"Ollama"** community template
4. Expose Port `11434`
5. Deploy & copy your **Public IP**

### Step 2: 🧱 Deploy FastAPI on Render

1. Push your code to GitHub
2. Go to [Render](https://render.com)
3. Click **New → Web Service**
4. Connect your repo & set:
   - **Start Command**:  
     ```
     uvicorn data.src.api_server:app --host 0.0.0.0 --port 10000
     ```
   - **Build Command**: `pip install -r requirements.txt`
5. Add ENV var:
   - `OLLAMA_BASE_URL=http://<RunPod-IP>:11434`

---

## 🧪 Local Development

```bash
# Step 1: Install dependencies
pip install -r requirements.txt

# Step 2: Generate vector store
python data/src/vector_store.py

# Step 3: Run chatbot locally (assuming Ollama is running)
python data/src/api_server.py
```

---

## 🔗 Usage

Access the chatbot UI at:

```
https://your-render-subdomain.onrender.com
```

Ask questions like:
- “Where is the Rain Vortex?”
- “How to access Jewel from Terminal 3?”

---

## 🧠 Model & Tech Stack

| Component        | Tech                            |
|------------------|----------------------------------|
| LLM              | Mistral / LLaMA 3 via Ollama     |
| Embedding Model  | `all-MiniLM-L6-v2`               |
| Vector DB        | FAISS                           |
| Backend          | FastAPI                         |
| Frontend         | HTML + JS (Simple UI)           |
| Deployment       | Render + RunPod                 |

---

## 📄 License

MIT License.  
Data and chatbot logic are tailored for educational and non-commercial use.
