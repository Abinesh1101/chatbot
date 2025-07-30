# ✈️ Changi & Jewel Airport AI Chatbot

This project is a **Retrieval-Augmented Generation (RAG)** chatbot that uses a Large Language Model (LLM) from **Groq API** and a **FAISS vector store** to answer questions based on content scraped from:
- 🌐 [Changi Airport Website](https://www.changiairport.com/in/en.html)
- 🌐 [Jewel Changi Airport Website](https://www.jewelchangiairport.com/)

---

## 🚀 Features

- 💬 Streamlit-based chatbot UI
- 🧠 LLM (LLaMA3-8B via Groq API) for contextual answers
- 🗂️ FAISS vector DB for document retrieval
- 🧹 Custom web scraping + preprocessing pipeline
- 🔐 Secure handling of API keys via `.streamlit/secrets.toml`
- ☁️ Deployed on [Streamlit Cloud](https://chatbot-h2gwc3nand9kbigxvk5qzs.streamlit.app/)

---

## 🛠️ Project Structure

```
chatbot/
├── .streamlit/
│   └── secrets.toml
├── data/
│   ├── changi_faiss_store/
│   │   ├── index.faiss
│   │   └── index.pkl
│   └── src/
│       ├── final_chatbot2.py  ← Main app file
│       ├── vector_store.py
│       ├── text_processor.py
│       └── scrapers.py
├── requirements.txt
└── README.md
```

---

## 📦 Setup Instructions

### 1. Clone the repo
```bash
git clone https://github.com/Abinesh1101/chatbot.git
cd chatbot
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Set up secrets
Create a file at `.streamlit/secrets.toml` with:
```toml
GROQ_API_KEY = "your_groq_api_key_here"
```

### 4. Run locally
```bash
streamlit run data/src/final_chatbot2.py
```

---

## 🧠 Technologies Used

- **LLM:** Groq's LLaMA3-8B-8192
- **Vector DB:** FAISS
- **Embedding:** `all-MiniLM-L6-v2` via HuggingFace
- **Web UI:** Streamlit
- **LangChain** for chaining and RAG

---

## 📤 Deployment (Streamlit Cloud)

1. Push your code to GitHub
2. Create a new Streamlit Cloud app from the repo
3. In Settings → Secrets, add your `GROQ_API_KEY`
4. Done! 🎉

---

## 🙌 Acknowledgements

Special thanks to the teams behind:
- [Groq](https://console.groq.com)
- [LangChain](https://www.langchain.com/)
- [Streamlit](https://streamlit.io/)
- [Changi Airport](https://www.changiairport.com/)
- [Jewel Changi](https://www.jewelchangiairport.com/)