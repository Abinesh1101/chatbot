import os
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document

# === Websites to scrape
URLS = [
    "https://www.changiairport.com/en/airport-guide.html",
    "https://www.jewelchangiairport.com/en/attractions.html"
]

# === Step 1: Load website content
docs = []
for url in URLS:
    loader = WebBaseLoader(url)
    page_docs = loader.load()
    docs.extend(page_docs)

# === Step 2: Split text
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(docs)

# === Step 3: Embed and store with FAISS
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(chunks, embedding=embedding_model)

# === Step 4: Save vector store in proper format
SAVE_PATH = r"C:\Users\abiun\Downloads\CHATBOT\data\changi_faiss_store"
vectorstore.save_local(SAVE_PATH)

print("✅ FAISS vector store created successfully!")
