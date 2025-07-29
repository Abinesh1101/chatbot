import streamlit as st
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
import streamlit as st
from langchain_community.embeddings import HuggingFaceEmbeddings

# === Groq API key (safe since it's your local dev)
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]

# === Path to FAISS vector store (rebuilt safely by you)
FAISS_PATH = "data/changi_faiss_store"
# === Prompt Template
custom_prompt = PromptTemplate(
    template="""
You are an intelligent assistant answering queries based only on the provided context.

Context:
{context}

Question:
{question}

Answer in a concise and helpful manner.
""",
    input_variables=["context", "question"],
)

# === Streamlit UI setup
st.set_page_config(page_title="Changi RAG Chatbot", page_icon="🛫")
st.title("🛫 Changi Airport & Jewel Chatbot")
st.caption("Ask anything based on content from changiairport.com or jewelchangiairport.com")

# === Load chain with FAISS + Groq + Embeddings
@st.cache_resource
def load_chain():
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    vectorstore = FAISS.load_local(
        FAISS_PATH,
        index_name="index",
        embeddings=embeddings,
        allow_dangerous_deserialization=True  # 👈 required for local .pkl
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    llm = ChatGroq(model="llama3-8b-8192", groq_api_key=GROQ_API_KEY)

    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": custom_prompt}
    )

qa_chain = load_chain()

# === User Query Input
query = st.text_input("💬 Ask your question here")

if st.button("Get Answer") and query.strip():
    with st.spinner("🧠 Thinking..."):
        try:
            response = qa_chain.run(query)
            st.success(response)
        except Exception as e:
            st.error("An error occurred while answering your query.")
            st.exception(e)
