from fastapi import FastAPI, Request
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import os
from final_chatbot1 import ChangiChatbot

# Define request schema
class QueryRequest(BaseModel):
    question: str

# Create FastAPI app
app = FastAPI(title="Changi Airport Chatbot API")

# Allow frontend (CORS config)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the chatbot on startup
@app.on_event("startup")
def load_chatbot():
    global chatbot
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(current_dir))
    vector_store_path = os.path.join(project_root, "data", "vector_store_20250726_154352")
    chatbot = ChangiChatbot(vector_store_path, ollama_model="llama3.2:1b")

@app.post("/chat")
async def chat_endpoint(request: QueryRequest):
    response = chatbot.chat(request.question)
    return response

if __name__ == "__main__":
    import webbrowser
    print("📄 Opening Swagger Docs at http://127.0.0.1:8000/docs")
    webbrowser.open("http://127.0.0.1:8000/docs")