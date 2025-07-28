import os
import json
import logging 
from datetime import datetime
from typing import List, Dict, Any, Optional
import requests
from vector_store import VectorStore
import warnings
warnings.filterwarnings("ignore")

class ChangiChatbot:
    def __init__(self, vector_store_path: str, ollama_model: str = "llama3.2:1b"):
        """
        Initialize Changi Airport RAG Chatbot with Ollama llama3.2:1b
        
        Args:
            vector_store_path (str): Path to saved vector store
            ollama_model (str): Ollama model name
        """
        self.ollama_model = ollama_model
        self.ollama_url = "http://localhost:11434/api/generate"
        self.vector_store_path = vector_store_path
        
        # Setup logging
        self.setup_logging()
        
        # Test Ollama connection
        self.test_ollama_connection()
        
        # Initialize vector store
        self.vector_store = self.load_vector_store()
        
        # Conversation history
        self.conversation_history = []
        
        print("🤖 Changi Airport Chatbot is ready!")
        
    def setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def load_vector_store(self) -> VectorStore:
        """Load the vector store"""
        try:
            print("📚 Loading vector store...")
            vector_store = VectorStore(model_name="all-MiniLM-L6-v2")
            
            if vector_store.load_vector_store(self.vector_store_path):
                print("✅ Vector store loaded successfully!")
                return vector_store
            else:
                raise Exception("Failed to load vector store")
                
        except Exception as e:
            self.logger.error(f"Error loading vector store: {str(e)}")
            raise e
    
    def test_ollama_connection(self):
        """Test connection to Ollama server"""
        try:
            # Test if Ollama is running
            test_url = "http://localhost:11434/api/tags"
            response = requests.get(test_url, timeout=5)
            
            if response.status_code == 200:
                models = response.json().get('models', [])
                model_names = [model['name'] for model in models]
                
                if self.ollama_model in model_names:
                    print(f"✅ Ollama connected! Using model: {self.ollama_model}")
                else:
                    print(f"⚠️ Model {self.ollama_model} not found. Available models:")
                    for model in model_names:
                        print(f"  - {model}")
                    print(f"\nTo install llama3.2:1b, run: ollama pull llama3.2:1b")
                    
            else:
                raise Exception("Ollama server not responding")
                
        except Exception as e:
            print(f"❌ Error connecting to Ollama: {str(e)}")
            print("Make sure Ollama is running:")
            print("1. Start Ollama: ollama serve")
            print("2. Install model: ollama pull llama3.2:1b")
            raise e
    
    def retrieve_relevant_context(self, query: str, k: int = 3) -> List[Dict[str, Any]]:
        """
        Retrieve relevant context from vector store
        
        Args:
            query (str): User query
            k (int): Number of relevant chunks to retrieve
            
        Returns:
            List[Dict]: Relevant context chunks
        """
        try:
            results = self.vector_store.search_similar(query, k=k)
            
            self.logger.info(f"Retrieved {len(results)} relevant chunks for query: '{query}'")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error retrieving context: {str(e)}")
            return []
    
    def format_context(self, context_chunks: List[Dict[str, Any]]) -> str:
        """
        Format retrieved context for the LLM prompt
        
        Args:
            context_chunks (List[Dict]): Retrieved context chunks
            
        Returns:
            str: Formatted context
        """
        if not context_chunks:
            return "No relevant information found in the knowledge base."
        
        context_parts = []
        for i, chunk in enumerate(context_chunks, 1):
            source = chunk['chunk']['source_url'].split('/')[-1].replace('.html', '')
            content = chunk['chunk']['content'][:400]  # Limit content length
            
            context_parts.append(f"[Source {i} - {source}]\n{content}\n")
        
        return "\n".join(context_parts)
    
    def create_rag_prompt(self, query: str, context: str, conversation_history: List[str] = None) -> str:
        """
        Create RAG prompt for llama3.2:1b
        
        Args:
            query (str): User query
            context (str): Retrieved context
            conversation_history (List[str]): Previous conversation
            
        Returns:
            str: Formatted prompt
        """
        # Build conversation context
        conv_context = ""
        if conversation_history and len(conversation_history) > 0:
            recent_history = conversation_history[-4:]  # Last 2 exchanges
            conv_context = "\n".join(recent_history) + "\n\n"
        
        prompt = f"""<|system|>
You are a helpful AI assistant for Changi Airport and Jewel Changi Airport. You provide accurate, friendly, and informative responses based on official information from these airports.

Guidelines:
- Use only the provided context to answer questions
- Be concise but comprehensive
- If information is not in the context, politely say so
- Maintain a helpful and professional tone
- Focus on practical information for travelers

Context Information:
{context}

{conv_context}<|user|>
{query}<|assistant|>
"""
        
        return prompt
    
    def generate_response(self, prompt: str) -> str:
        """
        Generate response using Ollama llama3.2:1b
        
        Args:
            prompt (str): Formatted prompt
            
        Returns:
            str: Generated response
        """
        try:
            payload = {
                "model": self.ollama_model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "top_k": 20,  # Reduced from 40
                    "num_predict": 200,  # Reduced from 400 for faster response
                    "repeat_penalty": 1.1,
                    "stop": ["<|user|>", "<|system|>"],
                    "num_thread": 4  # Limit threads for stability
                }
            }
            
            response = requests.post(
                self.ollama_url,
                json=payload,
                timeout=60  # Increased timeout to 60 seconds
            )
            
            if response.status_code == 200:
                result = response.json()
                generated_text = result.get('response', '').strip()
                
                # Clean up response
                generated_text = self.clean_response(generated_text)
                
                return generated_text
            else:
                self.logger.error(f"Ollama API error: {response.status_code}")
                return "I apologize, but I'm having trouble generating a response right now. Please try again."
                
        except requests.exceptions.Timeout:
            self.logger.error("Timeout waiting for Ollama response")
            return "The response is taking too long. Please try a simpler question."
        except Exception as e:
            self.logger.error(f"Error generating response: {str(e)}")
            return "I apologize, but I'm having trouble generating a response right now. Please try rephrasing your question."
    
    def clean_response(self, response: str) -> str:
        """Clean and format the generated response"""
        # Remove special tokens
        response = response.replace("<|assistant|>", "").replace("<|user|>", "").replace("<|system|>", "")
        
        # Remove repetitive patterns
        lines = response.split('\n')
        unique_lines = []
        for line in lines:
            if line.strip() and line not in unique_lines:
                unique_lines.append(line)
        
        response = '\n'.join(unique_lines[:10])  # Limit to 10 lines
        
        return response.strip()
    
    def chat(self, user_query: str) -> Dict[str, Any]:
        """
        Main chat function
        
        Args:
            user_query (str): User's question
            
        Returns:
            Dict: Response with metadata
        """
        try:
            start_time = datetime.now()
            
            # Retrieve relevant context
            context_chunks = self.retrieve_relevant_context(user_query, k=3)
            context = self.format_context(context_chunks)
            
            # Create RAG prompt
            prompt = self.create_rag_prompt(user_query, context, self.conversation_history)
            
            # Generate response
            response = self.generate_response(prompt)
            
            # Update conversation history
            self.conversation_history.append(f"User: {user_query}")
            self.conversation_history.append(f"Assistant: {response}")
            
            # Keep only recent history
            if len(self.conversation_history) > 10:
                self.conversation_history = self.conversation_history[-10:]
            
            # Calculate response time
            response_time = (datetime.now() - start_time).total_seconds()
            
            # Prepare response
            chat_response = {
                "query": user_query,
                "response": response,
                "context_sources": [chunk['chunk']['source_url'] for chunk in context_chunks],
                "relevance_scores": [chunk['score'] for chunk in context_chunks],
                "response_time": response_time,
                "timestamp": datetime.now().isoformat()
            }
            
            self.logger.info(f"Query processed in {response_time:.2f}s")
            
            return chat_response
            
        except Exception as e:
            self.logger.error(f"Error in chat function: {str(e)}")
            return {
                "query": user_query,
                "response": "I apologize, but I encountered an error processing your question. Please try again.",
                "context_sources": [],
                "relevance_scores": [],
                "response_time": 0,
                "timestamp": datetime.now().isoformat(),
                "error": str(e)
            }
    
    def interactive_chat(self):
        """Start interactive chat session"""
        print(f"\n{'='*60}")
        print("🛬 Welcome to Changi Airport Assistant! 🛬")
        print("Ask me anything about Changi Airport or Jewel Changi Airport")
        print("Type 'quit', 'exit', or 'bye' to end the conversation")
        print(f"{'='*60}\n")
        
        while True:
            try:
                user_input = input("You: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'bye', 'q']:
                    print("👋 Thank you for using Changi Airport Assistant! Have a great day!")
                    break
                
                if not user_input:
                    print("Please enter a question.")
                    continue
                
                print("🤖 Thinking...")
                
                response = self.chat(user_input)
                
                print(f"\nAssistant: {response['response']}")
                
                if response['context_sources']:
                    print(f"\n📚 Sources:")
                    for i, source in enumerate(response['context_sources'], 1):
                        score = response['relevance_scores'][i-1]
                        print(f"  {i}. {source} (relevance: {score:.3f})")
                
                print(f"\n⏱️ Response time: {response['response_time']:.2f}s")
                print("-" * 60)
                
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"Error: {str(e)}")
                continue

def main():
    """Main function to run the chatbot"""
    # Configuration - Fix path resolution
    import os
    
    # Get the project root directory (handle different run locations)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Check if we're in src directory or root directory
    if "src" in current_dir:
        # If running from src, go up to project root
        project_root = os.path.dirname(os.path.dirname(current_dir))
    else:
        # If running from project root
        project_root = os.path.dirname(current_dir)
    
    vector_store_path = os.path.join(project_root, "data", "vector_store_20250726_154352")
    
    print(f"Looking for vector store at: {vector_store_path}")
    
    # Check if vector store exists
    if not os.path.exists(f"{vector_store_path}.index"):
        print("❌ Vector store not found!")
        print("Please run vector_store.py first to create the vector database.")
        print("\nAvailable vector stores:")
        data_dir = os.path.join(project_root, "data")
        if os.path.exists(data_dir):
            for file in os.listdir(data_dir):
                if file.endswith(".index"):
                    base_name = file.replace(".index", "")
                    print(f"  - data/{base_name}")
        return
    
    try:
        # Initialize chatbot with lighter model
        chatbot = ChangiChatbot(vector_store_path, ollama_model="llama3.2:1b")
        
        # Start interactive chat
        chatbot.interactive_chat()
        
    except Exception as e:
        print(f"❌ Error initializing chatbot: {str(e)}")
        print("\nTroubleshooting tips:")
        print("1. Make sure Ollama is running: ollama serve")
        print("2. Install llama3.2:1b: ollama pull llama3.2:1b")
        print("3. Check if the vector store path is correct")

if __name__ == "__main__":
    main()