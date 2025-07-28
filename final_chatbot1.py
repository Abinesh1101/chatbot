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
            content = chunk['chunk']['content'][:500]  # Increased content length for better context
            relevance = chunk['score']
            
            context_parts.append(f"[Source {i} - {source} (Relevance: {relevance:.3f})]\n{content}\n")
        
        return "\n".join(context_parts)
    
    def validate_context_quality(self, context_chunks: List[Dict[str, Any]], query: str) -> bool:
        """
        Validate if retrieved context is relevant enough to answer the query
        
        Args:
            context_chunks: Retrieved context chunks
            query: User query
            
        Returns:
            bool: True if context quality is sufficient
        """
        if not context_chunks:
            return False
        
        # Check if highest relevance score is above threshold
        max_score = max(chunk['score'] for chunk in context_chunks)
        return max_score > 0.3  # Threshold for relevance
    
    def create_rag_prompt(self, query: str, context: str, conversation_history: List[str] = None) -> str:
        """
        Create advanced RAG prompt with universal accuracy principles
        
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
You are an expert assistant that provides accurate, helpful information. Follow these universal principles for every response:

CORE ACCURACY PRINCIPLES:
1. SEMANTIC ALIGNMENT: Ensure your response directly addresses what the user is asking about. If they ask about luggage, talk about luggage - not unrelated services.

2. CONTEXT DEPENDENCY: Use ONLY the information provided in the context. If context mentions "car services" but the question is about luggage storage, do not connect them unless explicitly linked.

3. LOGICAL CONSISTENCY: Before including any information, ask yourself: "Does this logically answer the user's specific question?" If not, exclude it.

4. PRECISION OVER COMPLETENESS: Better to give a short accurate answer than a long irrelevant one.

5. CONTEXT VALIDATION: Each piece of information you include must be:
   - Directly relevant to the query topic
   - Actually present in the provided context
   - Logically connected to what the user needs

RESPONSE METHODOLOGY:
Step 1: Identify the EXACT topic the user is asking about
Step 2: Scan context for information ONLY related to that specific topic
Step 3: Ignore any context information that doesn't directly address the query
Step 4: Structure response with most relevant information first
Step 5: If insufficient relevant context exists, state this clearly

UNIVERSAL QUALITY CHECKS:
- Does every sentence in my response help answer the specific question?
- Am I mixing different services or topics inappropriately?
- Would a user find this response directly helpful for their stated need?
- Are all details (names, locations, times, procedures) verified in the context?

RESPONSE FORMAT:
- Start with the most direct answer possible
- Add supporting details only if they enhance the main answer
- Keep responses focused and actionable
- End with clear next steps when appropriate

Context Information:
{context}

{conv_context}<|user|>
{query}<|assistant|>
"""
        
        return prompt
    
    def generate_response(self, prompt: str) -> str:
        """
        Generate response using Ollama llama3.2:1b with enhanced parameters
        
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
                    "temperature": 0.3,  # Lower temperature for more focused responses
                    "top_p": 0.8,        # More focused token selection
                    "top_k": 15,         # Reduced for better precision
                    "num_predict": 250,  # Slightly increased for complete answers
                    "repeat_penalty": 1.2,  # Higher penalty to avoid repetition
                    "stop": ["<|user|>", "<|system|>", "User:", "Assistant:"],
                    "num_thread": 4
                }
            }
            
            response = requests.post(
                self.ollama_url,
                json=payload,
                timeout=60
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
        """Enhanced response cleaning for better quality"""
        # Remove special tokens and unwanted patterns
        response = response.replace("<|assistant|>", "").replace("<|user|>", "").replace("<|system|>", "")
        response = response.replace("Assistant:", "").replace("User:", "").strip()
        
        # Remove repetitive patterns and ensure uniqueness
        lines = response.split('\n')
        cleaned_lines = []
        seen_content = set()
        
        for line in lines:
            line = line.strip()
            if line and line not in seen_content and not line.startswith('---'):
                cleaned_lines.append(line)
                seen_content.add(line)
        
        # Limit response length and remove trailing repetitions
        response = '\n'.join(cleaned_lines[:8])  # Max 8 lines
        
        # Remove any trailing incomplete sentences
        sentences = response.split('.')
        if len(sentences) > 1 and len(sentences[-1].strip()) < 10:
            response = '.'.join(sentences[:-1]) + '.'
        
        return response.strip()
    
    def validate_response_accuracy(self, response: str, query: str, context: str) -> tuple:
        """
        Universal response validation system
        
        Args:
            response: Generated response
            query: Original user query
            context: Source context used
            
        Returns:
            tuple: (is_valid, corrected_response_or_error_message)
        """
        # Extract key topic from query
        query_lower = query.lower()
        response_lower = response.lower()
        
        # Define topic keywords and their expected response content
        topic_validations = {
            'luggage|baggage|check.?in|storage': ['luggage', 'baggage', 'check', 'storage', 'facility'],
            'food|restaurant|dining|eat': ['food', 'restaurant', 'dining', 'eat', 'cuisine'],
            'shopping|shop|store|buy': ['shop', 'store', 'retail', 'buy', 'brand'],
            'parking|car|vehicle': ['parking', 'car', 'vehicle', 'lot', 'rate'],
            'wifi|internet|connection': ['wifi', 'internet', 'connection', 'network'],
            'toilet|restroom|bathroom': ['toilet', 'restroom', 'bathroom', 'facilities'],
            'transportation|transport|bus|train|taxi|transfer|terminal': ['transport', 'bus', 'train', 'taxi', 'mrt', 'transfer', 'terminal', 'shuttle']
        }
        
        # Check semantic alignment
        query_topic = None
        for pattern, expected_words in topic_validations.items():
            import re
            if re.search(pattern, query_lower):
                query_topic = expected_words
                break
        
        if query_topic:
            # Check if response contains relevant topic words
            topic_match = any(word in response_lower for word in query_topic)
            
            # Check for irrelevant content mixing (but be less strict)
            irrelevant_mixing = False
            for pattern, words in topic_validations.items():
                # Skip current topic and transportation (as it overlaps with many topics)
                if 'transport' in pattern or any(word in query_lower for word in words):
                    continue
                # Only flag if response has significant content from unrelated topics
                unrelated_matches = sum(1 for word in words if word in response_lower)
                if unrelated_matches > 2:  # More than 2 unrelated topic words
                    irrelevant_mixing = True
                    break
            
            if not topic_match:
                return False, "Response doesn't align with the question topic."
            
            if irrelevant_mixing:
                return False, "Response mixes irrelevant information."
        
        # Check for context hallucination
        if "I don't have" not in response and context and "No relevant information found" in context:
            return False, "Response provides information not present in context."
        
        return True, response
    
    def post_process_response(self, response: str, query: str, context_chunks: List[Dict[str, Any]]) -> str:
        """
        Enhanced post-processing with universal validation
        
        Args:
            response: Generated response
            query: Original user query
            context_chunks: Context used for generation
            
        Returns:
            str: Validated and corrected response
        """
        # First, validate response accuracy
        context_text = self.format_context(context_chunks)
        is_valid, validation_result = self.validate_response_accuracy(response, query, context_text)
        
        if not is_valid:
            # If response is invalid, provide safe fallback
            if context_chunks and max(chunk['score'] for chunk in context_chunks) > 0.4:
                # Extract most relevant information
                best_chunk = max(context_chunks, key=lambda x: x['score'])
                relevant_content = best_chunk['chunk']['content'][:300]
                return f"Based on the available information: {relevant_content}. For more specific details, please contact the relevant service directly."
            else:
                return "I don't have specific information about that in my knowledge base. Please contact the relevant service for accurate details."
        
        # Additional quality checks for valid responses
        lines = response.split('\n')
        filtered_lines = []
        
        for line in lines:
            line = line.strip()
            if line and not any(skip_phrase in line.lower() for skip_phrase in 
                              ['as an ai', 'i cannot', 'please note that', 'disclaimer:']):
                filtered_lines.append(line)
        
        return '\n'.join(filtered_lines[:6])  # Limit to 6 lines max
    
    def chat(self, user_query: str) -> Dict[str, Any]:
        """
        Enhanced chat function with improved accuracy controls
        
        Args:
            user_query (str): User's question
            
        Returns:
            Dict: Response with metadata
        """
        try:
            start_time = datetime.now()
            
            # Retrieve relevant context
            context_chunks = self.retrieve_relevant_context(user_query, k=3)
            
            # Validate context quality
            context_quality = self.validate_context_quality(context_chunks, user_query)
            
            if not context_quality:
                return {
                    "query": user_query,
                    "response": "I don't have specific information about that in my knowledge base. Please contact Changi Airport directly for the most accurate and up-to-date information.",
                    "context_sources": [],
                    "relevance_scores": [],
                    "response_time": (datetime.now() - start_time).total_seconds(),
                    "timestamp": datetime.now().isoformat()
                }
            
            context = self.format_context(context_chunks)
            
            # Create enhanced RAG prompt
            prompt = self.create_rag_prompt(user_query, context, self.conversation_history)
            
            # Generate response
            response = self.generate_response(prompt)
            
            # Enhanced post-processing with validation
            response = self.post_process_response(response, user_query, context_chunks)
            
            # Final accuracy check
            if len(response.split()) < 5:  # Too short response
                return {
                    "query": user_query,
                    "response": "I need more specific information to provide a helpful answer. Could you please rephrase your question with more details?",
                    "context_sources": [chunk['chunk']['source_url'] for chunk in context_chunks],
                    "relevance_scores": [chunk['score'] for chunk in context_chunks],
                    "response_time": (datetime.now() - start_time).total_seconds(),
                    "timestamp": datetime.now().isoformat()
                }
            
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
                "timestamp": datetime.now().isoformat(),
                "context_quality": context_quality
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