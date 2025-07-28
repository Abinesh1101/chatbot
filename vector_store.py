import json
import os
import pickle
import numpy as np
from datetime import datetime
from typing import List, Dict, Any, Tuple
import logging
from sentence_transformers import SentenceTransformer
import faiss
from tqdm import tqdm

class VectorStore:
    def __init__(self, model_name="all-MiniLM-L6-v2", vector_dim=384):
        """
        Initialize Vector Store with FAISS and SentenceTransformers
        
        Args:
            model_name (str): SentenceTransformer model name
            vector_dim (int): Dimension of embeddings (384 for all-MiniLM-L6-v2)
        """
        self.model_name = model_name
        self.vector_dim = vector_dim
        
        # Initialize the embedding model
        print(f"Loading embedding model: {model_name}...")
        self.embedding_model = SentenceTransformer(model_name)
        print("✅ Embedding model loaded successfully!")
        
        # Initialize FAISS index
        self.index = None
        self.chunks_metadata = []
        
        # Setup logging
        self.setup_logging()
        
    def setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def load_processed_chunks(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Load processed chunks from JSON file
        
        Args:
            file_path (str): Path to processed chunks JSON file
            
        Returns:
            List[Dict]: List of processed chunks
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                chunks = json.load(f)
            self.logger.info(f"Loaded {len(chunks)} chunks from {file_path}")
            return chunks
        except Exception as e:
            self.logger.error(f"Error loading processed chunks: {str(e)}")
            return []
    
    def generate_embeddings(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """
        Generate embeddings for a list of texts
        
        Args:
            texts (List[str]): List of texts to embed
            batch_size (int): Batch size for processing
            
        Returns:
            np.ndarray: Array of embeddings
        """
        embeddings = []
        
        print(f"Generating embeddings for {len(texts)} texts...")
        
        # Process in batches with progress bar
        for i in tqdm(range(0, len(texts), batch_size), desc="Generating embeddings"):
            batch_texts = texts[i:i + batch_size]
            
            try:
                # Generate embeddings for batch
                batch_embeddings = self.embedding_model.encode(
                    batch_texts,
                    convert_to_numpy=True,
                    show_progress_bar=False
                )
                embeddings.append(batch_embeddings)
                
            except Exception as e:
                self.logger.error(f"Error generating embeddings for batch {i//batch_size}: {str(e)}")
                # Create zero embeddings for failed batch
                zero_embeddings = np.zeros((len(batch_texts), self.vector_dim))
                embeddings.append(zero_embeddings)
        
        # Concatenate all embeddings
        all_embeddings = np.vstack(embeddings)
        
        self.logger.info(f"Generated embeddings shape: {all_embeddings.shape}")
        return all_embeddings
    
    def create_faiss_index(self, embeddings: np.ndarray) -> faiss.Index:
        """
        Create FAISS index from embeddings
        
        Args:
            embeddings (np.ndarray): Array of embeddings
            
        Returns:
            faiss.Index: FAISS index
        """
        print("Creating FAISS index...")
        
        # Create FAISS index (using IndexFlatIP for cosine similarity)
        index = faiss.IndexFlatIP(self.vector_dim)
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        
        # Add embeddings to index
        index.add(embeddings.astype('float32'))
        
        self.logger.info(f"FAISS index created with {index.ntotal} vectors")
        print(f"✅ FAISS index created successfully with {index.ntotal} vectors!")
        
        return index
    
    def build_vector_store(self, chunks_file: str, save_path: str = None) -> bool:
        """
        Build complete vector store from processed chunks
        
        Args:
            chunks_file (str): Path to processed chunks JSON file
            save_path (str): Path to save vector store (optional)
            
        Returns:
            bool: Success status
        """
        try:
            # Load processed chunks
            chunks = self.load_processed_chunks(chunks_file)
            
            if not chunks:
                self.logger.error("No chunks loaded")
                return False
            
            # Extract texts for embedding
            texts = [chunk['content'] for chunk in chunks]
            
            # Generate embeddings
            embeddings = self.generate_embeddings(texts)
            
            # Create FAISS index
            self.index = self.create_faiss_index(embeddings)
            
            # Store metadata
            self.chunks_metadata = chunks
            
            # Save vector store
            if save_path:
                self.save_vector_store(save_path)
            else:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                default_path = f"data/vector_store_{timestamp}"
                self.save_vector_store(default_path)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error building vector store: {str(e)}")
            return False
    
    def save_vector_store(self, base_path: str):
        """
        Save vector store to disk
        
        Args:
            base_path (str): Base path for saving files
        """
        try:
            os.makedirs("data", exist_ok=True)
            
            # Save FAISS index
            index_path = f"{base_path}.index"
            faiss.write_index(self.index, index_path)
            
            # Save metadata
            metadata_path = f"{base_path}_metadata.pkl"
            with open(metadata_path, 'wb') as f:
                pickle.dump(self.chunks_metadata, f)
            
            # Save configuration
            config = {
                'model_name': self.model_name,
                'vector_dim': self.vector_dim,
                'total_vectors': len(self.chunks_metadata),
                'created_at': datetime.now().isoformat(),
                'index_path': index_path,
                'metadata_path': metadata_path
            }
            config_path = f"{base_path}_config.json"
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)
            
            self.logger.info(f"Vector store saved:")
            self.logger.info(f"  - Index: {index_path}")
            self.logger.info(f"  - Metadata: {metadata_path}")
            self.logger.info(f"  - Config: {config_path}")
            
            # Print summary
            print(f"\n{'='*50}")
            print("VECTOR STORE CREATION SUMMARY")
            print(f"{'='*50}")
            print(f"Model used: {self.model_name}")
            print(f"Vector dimension: {self.vector_dim}")
            print(f"Total vectors: {len(self.chunks_metadata)}")
            print(f"Index saved to: {index_path}")
            print(f"Metadata saved to: {metadata_path}")
            print(f"Config saved to: {config_path}")
            print(f"{'='*50}")
            
        except Exception as e:
            self.logger.error(f"Error saving vector store: {str(e)}")
    
    def load_vector_store(self, base_path: str) -> bool:
        """
        Load vector store from disk
        
        Args:
            base_path (str): Base path for loading files
            
        Returns:
            bool: Success status
        """
        try:
            # Load configuration
            config_path = f"{base_path}_config.json"
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            # Verify model compatibility
            if config['model_name'] != self.model_name:
                self.logger.warning(f"Model mismatch: expected {self.model_name}, got {config['model_name']}")
            
            # Load FAISS index
            index_path = f"{base_path}.index"
            self.index = faiss.read_index(index_path)
            
            # Load metadata
            metadata_path = f"{base_path}_metadata.pkl"
            with open(metadata_path, 'rb') as f:
                self.chunks_metadata = pickle.load(f)
            
            self.logger.info(f"Vector store loaded successfully:")
            self.logger.info(f"  - Vectors: {len(self.chunks_metadata)}")
            self.logger.info(f"  - Model: {config['model_name']}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading vector store: {str(e)}")
            return False
    
    def search_similar(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for similar chunks using the query
        
        Args:
            query (str): Search query
            k (int): Number of results to return
            
        Returns:
            List[Dict]: Similar chunks with scores
        """
        if not self.index or not self.chunks_metadata:
            self.logger.error("Vector store not loaded")
            return []
        
        try:
            # Generate query embedding
            query_embedding = self.embedding_model.encode([query], convert_to_numpy=True)
            
            # Normalize for cosine similarity
            faiss.normalize_L2(query_embedding)
            
            # Search in FAISS index
            scores, indices = self.index.search(query_embedding.astype('float32'), k)
            
            # Prepare results
            results = []
            for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
                if idx >= 0 and idx < len(self.chunks_metadata):
                    result = {
                        'rank': i + 1,
                        'score': float(score),
                        'chunk': self.chunks_metadata[idx],
                        'content': self.chunks_metadata[idx]['content'][:200] + "..." if len(self.chunks_metadata[idx]['content']) > 200 else self.chunks_metadata[idx]['content']
                    }
                    results.append(result)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error searching: {str(e)}")
            return []
    
    def test_search(self, test_queries: List[str] = None):
        """
        Test the vector store with sample queries
        
        Args:
            test_queries (List[str]): List of test queries
        """
        if not test_queries:
            test_queries = [
                "Changi Airport opening hours",
                "Jewel Changi shopping",
                "Airport facilities and services",
                "Immigration and customs",
                "Terminal information"
            ]
        
        print(f"\n{'='*50}")
        print("TESTING VECTOR STORE")
        print(f"{'='*50}")
        
        for query in test_queries:
            print(f"\nQuery: '{query}'")
            print("-" * 30)
            
            results = self.search_similar(query, k=3)
            
            for result in results:
                print(f"Rank {result['rank']} (Score: {result['score']:.4f})")
                print(f"Source: {result['chunk']['source_url']}")
                print(f"Content: {result['content']}")
                print()

# Usage example
if __name__ == "__main__":
    # Initialize vector store
    vector_store = VectorStore(model_name="all-MiniLM-L6-v2")
    
    # Build vector store from processed chunks
    chunks_file = r"C:\Users\abiun\Downloads\CHATBOT\data\processed_chunks_20250726_152421.json"  # Update with your actual file
    
    if os.path.exists(chunks_file):
        print("Building vector store...")
        success = vector_store.build_vector_store(chunks_file)
        
        if success:
            print("✅ Vector store created successfully!")
            
            # Test the vector store
            vector_store.test_search()
        else:
            print("❌ Failed to create vector store")
    else:
        print(f"Chunks file {chunks_file} not found.")
        print("Available files in data directory:")
        if os.path.exists("data"):
            for file in os.listdir("data"):
                if file.startswith('processed_chunks') and file.endswith('.json'):
                    print(f"  - {file}")