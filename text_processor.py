import json
import re
import os
from datetime import datetime
from typing import List, Dict, Any
from langchain.text_splitter import RecursiveCharacterTextSplitter
import logging

class TextProcessor:
    def __init__(self, chunk_size=1000, chunk_overlap=200):
        """
        Initialize TextProcessor with chunking parameters
        
        Args:
            chunk_size (int): Maximum size of each chunk
            chunk_overlap (int): Overlap between consecutive chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Initialize the text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        # Setup logging
        self.setup_logging()
        
    def setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def load_scraped_data(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Load scraped data from JSON file
        
        Args:
            file_path (str): Path to the scraped data JSON file
            
        Returns:
            List[Dict]: List of scraped page data
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            self.logger.info(f"Loaded {len(data)} pages from {file_path}")
            return data
        except Exception as e:
            self.logger.error(f"Error loading scraped data: {str(e)}")
            return []
    
    def clean_text(self, text: str) -> str:
        """
        Enhanced text cleaning for better processing
        
        Args:
            text (str): Raw text to clean
            
        Returns:
            str: Cleaned text
        """
        if not text:
            return ""
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep punctuation
        text = re.sub(r'[^\w\s\.\,\!\?\;\:\-\(\)]', ' ', text)
        
        # Remove repeated punctuation
        text = re.sub(r'([.!?]){2,}', r'\1', text)
        
        # Clean up spacing around punctuation
        text = re.sub(r'\s+([.!?,:;])', r'\1', text)
        text = re.sub(r'([.!?])\s*([A-Z])', r'\1 \2', text)
        
        # Remove lines with too few words (likely navigation/junk)
        lines = text.split('\n')
        cleaned_lines = []
        for line in lines:
            words = line.strip().split()
            if len(words) >= 3:  # Keep lines with at least 3 words
                cleaned_lines.append(line.strip())
        
        text = '\n'.join(cleaned_lines)
        
        # Final cleanup
        text = text.strip()
        
        return text
    
    def extract_relevant_content(self, page_data: Dict[str, Any]) -> str:
        """
        Extract and combine relevant content from a page
        
        Args:
            page_data (Dict): Single page data from scraper
            
        Returns:
            str: Combined relevant content
        """
        content_parts = []
        
        # Add title
        if page_data.get('title'):
            title = page_data['title'].strip()
            if title and title.lower() != 'untitled':
                content_parts.append(f"Title: {title}")
        
        # Add meta description if available
        if page_data.get('metadata', {}).get('description'):
            desc = page_data['metadata']['description'].strip()
            if desc:
                content_parts.append(f"Description: {desc}")
        
        # Add main content
        if page_data.get('content'):
            cleaned_content = self.clean_text(page_data['content'])
            if cleaned_content:
                content_parts.append(f"Content: {cleaned_content}")
        
        # Add URL for reference
        if page_data.get('url'):
            content_parts.append(f"Source URL: {page_data['url']}")
        
        return '\n\n'.join(content_parts)
    
    def process_scraped_data(self, scraped_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Process scraped data into chunks suitable for vector storage
        
        Args:
            scraped_data (List[Dict]): Raw scraped data
            
        Returns:
            List[Dict]: Processed chunks with metadata
        """
        processed_chunks = []
        
        for i, page_data in enumerate(scraped_data):
            try:
                # Extract relevant content
                content = self.extract_relevant_content(page_data)
                
                if not content or len(content.strip()) < 100:
                    self.logger.warning(f"Skipping page {i}: content too short")
                    continue
                
                # Split content into chunks
                chunks = self.text_splitter.split_text(content)
                
                # Create chunk objects with metadata
                for j, chunk in enumerate(chunks):
                    if len(chunk.strip()) < 50:  # Skip very small chunks
                        continue
                    
                    chunk_data = {
                        'chunk_id': f"{page_data['url']}_{j}",
                        'content': chunk.strip(),
                        'source_url': page_data['url'],
                        'page_title': page_data.get('title', ''),
                        'chunk_index': j,
                        'total_chunks': len(chunks),
                        'original_length': page_data.get('length', 0),
                        'word_count': len(chunk.split()),
                        'processed_at': datetime.now().isoformat(),
                        'metadata': {
                            'domain': self.extract_domain(page_data['url']),
                            'has_title': bool(page_data.get('title')),
                            'has_description': bool(page_data.get('metadata', {}).get('description')),
                            'original_word_count': page_data.get('word_count', 0)
                        }
                    }
                    
                    processed_chunks.append(chunk_data)
                
                self.logger.info(f"Processed page {i+1}/{len(scraped_data)}: {len(chunks)} chunks created")
                
            except Exception as e:
                self.logger.error(f"Error processing page {i}: {str(e)}")
                continue
        
        self.logger.info(f"Total chunks created: {len(processed_chunks)}")
        return processed_chunks
    
    def extract_domain(self, url: str) -> str:
        """Extract domain from URL"""
        from urllib.parse import urlparse
        return urlparse(url).netloc
    
    def save_processed_data(self, processed_chunks: List[Dict[str, Any]], filename: str = None):
        """
        Save processed chunks to JSON file
        
        Args:
            processed_chunks (List[Dict]): Processed chunk data
            filename (str): Output filename (optional)
        """
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"processed_chunks_{timestamp}.json"
        
        os.makedirs("data", exist_ok=True)
        filepath = os.path.join("data", filename)
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(processed_chunks, f, ensure_ascii=False, indent=2)
            
            # Save processing statistics
            stats = self.generate_processing_stats(processed_chunks)
            stats_filename = filename.replace('.json', '_stats.json')
            stats_filepath = os.path.join("data", stats_filename)
            
            with open(stats_filepath, 'w', encoding='utf-8') as f:
                json.dump(stats, f, ensure_ascii=False, indent=2)
            
            self.logger.info(f"Processed data saved to {filepath}")
            self.logger.info(f"Processing stats saved to {stats_filepath}")
            
            # Print summary
            print(f"\n{'='*50}")
            print("TEXT PROCESSING SUMMARY")
            print(f"{'='*50}")
            print(f"Total chunks created: {len(processed_chunks)}")
            print(f"Average chunk size: {stats['avg_chunk_length']:.1f} characters")
            print(f"Average words per chunk: {stats['avg_words_per_chunk']:.1f}")
            print(f"Unique domains: {stats['unique_domains']}")
            print(f"Processed data saved to: {filepath}")
            print(f"{'='*50}")
            
        except Exception as e:
            self.logger.error(f"Failed to save processed data: {str(e)}")
    
    def generate_processing_stats(self, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate statistics for processed chunks"""
        if not chunks:
            return {}
        
        chunk_lengths = [len(chunk['content']) for chunk in chunks]
        word_counts = [chunk['word_count'] for chunk in chunks]
        domains = set(chunk['metadata']['domain'] for chunk in chunks)
        
        stats = {
            'total_chunks': len(chunks),
            'avg_chunk_length': sum(chunk_lengths) / len(chunks),
            'min_chunk_length': min(chunk_lengths),
            'max_chunk_length': max(chunk_lengths),
            'avg_words_per_chunk': sum(word_counts) / len(chunks),
            'total_words': sum(word_counts),
            'unique_domains': len(domains),
            'domains': list(domains),
            'chunk_size_config': self.chunk_size,
            'chunk_overlap_config': self.chunk_overlap,
            'processing_date': datetime.now().isoformat()
        }
        
        return stats
    
    def process_from_file(self, input_file: str, output_file: str = None) -> List[Dict[str, Any]]:
        """
        Complete processing pipeline from scraped data file
        
        Args:
            input_file (str): Path to scraped data JSON file
            output_file (str): Output filename (optional)
            
        Returns:
            List[Dict]: Processed chunks
        """
        # Load scraped data
        scraped_data = self.load_scraped_data(input_file)
        
        if not scraped_data:
            self.logger.error("No data to process")
            return []
        
        # Process the data
        processed_chunks = self.process_scraped_data(scraped_data)
        
        # Save processed data
        if processed_chunks:
            self.save_processed_data(processed_chunks, output_file)
        
        return processed_chunks

# Usage example
if __name__ == "__main__":
    # Initialize processor
    processor = TextProcessor(chunk_size=1000, chunk_overlap=200)
    
    # Process scraped data (replace with your actual scraped data file)
    input_file = r"C:\Users\abiun\Downloads\CHATBOT\data\scraped_data_20250726_150805.json"  # Update with actual filename
    
    if os.path.exists(input_file):
        processed_chunks = processor.process_from_file(input_file)
        print(f"Processing complete! Created {len(processed_chunks)} chunks.")
    else:
        print(f"Input file {input_file} not found. Please run the scraper first.")
        print("Available files in data directory:")
        if os.path.exists("data"):
            for file in os.listdir("data"):
                if file.endswith('.json'):
                    print(f"  - {file}")