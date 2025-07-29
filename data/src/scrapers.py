import requests
from bs4 import BeautifulSoup
import time
import json
from urllib.parse import urljoin, urlparse
from urllib.robotparser import RobotFileParser
import os
import re
import hashlib
import random
from datetime import datetime
import logging

class WebScraper:
    def __init__(self, base_urls, max_pages=50, config_file=None):
        # Load configuration
        self.config = self.load_config(config_file)
        
        self.base_urls = base_urls
        self.max_pages = max_pages
        self.visited_urls = set()
        self.scraped_data = []
        self.content_hashes = set()
        self.robots_cache = {}
        
        # Setup session with better headers
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': self.config.get('user_agent', 'WebScraper Bot 1.0 (+https://example.com/contact)'),
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1'
        })
        
        # Setup logging
        self.setup_logging()

    def load_config(self, config_file):
        """Load configuration from file or use defaults"""
        default_config = {
            'delay_min': 1,
            'delay_max': 3,
            'min_content_length': 500,
            'min_word_count': 50,
            'user_agent': 'WebScraper Bot 1.0 (+https://example.com/contact)',
            'timeout': 10,
            'max_retries': 3,
            'respect_robots': True
        }
        
        if config_file and os.path.exists(config_file):
            try:
                with open(config_file, 'r') as f:
                    user_config = json.load(f)
                    default_config.update(user_config)
            except Exception as e:
                print(f"Error loading config file: {e}. Using defaults.")
        
        return default_config

    def setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('scraper.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def check_robots_txt(self, base_url):
        """Check if scraping is allowed according to robots.txt"""
        if not self.config.get('respect_robots', True):
            return True
            
        base_domain = urlparse(base_url).netloc
        
        if base_domain in self.robots_cache:
            return self.robots_cache[base_domain]
        
        try:
            robots_url = f"https://{base_domain}/robots.txt"
            rp = RobotFileParser()
            rp.set_url(robots_url)
            rp.read()
            
            user_agent = self.config.get('user_agent', '*')
            can_fetch = rp.can_fetch(user_agent, base_url)
            
            self.robots_cache[base_domain] = can_fetch
            self.logger.info(f"Robots.txt check for {base_domain}: {'Allowed' if can_fetch else 'Disallowed'}")
            
            return can_fetch
            
        except Exception as e:
            self.logger.warning(f"Could not fetch robots.txt for {base_domain}: {e}")
            # Assume allowed if we can't check
            self.robots_cache[base_domain] = True
            return True

    def is_valid_url(self, url, base_domain):
        """Enhanced URL validation"""
        parsed = urlparse(url)
        
        # Check domain
        if parsed.netloc != base_domain:
            return False
        
        # Skip file extensions
        file_extensions = ['.pdf', '.jpg', '.jpeg', '.png', '.gif', '.css', '.js', '.ico', 
                          '.zip', '.exe', '.doc', '.docx', '.xls', '.xlsx', '.ppt', '.pptx']
        if any(ext in url.lower() for ext in file_extensions):
            return False
        
        # Skip certain URL patterns
        skip_patterns = ['/search', '/login', '/register', '/cart', '/checkout', '/admin',
                        '/wp-admin', '/wp-content', '/wp-includes', '?print=', '?share=',
                        '/feed', '/rss', '/sitemap', '?replytocom=']
        if any(pattern in url.lower() for pattern in skip_patterns):
            return False
        
        # Skip URLs with too many parameters
        if url.count('?') > 0 and url.count('&') > 3:
            return False
        
        # Skip anchor links
        if '#' in url and url.split('#')[0] in self.visited_urls:
            return False
            
        return True

    def get_content_hash(self, content):
        """Generate hash for content deduplication"""
        # Normalize content for better duplicate detection
        normalized = re.sub(r'\s+', ' ', content.lower().strip())
        return hashlib.md5(normalized.encode()).hexdigest()

    def clean_text(self, text):
        """Enhanced text cleaning"""
        # Remove legal/privacy/footer junk
        patterns_to_remove = [
            r'©.*?rights reserved',
            r'terms.*?conditions',
            r'privacy policy',
            r'×Close.*?\d+\s*/\s*\d+',
            r'skip to.*?content',
            r'cookie.*?policy',
            r'gdpr.*?compliance',
            r'all rights reserved',
            r'powered by.*',
            r'designed by.*',
            r'follow us on.*',
            r'share this.*',
            r'print this.*',
            r'download.*pdf'
        ]
        
        for pattern in patterns_to_remove:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE)
        
        # Clean excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\n\s*\n', '\n', text)
        
        return text.strip()

    def extract_text_content(self, soup):
        """Enhanced content extraction"""
        # Remove unwanted elements
        unwanted_tags = ["script", "style", "nav", "footer", "header", "aside", 
                        "advertisement", "ads", "sidebar", "menu", "breadcrumb"]
        for tag in unwanted_tags:
            for element in soup.find_all(tag):
                element.decompose()
        
        # Remove elements by class/id patterns
        unwanted_patterns = ['cookie', 'gdpr', 'popup', 'modal', 'advertisement', 
                           'social', 'share', 'comment', 'sidebar']
        for pattern in unwanted_patterns:
            for element in soup.find_all(attrs={'class': re.compile(pattern, re.I)}):
                element.decompose()
            for element in soup.find_all(attrs={'id': re.compile(pattern, re.I)}):
                element.decompose()
        
        # Try to find main content areas first
        main_content = None
        content_selectors = ['main', 'article', '[role="main"]', '.content', '.post', '.entry']
        
        for selector in content_selectors:
            main_content = soup.select_one(selector)
            if main_content:
                break
        
        # Use main content if found, otherwise use body
        content_source = main_content if main_content else soup
        
        # Extract text
        text = content_source.get_text(separator=' ', strip=True)
        
        # Clean the text
        text = self.clean_text(text)
        
        return text

    def extract_metadata(self, soup):
        """Extract additional metadata from the page"""
        metadata = {}
        
        # Meta description
        meta_desc = soup.find('meta', attrs={'name': 'description'})
        if meta_desc:
            metadata['description'] = meta_desc.get('content', '')
        
        # Meta keywords
        meta_keywords = soup.find('meta', attrs={'name': 'keywords'})
        if meta_keywords:
            metadata['keywords'] = meta_keywords.get('content', '')
        
        # Language
        html_tag = soup.find('html')
        if html_tag and html_tag.get('lang'):
            metadata['language'] = html_tag.get('lang')
        
        # Headings
        headings = []
        for i in range(1, 7):
            for heading in soup.find_all(f'h{i}'):
                headings.append({
                    'level': i,
                    'text': heading.get_text(strip=True)
                })
        metadata['headings'] = headings[:10]  # Limit to first 10 headings
        
        return metadata

    def scrape_page(self, url):
        """Enhanced page scraping with retries"""
        for attempt in range(self.config.get('max_retries', 3)):
            try:
                response = self.session.get(url, timeout=self.config.get('timeout', 10))
                
                # Handle different status codes
                if response.status_code == 429:  # Too Many Requests
                    self.logger.warning(f"Rate limited on {url}, waiting 60 seconds...")
                    time.sleep(60)
                    continue
                elif response.status_code in [403, 404, 410]:
                    self.logger.warning(f"Access denied or not found: {url} (Status: {response.status_code})")
                    return None
                elif response.status_code != 200:
                    self.logger.warning(f"Unexpected status code {response.status_code} for {url}")
                    continue
                
                response.raise_for_status()
                
                # Parse content
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Extract title
                title_tag = soup.find('title')
                title = title_tag.get_text().strip() if title_tag else urlparse(url).path
                
                # Extract main content
                content = self.extract_text_content(soup)
                
                # Check content quality
                word_count = len(content.split())
                if (len(content) < self.config.get('min_content_length', 500) or 
                    word_count < self.config.get('min_word_count', 50)):
                    self.logger.debug(f"Content too short for {url}: {len(content)} chars, {word_count} words")
                    return None
                
                # Check for duplicate content
                content_hash = self.get_content_hash(content)
                if content_hash in self.content_hashes:
                    self.logger.debug(f"Duplicate content detected for {url}")
                    return None
                
                self.content_hashes.add(content_hash)
                
                # Extract metadata
                metadata = self.extract_metadata(soup)
                
                page_data = {
                    'url': url,
                    'title': title,
                    'content': content,
                    'length': len(content),
                    'word_count': word_count,
                    'scraped_at': datetime.now().isoformat(),
                    'metadata': metadata,
                    'content_hash': content_hash
                }
                
                self.logger.info(f"Successfully scraped: {url} ({len(content)} chars)")
                return page_data
                
            except requests.exceptions.Timeout:
                self.logger.warning(f"Timeout on {url}, attempt {attempt + 1}")
                if attempt < self.config.get('max_retries', 3) - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
            except Exception as e:
                self.logger.error(f"Error scraping {url} (attempt {attempt + 1}): {str(e)}")
                if attempt < self.config.get('max_retries', 3) - 1:
                    time.sleep(2 ** attempt)
        
        return None

    def get_internal_links(self, soup, base_url, base_domain):
        """Enhanced link extraction"""
        links = set()
        
        for link in soup.find_all('a', href=True):
            href = link['href']
            full_url = urljoin(base_url, href)
            
            # Clean URL (remove fragments, normalize)
            parsed = urlparse(full_url)
            clean_url = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
            if parsed.query:
                clean_url += f"?{parsed.query}"
            
            if (self.is_valid_url(clean_url, base_domain) and 
                clean_url not in self.visited_urls and
                len(links) < 10):  # Limit links per page
                links.add(clean_url)
        
        return links

    def smart_delay(self):
        """Implement smart delay with randomization"""
        delay = random.uniform(
            self.config.get('delay_min', 1),
            self.config.get('delay_max', 3)
        )
        time.sleep(delay)

    def scrape_website(self, base_url):
        """Enhanced website scraping with better flow control"""
        # Check robots.txt first
        if not self.check_robots_txt(base_url):
            self.logger.warning(f"Robots.txt disallows scraping {base_url}")
            return
        
        base_domain = urlparse(base_url).netloc
        urls_to_visit = [base_url]
        pages_scraped = 0
        
        self.logger.info(f"Starting to scrape: {base_url}")
        
        while urls_to_visit and pages_scraped < self.max_pages:
            current_url = urls_to_visit.pop(0)
            
            if current_url in self.visited_urls:
                continue
            
            self.visited_urls.add(current_url)
            self.logger.info(f"Scraping ({pages_scraped + 1}/{self.max_pages}): {current_url}")
            
            try:
                # Get page content for link extraction
                response = self.session.get(current_url, timeout=self.config.get('timeout', 10))
                response.raise_for_status()
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Scrape current page
                page_data = self.scrape_page(current_url)
                if page_data:
                    self.scraped_data.append(page_data)
                    pages_scraped += 1
                    
                    # Save incrementally every 10 pages
                    if pages_scraped % 10 == 0:
                        self.save_incremental_data()
                
                # Get more links to scrape
                if pages_scraped < self.max_pages:
                    new_links = self.get_internal_links(soup, current_url, base_domain)
                    # Add new links to the beginning for breadth-first crawling
                    urls_to_visit = list(new_links) + urls_to_visit
                
                # Smart delay between requests
                self.smart_delay()
                
            except requests.exceptions.RequestException as e:
                self.logger.error(f"Request error for {current_url}: {str(e)}")
                continue
            except Exception as e:
                self.logger.error(f"Unexpected error processing {current_url}: {str(e)}")
                continue
        
        self.logger.info(f"Completed scraping {base_url}. Scraped {pages_scraped} pages.")

    def scrape_all_websites(self):
        """Scrape all websites with enhanced error handling"""
        total_start_time = time.time()
        
        for i, base_url in enumerate(self.base_urls):
            self.logger.info(f"Starting website {i+1}/{len(self.base_urls)}: {base_url}")
            
            try:
                self.scrape_website(base_url)
            except Exception as e:
                self.logger.error(f"Failed to scrape {base_url}: {str(e)}")
            
            # Longer delay between websites
            if i < len(self.base_urls) - 1:
                delay = random.uniform(5, 10)
                self.logger.info(f"Waiting {delay:.1f} seconds before next website...")
                time.sleep(delay)
        
        total_time = time.time() - total_start_time
        self.logger.info(f"Scraping completed in {total_time:.2f} seconds")
        
        return self.scraped_data

    def save_incremental_data(self, filename_prefix="scraped_data"):
        """Save data incrementally to avoid loss"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{filename_prefix}_incremental_{timestamp}.json"
        
        os.makedirs("data", exist_ok=True)
        filepath = os.path.join("data", filename)
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(self.scraped_data, f, ensure_ascii=False, indent=2)
            self.logger.info(f"Incremental data saved to {filepath}")
        except Exception as e:
            self.logger.error(f"Failed to save incremental data: {str(e)}")

    def save_data(self, filename="scraped_data.json"):
        """Enhanced data saving with statistics"""
        os.makedirs("data", exist_ok=True)
        
        # Add timestamp to filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = filename.split('.')[0]
        extension = filename.split('.')[-1] if '.' in filename else 'json'
        timestamped_filename = f"{base_name}_{timestamp}.{extension}"
        
        filepath = os.path.join("data", timestamped_filename)
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(self.scraped_data, f, ensure_ascii=False, indent=2)
            
            # Save statistics
            stats = self.generate_statistics()
            stats_filepath = os.path.join("data", f"statistics_{timestamp}.json")
            with open(stats_filepath, 'w', encoding='utf-8') as f:
                json.dump(stats, f, ensure_ascii=False, indent=2)
            
            self.logger.info(f"Data saved to {filepath}")
            self.logger.info(f"Statistics saved to {stats_filepath}")
            
            # Print summary
            print(f"\n{'='*50}")
            print("SCRAPING SUMMARY")
            print(f"{'='*50}")
            print(f"Total pages scraped: {len(self.scraped_data)}")
            print(f"Total characters: {stats['total_characters']:,}")
            print(f"Total words: {stats['total_words']:,}")
            print(f"Average page length: {stats['avg_characters_per_page']:.1f} characters")
            print(f"Data saved to: {filepath}")
            print(f"{'='*50}")
        
        except Exception as e:
            self.logger.error(f"Failed to save data: {str(e)}")

    def generate_statistics(self):
        """Generate comprehensive statistics"""
        if not self.scraped_data:
            return {}
        
        total_chars = sum(page['length'] for page in self.scraped_data)
        total_words = sum(page['word_count'] for page in self.scraped_data)
        
        stats = {
            'total_pages': len(self.scraped_data),
            'total_characters': total_chars,
            'total_words': total_words,
            'avg_characters_per_page': total_chars / len(self.scraped_data),
            'avg_words_per_page': total_words / len(self.scraped_data),
            'min_page_length': min(page['length'] for page in self.scraped_data),
            'max_page_length': max(page['length'] for page in self.scraped_data),
            'urls_visited': len(self.visited_urls),
            'unique_domains': len(set(urlparse(url).netloc for url in self.visited_urls)),
            'scraping_date': datetime.now().isoformat()
        }
        
        return stats

# Usage
if __name__ == "__main__":
    # Configuration (optional - will use defaults if not provided)
    config = {
        "delay_min": 1,
        "delay_max": 3,
        "min_content_length": 500,
        "min_word_count": 50,
        "user_agent": "WebScraper Bot 1.0 (+https://example.com/contact)",
        "timeout": 10,
        "max_retries": 3,
        "respect_robots": True
    }
    
    # Save config to file (optional)
    with open('scraper_config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    # URLs to scrape
    urls = [
        "https://www.changiairport.com",
        "https://www.jewelchangiairport.com"
    ]
    
    # Create and run scraper
    scraper = WebScraper(urls, max_pages=30, config_file='scraper_config.json')
    scraped_data = scraper.scrape_all_websites()
    scraper.save_data()