# Data Pipeline Documentation

## Overview

[Web-Graze](https://github.com/shivendrra/web-graze) is a comprehensive Python library for scraping educational content from various internet sources. It provides automated data extraction pipelines for Encyclopedia Britannica, Wikipedia, YouTube transcripts, and image platforms, making it easy to collect large-scale datasets for research, education, or machine learning projects.

## Features

- **Multi-Source Scraping**: Extract content from Britannica, Wikipedia, and YouTube
- **Automated Query Management**: Pre-defined topic lists for systematic data collection
- **Rate Limiting & Error Handling**: Built-in protections against being blocked
- **Metrics & Monitoring**: Track scraping performance and success rates
- **Flexible Output Formats**: Support for text and JSON outputs
- **User-Agent Rotation**: Prevents detection and blocking

## Installation

```bash
pip install requests beautifulsoup4 google-api-python-client youtube-transcript-api tqdm
```

## Quick Start

```python
from .database import Wikipedia, Britannica, Youtube, Queries

# Get predefined queries
queries = Queries("search")()[:5]  # Get first 5 search topics

# Scrape Wikipedia
wiki = Wikipedia("data/wiki_content.txt", metrics=True)
wiki(queries)

# Scrape Britannica
britannica = Britannica("data/britannica_content.txt", max_limit=3, metrics=True)
britannica(queries)
```

## Core Components

### 1. Query Management (`Queries`)

The `Queries` class provides organized topic lists for different scraping purposes.

#### Usage
```python
from .database import Queries

# Available categories
search_queries = Queries("search")()    # For Britannica/Wikipedia
channel_ids = Queries("channel")()      # For YouTube channels
image_topics = Queries("images")()      # For image platforms
```

#### Categories
- **"search"**: 200+ educational topics (science, history, technology, arts)
- **"channel"**: 100+ YouTube channel IDs for educational content
- **"images"**: 100+ image search topics for visual content

### 2. Wikipedia Scraper (`Wikipedia`)

Extracts article content and linked pages from Wikipedia.

#### Basic Usage
```python
from .database import Wikipedia

# Initialize scraper
wiki = Wikipedia(
    filepath="output/wikipedia_data.txt",
    metrics=True
)

# Scrape specific topics
topics = ["Machine Learning", "Quantum Physics", "Renaissance Art"]
wiki(topics)
```

#### Advanced Usage
```python
# Include linked articles for deeper content
wiki(topics, extra_urls=True)
```

#### Parameters
- `filepath`: Output file path (creates directories if needed)
- `metrics`: Show scraping statistics (default: False)
- `extra_urls`: Scrape linked articles within each page (default: False)

### 3. Britannica Scraper (`Britannica`)

Scrapes Encyclopedia Britannica articles with pagination support.

#### Basic Usage
```python
from .database import Britannica

# Initialize scraper
britannica = Britannica(
    filepath="output/britannica_data.txt",
    max_limit=5,
    metrics=True
)

# Scrape topics
topics = ["Biology", "Chemistry", "Physics"]
britannica(topics)
```

#### Parameters
- `filepath`: Output file path
- `max_limit`: Pages to scrape per query (default: 10)
- `metrics`: Show scraping statistics (default: False)

#### Features
- **Pagination Support**: Automatically navigates through result pages
- **Rate Limiting**: Built-in delays to prevent blocking
- **User-Agent Rotation**: Randomized headers for stealth

### 4. YouTube Transcript Scraper (`Youtube`)

Extracts video transcripts from YouTube channels using the YouTube Data API.

#### Setup
First, obtain a YouTube Data API key:
1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a project and enable YouTube Data API v3
3. Create credentials (API key)

#### Basic Usage
```python
from .database import Youtube

# Initialize with API key
yt = Youtube(
    api_key="YOUR_YOUTUBE_API_KEY",
    filepath="output/youtube_transcripts.txt",
    max_results=50,
    metrics=True
)

# Scrape from channel IDs
channel_ids = ["UC_x5XG1OV2P6uZZ5FSM9Ttw", "UCsXVk37bltHxD1rDPwtNM8Q"]
yt(channel_ids)
```

#### Extract Video URLs Only
```python
# Save video URLs instead of transcripts
yt(channel_ids, videoUrls=True)  # Saves to .json file
```

#### Parameters
- `api_key`: YouTube Data API v3 key (required)
- `filepath`: Output file path
- `max_results`: Videos per channel (default: 50)
- `metrics`: Show scraping statistics (default: False)
- `videoUrls`: Save URLs instead of transcripts (default: False)

## Complete Workflow Examples

### 1. Educational Content Pipeline
```python
from .database import Wikipedia, Britannica, Queries

# Get science topics
queries = Queries("search")()
science_topics = [q for q in queries if any(term in q.lower() 
                  for term in ['physics', 'chemistry', 'biology', 'science'])]

# Create comprehensive dataset
wiki = Wikipedia("datasets/science_wiki.txt", metrics=True)
wiki(science_topics[:20], extra_urls=True)

britannica = Britannica("datasets/science_britannica.txt", max_limit=3, metrics=True)
britannica(science_topics[:20])
```

### 2. YouTube Educational Content
```python
from .database import Youtube, Queries

# Get educational channel IDs
channels = Queries("channel")()[:10]

# Extract transcripts
yt = Youtube(
    api_key="YOUR_API_KEY",
    filepath="datasets/edu_transcripts.txt",
    max_results=100,
    metrics=True
)
yt(channels)
```

### 3. Multi-Source Research Dataset
```python
from .database import Wikipedia, Britannica, Youtube, Queries

def create_research_dataset(topic_filter, output_dir):
    """Create a comprehensive dataset on specific topics"""
    
    # Get relevant queries
    all_queries = Queries("search")()
    filtered_queries = [q for q in all_queries if topic_filter.lower() in q.lower()]
    
    print(f"Found {len(filtered_queries)} topics related to '{topic_filter}'")
    
    # Wikipedia content
    wiki = Wikipedia(f"{output_dir}/wikipedia_{topic_filter}.txt", metrics=True)
    wiki(filtered_queries, extra_urls=True)
    
    # Britannica content
    britannica = Britannica(f"{output_dir}/britannica_{topic_filter}.txt", 
                           max_limit=5, metrics=True)
    britannica(filtered_queries)
    
    print(f"Dataset created in {output_dir}/")

# Usage
create_research_dataset("artificial intelligence", "datasets/ai_research")
```

## Best Practices

### 1. Rate Limiting
```python
import time
import random

# Add delays between requests
for topic in topics:
    scraper(topic)
    time.sleep(random.uniform(1, 3))  # Random delay 1-3 seconds
```

### 2. Error Handling
```python
try:
    wiki = Wikipedia("output/data.txt", metrics=True)
    wiki(topics)
except Exception as e:
    print(f"Scraping failed: {e}")
    # Log error or retry with different parameters
```

### 3. Data Organization
```python
import os
from datetime import datetime

# Organize by date and source
date_str = datetime.now().strftime("%Y%m%d")
base_dir = f"datasets/{date_str}"
os.makedirs(base_dir, exist_ok=True)

# Separate files by source
wiki = Wikipedia(f"{base_dir}/wikipedia_content.txt")
britannica = Britannica(f"{base_dir}/britannica_content.txt")
```

### 4. Batch Processing
```python
def batch_scrape(queries, batch_size=10):
    """Process queries in batches to avoid overwhelming servers"""
    
    for i in range(0, len(queries), batch_size):
        batch = queries[i:i+batch_size]
        print(f"Processing batch {i//batch_size + 1}: {len(batch)} queries")
        
        # Process batch
        wiki = Wikipedia(f"output/batch_{i//batch_size + 1}.txt")
        wiki(batch)
        
        # Rest between batches
        time.sleep(10)
```

## Configuration & Customization

### 1. Custom User Agents
```python
# For Britannica scraper
custom_agents = [
    "Your-Custom-User-Agent/1.0",
    "Another-Custom-Agent/2.0"
]

# Modify the USER_AGENTS list in _britannica.py
```

### 2. Custom Headers
```python
# For Wikipedia scraper
custom_headers = {
    "User-Agent": "Your-Research-Bot/1.0",
    "Accept": "text/html,application/xhtml+xml",
    "Accept-Language": "en-US,en;q=0.9"
}
```

### 3. Output Formatting
```python
# Custom text processing
def clean_text(text):
    """Custom text cleaning function"""
    import re
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove special characters
    text = re.sub(r'[^\w\s.,!?-]', '', text)
    
    return text.strip()
```

## Troubleshooting

### Common Issues

#### 1. YouTube API Quota Exceeded
```python
# Solution: Use multiple API keys or reduce max_results
yt = Youtube(api_key="key1", filepath="output.txt", max_results=25)
```

#### 2. Rate Limiting (429 Errors)
```python
# Solution: Increase delays or reduce concurrent requests
# The scrapers have built-in retry logic with exponential backoff
```

#### 3. Missing Transcripts
```python
# YouTube videos without captions won't be scraped
# Check logs for specific video IDs that failed
```

#### 4. Network Timeouts
```python
# Solution: Add custom timeout handling
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

session = requests.Session()
retry_strategy = Retry(total=3, backoff_factor=1)
adapter = HTTPAdapter(max_retries=retry_strategy)
session.mount("http://", adapter)
session.mount("https://", adapter)
```

## Performance Tips

### 1. Monitor Resource Usage
```python
import psutil
import os

def monitor_usage():
    process = psutil.Process(os.getpid())
    print(f"Memory usage: {process.memory_info().rss / 1024 / 1024:.2f} MB")
    print(f"CPU usage: {process.cpu_percent()}%")

# Call periodically during scraping
```

### 2. Optimize File I/O
```python
# Use buffered writing for better performance
class BufferedWriter:
    def __init__(self, filepath, buffer_size=1000):
        self.filepath = filepath
        self.buffer = []
        self.buffer_size = buffer_size
    
    def write(self, text):
        self.buffer.append(text)
        if len(self.buffer) >= self.buffer_size:
            self.flush()
    
    def flush(self):
        with open(self.filepath, 'a', encoding='utf-8') as f:
            f.write('\n'.join(self.buffer))
        self.buffer = []
```

### 3. Parallel Processing
```python
from concurrent.futures import ThreadPoolExecutor
import threading

def parallel_scrape(queries, max_workers=3):
    """Scrape multiple topics in parallel"""
    
    def scrape_topic(topic):
        wiki = Wikipedia(f"output/{topic.replace(' ', '_')}.txt")
        wiki([topic])
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        executor.map(scrape_topic, queries)
```

## Legal & Ethical Considerations

### 1. Respect robots.txt
Always check the website's robots.txt file before scraping:
- Wikipedia: https://en.wikipedia.org/robots.txt
- Britannica: https://www.britannica.com/robots.txt

### 2. Rate Limiting
- Don't overwhelm servers with rapid requests
- Use appropriate delays between requests
- Monitor for 429 (rate limit) responses

### 3. Terms of Service
- Review each platform's Terms of Service
- YouTube: https://www.youtube.com/terms
- Wikipedia: https://en.wikipedia.org/wiki/Wikipedia:Terms_of_use
- Britannica: Check their terms of service

### 4. Data Usage
- Respect copyright and fair use guidelines
- Attribute sources appropriately
- Don't redistribute copyrighted content without permission

## Advanced Usage

### 1. Custom Query Generation
```python
def generate_academic_queries():
    """Generate queries for academic research"""
    subjects = ["machine learning", "quantum computing", "biotechnology"]
    aspects = ["history", "applications", "future trends", "challenges"]
    
    queries = []
    for subject in subjects:
        for aspect in aspects:
            queries.append(f"{subject} {aspect}")
    
    return queries

custom_queries = generate_academic_queries()
```

### 2. Data Quality Validation
```python
def validate_scraped_content(filepath):
    """Validate the quality of scraped content"""
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Check content length
    if len(content) < 1000:
        print("Warning: Content seems too short")
    
    # Check for common error indicators
    error_indicators = ["404", "not found", "access denied", "error"]
    for indicator in error_indicators:
        if indicator.lower() in content.lower():
            print(f"Warning: Found error indicator: {indicator}")
    
    # Basic statistics
    word_count = len(content.split())
    print(f"Total words: {word_count}")
    print(f"Total characters: {len(content)}")
```

### 3. Content Preprocessing
```python
def preprocess_scraped_data(input_file, output_file):
    """Clean and preprocess scraped content"""
    import re
    
    with open(input_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Remove extra whitespace
    content = re.sub(r'\s+', ' ', content)
    
    # Remove URLs
    content = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', content)
    
    # Remove email addresses
    content = re.sub(r'\S+@\S+', '', content)
    
    # Split into sentences and filter
    sentences = content.split('.')
    filtered_sentences = [s.strip() for s in sentences if len(s.strip()) > 20]
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(filtered_sentences))
```

## Support

For issues, feature requests, or contributions:
- GitHub: https://github.com/delveopers/CookbookLLM
- Documentation: Check the source code for detailed docstrings
- Logs: Check generated log files for debugging information