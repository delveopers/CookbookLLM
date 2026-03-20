"""
@brief:
A Python module for scraping article content from Encyclopedia Britannica based on user-defined queries.

This script allows automated extraction of article text by simulating keyword searches and navigating result pages 
from the Britannica website (https://www.britannica.com). It supports paginated URL crawling, dynamic user-agent 
rotation to prevent throttling, and extracts the clean textual content from article pages.

Classes:
--------
- `Britannica`:
    Main interface class to perform scraping. Instantiate with output filepath and configuration options.
    Call the object with a list of queries to initiate the scrape.

Functions:
----------
- `build_britannica_url(query: str, page_no: int) -> str`:
    Constructs a Britannica search URL for a given query and page number.

- `get_target_url(target_url: str, headers: dict) -> list[str]`:
    Fetches the result page and extracts links to individual article pages.

Example:
--------
```python
from britannica_scraper import Britannica

scraper = Britannica(filepath="results/britannica_articles.txt", max_limit=5, metrics=True)
scraper(["Black Hole", "Photosynthesis", "Relativity"])
```

@author: @shivendrra
@src: https://github.com/shivendrra/web-graze/blob/main/graze/_britannica.py
@license: MIT
@git-repo: https://github.com/shivendrra/web-graze
"""

import os, logging, requests, timeit, time, re, random
from bs4 import BeautifulSoup
from tqdm import tqdm

logging.basicConfig(filename="britannica_scraper.log", level=logging.ERROR)

def build_britannica_url(query, page_no):
  formatted_query = '%20'.join(query.split())
  return f"https://www.britannica.com/search?query={formatted_query}&page={page_no}"

def get_target_url(target_url, headers, max_retries=5):
  for _ in range(max_retries):
    r = requests.get(target_url, headers=headers)
    if r.status_code == 200:
      soup = BeautifulSoup(r.content, 'html.parser')
      fetched = soup.find_all('a', href=True)
      urls = []
      for a in fetched:
        href = a.get('href')
        if href and href.startswith(('/topic/', '/science/', '/technology/', '/arts-culture/')):
          urls.append(href)
      return list(dict.fromkeys(urls))
    elif r.status_code == 429:
      time.sleep(random.uniform(10, 30))
    else:
      return []
  return []

USER_AGENTS = [
  "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.114 Safari/537.36",
]

class Britannica:
  def __init__(self, filepath:str, max_limit:int=10, metrics:bool=False) -> None:
    self.directory, filename_with_ext = os.path.split(filepath)
    self.filename, _ = os.path.splitext(filename_with_ext)
    if self.directory and not os.path.exists(self.directory):
      os.makedirs(self.directory, exist_ok=True)
    self.max_limit = max_limit
    self.headers = {
      'User-Agent': random.choice(USER_AGENTS),
      'Referer': 'https://www.google.com/',
      'Accept-Language': 'en-US,en;q=0.9',
      'Accept-Encoding': 'gzip, deflate, br',
    }
    self.metrics = metrics
    self.total_urls = 0
    self.total_pages = 0

  def __call__(self, queries:list[str]):
    if not queries:
      raise ValueError("Search queries can't be empty.")
    self.total_time = timeit.default_timer()
    for query in tqdm(queries, desc="Generating Britannica URLs"):
      page_no = 1
      for _ in range(self.max_limit):
        target_url = build_britannica_url(query, page_no)
        new_urls = get_target_url(target_url, self.headers)
        if not new_urls:
          break
        self.write_urls_to_file(new_urls)
        self.total_urls += len(new_urls)
        page_no += 1
        self.headers['User-Agent'] = random.choice(USER_AGENTS)
        time.sleep(random.uniform(1, 3))
    self.total_time = timeit.default_timer() - self.total_time
    if self.metrics:
      self.get_metrics()

  def text_extractor(self, url_snippet):
    target_url = f"https://www.britannica.com{url_snippet}" if url_snippet.startswith('/') else url_snippet
    try:
      r = requests.get(target_url, headers=self.headers)
    except Exception as e:
      logging.error(f"Request error for {target_url}: {e}")
      return None
    if r.status_code == 200:
      soup = BeautifulSoup(r.content, 'html.parser')
      paragraphs = soup.find_all('p')
      page_text = []
      for p in paragraphs:
        txt = p.get_text().strip()
        if "Our editors will review what you've submitted" in txt:
          continue
        if txt:
          page_text.append(txt)
      page = '\n'.join(page_text)
      page = re.sub(r'&[a-zA-Z]+;', '', page)
      self.total_pages += 1
      return page
    return None

  def write_urls_to_file(self, url_snippets):
    filepath = os.path.join(self.directory, f"{self.filename}.txt") if self.directory else f"{self.filename}.txt"
    with open(filepath, 'a', encoding='utf-8') as f:
      for snippet in url_snippets:
        page = self.text_extractor(snippet)
        if page:
          f.write(page + "\n")

  def get_metrics(self):
    print("\nBritannica scraping metrics:\n")
    print("------------------------------------------------------")
    print(f"Total URLs fetched: {self.total_urls}")
    print(f"Total pages extracted: {self.total_pages}")
    if self.total_time < 60:
      print(f"Total time taken: {self.total_time:.2f} seconds")
    elif self.total_time < 3600:
      print(f"Total time taken: {self.total_time/60:.2f} minutes")
    else:
      print(f"Total time taken: {self.total_time/3600:.2f} hours")
    print("------------------------------------------------------")
