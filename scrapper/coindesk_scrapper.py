import csv
import re
from bs4 import BeautifulSoup
import requests

###
coin = "ethereum"
filename = "../data/coindesk_eth.csv"
base_url = "https://www.coindesk.com/"
coin_url = "tag/ethereum/"
max_page = 369
###

# Function to fetch and parse a page
def fetch_page(url):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.content, 'html.parser')
    return soup

# Extract URLs based on the href
# in the format: 'market/2024/title' and so on
def extract_relevant_links(soup):
    urls = set()
    pattern = re.compile(r'/((markets|business|tech)/\d{4})/')
    for link in soup.find_all('a'):
        href = link.get('href')
        if href and pattern.search(href):
            urls.add(href)
    return urls

def fetch_article(url):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.content, 'html.parser')

    # Example: Adjust the following based on actual HTML structure of articles
    title = soup.find('h1').get_text()
    #pub_date = soup.find('time')['datetime']
    content = ' '.join([p.get_text() for p in soup.find_all('p')])

    year_match = re.search(r'/(\d{4})/', url)
    article_year = year_match.group(1) if year_match else 'Unknown year'
    
    return {
        'title': title,
        'year': article_year,
        'content': content,
        'url': url
    }

def save_to_csv(articles, filename):
    keys = articles[0].keys()
    with open(filename, 'a', newline='', encoding='utf-8') as output_file:
        dict_writer = csv.DictWriter(output_file, fieldnames=keys)
        # Check if the file is empty to write the header
        if output_file.tell() == 0:
            dict_writer.writeheader()
        dict_writer.writerows(articles)

# Loop through paginated pages
for page in range(1,max_page):
  print("page:", page)
  # Scrape articles from the filtered URLs
  articles = []

  url = f"{base_url + coin_url}/{page}/"    

  soup = fetch_page(url)
  relevant_urls = list(extract_relevant_links(soup))

  for relative_url in relevant_urls: 
    # Ensure the URL is properly constructed
    if relative_url.startswith('http'):
        full_url = relative_url
    else:
        full_url = base_url + relative_url.lstrip('/')

    print(full_url)
    
    try:
        article = fetch_article(full_url)
        articles.append(article)
    except Exception as e:
        print(f"Failed to fetch {full_url}: {e}")
  
  # Print the scraped articles
  # for article in articles:
  #     print(f"Title: {article['title']}")
  #     print(f"Published: {article['year']}")
  #     print(f"URL: {article['url']}")
  #     print(f"Content: {article['content'][:200]}...")  # Print first 200 characters of content
  #     print("\n")

  if articles:
    save_to_csv(articles, filename)

  

