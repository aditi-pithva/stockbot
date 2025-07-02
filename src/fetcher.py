# src/fetcher.py

import yfinance as yf
import requests
from bs4 import BeautifulSoup
import re
import unicodedata

def clean_text(text):
    """
    Clean text by removing unicode artifacts, extra whitespace, and formatting issues
    """
    if not text:
        return ""
    
    # First, try to fix common encoding issues
    try:
        # Handle common UTF-8 encoding issues
        text = text.replace('Ã¢', '—')  # Em dash
        text = text.replace('Ã¢Â€Â™', "'")  # Apostrophe
        text = text.replace('Ã¢Â€Â"', '—')  # Em dash
        text = text.replace('Ã¢Â€Â¦', '...')  # Ellipsis
        text = text.replace('Ã¢Â€Âœ', '"')  # Left quote
        text = text.replace('Ã¢Â€Â', '"')   # Right quote
    except:
        pass
    
    # Decode unicode escape sequences
    try:
        text = text.encode().decode('unicode_escape')
    except:
        pass
    
    # Remove unicode non-breaking spaces and other problematic characters
    text = text.replace('\u00a0', ' ')  # Non-breaking space
    text = text.replace('\u2013', '-')  # En dash
    text = text.replace('\u2014', '—')  # Em dash
    text = text.replace('\u2019', "'")  # Right single quotation mark
    text = text.replace('\u201c', '"')  # Left double quotation mark
    text = text.replace('\u201d', '"')  # Right double quotation mark
    text = text.replace('\u2026', '...')  # Horizontal ellipsis
    text = text.replace('\u2010', '-')  # Hyphen
    text = text.replace('\u2011', '-')  # Non-breaking hyphen
    
    # Fix common HTML entities that might leak through
    text = text.replace('&nbsp;', ' ')
    text = text.replace('&amp;', '&')
    text = text.replace('&lt;', '<')
    text = text.replace('&gt;', '>')
    text = text.replace('&quot;', '"')
    text = text.replace('&#39;', "'")
    
    # Remove other unicode control characters
    text = ''.join(char for char in text if unicodedata.category(char)[0] != 'C')
    
    # Clean up extra whitespace and line breaks
    text = re.sub(r'\s+', ' ', text)  # Multiple spaces to single space
    text = re.sub(r'\n+', ' ', text)  # Multiple newlines to space
    text = re.sub(r'\t+', ' ', text)  # Tabs to space
    
    # Remove repeated punctuation
    text = re.sub(r'\.{3,}', '...', text)  # Multiple dots to ellipsis
    text = re.sub(r'-{2,}', '—', text)     # Multiple dashes to em dash
    
    # Clean up timestamp patterns and source attributions
    text = re.sub(r'\d+\s*(hours?|minutes?|days?|weeks?)\s*ago', '', text, flags=re.IGNORECASE)
    text = re.sub(r'(Yahoo Finance|Reuters|Bloomberg|CNN|CNBC|MarketWatch|Seeking Alpha|More|GMT|EST|PST)', '', text, flags=re.IGNORECASE)
    
    # Remove common web artifacts
    text = re.sub(r'(Read more|Continue reading|Click here)', '', text, flags=re.IGNORECASE)
    text = re.sub(r'(\(NASDAQ:\w+\)|\(NYSE:\w+\))', lambda m: m.group(1), text)  # Keep ticker symbols
    
    # Remove extra spaces again after cleaning
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def fetch_news(ticker, num_articles=10):
    """
    Scrapes top news headlines for the given ticker using multiple sources.
    Filters news to ensure it's specifically about the requested ticker.
    """
    # Map common tickers to company names for better search
    ticker_to_company = {
        'TSLA': 'Tesla',
        'AAPL': 'Apple',
        'GOOGL': 'Google Alphabet',
        'MSFT': 'Microsoft',
        'AMZN': 'Amazon',
        'META': 'Meta Facebook',
        'NVDA': 'NVIDIA',
        'AMD': 'AMD',
        'NFLX': 'Netflix',
        'PYPL': 'PayPal',
        'CRM': 'Salesforce',
        'UBER': 'Uber',
        'ABNB': 'Airbnb',
        'HOOD': 'Robinhood',
        'PLTR': 'Palantir',
        'NKE': 'Nike',
        'CNC': 'Centene'
    }
    
    # Get company name or use ticker
    company_name = ticker_to_company.get(ticker.upper(), ticker)
    
    # Search for ticker-specific news
    search_terms = [
        f"{ticker} stock",
        f"{company_name} stock",
        f"{ticker} earnings",
        f"{company_name} news"
    ]
    
    all_articles = []
    
    for search_term in search_terms:
        try:
            query = search_term.replace(' ', '+')
            url = f"https://news.google.com/search?q={query}&hl=en-US&gl=US&ceid=US:en"
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            
            response = requests.get(url, headers=headers, timeout=10)
            soup = BeautifulSoup(response.text, 'lxml')
            
            # Extract article headlines and snippets
            articles = soup.select("article")
            for article in articles[:5]:  # Limit per search term
                raw_text = article.get_text().strip()
                cleaned_text = clean_text(raw_text)
                
                if cleaned_text and len(cleaned_text) > 20:  # Filter out very short text
                    # Check if article is actually about the ticker/company
                    text_upper = cleaned_text.upper()
                    if (ticker.upper() in text_upper or 
                        company_name.upper() in text_upper or
                        any(word.upper() in text_upper for word in company_name.split())):
                        all_articles.append(cleaned_text)
            
        except Exception as e:
            print(f"⚠️ Error fetching news for {search_term}: {e}")
            continue
    
    # Remove duplicates and filter further
    unique_articles = []
    seen_articles = set()
    
    for article in all_articles:
        # Clean the article text again to ensure consistency
        cleaned_article = clean_text(article)
        
        # Create a simple hash to check for duplicates
        article_hash = cleaned_article[:100].lower().replace(' ', '')
        if article_hash not in seen_articles and len(cleaned_article) > 30:
            seen_articles.add(article_hash)
            unique_articles.append(cleaned_article)
    
    # Final filtering: ensure articles are relevant
    filtered_articles = []
    ticker_upper = ticker.upper()
    company_words = company_name.upper().split()
    
    for article in unique_articles:
        article_upper = article.upper()
        # Check if article mentions ticker or company name prominently
        relevance_score = 0
        
        # High relevance: ticker symbol mentioned
        if ticker_upper in article_upper:
            relevance_score += 3
        
        # Medium relevance: company name mentioned
        for word in company_words:
            if len(word) > 2 and word in article_upper:
                relevance_score += 1
        
        # Filter out general market news
        general_terms = ['DOW JONES', 'S&P 500', 'NASDAQ', 'MARKET MOVES', 'STOCKS MAKING']
        is_general = any(term in article_upper for term in general_terms)
        
        # Only include if relevance score is high enough and not general market news
        if relevance_score >= 2 and not is_general:
            filtered_articles.append(article)
        elif relevance_score >= 4:  # Very high relevance, include even if general terms present
            filtered_articles.append(article)
    
    # Return top articles
    result = filtered_articles[:num_articles]
    
    if result:
        print(f"✅ Found {len(result)} ticker-specific articles for {ticker}")
    else:
        print(f"⚠️ No specific articles found for {ticker}, trying fallback search...")
        # Fallback: try Yahoo Finance news
        result = fetch_yahoo_news(ticker, num_articles)
    
    return result

def fetch_price_data(ticker, period="1mo"):
    """
    Fetch historical closing price data for a stock over a period.
    """
    try:
        return yf.Ticker(ticker).history(period=period)['Close']
    except Exception as e:
        print(f"❌ Error fetching price data for {ticker}: {e}")
        return None

def fetch_index_price(ticker):
    """
    Fetch the latest closing price for a stock index (e.g., ^GSPC).
    """
    try:
        return f"{yf.Ticker(ticker).history(period='1d')['Close'][-1]:,.2f}"
    except Exception as e:
        print(f"❌ Error fetching index price for {ticker}: {e}")
        return "N/A"

def fetch_stock_data(ticker):
    """
    Fetch last 7 days of OHLCV data (Open, High, Low, Close, Volume).
    Used for generating 25-feature input for prediction.
    """
    try:
        df = yf.Ticker(ticker).history(period="7d", interval="1d")
        if df.empty or len(df) < 5:
            print("❌ Not enough data returned for prediction.")
            return None
        return df[["Open", "High", "Low", "Close", "Volume"]]
    except Exception as e:
        print(f"❌ Error fetching stock data for {ticker}: {e}")
        return None

def fetch_yahoo_news(ticker, num_articles=5):
    """
    Fallback news fetcher using Yahoo Finance news for the ticker.
    """
    try:
        import yfinance as yf
        
        # Get company info and news from yfinance
        stock = yf.Ticker(ticker)
        news = stock.news
        
        articles = []
        for item in news[:num_articles]:
            title = item.get('title', '')
            summary = item.get('summary', '')
            
            # Combine title and summary
            article_text = f"{title}. {summary}" if summary else title
            
            # Clean the text to remove formatting artifacts
            cleaned_text = clean_text(article_text)
            
            if cleaned_text and len(cleaned_text.strip()) > 20:
                articles.append(cleaned_text.strip())
        
        print(f"✅ Found {len(articles)} Yahoo Finance articles for {ticker}")
        return articles
        
    except Exception as e:
        print(f"⚠️ Yahoo Finance fallback failed for {ticker}: {e}")
        return []
