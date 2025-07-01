# src/fetcher.py

import yfinance as yf
import requests
from bs4 import BeautifulSoup

def fetch_news(company_name, num_articles=10):
    """
    Scrapes top news headlines for the given company using Google News.
    """
    query = company_name.replace(' ', '+')
    url = f"https://news.google.com/search?q={query}+stock&hl=en-US&gl=US&ceid=US:en"
    headers = {'User-Agent': 'Mozilla/5.0'}
    
    try:
        response = requests.get(url, headers=headers, timeout=10)  # Added timeout
        soup = BeautifulSoup(response.text, 'lxml')
        return [item.text.strip() for item in soup.select("article")[:num_articles]]
    except Exception as e:
        print(f"❌ Error fetching news: {e}")
        return []

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
