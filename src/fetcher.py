import yfinance as yf
import requests
from bs4 import BeautifulSoup

def fetch_news(company_name, num_articles=10):
    query = company_name.replace(' ', '+')
    url = f"https://news.google.com/search?q={query}+stock&hl=en-US&gl=US&ceid=US:en"
    headers = {'User-Agent': 'Mozilla/5.0'}
    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.text, 'lxml')
    return [item.text.strip() for item in soup.select("article")[:num_articles]]

def fetch_price_data(ticker, period="1mo"):
    try:
        return yf.Ticker(ticker).history(period=period)['Close']
    except:
        return None

def fetch_index_price(ticker):
    try:
        return f"{yf.Ticker(ticker).history(period='1d')['Close'][-1]:,.2f}"
    except:
        return "N/A"
