import requests
from bs4 import BeautifulSoup
from transformers import pipeline
from src.predictor import get_price_data, predict_stock_trend

# Load LLM summarizer and sentiment model
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
sentiment_model = pipeline("sentiment-analysis", model="yiyanghkust/finbert-tone")

def extract_news_text(url):
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        response = requests.get(url, headers=headers)
        soup = BeautifulSoup(response.content, "html.parser")
        paragraphs = soup.find_all("p")
        text = " ".join(p.text for p in paragraphs if len(p.text) > 50)
        return text.strip()
    except Exception as e:
        print(f"Failed to extract from {url}: {e}")
        return ""

def fetch_google_news(ticker, limit=20):
    from googlesearch import search
    query = f"{ticker} stock site:finance.yahoo.com"
    return list(search(query, num=limit, stop=limit))

def summarize_stock(ticker):
    try:
        # Step 1: Get price trend and prediction
        prices = get_price_data(ticker)
        prediction = predict_stock_trend(prices)

        # Step 2: Fetch and extract news articles
        urls = fetch_google_news(ticker, limit=20)
        news_articles = [extract_news_text(url) for url in urls]
        news_articles = [article for article in news_articles if article]

        if not news_articles:
            return {"error": "No news content found."}

        # Step 3: Sentiment analysis
        sentiments = sentiment_model([t[:512] for t in news_articles])
        counts = {"POSITIVE": 0, "NEGATIVE": 0, "NEUTRAL": 0}
        for res in sentiments:
            counts[res["label"]] += 1
        dominant_sentiment = max(counts, key=counts.get)

        # Step 4: Compose prompt for LLM summarizer
        combined_news = " ".join(news_articles[:10])
        prompt = f"""
Stock: {ticker.upper()}
Sentiment: {dominant_sentiment.lower()}
Predicted Signal: {prediction.lower()}

Recent news:
{combined_news}

Please summarize the overall trend and events affecting {ticker.upper()}, and give an investment suggestion (BUY, HOLD, or SELL) with reasoning.
"""

        result = summarizer(prompt, max_length=180, min_length=60, do_sample=False)
        return {
            "summary": result[0]["summary_text"],
            "sentiment": dominant_sentiment,
            "prediction": prediction,
            "article_count": len(news_articles)
        }

    except Exception as e:
        return {"error": str(e)}
