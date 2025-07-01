# src/summerizer.py
import re
import torch
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline
from src.fetcher import fetch_news, fetch_stock_data
from src.predictor import predict
from src.utils import generate_feature_vector

# Load models
embedder = SentenceTransformer("all-MiniLM-L6-v2")
sentiment_model = pipeline("sentiment-analysis", model="yiyanghkust/finbert-tone")
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

def generate_summary(ticker, user_query="What's happening with this stock?"):
    try:
        print(f"\n📈 Fetching price data for: {ticker}")
        df = fetch_stock_data(ticker)
        if df is None or df.empty:
            print("❌ No price data available.")
            return {"error": "No price data available."}

        print("✅ Got price data")
        feature_vec = generate_feature_vector(df)
        prediction = predict(feature_vec)
        print(f"🤖 Model Prediction: {prediction}")

        print("📰 Fetching news articles...")
        news_articles = fetch_news(ticker)
        print(f"✅ News Articles Found: {len(news_articles)}")
        if not news_articles:
            return {"error": "No news found."}

        article_embeddings = embedder.encode(news_articles, convert_to_tensor=True)
        query_embedding = embedder.encode(user_query, convert_to_tensor=True)

        cosine_scores = util.cos_sim(query_embedding, article_embeddings)[0]
        top_indices = torch.topk(cosine_scores, k=3).indices
        top_articles = [news_articles[i] for i in top_indices]

        sentiments = sentiment_model([text[:512] for text in top_articles])
        sentiment_counts = {"POSITIVE": 0, "NEGATIVE": 0, "NEUTRAL": 0}
        for s in sentiments:
            label = s["label"].upper()
            if label in sentiment_counts:
                sentiment_counts[label] += 1
            else:
                print(f"⚠️ Unexpected sentiment label: {s['label']}")

        dominant_sentiment = max(sentiment_counts, key=sentiment_counts.get)

        annotated_articles = "\n\n".join(text for text in top_articles)

        prompt = (
        f"{annotated_articles}\n\n"
        f"Summarize how the news might impact {ticker.upper()} stock in simple terms."
        )

        summary_output = summarizer(prompt, max_length=300, min_length=200, do_sample=False)
        summary_text = summary_output[0]["summary_text"]
        summary_text = re.sub(r"(Summarize|Based on).*?(BUY|HOLD|SELL)[).]*", "", summary_text, flags=re.IGNORECASE).strip()

        return {
            "ticker": ticker,
            "summary": summary_text,
            "prediction": prediction,
            "sentiment": dominant_sentiment,
            "article_count": len(news_articles),
            "used_articles": len(top_articles)
        }

    except Exception as e:
        return {"error": f"Summary generation failed: {e}"}
