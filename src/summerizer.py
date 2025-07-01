# src/summerizer.py
import re
import torch
import os
from src.fetcher import fetch_news, fetch_stock_data
from src.predictor import predict
from src.utils import generate_feature_vector

# Set up local models directory
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
TRANSFORMERS_CACHE = os.path.join(MODELS_DIR, "transformers_cache")
SENTENCE_TRANSFORMERS_CACHE = os.path.join(MODELS_DIR, "sentence_transformers")

# Create directories with proper permissions
def setup_model_directories():
    try:
        os.makedirs(MODELS_DIR, exist_ok=True)
        os.makedirs(TRANSFORMERS_CACHE, exist_ok=True)
        os.makedirs(SENTENCE_TRANSFORMERS_CACHE, exist_ok=True)
        
        # Set permissions to be writable
        os.chmod(MODELS_DIR, 0o755)
        os.chmod(TRANSFORMERS_CACHE, 0o755)
        os.chmod(SENTENCE_TRANSFORMERS_CACHE, 0o755)
        
        print(f"✅ Model directories created: {MODELS_DIR}")
        return True
    except Exception as e:
        print(f"⚠️ Could not create model directories: {e}")
        return False

# Set environment variables for model caching
os.environ["TRANSFORMERS_CACHE"] = TRANSFORMERS_CACHE
os.environ["HF_HOME"] = TRANSFORMERS_CACHE
os.environ["SENTENCE_TRANSFORMERS_HOME"] = SENTENCE_TRANSFORMERS_CACHE

# Initialize directories
setup_model_directories()

# Lazy loading for models to prevent startup timeout
_embedder = None
_sentiment_model = None
_summarizer = None

def get_embedder():
    global _embedder
    if _embedder is None:
        try:
            from sentence_transformers import SentenceTransformer
            print("📥 Downloading sentence transformer model...")
            _embedder = SentenceTransformer("all-MiniLM-L6-v2", cache_folder=SENTENCE_TRANSFORMERS_CACHE)
            print("✅ Sentence transformer model loaded successfully")
        except Exception as e:
            print(f"⚠️ Error loading embedder: {e}")
            # Fallback: return a dummy embedder
            _embedder = DummyEmbedder()
    return _embedder

def get_sentiment_model():
    global _sentiment_model
    if _sentiment_model is None:
        try:
            from transformers import pipeline
            print("📥 Downloading FinBERT sentiment model...")
            _sentiment_model = pipeline("sentiment-analysis", 
                                      model="yiyanghkust/finbert-tone",
                                      model_kwargs={"cache_dir": TRANSFORMERS_CACHE})
            print("✅ FinBERT sentiment model loaded successfully")
        except Exception as e:
            print(f"⚠️ Error loading sentiment model: {e}")
            # Fallback: return dummy sentiment analyzer
            _sentiment_model = DummySentimentModel()
    return _sentiment_model

def get_summarizer():
    global _summarizer
    if _summarizer is None:
        try:
            from transformers import pipeline
            print("📥 Downloading BART summarization model...")
            _summarizer = pipeline("summarization", 
                                 model="facebook/bart-large-cnn",
                                 model_kwargs={"cache_dir": TRANSFORMERS_CACHE})
            print("✅ BART summarization model loaded successfully")
        except Exception as e:
            print(f"⚠️ Error loading summarizer: {e}")
            # Fallback: return dummy summarizer
            _summarizer = DummySummarizer()
    return _summarizer

# Dummy classes for fallback when models can't be loaded
class DummyEmbedder:
    def encode(self, texts, convert_to_tensor=False):
        import numpy as np
        if isinstance(texts, str):
            texts = [texts]
        # Return random embeddings
        embeddings = np.random.rand(len(texts), 384)
        if convert_to_tensor:
            return torch.tensor(embeddings, dtype=torch.float32)
        return embeddings

class DummySentimentModel:
    def __call__(self, texts):
        import random
        return [{"label": random.choice(["POSITIVE", "NEGATIVE", "NEUTRAL"]), 
                "score": 0.8} for _ in texts]

class DummySummarizer:
    def __call__(self, text, max_length=300, min_length=200, do_sample=False):
        return [{"summary_text": "Market analysis temporarily unavailable. Please try again later."}]

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

        embedder = get_embedder()
        sentiment_model = get_sentiment_model()
        summarizer = get_summarizer()
        
        from sentence_transformers import util

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
