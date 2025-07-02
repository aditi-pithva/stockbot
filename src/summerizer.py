# src/summerizer.py
import re
import torch
import os
import warnings
from src.fetcher import fetch_news, fetch_stock_data
from src.predictor import predict
from src.utils import generate_feature_vector

# Suppress specific warnings
warnings.filterwarnings("ignore", message=".*Could not cache non-existence of file.*")
warnings.filterwarnings("ignore", message=".*Device set to use cpu.*")

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
        
        # Set permissions to be writable (if possible)
        try:
            os.chmod(MODELS_DIR, 0o755)
            os.chmod(TRANSFORMERS_CACHE, 0o755)
            os.chmod(SENTENCE_TRANSFORMERS_CACHE, 0o755)
        except (OSError, PermissionError):
            # Ignore permission errors - directories may still be usable
            pass
        
        print(f"✅ Model directories created: {MODELS_DIR}")
        return True
    except Exception as e:
        print(f"⚠️ Could not create model directories: {e}")
        return False

# Set environment variables for model caching
os.environ["TRANSFORMERS_CACHE"] = TRANSFORMERS_CACHE
os.environ["HF_HOME"] = TRANSFORMERS_CACHE
os.environ["SENTENCE_TRANSFORMERS_HOME"] = SENTENCE_TRANSFORMERS_CACHE
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Suppress tokenizer warnings

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
            import threading
            import time
            print("📥 Loading sentence transformer model (this may take a moment)...")
            
            # Windows-compatible timeout using threading
            result = [None]
            exception = [None]
            
            def load_model():
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        result[0] = SentenceTransformer("all-MiniLM-L6-v2", 
                                                      cache_folder=SENTENCE_TRANSFORMERS_CACHE,
                                                      device='cpu')
                except Exception as e:
                    exception[0] = e
            
            thread = threading.Thread(target=load_model)
            thread.daemon = True
            thread.start()
            thread.join(timeout=30)  # 30 second timeout
            
            if thread.is_alive():
                print("⚠️ Model loading timeout - using fallback")
                _embedder = DummyEmbedder()
            elif exception[0]:
                print(f"⚠️ Error loading embedder: {exception[0]}")
                _embedder = DummyEmbedder()
            else:
                _embedder = result[0]
                print("✅ Sentence transformer model loaded successfully")
                
        except Exception as e:
            print(f"⚠️ Error loading embedder: {e}")
            _embedder = DummyEmbedder()
    return _embedder

def get_sentiment_model():
    global _sentiment_model
    if _sentiment_model is None:
        try:
            from transformers import pipeline
            import threading
            print("📥 Loading FinBERT sentiment model (this may take a moment)...")
            
            result = [None]
            exception = [None]
            
            def load_model():
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        result[0] = pipeline("sentiment-analysis", 
                                          model="yiyanghkust/finbert-tone",
                                          model_kwargs={"cache_dir": TRANSFORMERS_CACHE},
                                          device=-1)
                except Exception as e:
                    exception[0] = e
            
            thread = threading.Thread(target=load_model)
            thread.daemon = True
            thread.start()
            thread.join(timeout=45)  # 45 second timeout
            
            if thread.is_alive():
                print("⚠️ Sentiment model loading timeout - using fallback")
                _sentiment_model = DummySentimentModel()
            elif exception[0]:
                print(f"⚠️ Error loading sentiment model: {exception[0]}")
                _sentiment_model = DummySentimentModel()
            else:
                _sentiment_model = result[0]
                print("✅ FinBERT sentiment model loaded successfully")
                
        except Exception as e:
            print(f"⚠️ Error loading sentiment model: {e}")
            _sentiment_model = DummySentimentModel()
    return _sentiment_model

def get_summarizer():
    global _summarizer
    if _summarizer is None:
        try:
            from transformers import pipeline
            import threading
            print("📥 Loading BART summarization model (this may take a moment)...")
            
            result = [None]
            exception = [None]
            
            def load_model():
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        result[0] = pipeline("summarization", 
                                           model="facebook/bart-large-cnn",
                                           model_kwargs={"cache_dir": TRANSFORMERS_CACHE},
                                           device=-1)
                except Exception as e:
                    exception[0] = e
            
            thread = threading.Thread(target=load_model)
            thread.daemon = True
            thread.start()
            thread.join(timeout=60)  # 60 second timeout
            
            if thread.is_alive():
                print("⚠️ Summarizer model loading timeout - using fallback")
                _summarizer = DummySummarizer()
            elif exception[0]:
                print(f"⚠️ Error loading summarizer: {exception[0]}")
                _summarizer = DummySummarizer()
            else:
                _summarizer = result[0]
                print("✅ BART summarization model loaded successfully")
                
        except Exception as e:
            print(f"⚠️ Error loading summarizer: {e}")
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
            return {
                "ticker": ticker,
                "summary": f"No recent news found for {ticker}. Based on technical analysis, the model predicts: {prediction}",
                "prediction": prediction,
                "sentiment": "NEUTRAL",
                "article_count": 0,
                "used_articles": 0
            }

        # Add progress indicators and timeout handling
        print("🤖 Loading AI models for analysis...")
        
        try:
            embedder = get_embedder()
            print("✅ Text embedding model ready")
            
            sentiment_model = get_sentiment_model()
            print("✅ Sentiment analysis model ready")
            
            summarizer = get_summarizer()
            print("✅ Text summarization model ready")
            
        except Exception as e:
            print(f"⚠️ Model loading failed: {e}")
            # Return basic analysis without AI models
            return {
                "ticker": ticker,
                "summary": f"Technical analysis for {ticker}: {prediction}. AI models temporarily unavailable for detailed news analysis.",
                "prediction": prediction,
                "sentiment": "NEUTRAL",
                "article_count": len(news_articles),
                "used_articles": 0
            }
        
        print("🔍 Analyzing news relevance...")
        
        # Add timeout for analysis using threading
        import threading
        
        analysis_result = [None]
        analysis_exception = [None]
        
        def run_analysis():
            try:
                from sentence_transformers import util

                article_embeddings = embedder.encode(news_articles, convert_to_tensor=True)
                query_embedding = embedder.encode(user_query, convert_to_tensor=True)

                cosine_scores = util.cos_sim(query_embedding, article_embeddings)[0]
                top_indices = torch.topk(cosine_scores, k=min(3, len(news_articles))).indices
                top_articles = [news_articles[i] for i in top_indices]

                print("💭 Analyzing sentiment...")
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

                print("📝 Generating summary...")
                # Calculate appropriate max_length based on input length
                input_length = len(prompt.split())
                max_length = min(300, max(50, int(input_length * 0.5)))  # At most 50% of input length
                min_length = min(max_length - 10, 30)  # Ensure min_length < max_length
                
                # Suppress the max_length warning
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    summary_output = summarizer(prompt, 
                                              max_length=max_length, 
                                              min_length=min_length, 
                                              do_sample=False)
                
                summary_text = summary_output[0]["summary_text"]
                summary_text = re.sub(r"(Summarize|Based on).*?(BUY|HOLD|SELL)[).]*", "", summary_text, flags=re.IGNORECASE).strip()

                analysis_result[0] = {
                    "ticker": ticker,
                    "summary": summary_text,
                    "prediction": prediction,
                    "sentiment": dominant_sentiment,
                    "article_count": len(news_articles),
                    "used_articles": len(top_articles)
                }
                
            except Exception as e:
                analysis_exception[0] = e
        
        # Run analysis with timeout
        analysis_thread = threading.Thread(target=run_analysis)
        analysis_thread.daemon = True
        analysis_thread.start()
        analysis_thread.join(timeout=45)  # 45 seconds for analysis
        
        if analysis_thread.is_alive():
            print("⚠️ Analysis timeout - providing basic summary")
            return {
                "ticker": ticker,
                "summary": f"Quick analysis for {ticker}: Found {len(news_articles)} news articles. Technical prediction: {prediction}. Detailed AI analysis took too long - try again for full analysis.",
                "prediction": prediction,
                "sentiment": "NEUTRAL",
                "article_count": len(news_articles),
                "used_articles": 0
            }
        elif analysis_exception[0]:
            print(f"⚠️ Analysis error: {analysis_exception[0]}")
            return {
                "ticker": ticker,
                "summary": f"Basic analysis for {ticker}: Found {len(news_articles)} news articles. Technical prediction: {prediction}. AI analysis encountered an error.",
                "prediction": prediction,
                "sentiment": "NEUTRAL",
                "article_count": len(news_articles),
                "used_articles": 0
            }
        else:
            print("✅ Analysis complete!")
            return analysis_result[0]

    except Exception as e:
        print(f"❌ Error in generate_summary: {e}")
        return {"error": f"Summary generation failed: {e}"}
