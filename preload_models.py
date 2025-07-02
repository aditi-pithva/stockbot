#!/usr/bin/env python3
"""
Preload AI models for StockBot to avoid delays during first use.
Run this script after installation to download and cache models.
"""

import os
import sys
import warnings
import threading
import time

# Suppress warnings during model loading
warnings.filterwarnings("ignore")

def print_progress(message):
    print(f"⏳ {message}")

def preload_sentence_transformer():
    """Preload sentence transformer model"""
    try:
        print_progress("Downloading Sentence Transformer model...")
        from sentence_transformers import SentenceTransformer
        
        # Set up cache directory
        models_dir = os.path.join(os.path.dirname(__file__), "models")
        cache_dir = os.path.join(models_dir, "sentence_transformers")
        os.makedirs(cache_dir, exist_ok=True)
        
        model = SentenceTransformer("all-MiniLM-L6-v2", 
                                   cache_folder=cache_dir,
                                   device='cpu')
        
        # Test the model
        test_embedding = model.encode("Test sentence", convert_to_tensor=True)
        print("✅ Sentence Transformer model ready")
        return True
        
    except Exception as e:
        print(f"❌ Failed to load Sentence Transformer: {e}")
        return False

def preload_sentiment_model():
    """Preload FinBERT sentiment model"""
    try:
        print_progress("Downloading FinBERT sentiment model...")
        from transformers import pipeline
        
        # Set up cache directory
        models_dir = os.path.join(os.path.dirname(__file__), "models")
        cache_dir = os.path.join(models_dir, "transformers_cache")
        os.makedirs(cache_dir, exist_ok=True)
        
        sentiment_model = pipeline("sentiment-analysis", 
                                 model="yiyanghkust/finbert-tone",
                                 model_kwargs={"cache_dir": cache_dir},
                                 device=-1)
        
        # Test the model
        test_result = sentiment_model("The stock market is performing well today")
        print("✅ FinBERT sentiment model ready")
        return True
        
    except Exception as e:
        print(f"❌ Failed to load FinBERT: {e}")
        return False

def preload_summarizer_model():
    """Preload BART summarization model"""
    try:
        print_progress("Downloading BART summarization model...")
        from transformers import pipeline
        
        # Set up cache directory
        models_dir = os.path.join(os.path.dirname(__file__), "models")
        cache_dir = os.path.join(models_dir, "transformers_cache")
        os.makedirs(cache_dir, exist_ok=True)
        
        summarizer = pipeline("summarization", 
                             model="facebook/bart-large-cnn",
                             model_kwargs={"cache_dir": cache_dir},
                             device=-1)
        
        # Test the model
        test_text = "The stock market has been volatile recently. Many investors are concerned about inflation and interest rates. Companies are reporting mixed earnings results."
        test_result = summarizer(test_text, max_length=50, min_length=20, do_sample=False)
        print("✅ BART summarization model ready")
        return True
        
    except Exception as e:
        print(f"❌ Failed to load BART: {e}")
        return False

def preload_with_timeout(func, timeout=120):
    """Run model loading with timeout"""
    result = [False]
    exception = [None]
    
    def target():
        try:
            result[0] = func()
        except Exception as e:
            exception[0] = e
    
    thread = threading.Thread(target=target)
    thread.daemon = True
    thread.start()
    thread.join(timeout=timeout)
    
    if thread.is_alive():
        print(f"⚠️ {func.__name__} timed out after {timeout} seconds")
        return False
    elif exception[0]:
        print(f"⚠️ {func.__name__} failed: {exception[0]}")
        return False
    else:
        return result[0]

def main():
    """Main preloading function"""
    print("🚀 StockBot Model Preloader")
    print("=" * 50)
    print("This will download and cache AI models for faster startup.")
    print("This may take 5-10 minutes depending on your internet speed.\n")
    
    # Check if user wants to continue
    try:
        response = input("Continue? (y/N): ").strip().lower()
        if response not in ['y', 'yes']:
            print("Canceled.")
            return
    except KeyboardInterrupt:
        print("\nCanceled.")
        return
    
    models_loaded = 0
    total_models = 3
    
    print("\n📥 Starting model downloads...\n")
    
    # Load models with timeouts
    if preload_with_timeout(preload_sentence_transformer, 180):
        models_loaded += 1
    
    if preload_with_timeout(preload_sentiment_model, 180):
        models_loaded += 1
    
    if preload_with_timeout(preload_summarizer_model, 180):
        models_loaded += 1
    
    print(f"\n🎉 Preloading complete! {models_loaded}/{total_models} models loaded successfully.")
    
    if models_loaded == total_models:
        print("✅ All models ready! StockBot will start much faster now.")
    elif models_loaded > 0:
        print("⚠️ Some models failed to load but StockBot will use fallbacks.")
    else:
        print("❌ No models loaded. StockBot will use fallback implementations.")
    
    print("\nYou can now run: streamlit run app.py")

if __name__ == "__main__":
    main()
