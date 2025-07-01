#!/usr/bin/env python3
"""
Pre-download script for ML models
This script downloads all required models to the local models directory
"""

import os
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / 'src'))

def download_models():
    """Download all required models to local cache"""
    print("🚀 Starting model download process...")
    
    # Check if required packages are installed
    try:
        import sentence_transformers
        import transformers
        print("✅ Required packages found")
    except ImportError as e:
        print(f"❌ Missing required packages: {e}")
        print("💡 Please run: pip install -r requirements.txt")
        return False
    
    # Set up directories
    models_dir = Path(__file__).parent / "models"
    models_dir.mkdir(exist_ok=True)
    
    transformers_cache = models_dir / "transformers_cache"
    sentence_transformers_cache = models_dir / "sentence_transformers"
    
    transformers_cache.mkdir(exist_ok=True)
    sentence_transformers_cache.mkdir(exist_ok=True)
    
    # Set environment variables
    os.environ["TRANSFORMERS_CACHE"] = str(transformers_cache)
    os.environ["HF_HOME"] = str(transformers_cache)
    os.environ["SENTENCE_TRANSFORMERS_HOME"] = str(sentence_transformers_cache)
    
    try:
        # Download SentenceTransformers model
        print("📥 Downloading SentenceTransformers model...")
        from sentence_transformers import SentenceTransformer
        embedder = SentenceTransformer("all-MiniLM-L6-v2", cache_folder=str(sentence_transformers_cache))
        print("✅ SentenceTransformers model downloaded successfully")
        
        # Download FinBERT sentiment model
        print("📥 Downloading FinBERT sentiment model...")
        from transformers import pipeline
        sentiment_model = pipeline("sentiment-analysis", 
                                 model="yiyanghkust/finbert-tone",
                                 model_kwargs={"cache_dir": str(transformers_cache)})
        print("✅ FinBERT sentiment model downloaded successfully")
        
        # Download BART summarization model
        print("📥 Downloading BART summarization model...")
        summarizer = pipeline("summarization", 
                            model="facebook/bart-large-cnn",
                            model_kwargs={"cache_dir": str(transformers_cache)})
        print("✅ BART summarization model downloaded successfully")
        
        print("🎉 All models downloaded successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error downloading models: {e}")
        print("🔄 Models will be downloaded automatically when the app starts")
        return False

if __name__ == "__main__":
    success = download_models()
    sys.exit(0 if success else 1)
