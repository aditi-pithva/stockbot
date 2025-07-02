---
title: StockBot
emoji: 🤖
colorFrom: yellow
colorTo: green
sdk: docker
app_port: 7860
tags:
- streamlit
- finance
- stocks
- chatbot
- machine-learning
pinned: false
short_description: AI-powered stock market chatbot with predictions
license: mit
---

# StockBot 📈

An AI-powered chatbot for real-time stock market analysis and predictions using machine learning.

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/your-username/stockbot.git
cd stockbot

# Install dependencies
pip install -r requirements.txt
```

### 2. Model Preloading (Recommended)

To avoid delays during first use, preload the AI models:

```bash
# Download and cache all AI models (5-10 minutes)
python preload_models.py
```

This will download:
- 📝 Sentence Transformer model (85MB)
- 💭 FinBERT sentiment model (440MB) 
- 📋 BART summarization model (1.6GB)

**Note**: Without preloading, the first query will take 5+ minutes to download models.

### 3. Run the Application

```bash
streamlit run app.py
```

The app will be available at `http://localhost:8501`

## 🏗️ System Architecture & Application Flow

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           STOCKBOT AI SYSTEM ARCHITECTURE                      │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  ┌─────────────────┐    ┌──────────────────┐    ┌─────────────────────────────┐ │
│  │  PRESENTATION   │    │   APPLICATION    │    │       DATA LAYER            │ │
│  │     LAYER       │    │      LAYER       │    │                             │ │
│  │                 │    │                  │    │                             │ │
│  │ ┌─────────────┐ │    │ ┌──────────────┐ │    │ ┌─────────────────────────┐ │ │
│  │ │ Streamlit   │ │◄──►│ │ Chat Handler │ │◄──►│ │    Market Data APIs     │ │ │
│  │ │ Dashboard   │ │    │ │ & NLP Router │ │    │ │  • Yahoo Finance API   │ │ │
│  │ └─────────────┘ │    │ └──────────────┘ │    │ │  • Google News Scraper │ │ │
│  │                 │    │                  │    │ │  • Real-time Indices   │ │ │
│  │ ┌─────────────┐ │    │ ┌──────────────┐ │    │ └─────────────────────────┘ │ │
│  │ │ Interactive │ │◄──►│ │   AI Engine  │ │    │                             │ │
│  │ │ Chat UI     │ │    │ │ Orchestrator │ │    │ ┌─────────────────────────┐ │ │
│  │ └─────────────┘ │    │ └──────────────┘ │    │ │    Model Storage        │ │ │
│  │                 │    │        │         │    │ │  • PyTorch Models      │ │ │
│  │ ┌─────────────┐ │    │        ▼         │    │ │  • Transformer Cache   │ │ │
│  │ │ Price Charts│ │◄──►│ ┌──────────────┐ │    │ │  • Scalers & Weights   │ │ │
│  │ │ & Analytics │ │    │ │  ML Pipeline │ │    │ └─────────────────────────┘ │ │
│  │ └─────────────┘ │    │ │  Integration │ │    │                             │ │
│  └─────────────────┘    │ └──────────────┘ │    └─────────────────────────────┘ │
│                         └──────────────────┘                                    │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### Detailed Application Flow

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                        STOCKBOT PROCESSING PIPELINE                            │
└─────────────────────────────────────────────────────────────────────────────────┘

    USER INPUT: "How is Tesla doing?"
           │
           ▼
    ┌─────────────────┐
    │  1. NLP ROUTER  │  ◄── Detect stock symbols, intent analysis
    │                 │      Extract ticker from natural language
    └─────────┬───────┘
              │
              ▼
    ┌─────────────────┐
    │ 2. DATA FETCHER │  ◄── Fetch OHLCV data, News articles
    │                 │      Yahoo Finance API + Google News scraping
    └─────────┬───────┘
              │
              ▼
    ┌─────────────────┐
    │ 3. FEATURE ENG. │  ◄── Generate 25 technical features
    │                 │      Moving averages, RSI, volatility, etc.
    └─────────┬───────┘
              │
              ▼
    ┌─────────────────┐
    │ 4. PREDICTOR    │  ◄── Neural Network Inference
    │                 │      Input: 25 features → Output: BUY/SELL/HOLD
    └─────────┬───────┘
              │
              ▼
    ┌─────────────────┐
    │ 5. NEWS FILTER  │  ◄── Filter ticker-specific articles
    │                 │      Remove general market noise
    └─────────┬───────┘
              │
              ▼
    ┌─────────────────┐
    │ 6. EMBEDDER     │  ◄── SentenceTransformer semantic search
    │                 │      Find most relevant articles to user query
    └─────────┬───────┘
              │
              ▼
    ┌─────────────────┐
    │ 7. SENTIMENT    │  ◄── FinBERT sentiment classification
    │                 │      Analyze financial tone: POSITIVE/NEGATIVE/NEUTRAL
    └─────────┬───────┘
              │
              ▼
    ┌─────────────────┐
    │ 8. SUMMARIZER   │  ◄── BART text summarization
    │                 │      Generate concise investment insights
    └─────────┬───────┘
              │
              ▼
    ┌─────────────────┐
    │ 9. VISUALIZER   │  ◄── Plotly chart generation
    │                 │      Create interactive price charts
    └─────────┬───────┘
              │
              ▼
    ┌─────────────────┐
    │10. RESPONSE     │  ◄── Format final output
    │   FORMATTER     │      Combine prediction + sentiment + summary
    └─────────┬───────┘
              │
              ▼
         USER RESPONSE: "📊 TSLA Prediction: BUY 
                        💭 Sentiment: POSITIVE
                        📈 Summary: Tesla shows strong momentum..."
```

### Component Details

#### 🧠 AI Model Pipeline

```
┌─────────────────────────────────────────────────────────────────────┐
│                        ML MODEL ARCHITECTURE                       │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐ │
│  │  STOCK PRICE    │    │  SENTIMENT       │    │  TEXT           │ │
│  │  PREDICTOR      │    │  ANALYZER        │    │  SUMMARIZER     │ │
│  │                 │    │                  │    │                 │ │
│  │ ┌─────────────┐ │    │ ┌──────────────┐ │    │ ┌─────────────┐ │ │
│  │ │ Input Layer │ │    │ │   FinBERT    │ │    │ │    BART     │ │ │
│  │ │ 25 Features │ │    │ │  (Pre-trained│ │    │ │ (facebook/  │ │ │
│  │ │             │ │    │ │  Fine-tuned) │ │    │ │ bart-large- │ │ │
│  │ └─────────────┘ │    │ └──────────────┘ │    │ │     cnn)    │ │ │
│  │        │        │    │        │         │    │ └─────────────┘ │ │
│  │        ▼        │    │        ▼         │    │        │        │ │
│  │ ┌─────────────┐ │    │ ┌──────────────┐ │    │        ▼        │ │
│  │ │Hidden Layer │ │    │ │ Transformer  │ │    │ ┌─────────────┐ │ │
│  │ │ 32 Neurons  │ │    │ │   Layers     │ │    │ │ Encoder-    │ │ │
│  │ │ ReLU + Drop │ │    │ │ (12 layers)  │ │    │ │ Decoder     │ │ │
│  │ └─────────────┘ │    │ └──────────────┘ │    │ │ Architecture│ │ │
│  │        │        │    │        │         │    │ └─────────────┘ │ │
│  │        ▼        │    │        ▼         │    │        │        │ │
│  │ ┌─────────────┐ │    │ ┌──────────────┐ │    │        ▼        │ │
│  │ │Output Layer │ │    │ │Classification│ │    │ ┌─────────────┐ │ │
│  │ │ 3 Classes   │ │    │ │    Head      │ │    │ │ Generated   │ │ │
│  │ │BUY/SELL/HOLD│ │    │ │ POS/NEG/NEU  │ │    │ │  Summary    │ │ │
│  │ └─────────────┘ │    │ └──────────────┘ │    │ │ (200-300w)  │ │ │
│  └─────────────────┘    └──────────────────┘    │ └─────────────┘ │ │
│                                                  └─────────────────┘ │
├─────────────────────────────────────────────────────────────────────┤
│  ┌─────────────────────────────────────────────────────────────────┐ │
│  │              SENTENCE TRANSFORMER EMBEDDER                     │ │
│  │                                                                 │ │
│  │  ┌─────────────┐    ┌──────────────┐    ┌─────────────────────┐ │ │
│  │  │ User Query  │───▶│ all-MiniLM-  │───▶│ 384-dim Embedding  │ │ │
│  │  │ "Tesla news"│    │ L6-v2 Model  │    │ Vector             │ │ │
│  │  └─────────────┘    └──────────────┘    └─────────────────────┘ │ │
│  │         │                                          │             │ │
│  │         ▼                                          ▼             │ │
│  │  ┌─────────────┐    ┌──────────────┐    ┌─────────────────────┐ │ │
│  │  │News Articles│───▶│Same Embedding│───▶│Cosine Similarity    │ │ │
│  │  │   (1-10)    │    │    Model     │    │ Ranking & Selection │ │ │
│  │  └─────────────┘    └──────────────┘    └─────────────────────┘ │ │
│  └─────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────┘
```

### Data Flow Sequence

```
┌─────────────────────────────────────────────────────────────────────┐
│                     REAL-TIME DATA PROCESSING                      │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  USER QUERY: "What's happening with Apple stock?"                  │
│       │                                                             │
│       ▼                                                             │
│  [1] NLP Processing                                                 │
│       ├─ Extract ticker: "AAPL"                                     │
│       ├─ Detect intent: "analysis"                                  │
│       └─ Context: "recent performance"                              │
│       │                                                             │
│       ▼                                                             │
│  [2] Data Collection (Parallel Processing)                         │
│       ├─ Yahoo Finance API ──┐                                      │
│       │   • Get 7-day OHLCV  │                                      │
│       │   • Get current price │                                      │
│       │                      │                                      │
│       ├─ Google News Scraper─┤                                      │
│       │   • Search "AAPL"    │                                      │
│       │   • Filter relevant  │                                      │
│       │   • Clean text       │                                      │
│       │                      │                                      │
│       └─ Market Indices ─────┘                                      │
│           • S&P 500, NASDAQ                                         │
│           • Sector performance                                      │
│       │                                                             │
│       ▼                                                             │
│  [3] Feature Engineering                                            │
│       ├─ Price features: [open, high, low, close, volume]          │
│       ├─ Technical indicators: [SMA_3, SMA_5, RSI, volatility]     │
│       ├─ Derived metrics: [price_change, volume_ratio, momentum]   │
│       └─ Normalization: StandardScaler.transform()                 │
│       │                                                             │
│       ▼                                                             │
│  [4] ML Model Inference                                             │
│       ├─ Input: [25-feature vector]                                 │
│       ├─ Forward pass: features → hidden(32) → output(3)           │
│       ├─ Softmax activation: [P(BUY), P(HOLD), P(SELL)]           │
│       └─ Argmax selection: "BUY" (confidence: 0.73)                │
│       │                                                             │
│       ▼                                                             │
│  [5] News Processing Pipeline                                       │
│       ├─ Article filtering: Keep only AAPL-specific               │
│       ├─ Semantic search: Query-article similarity scoring        │
│       ├─ Top-K selection: Select 3 most relevant articles         │
│       └─ Text cleaning: Remove artifacts, normalize encoding      │
│       │                                                             │
│       ▼                                                             │
│  [6] AI Analysis Chain                                              │
│       ├─ Sentiment Analysis:                                       │
│       │   FinBERT("Apple reports strong Q4...") → POSITIVE(0.89)   │
│       │                                                             │
│       ├─ Text Summarization:                                       │
│       │   BART(articles + context) → "Apple shows momentum..."     │
│       │                                                             │
│       └─ Response Generation:                                       │
│           Combine prediction + sentiment + summary                 │
│       │                                                             │
│       ▼                                                             │
│  [7] Visualization & Output                                         │
│       ├─ Generate Plotly chart: 30-day price trend                │
│       ├─ Format response: Markdown with emojis                     │
│       ├─ Cache results: Store for 5-minute TTL                     │
│       └─ Stream to UI: Progressive response rendering              │
│                                                                     │
│  FINAL OUTPUT:                                                      │
│  🧠 **Summary**: Apple shows strong momentum with positive         │
│     earnings sentiment and technical indicators...                 │
│  📊 **Prediction**: BUY (Confidence: 73%)                         │
│  💭 **Sentiment**: POSITIVE                                        │
│  📈 [Interactive Price Chart]                                      │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### Caching & Performance Strategy

```
┌─────────────────────────────────────────────────────────────────────┐
│                        CACHING ARCHITECTURE                        │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────┐ │
│  │   MODEL CACHE   │  │   DATA CACHE    │  │   RESPONSE CACHE    │ │
│  │                 │  │                 │  │                     │ │
│  │ • Loaded Models │  │ • Market Data   │  │ • Generated         │ │
│  │   (Memory)      │  │   (5 min TTL)   │  │   Summaries         │ │
│  │                 │  │                 │  │   (Session-based)   │ │
│  │ • PyTorch State │  │ • News Articles │  │                     │ │
│  │ • Transformers  │  │   (10 min TTL)  │  │ • Chart Data        │ │
│  │ • Tokenizers    │  │                 │  │   (User session)    │ │
│  │                 │  │ • Index Prices  │  │                     │ │
│  │ • Lazy Loading  │  │   (5 min TTL)   │  │ • Query Results     │ │
│  │ • Fallbacks     │  │                 │  │   (Temporary)       │ │
│  └─────────────────┘  └─────────────────┘  └─────────────────────┘ │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### Error Handling & Fallbacks

```
┌─────────────────────────────────────────────────────────────────────┐
│                     ROBUST FALLBACK SYSTEM                         │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  AI Model Loading Issues:                                           │
│  ┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐ │
│  │ Primary: Full   │───▶│ Fallback: Basic  │───▶│ Ultimate: Rule- │ │
│  │ AI Pipeline     │    │ Text Processing  │    │ Based Responses │ │
│  │ (All models)    │    │ (Manual summary) │    │ (Static analysis│ │
│  └─────────────────┘    └──────────────────┘    └─────────────────┘ │
│                                                                     │
│  Data Source Failures:                                             │
│  ┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐ │
│  │ Yahoo Finance   │───▶│ Alternative APIs │───▶│ Cached/Static   │ │
│  │ (Primary)       │    │ (Backup sources) │    │ Data Sources    │ │
│  └─────────────────┘    └──────────────────┘    └─────────────────┘ │
│                                                                     │
│  Network/Timeout Issues:                                           │
│  ┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐ │
│  │ Real-time Fetch │───▶│ Cached Results   │───▶│ Graceful Error  │ │
│  │ (30s timeout)   │    │ (Recent data)    │    │ Messages        │ │
│  └─────────────────┘    └──────────────────┘    └─────────────────┘ │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

## 🧠 Model Training & Development Pipeline

### Training Workflow

```
┌─────────────────────────────────────────────────────────────────────┐
│                     MODEL TRAINING PIPELINE                        │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  [1] Data Collection & Preparation                                  │
│       ┌─────────────────┐    ┌──────────────────┐                   │
│       │ Historical Data │───▶│ Feature Engineer │                   │
│       │ • OHLCV prices  │    │ • 25 indicators  │                   │
│       │ • Multiple tickers   │ • Technical analysis                │
│       │ • 7-day windows │    │ • Normalization  │                   │
│       └─────────────────┘    └──────────────────┘                   │
│                                       │                             │
│  [2] Label Generation                 ▼                             │
│       ┌─────────────────────────────────────────┐                   │
│       │ Next-day price movement classification: │                   │
│       │ • BUY:  price increase > +2%           │                   │
│       │ • SELL: price decrease > -2%           │                   │
│       │ • HOLD: price change between ±2%       │                   │
│       └─────────────────────────────────────────┘                   │
│                                       │                             │
│  [3] Model Architecture               ▼                             │
│       ┌─────────────────────────────────────────┐                   │
│       │ Neural Network Design:                  │                   │
│       │                                         │                   │
│       │ Input(25) → Linear(25→32) → ReLU →     │                   │
│       │ Dropout(0.3) → Linear(32→3) → Softmax  │                   │
│       │                                         │                   │
│       │ • Loss: CrossEntropyLoss               │                   │
│       │ • Optimizer: Adam (lr=0.001)           │                   │
│       │ • Regularization: Dropout + L2         │                   │
│       └─────────────────────────────────────────┘                   │
│                                       │                             │
│  [4] Hyperparameter Optimization     ▼                             │
│       ┌─────────────────────────────────────────┐                   │
│       │ Bayesian Optimization:                  │                   │
│       │ • Learning Rate: [1e-5, 1e-3]         │                   │
│       │ • Dropout Rate: [0.1, 0.5]            │                   │
│       │ • Optimization Iterations: 5-10        │                   │
│       │ • Cross-validation: 5-fold             │                   │
│       └─────────────────────────────────────────┘                   │
│                                       │                             │
│  [5] Training & Validation            ▼                             │
│       ┌─────────────────────────────────────────┐                   │
│       │ Training Process:                       │                   │
│       │ • Epochs: 10-25 with early stopping   │                   │
│       │ • Batch processing: Full dataset       │                   │
│       │ • Validation monitoring                │                   │
│       │ • Learning curves tracking             │                   │
│       │ • Overfitting detection               │                   │
│       └─────────────────────────────────────────┘                   │
│                                       │                             │
│  [6] Model Evaluation                 ▼                             │
│       ┌─────────────────────────────────────────┐                   │
│       │ Performance Metrics:                    │                   │
│       │ • Accuracy: ~68-75%                    │                   │
│       │ • Precision/Recall per class          │                   │
│       │ • Confusion matrix analysis           │                   │
│       │ • Feature importance ranking          │                   │
│       │ • Cross-validation stability          │                   │
│       └─────────────────────────────────────────┘                   │
│                                       │                             │
│  [7] Model Deployment                 ▼                             │
│       ┌─────────────────────────────────────────┐                   │
│       │ Production Pipeline:                    │                   │
│       │ • Save model state dict (.pt)         │                   │
│       │ • Export feature scaler (.pkl)        │                   │
│       │ • Validation testing                   │                   │
│       │ • Integration with app.py              │                   │
│       └─────────────────────────────────────────┘                   │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### Stock Prediction Model Architecture

**Model Type**: Feedforward Neural Network (PyTorch)

```python
class StockClassifier(nn.Module):
    def __init__(self, input_dim=25, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32),    # Input: 25 features
            nn.ReLU(),                   # Activation
            nn.Dropout(dropout),         # Regularization
            nn.Linear(32, 3)             # Output: BUY/SELL/HOLD
        )
```

**Training Configuration**:
- **Input Dimensions**: 25 engineered features
- **Hidden Layer**: 32 neurons with ReLU activation
- **Dropout Rate**: 0.3 for overfitting prevention
- **Output Classes**: 3 (BUY=0, SELL=1, HOLD=2)
- **Loss Function**: CrossEntropyLoss
- **Optimizer**: Adam with learning rate scheduling

### Feature Engineering Pipeline

**Raw Data Sources**: 7-day OHLCV (Open, High, Low, Close, Volume) data

**25 Engineered Features**:
1. **Price Features (5)**: Open, High, Low, Close, Adjusted Close
2. **Volume Features (3)**: Volume, Volume MA, Volume change
3. **Price Changes (4)**: Daily change, % change, High-Low spread, Close-Open
4. **Moving Averages (4)**: 3-day, 5-day, 7-day, 10-day SMA
5. **Technical Indicators (9)**: RSI-style momentum, volatility measures, trend strength

**Feature Preprocessing**:
```python
# Normalization using StandardScaler
scaler = StandardScaler()
scaled_features = scaler.fit_transform(raw_features)
```

### Training Data Preparation

**Data Collection**:
- Historical stock data from Yahoo Finance API
- 7-day sliding windows for feature extraction
- Multiple stocks for diverse training examples

**Labeling Strategy**:
- **BUY**: Next day price increase > 2%
- **SELL**: Next day price decrease > 2%
- **HOLD**: Price change between -2% and +2%

**Data Split**:
- Training: 70%
- Validation: 15%
- Testing: 15%

### Visualization & Analysis

The training notebook (`train/train_predictor.ipynb`) includes comprehensive heatmap visualizations:

1. **📊 Feature Correlation Heatmap** - Identify feature relationships
2. **📈 Learning Curves Heatmap** - Training dynamics and overfitting detection  
3. **🎯 Confusion Matrix Heatmap** - Model performance by class
4. **⚙️ Hyperparameter Optimization Landscape** - Performance surface visualization
5. **🧠 Neural Network Weights Heatmap** - Internal model representations
6. **📋 Cross-Validation Heatmap** - Model stability across data splits
7. **📊 Feature Importance Rankings** - Weight-based feature analysis
8. **🔄 Training Dynamics Dashboard** - Comprehensive training overview

### Sentiment Analysis Pipeline

**Model**: `yiyanghkust/finbert-tone` (Pre-trained FinBERT)

**Process Flow**:
1. **News Retrieval**: Scrape Google News for stock-specific articles
2. **Relevance Filtering**: Use SentenceTransformers to match user queries
3. **Sentiment Classification**: FinBERT assigns sentiment scores
4. **Aggregation**: Combine sentiments for overall market mood

## 📁 Project Structure & File Organization

```
stockbot/
├── 📄 app.py                    # Main entry point (Hugging Face Spaces)
├── 📄 Dockerfile               # Container configuration
├── 📄 requirements.txt         # Python dependencies
├── 📄 README.md                # Project documentation
├── 📄 LICENSE                  # MIT license
├── 📄 .gitignore               # Git ignore rules
├── 📄 preload_models.py        # Model pre-download utility
├── 📄 download_models.py       # Model download script
├── 📄 start.sh                 # Docker startup script
├── 📄 test_summarizer.py       # Unit tests
│
├── 📁 src/                     # Source code directory
│   ├── 📄 __init__.py          # Package initialization
│   ├── 📄 app.py               # Main Streamlit application
│   ├── 📄 fetcher.py           # Data fetching & news scraping
│   ├── 📄 predictor.py         # ML model inference
│   ├── 📄 summerizer.py        # AI summarization pipeline
│   └── 📄 utils.py             # Utility functions & plotting
│
├── 📁 models/                  # Model storage directory
│   ├── 📄 scaler.pkl           # Feature scaler (StandardScaler)
│   ├── 📄 tenson.pt            # Trained PyTorch model weights
│   ├── 📁 sentence_transformers/  # SentenceTransformer cache
│   └── 📁 transformers_cache/     # Hugging Face model cache
│
├── 📁 train/                   # Training & development
│   └── 📄 train_predictor.ipynb   # Model training notebook
│
└── 📁 .ipynb_checkpoints/      # Jupyter notebook checkpoints
    └── 📄 Untitled-checkpoint.ipynb
```

### File Descriptions

#### 🎯 **Core Application Files**

- **`src/app.py`** - Main Streamlit interface with chat UI, market dashboard, and visualization
- **`src/fetcher.py`** - Data acquisition layer for stocks, news, and market indices  
- **`src/predictor.py`** - Neural network model loading and inference pipeline
- **`src/summerizer.py`** - AI orchestration: sentiment analysis, text summarization, and response generation
- **`src/utils.py`** - Helper functions for feature engineering and chart plotting

#### 🤖 **AI Model Components**

- **`models/tenson.pt`** - PyTorch state dict for trained stock classifier
- **`models/scaler.pkl`** - Scikit-learn StandardScaler for feature normalization
- **`models/sentence_transformers/`** - Cached SentenceTransformer model (all-MiniLM-L6-v2)
- **`models/transformers_cache/`** - Cached Hugging Face models (FinBERT + BART)

#### 📚 **Training & Development**

- **`train/train_predictor.ipynb`** - Complete model training pipeline with 13+ heatmap visualizations
- **`preload_models.py`** - Interactive model download utility for faster startup
- **`test_summarizer.py`** - Unit tests for summarization pipeline

#### 🚀 **Deployment Configuration**

- **`Dockerfile`** - Multi-stage container build with model pre-caching
- **`app.py`** - Entry point wrapper for Hugging Face Spaces deployment  
- **`start.sh`** - Container startup script with health checks
- **`requirements.txt`** - Pinned Python dependencies

### Module Dependencies

```
┌─────────────────────────────────────────────────────────────────────┐
│                        MODULE DEPENDENCY GRAPH                     │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  app.py (Entry) ──────────┐                                        │
│                           │                                         │
│  src/app.py (Main) ───────┼─────────────────────┐                   │
│           │               │                     │                   │
│           ▼               ▼                     ▼                   │
│  ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────────┐   │
│  │  src/fetcher.py │ │ src/predictor.py│ │  src/summerizer.py  │   │
│  │                 │ │                 │ │                     │   │
│  │ • Yahoo Finance │ │ • PyTorch Model │ │ • SentenceTransform │   │
│  │ • Google News   │ │ • Feature Eng.  │ │ • FinBERT Sentiment │   │
│  │ • Price Data    │ │ • Classification│ │ • BART Summarizer   │   │
│  └─────────────────┘ └─────────────────┘ └─────────────────────┘   │
│           │               │                     │                   │
│           └───────────────┼─────────────────────┘                   │
│                           ▼                                         │
│                  ┌─────────────────┐                                │
│                  │  src/utils.py   │                                │
│                  │                 │                                │
│                  │ • Feature Utils │                                │
│                  │ • Plotly Charts │                                │
│                  │ • Data Helpers  │                                │
│                  └─────────────────┘                                │
│                                                                     │
│  External Dependencies:                                             │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │ streamlit, torch, transformers, sentence-transformers,     │   │
│  │ yfinance, requests, beautifulsoup4, plotly, pandas,        │   │
│  │ numpy, scikit-learn, matplotlib, seaborn                   │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

## 🔧 Installation & Setup Guide

### Prerequisites

```bash
# System Requirements
- Python 3.8+ (recommended: 3.10)
- 8GB+ RAM (for AI models)
- 5GB+ disk space (for model cache)
- Internet connection (for real-time data)
```

### Quick Installation

```bash
# 1. Clone the repository
git clone https://github.com/your-username/stockbot.git
cd stockbot

# 2. Create virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# OR
.venv\Scripts\activate     # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Pre-download AI models (optional but recommended)
python preload_models.py

# 5. Run the application
streamlit run app.py
```

### Development Setup

```bash
# Additional development dependencies
pip install jupyter notebook matplotlib seaborn bayesian-optimization

# Train your own model
jupyter notebook train/train_predictor.ipynb

# Run tests
python test_summarizer.py

# Build Docker container
docker build -t stockbot .
docker run -p 7860:7860 stockbot
```

### Environment Variables (Optional)

```bash
# Custom model cache directories
export TRANSFORMERS_CACHE=/path/to/cache
export HF_HOME=/path/to/hf_cache
export SENTENCE_TRANSFORMERS_HOME=/path/to/st_cache

# Disable tokenizer warnings
export TOKENIZERS_PARALLELISM=false

# Suppress Python warnings
export PYTHONWARNINGS=ignore
```

## 📊 Model Performance

### Stock Prediction Metrics
- **Training Accuracy**: ~75% (educational model)
- **Validation Accuracy**: ~68%
- **Class Distribution**: Balanced across BUY/SELL/HOLD

### Optimization Techniques
- **Dropout Regularization**: Prevents overfitting
- **Feature Scaling**: StandardScaler normalization
- **Early Stopping**: Monitor validation loss
- **Cross-Validation**: K-fold validation for robustness

## 🔄 Real-time Inference Pipeline

```
User Query → Stock Symbol Detection → Feature Extraction → Model Prediction
     ↓
News Retrieval → Sentiment Analysis → Summary Generation → Response
```

**Caching Strategy**:
- Market data cached for 5 minutes
- Model predictions cached per session
- News data cached for 10 minutes

## 🛠️ Technology Stack

**Backend**:
- **Framework**: Python, PyTorch
- **Models**: Transformers, SentenceTransformers
- **Data**: Yahoo Finance, BeautifulSoup
- **Processing**: NumPy, Pandas, Scikit-learn

**Frontend**:
- **UI**: Streamlit
- **Visualization**: Plotly
- **Styling**: Custom CSS

**Deployment**:
- **Container**: Docker
- **Platform**: Hugging Face Spaces
- **CI/CD**: GitHub Actions

## 💡 Key Features

### Interactive Chat Interface
- Natural language processing for stock queries
- Context-aware responses
- Real-time data integration

### Market Dashboard
- Live stock prices and indices
- Trending stocks with price changes
- Global market snapshot

### AI-Powered Insights
- ML-based BUY/SELL/HOLD recommendations
- News sentiment analysis
- Technical indicator summaries

### Visual Analytics
- Interactive price charts
- Trend visualization
- Volume analysis

## 🚀 Usage Examples

```python
# User queries the chatbot can handle:
"How is Tesla performing?"
"Should I buy Apple stock?"
"What's the sentiment around NVIDIA?"
"Show me Bitcoin's recent trends"
"Compare Google and Microsoft"
```

## 📈 Future Improvements

1. **Enhanced Features**: Add more technical indicators (MACD, Bollinger Bands)
2. **Multi-timeframe Analysis**: 1-day, 1-week, 1-month predictions
3. **Portfolio Management**: Track multiple stocks simultaneously
4. **Advanced NLP**: Incorporate earnings calls and SEC filings
5. **Risk Assessment**: Add volatility and risk metrics

## ⚠️ Disclaimers

- **Educational Purpose**: This model is for learning and demonstration
- **Not Financial Advice**: Always consult financial professionals
- **Market Volatility**: Past performance doesn't guarantee future results
- **Data Limitations**: Predictions based on historical patterns only

## 🔧 Installation & Setup

```bash
# Clone repository
git clone <repository-url>
cd stockbot

# Install dependencies
pip install -r requirements.txt

# Run locally
streamlit run app.py
```

## 📋 Requirements

```
streamlit
yfinance
requests
beautifulsoup4
transformers
torch
scikit-learn
plotly
sentence-transformers
pandas
numpy
```

## 🔧 Troubleshooting

### Common Issues

**Stuck at "🔍 Analyzing..."**: 
- Run `python preload_models.py` first
- Models are downloading in the background
- Wait 5-10 minutes for first-time setup

**Permission Errors**:
- Models will fallback to basic analysis
- All core functionality remains available

**Memory Issues**:
- Use CPU-only mode (default)
- Models auto-fallback on memory errors

---
