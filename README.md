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

## 🏗️ Architecture Overview

### System Components

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Data Sources  │    │   ML Pipeline    │    │  User Interface │
│                 │    │                  │    │                 │
│ • Yahoo Finance │───▶│ • Stock Predictor│───▶│ • Streamlit UI  │
│ • Google News   │    │ • Sentiment Model│    │ • Chat Interface│
│ • Market APIs   │    │ • Text Summarizer│    │ • Price Charts  │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

### Core Models

1. **Stock Price Predictor**: Custom neural network for BUY/SELL/HOLD predictions
2. **Sentiment Analyzer**: FinBERT model for financial news sentiment
3. **Text Summarizer**: BART model for generating market insights
4. **Embedding Model**: SentenceTransformers for semantic search

## 🧠 Training Approach

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

### Sentiment Analysis Pipeline

**Model**: `yiyanghkust/finbert-tone` (Pre-trained FinBERT)

**Process Flow**:
1. **News Retrieval**: Scrape Google News for stock-specific articles
2. **Relevance Filtering**: Use SentenceTransformers to match user queries
3. **Sentiment Classification**: FinBERT assigns sentiment scores
4. **Aggregation**: Combine sentiments for overall market mood

### Text Summarization

**Model**: `facebook/bart-large-cnn`

**Implementation**:
- Input: Top 3 relevant news articles + user query context
- Output: Concise investment insights (200-300 words)
- Post-processing: Remove redundant phrases and formatting

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
