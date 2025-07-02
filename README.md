# StockBot

AI-powered stock market chatbot with neural network predictions and real-time sentiment analysis.

## Quick Start

```bash
# Clone and install
git clone <repository-url>
cd stockbot
pip install -r requirements.txt

# Run application
streamlit run app.py
```

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    STOCKBOT AI PIPELINE                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  USER QUERY: "How is Tesla doing?"                             │
│         │                                                       │
│         ▼                                                       │
│  ┌─────────────────┐    ┌──────────────────┐                   │
│  │  NLP ROUTER     │───▶│  DATA FETCHER    │                   │
│  │  Extract ticker │    │  Yahoo Finance   │                   │
│  │  Detect intent  │    │  Google News     │                   │
│  └─────────────────┘    └──────────────────┘                   │
│         │                         │                            │
│         ▼                         ▼                            │
│  ┌─────────────────┐    ┌──────────────────┐                   │
│  │ FEATURE ENGINEER│───▶│   ML PREDICTOR   │                   │
│  │ 19 technical    │    │ Neural Network   │                   │
│  │ indicators      │    │ BUY/SELL/HOLD    │                   │
│  └─────────────────┘    └──────────────────┘                   │
│         │                         │                            │
│         ▼                         ▼                            │
│  ┌─────────────────┐    ┌──────────────────┐                   │
│  │ NEWS PROCESSOR  │───▶│  AI SUMMARIZER   │                   │
│  │ Filter & rank   │    │ FinBERT + BART   │                   │
│  │ relevant news   │    │ Sentiment + Text │                   │
│  └─────────────────┘    └──────────────────┘                   │
│         │                         │                            │
│         └─────────┬─────────────────┘                          │
│                   ▼                                            │
│         ┌─────────────────┐                                    │
│         │ RESPONSE FORMAT │                                    │
│         │ Combine results │                                    │
│         │ Generate charts │                                    │
│         └─────────────────┘                                    │
│                   │                                            │
│                   ▼                                            │
│   📊 Prediction: BUY (73% confidence)                         │
│   💭 Sentiment: POSITIVE                                       │
│   📈 Summary: Tesla shows strong momentum...                   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### System Components

- **Frontend**: Streamlit web interface with interactive chat
- **Backend**: Python-based AI pipeline with caching
- **Data Sources**: Yahoo Finance API + Google News scraping
- **Models**: PyTorch neural networks + Transformer models

### Deployment Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    DEPLOYMENT PIPELINE                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────┐    ┌──────────────────┐                   │
│  │   DEVELOPMENT   │───▶│  GITHUB ACTIONS  │                   │
│  │   Local Coding  │    │  CI/CD Pipeline  │                   │
│  │   Model Training│    │  Automated Tests │                   │
│  └─────────────────┘    └──────────────────┘                   │
│                                   │                            │
│                                   ▼                            │
│  ┌─────────────────┐    ┌──────────────────┐                   │
│  │  DOCKER BUILD   │◄───│   BUILD TRIGGER  │                   │
│  │  Multi-stage    │    │   On Push/PR     │                   │
│  │  Model Caching  │    │   Auto Deploy    │                   │
│  └─────────────────┘    └──────────────────┘                   │
│           │                                                    │
│           ▼                                                    │
│  ┌─────────────────┐                                           │
│  │ HUGGING FACE    │                                           │
│  │ SPACES DEPLOY   │                                           │
│  │ Public Access   │                                           │
│  │ Auto Scaling    │                                           │
│  └─────────────────┘                                           │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

**Deployment Components**:
- **Docker**: Containerized application with multi-stage builds and model pre-caching
- **GitHub Actions**: Automated CI/CD pipeline for testing and deployment
- **Hugging Face Spaces**: Cloud hosting platform with automatic scaling and public access

## 🤖 AI Models

### 1. Stock Price Predictor
- **Type**: Custom PyTorch Neural Network
- **Architecture**: 19 features → 128 → 64 → 32 → 3 classes
- **Input**: Technical indicators (RSI, momentum, volatility, volume)
- **Output**: BUY/SELL/HOLD with confidence scores
- **Training**: 912 samples with Bayesian optimization
- **Accuracy**: ~68% validation accuracy

```python
class EnhancedStockPredictor(nn.Module):
    def __init__(self, input_dim=19):
        super().__init__()
        self.input_norm = nn.LayerNorm(input_dim)
        self.fc1 = nn.Linear(input_dim, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.dropout1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 3)  # BUY/SELL/HOLD
```

### 2. Sentiment Analysis
- **Model**: `yiyanghkust/finbert-tone`
- **Type**: Pre-trained FinBERT (Financial BERT)
- **Purpose**: Analyze financial news sentiment
- **Output**: POSITIVE/NEGATIVE/NEUTRAL with confidence

### 3. Text Summarization
- **Model**: `facebook/bart-large-cnn`
- **Type**: BART (Bidirectional Auto-Regressive Transformer)
- **Purpose**: Generate concise investment summaries
- **Input**: Multiple news articles + context
- **Output**: 200-300 word summary

### 4. Semantic Search
- **Model**: `sentence-transformers/all-MiniLM-L6-v2`
- **Type**: SentenceTransformer embedding model
- **Purpose**: Match user queries with relevant news
- **Method**: Cosine similarity ranking

## 📊 Feature Engineering

### Technical Indicators (19 features)
```python
features = {
    # Price Features
    'close_price', 'volume', 'daily_return', 'weekly_return',
    
    # Momentum Indicators  
    'price_momentum_3d', 'rsi', 'bb_position',
    
    # Volatility Measures
    'price_volatility', 'high_low_ratio', 'close_to_high_ratio',
    
    # Volume Analysis
    'volume_ratio', 'vpt',
    
    # Trend Features
    'distance_to_high', 'distance_to_low', 'gap_up', 'gap_down',
    
    # Intraday Metrics
    'intraday_return', 'intraday_high_reach', 'intraday_low_reach',
    
    # Market Strength
    'bullish_days_ratio'
}
```

### Data Pipeline
1. **Collection**: 7-day OHLCV data from Yahoo Finance
2. **Engineering**: Calculate 19 technical indicators
3. **Scaling**: StandardScaler normalization
4. **Correlation**: Feature selection (removed 14 correlated features)
5. **Training**: Bayesian hyperparameter optimization

## 📁 Project Structure

```
stockbot/
├── src/
│   ├── app.py              # Streamlit interface
│   ├── predictor.py        # Neural network model
│   ├── summerizer.py       # AI orchestration
│   ├── fetcher.py          # Data collection
│   └── utils.py            # Feature engineering
├── models/
│   ├── tensor.pt           # Trained model weights
│   ├── scaler.pkl          # Feature scaler
│   └── feature_info.json   # Model metadata
├── train/
│   ├── train_predictor.ipynb        # Training notebook
│   └── enhanced_feature_engineering.py  # Feature pipeline
├── app.py                  # Entry point
├── requirements.txt        # Dependencies
└── Dockerfile             # Container config
```

## 🔧 Technical Stack

**Core**: Python 3.8+, PyTorch, Transformers, Streamlit  
**Data**: yfinance, pandas, numpy, scikit-learn  
**NLP**: sentence-transformers, transformers, beautifulsoup4  
**Visualization**: plotly, matplotlib, seaborn  
**Deployment**: Docker, Hugging Face Spaces

## 📈 Usage Examples

```python
# Natural language queries
"How is Tesla performing?"           # → Analysis + prediction
"Should I buy Apple stock?"          # → ML recommendation  
"What's the sentiment on NVIDIA?"    # → News sentiment
"Show me Bitcoin trends"             # → Price visualization
```

## ⚠️ Disclaimer

This is an educational project. Not financial advice. Always consult financial professionals before making investment decisions.
