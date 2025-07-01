import streamlit as st
import re
import sys
import os

# Configure page first
st.set_page_config(page_title="StockBot Chat", layout="wide")

# Add startup indicator
if "app_started" not in st.session_state:
    st.session_state.app_started = True
    print("🚀 StockBot starting up...")

# Error handling for imports
try:
    from src.fetcher import fetch_news, fetch_price_data, fetch_index_price
    from src.summerizer import generate_summary
    from src.utils import plot_price_chart
    print("✅ All modules imported successfully")
except ImportError as e:
    st.error(f"Import Error: {e}")
    print(f"❌ Import Error: {e}")
    st.stop()

# Enlarged Chat + Input
st.markdown("""
    <style>
    [data-testid="chatMessageContent"] * {
        font-size: 1.3rem !important;
        line-height: 1.7;
    }
    input[type="text"] {
        font-size: 1.2rem !important;
    }
    </style>
""", unsafe_allow_html=True)

# Title & Animation
st.markdown("""
    <h1 style='font-size: 2.8rem;'>💬 <b>StockBot</b> — <span style='color: gold;'>Your Real-Time Investment Chatbot</span></h1>
    <style>
        [data-testid="stMetricValue"] {
            animation: pulse 2s infinite;
        }
        @keyframes pulse {
            0% { color: #ffcc00; }
            50% { color: #ffffff; }
            100% { color: #ffcc00; }
        }
    </style>
""", unsafe_allow_html=True)

# ──────────────────────────────
# Global Market Snapshot
# ──────────────────────────────
st.markdown("<div style='text-align: center;'><h2>🌐 <b>Global Market Snapshot</b></h2></div>", unsafe_allow_html=True)

card_style = """
    <div style='
        background-color: #1c1c1c;
        padding: 20px;
        border-radius: 15px;
        margin: 10px;
        text-align: center;
        box-shadow: 0 4px 8px rgba(0,0,0,0.4);
    '>
        <h3 style='margin-bottom:5px;'>{title}</h3>
        <h1 style='color: gold;'>{value}</h1>
    </div>
"""

# Initialize market data with fallbacks
@st.cache_data(ttl=300)  # Cache for 5 minutes
def get_market_data():
    try:
        return {
            "SP500": fetch_index_price("^GSPC"),
            "NASDAQ": fetch_index_price("^IXIC"), 
            "DOW": fetch_index_price("^DJI"),
            "BTC": fetch_index_price("BTC-USD"),
            "ETH": fetch_index_price("ETH-USD"),
            "AAPL": fetch_index_price("AAPL")
        }
    except Exception as e:
        st.error(f"⚠️ Market data temporarily unavailable: {e}")
        return {
            "SP500": "Loading...",
            "NASDAQ": "Loading...", 
            "DOW": "Loading...",
            "BTC": "Loading...",
            "ETH": "Loading...",
            "AAPL": "Loading..."
        }

market_data = get_market_data()

col1, col2, col3 = st.columns(3)
with col1:
    st.markdown(card_style.format(title="📈 S&P 500", value=market_data["SP500"]), unsafe_allow_html=True)
    st.markdown(card_style.format(title="💹 NASDAQ", value=market_data["NASDAQ"]), unsafe_allow_html=True)
with col2:
    st.markdown(card_style.format(title="🏦 Dow Jones", value=market_data["DOW"]), unsafe_allow_html=True)
    st.markdown(card_style.format(title="₿ Bitcoin", value="$" + market_data["BTC"]), unsafe_allow_html=True)
with col3:
    st.markdown(card_style.format(title="🔷 Ethereum", value="$" + market_data["ETH"]), unsafe_allow_html=True)
    st.markdown(card_style.format(title="🍏 Apple", value="$" + market_data["AAPL"]), unsafe_allow_html=True)

# ──────────────────────────────
# Trending Stocks
# ──────────────────────────────
st.markdown("### 🔥 Trending Stocks This Week")

@st.cache_data(ttl=600)  # Cache for 10 minutes
def get_trending_stock_data(tickers):
    import yfinance as yf
    data = {}
    for ticker in tickers:
        try:
            info = yf.Ticker(ticker)
            history = info.history(period="5d")
            if len(history) >= 2:
                last_close = history['Close'][-1]
                prev_close = history['Close'][-2]
                change = last_close - prev_close
                pct_change = (change / prev_close) * 100
                data[ticker] = {
                    "price": f"${last_close:,.2f}",
                    "change": f"{'▲' if change > 0 else '▼'} {pct_change:.2f}%",
                    "color": "green" if change > 0 else "red"
                }
            else:
                data[ticker] = {"price": "N/A", "change": "↕", "color": "gray"}
        except:
            data[ticker] = {"price": "N/A", "change": "↕", "color": "gray"}
    return data

trending = ["TSLA", "NVDA", "GOOGL", "META", "AMD"]
try:
    trends = get_trending_stock_data(trending)
    trend_cols = st.columns(len(trending))
    for i, ticker in enumerate(trending):
        with trend_cols[i]:
            st.markdown(f"**📊 {ticker}**")
            st.markdown(f"<h2>{trends[ticker]['price']}</h2>", unsafe_allow_html=True)
            st.markdown(f"<span style='color:{trends[ticker]['color']};'>{trends[ticker]['change']}</span>", unsafe_allow_html=True)
except Exception as e:
    st.warning("⚠️ Trending stocks data temporarily unavailable")

# ──────────────────────────────
# Chat History
# ──────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and "chart" in msg:
            st.plotly_chart(msg["chart"], use_container_width=True)

# ──────────────────────────────
# Chatbot Input Handler
# ──────────────────────────────
user_input = st.chat_input("Ask me about Tesla, Apple, or any stock!")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    def is_small_talk(text):
        text = text.lower()
        if any(word in text for word in ["hi", "hello", "hey"]): return "greeting"
        if any(word in text for word in ["thanks", "thank you"]): return "thanks"
        if any(word in text for word in ["bye", "goodbye", "see you"]): return "bye"
        return None

    intent = is_small_talk(user_input)
    reply = ""
    chart = None

    if intent == "greeting":
        reply = "👋 Hey! I'm StockBot. Ask me about Tesla, Apple or any stock trend."
    elif intent == "thanks":
        reply = "You're welcome! ✨ Happy investing."
    elif intent == "bye":
        reply = "👋 Goodbye! Stay smart with your investments."
    else:
        ticker_map = {
            "tesla": "TSLA", "apple": "AAPL", "microsoft": "MSFT", "google": "GOOGL",
            "meta": "META", "amazon": "AMZN", "nvidia": "NVDA", "amd": "AMD"
        }
        found = None
        for name, symbol in ticker_map.items():
            if name in user_input.lower():
                found = symbol
                break
        if not found:
            match = re.search(r'\b[A-Z]{2,5}\b', user_input.upper())
            if match:
                found = match.group(0)

        if not found:
            reply = "❌ Sorry, I couldn’t find that stock. Try asking about Tesla, Apple, etc."
        else:
            with st.spinner(f"🔍 Analyzing {found}..."):
                result = generate_summary(found, user_input)
                if "error" in result:
                    reply = f"❌ {result['error']}"
                else:
                    prices = fetch_price_data(found)
                    chart = plot_price_chart(prices, found) if prices is not None else None

                    reply = f"""
🧠 **Summary**: {result['summary']}

📊 **Prediction**: {result['prediction']}

📈 Check the chart below to see recent trends before investing.
"""

    with st.chat_message("assistant"):
        st.markdown(reply)
        if chart:
            st.plotly_chart(chart, use_container_width=True)

    entry = {"role": "assistant", "content": reply}
    if chart:
        entry["chart"] = chart
    st.session_state.messages.append(entry)
