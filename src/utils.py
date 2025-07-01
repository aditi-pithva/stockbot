import re
import plotly.graph_objects as go

def detect_small_talk(text):
    """
    Detects greetings, thanks, or farewells in the input text.
    Returns a tag string or None.
    """
    text = text.lower()
    if any(greet in text for greet in ["hi", "hello", "hey"]):
        return "greeting"
    if any(thank in text for thank in ["thanks", "thank you"]):
        return "thanks"
    if any(bye in text for bye in ["bye", "goodbye", "see you"]):
        return "bye"
    return None

def extract_ticker(text, fallback_map):
    """
    Extract a stock ticker symbol from a message.
    Tries known fallback map first, then uses regex.
    """
    for name, symbol in fallback_map.items():
        if name in text.lower():
            return symbol
    match = re.search(r'\b[A-Z]{2,5}\b', text)
    return match.group(0) if match else None

def plot_price_chart(prices, ticker):
    """
    Plots a line chart of the stock price over time using Plotly.
    """
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=prices.index, y=prices, mode='lines+markers', name=ticker))
    fig.update_layout(
        title=f"{ticker} Stock Price (Last 30 Days)",
        xaxis_title="Date",
        yaxis_title="Price",
        template="plotly_dark"
    )
    return fig

def generate_feature_vector(df):
    """
    Convert 7-day OHLCV data into a 25-feature vector.
    Returns: list of 25 float values or None if data is invalid.
    """
    if df is None or len(df) < 5:
        return None

    vec = []
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        series = df[col]
        vec.extend([
            series.mean(),
            series.std(),
            series.max(),
            series.min(),
            series.iloc[-1]  # latest value
        ])
    return vec