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
    Convert 7-day OHLCV data into enhanced feature vector with technical indicators.
    Returns: tuple of (feature_vector, feature_names) or (None, None) if data is invalid.
    """
    if df is None or len(df) < 5:
        return None, None

    # Import here to avoid circular imports
    import sys
    import os
    import pandas as pd
    import numpy as np
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    
    try:
        from enhanced_feature_engineering import calculate_technical_indicators
        
        # Use enhanced technical indicators
        features_dict = calculate_technical_indicators(df)
        if features_dict is None:
            return None, None
        
        # Convert to list and feature names (exclude metadata)
        exclude_keys = ['ticker', 'date', 'label']
        feature_names = [k for k in features_dict.keys() if k not in exclude_keys]
        feature_vector = [features_dict[k] for k in feature_names]
        
        # Handle any NaN values
        feature_vector = [0.0 if pd.isna(x) or np.isinf(x) else float(x) for x in feature_vector]
        
        return feature_vector, feature_names
        
    except ImportError:
        # Fallback to basic features if enhanced module not available
        print("Enhanced features not available, using basic features")
        return generate_basic_feature_vector(df)

def generate_basic_feature_vector(df):
    """
    Fallback function for basic feature generation (original implementation)
    """
    import pandas as pd
    import numpy as np
    
    vec = []
    feature_names = []
    
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        series = df[col]
        col_lower = col.lower()
        
        values = [
            series.mean(),
            series.std(),
            series.max(),
            series.min(),
            series.iloc[-1]  # latest value
        ]
        
        names = [
            f"{col_lower}_mean",
            f"{col_lower}_std", 
            f"{col_lower}_max",
            f"{col_lower}_min",
            f"{col_lower}_latest"
        ]
        
        vec.extend(values)
        feature_names.extend(names)
    
    return vec, feature_names

def get_selected_features():
    """
    Get the list of selected features from the trained model.
    Returns the selected feature names or all 25 features if no selection info.
    """
    import json
    import os
    
    try:
        feature_info_path = os.path.join(os.path.dirname(__file__), "..", "models", "feature_info.json")
        with open(feature_info_path, "r") as f:
            feature_info = json.load(f)
        return feature_info.get('selected_features', None)
    except (FileNotFoundError, json.JSONDecodeError):
        return None

def filter_features_for_prediction(feature_vector, feature_names):
    """
    Filter feature vector to only include selected features for prediction.
    """
    selected_features = get_selected_features()
    
    if selected_features is None:
        # No feature selection info, return all features
        return feature_vector
    
    # Filter to selected features
    try:
        selected_indices = [feature_names.index(feat) for feat in selected_features if feat in feature_names]
        filtered_vector = [feature_vector[i] for i in selected_indices]
        print(f"Filtered from {len(feature_vector)} to {len(filtered_vector)} features")
        return filtered_vector
    except (ValueError, IndexError) as e:
        print(f"Error filtering features: {e}")
        return feature_vector