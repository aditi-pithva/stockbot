import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

def calculate_technical_indicators(df):
    if len(df) < 7:
        return None
    
    features = {}
    
    features['close_price'] = df['Close'].iloc[-1]
    features['open_price'] = df['Open'].iloc[-1]
    features['high_price'] = df['High'].iloc[-1]
    features['low_price'] = df['Low'].iloc[-1]
    features['volume'] = df['Volume'].iloc[-1]
    
    features['daily_return'] = (df['Close'].iloc[-1] - df['Close'].iloc[-2]) / df['Close'].iloc[-2]
    features['weekly_return'] = (df['Close'].iloc[-1] - df['Close'].iloc[0]) / df['Close'].iloc[0]
    features['price_momentum_3d'] = (df['Close'].iloc[-1] - df['Close'].iloc[-4]) / df['Close'].iloc[-4] if len(df) >= 4 else 0
    
    features['sma_3'] = df['Close'].rolling(3).mean().iloc[-1]
    features['sma_7'] = df['Close'].rolling(7).mean().iloc[-1] if len(df) >= 7 else df['Close'].mean()
    features['ema_3'] = df['Close'].ewm(span=3).mean().iloc[-1]
    
    features['price_to_sma3'] = df['Close'].iloc[-1] / features['sma_3']
    features['price_to_sma7'] = df['Close'].iloc[-1] / features['sma_7']
    
    features['price_volatility'] = df['Close'].pct_change().std()
    features['high_low_ratio'] = df['High'].iloc[-1] / df['Low'].iloc[-1]
    features['close_to_high_ratio'] = df['Close'].iloc[-1] / df['High'].iloc[-1]
    features['close_to_low_ratio'] = df['Close'].iloc[-1] / df['Low'].iloc[-1]
    
    features['volume_sma'] = df['Volume'].rolling(3).mean().iloc[-1]
    features['volume_ratio'] = df['Volume'].iloc[-1] / features['volume_sma'] if features['volume_sma'] > 0 else 1
    features['price_volume'] = df['Close'].iloc[-1] * df['Volume'].iloc[-1]
    
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=7).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=7).mean()
    rs = gain / loss
    features['rsi'] = 100 - (100 / (1 + rs.iloc[-1])) if not np.isnan(rs.iloc[-1]) and rs.iloc[-1] != 0 else 50
    
    ema_12 = df['Close'].ewm(span=3).mean()  # Shorter period due to 7-day data
    ema_26 = df['Close'].ewm(span=5).mean()
    features['macd'] = (ema_12.iloc[-1] - ema_26.iloc[-1]) / df['Close'].iloc[-1]
    
    sma_bb = df['Close'].rolling(5).mean()
    std_bb = df['Close'].rolling(5).std()
    upper_bb = sma_bb + (std_bb * 2)
    lower_bb = sma_bb - (std_bb * 2)
    features['bb_position'] = (df['Close'].iloc[-1] - lower_bb.iloc[-1]) / (upper_bb.iloc[-1] - lower_bb.iloc[-1]) if (upper_bb.iloc[-1] - lower_bb.iloc[-1]) != 0 else 0.5
    
    recent_high = df['High'].rolling(7).max().iloc[-1]
    recent_low = df['Low'].rolling(7).min().iloc[-1]
    features['distance_to_high'] = (recent_high - df['Close'].iloc[-1]) / df['Close'].iloc[-1]
    features['distance_to_low'] = (df['Close'].iloc[-1] - recent_low) / df['Close'].iloc[-1]
    
    x = np.arange(len(df))
    y = df['Close'].values
    slope = np.polyfit(x, y, 1)[0]
    features['trend_slope'] = slope / df['Close'].iloc[-1]
    
    features['gap_up'] = max(0, (df['Open'].iloc[-1] - df['Close'].iloc[-2]) / df['Close'].iloc[-2]) if len(df) >= 2 else 0
    features['gap_down'] = max(0, (df['Close'].iloc[-2] - df['Open'].iloc[-1]) / df['Close'].iloc[-2]) if len(df) >= 2 else 0
    
    features['intraday_return'] = (df['Close'].iloc[-1] - df['Open'].iloc[-1]) / df['Open'].iloc[-1]
    features['intraday_high_reach'] = (df['High'].iloc[-1] - df['Open'].iloc[-1]) / df['Open'].iloc[-1]
    features['intraday_low_reach'] = (df['Open'].iloc[-1] - df['Low'].iloc[-1]) / df['Open'].iloc[-1]
    
    up_days = (df['Close'] > df['Open']).sum()
    features['bullish_days_ratio'] = up_days / len(df)
    
    features['vpt'] = ((df['Close'].pct_change() * df['Volume']).cumsum()).iloc[-1] / df['Volume'].sum() if df['Volume'].sum() > 0 else 0
    
    return features

def generate_enhanced_dataset(tickers, start_date, end_date, save_path="enhanced_stock_features.csv"):
    print(f"Generating enhanced dataset for {len(tickers)} tickers...")
    
    all_data = []
    failed_tickers = []
    
    for i, ticker in enumerate(tickers):
        try:
            print(f"Processing {ticker} ({i+1}/{len(tickers)})...")
            
            stock = yf.Ticker(ticker)
            hist = stock.history(start=start_date, end=end_date)
            
            if len(hist) < 7:
                print(f"Insufficient data for {ticker}")
                failed_tickers.append(ticker)
                continue
            
            for j in range(7, len(hist)):
                window_data = hist.iloc[j-7:j]
                features = calculate_technical_indicators(window_data)
                
                if features is None:
                    continue
                
                features['ticker'] = ticker
                features['date'] = hist.index[j-1].strftime('%Y-%m-%d')
                
                if j < len(hist) - 1:
                    future_return = (hist['Close'].iloc[j] - hist['Close'].iloc[j-1]) / hist['Close'].iloc[j-1]
                    
                    if future_return > 0.02:  # > 2% gain
                        features['label'] = 'BUY'
                    elif future_return < -0.02:  # > 2% loss
                        features['label'] = 'SELL'
                    else:
                        features['label'] = 'HOLD'
                    
                    all_data.append(features)
            
        except Exception as e:
            print(f"Error processing {ticker}: {e}")
            failed_tickers.append(ticker)
    
    # Convert to DataFrame
    df = pd.DataFrame(all_data)
    
    if len(df) > 0:
        # Save dataset
        df.to_csv(save_path, index=False)
        print(f"Dataset saved: {save_path}")
        print(f"Total samples: {len(df)}")
        print(f"Features: {len(df.columns) - 3}")  # -3 for ticker, date, label
        print(f"Label distribution:")
        print(df['label'].value_counts())
        print(f"Failed tickers: {failed_tickers}")
        return df
    else:
        print("No data generated")
        return None

if __name__ == "__main__":
    # Test with a few popular stocks
    test_tickers = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'NVDA', 'META', 'AMZN', 'NFLX']
    
    # Generate data for the last 6 months
    end_date = datetime.now()
    start_date = end_date - timedelta(days=180)
    
    print("Testing enhanced feature generation...")
    df = generate_enhanced_dataset(
        tickers=test_tickers,
        start_date=start_date.strftime('%Y-%m-%d'),
        end_date=end_date.strftime('%Y-%m-%d'),
        save_path="test_enhanced_features.csv"
    )
    
    if df is not None:
        print("\nSample features:")
        feature_cols = [col for col in df.columns if col not in ['ticker', 'date', 'label']]
        print(f"Feature count: {len(feature_cols)}")
        print(feature_cols[:10])  # Show first 10 features
        
        # Check for missing values
        missing_pct = (df[feature_cols].isnull().sum() / len(df)) * 100
        print(f"\nMissing values: {missing_pct[missing_pct > 0].head()}")
        
        print(f"\nEnhanced dataset ready with {len(feature_cols)} features!")
    else:
        print("Failed to generate dataset")
