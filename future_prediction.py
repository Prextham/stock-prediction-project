# ==============================================================================
# FINAL, COMPLETE VALIDATION & FORECASTING SCRIPT
# ==============================================================================
# This script performs two key tasks:
# 1. Backtesting: Trains the Ultimate CNN-LSTM on a historical period and evaluates
#    its performance on a recent test period with full accuracy metrics.
# 2. Forecasting: Re-trains the full ensemble on all available data up to a
#    user-defined date to provide a single, high-confidence forecast for the next day.
# ==============================================================================

# --- Step 1: Imports ---
import yfinance as yf
import pandas as pd
import numpy as np
from pytrends.request import TrendReq
import pandas_ta as ta
from polygon import RESTClient as PolygonClient
from newsapi import NewsApiClient
from alpha_vantage.fundamentaldata import FundamentalData
from datetime import date, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
import math
import os
import pmdarima as pm

# Suppress TensorFlow INFO/WARNING messages for a cleaner output
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Dense, Dropout, LayerNormalization, MultiHeadAttention, GlobalAveragePooling1D, Conv1D, MaxPooling1D, LSTM

# ==============================================================================
# --- Step 2: Configuration & User Input ---
# ==============================================================================
print("--- Stock Prediction Validator & Forecaster ---")
STOCK_TICKER = input("Enter the stock ticker (e.g., GOOGL, NVDA, AAPL): ").upper()
COMPANY_NAME = input(f"Enter the company name for news search (e.g., Google, NVIDIA): ")
POLYGON_API_KEY = input("Enter your Polygon.io API key: ")
ALPHA_VANTAGE_API_KEY = input("Enter your Alpha Vantage API key: ")
NEWS_API_KEY = input("Enter your NewsAPI.org API key: ")

# --- Interactive Dates ---
VALIDATION_DATE = input(f"Enter the date you want to predict (YYYY-MM-DD): ") 
# --- THIS IS THE CORRECTED LINE ---
TRAINING_END_DATE = (pd.to_datetime(VALIDATION_DATE) - timedelta(days=1)).strftime('%Y-%m-%d')

# -- Model Hyperparameters --
START_DATE = "2020-01-01"
SEQUENCE_LENGTH = 60
SPLIT_RATIO = 0.8
EPOCHS = 50 
BATCH_SIZE = 32
ADJUSTMENT_MULTIPLIER = 1.03 # <-- NEW: Your multiplier

# ==============================================================================
# --- Step 3: Data Collection and Feature Engineering ---
# ==============================================================================
def get_master_dataframe(ticker, company_name, polygon_key, alpha_vantage_key, newsapi_key, start, end):
    """Fetches all data and engineers all features."""
    print(f"\n--- Phase 1: Collecting data from {start} to {end} ---")
    try:
        raw_price_df = yf.download(ticker, start=start, end=end, progress=False)
        vix_df = yf.download('^VIX', start=start, end=end, progress=False)
    except Exception as e:
        print(f"FATAL ERROR: Could not fetch price data: {e}. Exiting.")
        return None
    try:
        pytrends = TrendReq(hl='en-US', tz=360, timeout=(10, 25), retries=3)
        pytrends.build_payload(kw_list=[company_name], timeframe=f'{start} {end}')
        trends_df = pytrends.interest_over_time()
    except Exception:
        trends_df = pd.DataFrame() 
    all_news = []
    try:
        polygon_client = PolygonClient(polygon_key)
        for a in polygon_client.list_ticker_news(ticker, limit=1000, published_utc_gte=start):
            all_news.append({'date': a.published_utc, 'title': a.title, 'sentiment': a.sentiment})
    except Exception: pass
    try:
        newsapi = NewsApiClient(api_key=newsapi_key)
        news_query = f'"{company_name}" OR "{ticker}"'
        all_articles = newsapi.get_everything(q=news_query, language='en', sort_by='publishedAt', page_size=100)
        for a in all_articles['articles']:
            all_news.append({'date': a['publishedAt'], 'title': a['title'], 'sentiment': 'finbert'})
    except Exception: pass
    raw_news_df = pd.DataFrame(all_news)
    try:
        fd = FundamentalData(key=alpha_vantage_key, output_format='pandas')
        earnings_data, _ = fd.get_earnings(ticker)
        earnings_data['reportedEPS'] = pd.to_numeric(earnings_data['reportedEPS'], errors='coerce')
        earnings_df = earnings_data[['fiscalDateEnding', 'reportedEPS']].copy()
        earnings_df.dropna(inplace=True)
    except Exception:
        earnings_df = pd.DataFrame()
    master_df = raw_price_df.copy()
    if isinstance(master_df.columns, pd.MultiIndex):
        master_df.columns = master_df.columns.get_level_values(0)
    master_df['VIX_Close'] = vix_df['Close']
    if not trends_df.empty and company_name in trends_df.columns:
        master_df = master_df.join(trends_df[company_name])
        master_df.rename(columns={company_name: 'google_trends'}, inplace=True)
    if not earnings_df.empty:
        earnings_df.set_index(pd.to_datetime(earnings_df['fiscalDateEnding']), inplace=True)
        eps_daily = earnings_df['reportedEPS'].reindex(master_df.index, method='ffill')
        master_df['EPS'] = eps_daily
    master_df.ta.rsi(length=14, append=True)
    master_df.ta.macd(append=True)
    master_df['returns'] = master_df['Close'].pct_change()
    if not raw_news_df.empty:
        raw_news_df['Date'] = pd.to_datetime(raw_news_df['date']).dt.date
        finbert_articles = raw_news_df[raw_news_df['sentiment'] == 'finbert'].copy()
        if not finbert_articles.empty:
            from transformers import pipeline
            sentiment_analyzer = pipeline('sentiment-analysis', model="ProsusAI/finbert", device=-1)
            sentiment_map = {'positive': 1, 'negative': -1, 'neutral': 0}
            finbert_articles['sentiment'] = [sentiment_map[s['label']] * s['score'] for s in sentiment_analyzer(finbert_articles['title'].tolist())]
        polygon_articles = raw_news_df[raw_news_df['sentiment'] != 'finbert'].copy()
        processed_news = pd.concat([finbert_articles, polygon_articles])
        processed_news['sentiment'] = pd.to_numeric(processed_news['sentiment'], errors='coerce')
        daily_sentiment = processed_news.groupby('Date')['sentiment'].mean().to_frame()
        daily_sentiment.index = pd.to_datetime(daily_sentiment.index)
        master_df = master_df.join(daily_sentiment)
        master_df.rename(columns={'sentiment': 'avg_sentiment'}, inplace=True)
    master_df.fillna(0, inplace=True)
    master_df.dropna(inplace=True)
    return master_df

# ==============================================================================
# --- Step 4: Model Training and Evaluation Functions ---
# ==============================================================================

def train_dl_model(model_name, X_train, y_train, X_test, y_test):
    """Helper function to build and train deep learning models."""
    print(f"--- Training {model_name} Model ---")
    if model_name == 'CNN-LSTM':
        model = Sequential([
            Conv1D(filters=128, kernel_size=3, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])),
            MaxPooling1D(pool_size=2), LSTM(units=100), Dropout(0.2),
            Dense(units=50, activation='relu'), Dense(units=1)
        ])
    elif model_name == 'Transformer':
        def transformer_encoder(inputs, head_size=256, num_heads=4, ff_dim=4, dropout=0.1):
            x = MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=dropout)(inputs, inputs)
            x = LayerNormalization(epsilon=1e-6)(inputs + x)
            ffn = Dense(ff_dim, activation="relu")(x); ffn = Dense(inputs.shape[-1])(ffn)
            return LayerNormalization(epsilon=1e-6)(x + ffn)
        input_layer = Input(shape=(X_train.shape[1], X_train.shape[2]))
        x = transformer_encoder(input_layer)
        x = GlobalAveragePooling1D(data_format="channels_last")(x)
        x = Dense(64, activation="relu")(x); output_layer = Dense(1)(x)
        model = Model(inputs=input_layer, outputs=output_layer)
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_data=(X_test, y_test), verbose=0)
    return model

def run_backtest(master_df, ticker):
    """Performs the historical validation (backtest) on the most recent 20% of data."""
    print("\n--- PART 1: Performing Historical Validation (Backtest) ---")
    
    # Prepare data and split
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(master_df)
    scaled_df = pd.DataFrame(scaled_data, columns=master_df.columns, index=master_df.index)
    
    TARGET_COLUMN_INDEX = scaled_df.columns.get_loc('Close')
    X, y = [], []
    for i in range(SEQUENCE_LENGTH, len(scaled_data)):
        X.append(scaled_df.iloc[i-SEQUENCE_LENGTH:i].values)
        y.append(scaled_df.iloc[i, TARGET_COLUMN_INDEX])
    X, y = np.array(X), np.array(y)
    
    split_index = int(len(X) * SPLIT_RATIO)
    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]
    
    # Train only the CNN-LSTM for the backtest
    model_cnn_lstm = train_dl_model('CNN-LSTM', X_train, y_train, X_test, y_test)
    
    # Generate predictions
    preds_cnn_lstm_scaled = model_cnn_lstm.predict(X_test)
    
    # Inverse transform
    actual_prices = master_df['Close'].iloc[-len(X_test):]
    dummy = np.zeros((len(preds_cnn_lstm_scaled), X_train.shape[2]))
    dummy[:, TARGET_COLUMN_INDEX] = preds_cnn_lstm_scaled.ravel()
    predictions = scaler.inverse_transform(dummy)[:, TARGET_COLUMN_INDEX]
    
    # --- NEW: Apply multiplier to the backtest prediction ---
    predictions_adjusted = predictions * ADJUSTMENT_MULTIPLIER
    
    results_df = pd.DataFrame({'Actual': actual_prices, 'Predicted': predictions, 'Adjusted': predictions_adjusted})
    
    # Calculate and plot metrics
    actual = results_df['Actual']
    predicted = results_df['Adjusted'] # Evaluate the adjusted prediction
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    mae = mean_absolute_error(actual, predicted)
    mape = mean_absolute_percentage_error(actual, predicted)
    r2 = r2_score(actual, predicted)
    dir_df = results_df.copy()
    dir_df['Actual_Move'] = dir_df['Actual'].diff()
    dir_df['Predicted_Move'] = dir_df['Adjusted'] - dir_df['Actual'].shift(1)
    dir_df['Correct_Direction'] = np.sign(dir_df['Actual_Move']) == np.sign(dir_df['Predicted_Move'])
    directional_accuracy = dir_df['Correct_Direction'].mean() * 100

    print("\n\n--- Historical Performance Metrics (ADJUSTED CNN-LSTM Backtest) ---")
    print(f"Directional Accuracy: {directional_accuracy:.2f}%")
    print(f"RMSE: ${rmse:.2f}, MAE: ${mae:.2f}, MAPE: {mape:.2%}, R²: {r2:.2f}")
    
    plt.figure(figsize=(20, 10))
    plt.plot(actual, color='blue', label='Actual Stock Price')
    plt.plot(predicted, color='green', linestyle='--', label=f'Adjusted Prediction (RMSE: ${rmse:.2f})')
    plt.plot(results_df['Predicted'], color='red', linestyle=':', label='Original Prediction', alpha=0.5)
    plt.title(f'{ticker} - Historical Backtest Performance (CNN-LSTM with Adjustment)', fontsize=16)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Stock Price (USD)', fontsize=12)
    plt.legend(fontsize=12)
    plt.show()

def run_forecast(master_df, ticker, validation_date):
    """Re-trains the full ensemble on all data and forecasts the validation date."""
    print("\n\n--- PART 2: Generating Final Forecast for Tomorrow ---")
    price_data = master_df['Close']
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(master_df)
    scaled_data = scaler.transform(master_df)
    TARGET_COLUMN_INDEX = master_df.columns.get_loc('Close')
    
    X_full, y_full = [], []
    for i in range(SEQUENCE_LENGTH, len(scaled_data)):
        X_full.append(scaled_data[i-SEQUENCE_LENGTH:i, :])
        y_full.append(scaled_data[i, TARGET_COLUMN_INDEX])
    X_full, y_full = np.array(X_full), np.array(y_full)

    final_arima = pm.auto_arima(price_data.values, trace=False, suppress_warnings=True)
    final_cnn_lstm = train_dl_model('CNN-LSTM', X_full, y_full, X_full, y_full)
    final_transformer = train_dl_model('Transformer', X_full, y_full, X_full, y_full)

    last_sequence_scaled = np.array([scaled_data[-SEQUENCE_LENGTH:]])
    
    pred_cnn_lstm_scaled = np.clip(final_cnn_lstm.predict(last_sequence_scaled, verbose=0)[0,0], 0, 1)
    pred_transformer_scaled = np.clip(final_transformer.predict(last_sequence_scaled, verbose=0)[0,0], 0, 1)
    pred_arima_price = final_arima.predict(n_periods=1)[0]
    pred_arima_scaled = np.clip((pred_arima_price - scaler.min_[TARGET_COLUMN_INDEX]) / scaler.scale_[TARGET_COLUMN_INDEX], 0, 1)
    
    w_cnn_lstm = 0.70; w_transformer = 0.15; w_arima = 0.15
    ensembled_pred_scaled = (w_cnn_lstm * pred_cnn_lstm_scaled + w_transformer * pred_transformer_scaled + w_arima * pred_arima_scaled)
    
    dummy = np.zeros((1, len(master_df.columns)))
    dummy[0, TARGET_COLUMN_INDEX] = ensembled_pred_scaled
    forecast_price = scaler.inverse_transform(dummy)[0, TARGET_COLUMN_INDEX]
    
    # --- NEW: Apply multiplier to the final forecast ---
    forecast_price_adjusted = forecast_price * ADJUSTMENT_MULTIPLIER

    # Get the actual price for comparison
    try:
        validation_plus_one = (pd.to_datetime(validation_date) + timedelta(days=1)).strftime('%Y-%m-%d')
        actual_price_df = yf.download(ticker, start=validation_date, end=validation_plus_one, progress=False)
        actual_price_today = float(actual_price_df['Close'].iloc[0])
        validation_possible = True
    except Exception:
        actual_price_today = 0
        validation_possible = False
    
    last_actual_price = master_df['Close'].iloc[-1]
    
    print("\n\n=============================================")
    print(f"Out-of-Sample Validation for {ticker} on {validation_date}:")
    print(f"Original Predicted Close: ${forecast_price:.2f}")
    print(f"**Adjusted Predicted Close (x{ADJUSTMENT_MULTIPLIER}): ${forecast_price_adjusted:.2f}**")
    if validation_possible:
        print(f"ACTUAL Close:    ${actual_price_today:.2f}")
        print(f"Adjusted Prediction Error: ${forecast_price_adjusted - actual_price_today:+.2f}")
    print("=============================================")

    plt.figure(figsize=(16, 8))
    plt.plot(master_df.index[-60:], master_df['Close'][-60:], label='Actual Historical Price', color='blue')
    forecast_date = pd.to_datetime(validation_date)
    plt.plot([forecast_date], [forecast_price_adjusted], label=f'Adjusted Predicted Price', color='green', marker='*', markersize=15)
    if validation_possible:
        plt.plot([forecast_date], [actual_price_today], label=f'Actual Price', color='black', marker='X', markersize=15)
    plt.title(f'{ticker} - Out-of-Sample Forecast (with Adjustment)', fontsize=16)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Stock Price (USD)', fontsize=12)
    plt.legend(fontsize=12)
    plt.show()

# ==============================================================================
# --- Step 5: Main Execution Block ---
# ==============================================================================
if __name__ == "__main__":
    sns.set_theme(style="whitegrid")
    
    master_df = get_master_dataframe(STOCK_TICKER, COMPANY_NAME, POLYGON_API_KEY, ALPHA_VANTAGE_API_KEY, NEWS_API_KEY, START_DATE, TRAINING_END_DATE)
    
    if master_df is not None and not master_df.empty:
        # 1. Run the historical backtest
        run_backtest(master_df, STOCK_TICKER)
        
        # 2. Run the forward-looking forecast
        run_forecast(master_df, STOCK_TICKER, VALIDATION_DATE)
        
        print("\nProcess finished.")
    else:
        print("\nCould not generate forecast because data collection failed.")

