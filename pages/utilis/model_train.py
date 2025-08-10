import yfinance as yf
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_squared_error, r2_score
from statsmodels.tsa.arima.model import ARIMA
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Function to fetch stock data
def get_data(ticker):
    stock_data = yf.download(ticker, start='2024-01-01')
    return stock_data['Close']

# Function to check stationarity using ADF test
def stationary_check(close_price):
    adf_test = adfuller(close_price)
    p_value = round(adf_test[1], 3)  # Extract p-value
    return p_value

# Function to calculate rolling mean
def get_rolling_mean(close_price):
    rolling_price = close_price.rolling(window=7).mean().dropna()
    return rolling_price

# Function to determine the differencing order (d) for ARIMA
def get_differencing_order(close_price):
    d = 0  # Initialize differencing order
    p_value = stationary_check(close_price)

    while p_value > 0.05 and d < 5:  # Limit differencing to avoid excessive loss of data
        d += 1
        close_price = close_price.diff().dropna()
        p_value = stationary_check(close_price)

    return d

# Function to fit ARIMA model
def fit_model(data, differencing_order):
    model = ARIMA(data, order=(3, differencing_order, 3))  # Adjusted order to avoid overfitting
    model_fit = model.fit()
    forecast = model_fit.get_forecast(steps=30)  # Forecast for 30 days
    predictions = forecast.predicted_mean
    return predictions

# Function to evaluate model performance
def evaluate_model(original_price, differencing_order):
    train_data, test_data = original_price[:-30], original_price[-30:]
    predictions = fit_model(train_data, differencing_order)
    rmse = np.sqrt(mean_squared_error(test_data, predictions))
    return round(rmse, 2)

# Function to scale data
def scaling(close_price):
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(np.array(close_price).reshape(-1, 1))
    return scaled_data, scaler

# Function to get forecasted values
def get_forecast(original_price, differencing_order):
    predictions = fit_model(original_price, differencing_order)
    start_date = datetime.now().strftime('%Y-%m-%d')
    end_date = (datetime.now() + timedelta(days=29)).strftime('%Y-%m-%d')
    forecast_index = pd.date_range(start=start_date, end=end_date, freq='D')
    forecast_df = pd.DataFrame(predictions, index=forecast_index, columns=['Close'])
    return forecast_df

# Function to inverse scale data
def inverse_scaling(scaler, scaled_data):
    close_price = scaler.inverse_transform(np.array(scaled_data).reshape(-1, 1))
    return close_price