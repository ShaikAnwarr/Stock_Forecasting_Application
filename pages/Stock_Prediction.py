import streamlit as st
import pandas as pd
from pages.utilis.model_train import evaluate_model, get_data, get_differencing_order, get_forecast, get_rolling_mean, inverse_scaling, scaling
from pages.utilis.model_train import get_data, get_rolling_mean, get_differencing_order, scaling, evaluate_model, get_forecast, inverse_scaling

from pages.utilis.plotly_figure import plotly_table, Moving_average_forecast # type: ignore

# Set Streamlit page configuration
st.set_page_config(
    page_title="Stock Prediction",
    page_icon="📉",
    layout="wide"
)

# Title
st.title("Stock Prediction")

# Layout with three columns
col1, col2, col3 = st.columns(3)

# Stock Ticker input
with col1:
    ticker = st.text_input('Stock Ticker', 'AAPL')

# Display stock prediction details
st.subheader(f"Predicting Next 30 Days Close Price for: {ticker}")

# Fetch stock data
close_price = get_data(ticker)

# Apply rolling mean and differencing order
rolling_price = get_rolling_mean(close_price)
differencing_order = get_differencing_order(rolling_price)

# Scale data
scaled_data, scaler = scaling(rolling_price)

# Evaluate model
rmse = evaluate_model(scaled_data, differencing_order)
st.write("**Model RMSE Score:**", rmse)

# Forecast future prices
forecast = get_forecast(scaled_data, differencing_order)
forecast['Close'] = inverse_scaling(scaler, forecast['Close'])

# Display forecast data
st.write('##### Forecast Data (Next 30 Days)')
fig_tail = plotly_table(forecast.sort_index(ascending=True).round(3))
fig_tail.update_layout(height=220)
st.plotly_chart(fig_tail, use_container_width=True)

# Concatenate rolling price with forecast
forecast_combined = pd.concat([rolling_price, forecast])

# Plot moving average forecast
st.plotly_chart(Moving_average_forecast(forecast_combined.iloc[-150:]), use_container_width=True)