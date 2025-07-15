import streamlit as st
import pandas as pd

from pages.utilis.model_train import (
    evaluate_model,
    get_data,
    get_rolling_mean,
    get_differencing_order,
    scaling,
    inverse_scaling,
    get_forecast
)

from pages.utilis.plotly_figure import plotly_table, Moving_average_forecast  # type: ignore

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

# ✅ SAFE GUARD: Check if data is empty or None
if close_price is None or len(close_price) == 0:
    st.error(f"❌ No price data found for '{ticker}'. You might be rate-limited by Yahoo Finance. Please try again later.")
    st.stop()

# Apply rolling mean and differencing order
rolling_price = get_rolling_mean(close_price)

# ✅ SAFE GUARD: Again check rolling_price before calling differencing
if rolling_price is None or len(rolling_price) == 0:
    st.error("❌ Not enough rolling price data to proceed. Please check the ticker or try again later.")
    st.stop()

try:
    differencing_order = get_differencing_order(rolling_price)
except ValueError as e:
    st.error(f"❌ {str(e)}")
    st.stop()

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
