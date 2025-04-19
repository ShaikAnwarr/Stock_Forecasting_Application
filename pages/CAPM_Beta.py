import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
import plotly.express as px
import statsmodels.api as sm
import datetime

# Page setup
st.set_page_config(page_title="CAPM Beta Calculator", page_icon="📊", layout="wide")
st.title("📊 CAPM Beta & Return Calculator")

# Sidebar for input
col1, col2 = st.columns([1, 1])
with col1:
    stock = st.selectbox("Choose a stock", ('TSLA', 'AAPL', 'NFLX', 'MSFT', 'MGM', 'AMZN', 'NVDA', 'GOOGL'))
with col2:
    num_years = st.number_input("Number of Years", 1, 10, value=1)

try:
    # Define date range
    end_date = datetime.datetime.now()
    start_date = end_date - datetime.timedelta(days=365 * num_years)

    # Download stock & market data
    stock_data = yf.download(stock, start=start_date, end=end_date)
    market_data = yf.download('^GSPC', start=start_date, end=end_date)

    # Reset index to get 'Date' column
    stock_data.reset_index(inplace=True)
    market_data.reset_index(inplace=True)

    # Calculate daily returns
    stock_data['Stock Return'] = stock_data['Close'].pct_change() * 100
    market_data['Market Return'] = market_data['Close'].pct_change() * 100

    # Merge on Date
    merged_data = pd.merge(stock_data[['Date', 'Stock Return']],
                           market_data[['Date', 'Market Return']],
                           on='Date')

    # Drop NaNs
    merged_data.dropna(inplace=True)

    # CAPM Regression
    X = sm.add_constant(merged_data['Market Return'])
    Y = merged_data['Stock Return']
    model = sm.OLS(Y, X).fit()

    # Get alpha, beta
    alpha = model.params['const']
    beta = model.params['Market Return']

    # CAPM Expected Return
    rf = 0.0  # risk-free rate
    rm = merged_data['Market Return'].mean() * 252  # Annualized market return
    expected_return = rf + beta * (rm - rf)

    # Display results
    st.subheader("📈 Results")
    st.markdown(f"**Beta (β): `{beta:.4f}`**")
    st.markdown(f"**Alpha (α): `{alpha:.4f}`**")
    st.markdown(f"**Expected Annual Return: `{expected_return:.2f}%`**")

    # Flatten column names if needed
    merged_data.columns = merged_data.columns.map(lambda x: x if isinstance(x, str) else x[0])

    # Plot
    fig = px.scatter(
        merged_data,
        x='Market Return',
        y='Stock Return',
        title=f'{stock} vs Market (CAPM Analysis)',
        trendline='ols',
        labels={'Market Return': 'Market Return (%)', 'Stock Return': 'Stock Return (%)'}
    )
    st.plotly_chart(fig, use_container_width=True)

    # Explanation
    st.markdown("---")
    st.markdown("### ℹ️ Interpretation")
    st.markdown("""
    - **Beta > 1** → More volatile than the market.
    - **Beta < 1** → Less volatile than the market.
    - **Beta ≈ 1** → Moves similar to the market.
    - **Alpha** shows extra return beyond market prediction.
    """)

except Exception as e:
    st.error(f"❌ Error fetching or processing data: `{e}`")
