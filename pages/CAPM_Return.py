import streamlit as st
import pandas as pd
import yfinance as yf
import pandas_datareader.data as web
from datetime import date
import datetime

st.set_page_config(page_title="CAPM",
                   page_icon="chart_with_upwards_trend",
                   layout='wide')

st.title("Capital Asset Pricing Model")

# Getting input from user
col1, col2 = st.columns([1, 1])
with col1:
    stock_list = st.multiselect("Choose 4 stocks", ('TSLA', 'AAPL', 'NFLX', 'MSFT', 'MGM', 'AMZN', 'NVDA', 'GOOGL'),
                                ['TSLA', 'AAPL', 'AMZN', 'GOOGL'])
with col2:
    num_years = int(st.number_input("Number of years", 1, 10, value=1))

# Downloading data for SP500
end = datetime.date.today()
start = datetime.date(end.year - num_years, end.month, end.day)
SP500 = web.DataReader(['sp500'], 'fred', start, end)

stocks_df = pd.DataFrame()

for stock in stock_list:
    data = yf.download(stock, period=f'{num_years}y')
    stocks_df[f'{stock}'] = data['Close']

stocks_df.reset_index(inplace=True)
SP500.reset_index(inplace=True)
SP500.columns = ['Date', 'sp500']

# Fix timezone issues
stocks_df['Date'] = stocks_df['Date'].dt.tz_localize(None)
SP500['Date'] = SP500['Date'].dt.tz_localize(None)

stocks_df['Date'] = stocks_df['Date'].astype('datetime64[ns]')
stocks_df['Date'] = stocks_df['Date'].apply(lambda x: str(x)[:10])
stocks_df['Date'] = pd.to_datetime(stocks_df['Date'])

stocks_df = pd.merge(stocks_df, SP500, on='Date', how='inner')

col1, col2 = st.columns([1, 1])
with col1:
    st.markdown("### Dataframe head")
    st.dataframe(stocks_df.head(), use_container_width=True)
with col2:
    st.markdown("### Dataframe tail")
    st.dataframe(stocks_df.tail(), use_container_width=True)