import streamlit as st
import pandas as pd
import yfinance as yf
import pandas_datareader.data as web
import datetime
import plotly.express as px
import numpy as np


# Set up Streamlit page
st.set_page_config(page_title="CAPM",
                   page_icon="chart_with_upwards_trend",
                   layout='wide')

st.title("Capital Asset Pricing Model")

# Getting input from user
col1, col2 = st.columns([1, 1])
with col1:
    stock_list = st.multiselect("Choose 4 stocks",
                                ('TSLA', 'AAPL', 'NFLX', 'MSFT', 'MGM', 'AMZN', 'NVDA', 'GOOGL'),
                                ['TSLA', 'AAPL', 'AMZN', 'GOOGL'])
with col2:
    num_years = int(st.number_input("Number of years", 1, 10, value=1))

# Downloading data for SP500
try:
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


    # Interactive Plot Function
    def interactive_plot(df):
        fig = px.line()
        for i in df.columns[1:]:
            fig.add_scatter(x=df['Date'], y=df[i], name=i)
        fig.update_layout(width=450, margin=dict(l=20, r=20, t=50, b=20),
                          legend=dict(orientation='h', yanchor='bottom',
                                      y=1.02, xanchor='right', x=1))
        return fig


    # Plot Price of all Stocks
    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown("### Price of all Stocks")
        st.plotly_chart(interactive_plot(stocks_df))


    # ----------------------------------------------------------------------------------------------

    # Function to normalize the prices based on the initial price
    def normalize(df):
        for i in df.columns[1:]:  # Fixed the typo here
            df[i] = df[i] / df[i][0]
        return df


    with col2:
        st.markdown("### Price of all Stocks (After Normalizing)")
        st.plotly_chart(interactive_plot(normalize(stocks_df)))


    # --------------------------------------------------------------------------------------------------

    # Function to calculate daily returns
    def daily_return(df):
        df_daily_return = df.copy()  # Create a copy to avoid modifying the original DataFrame
        for i in df.columns[1:]:
            for j in range(1, len(df)):
                df_daily_return[i][j] = ((df[i][j] - df[i][j - 1]) / df[i][j - 1]) * 100
            df_daily_return[i][0] = 0
        return df_daily_return


    # Calculate daily returns
    stocks_daily_return = daily_return(stocks_df)
    print(stocks_daily_return.head())


    # -----------------------------------------------------------

    # Function to create beta
    def calculate_beta(stocks_daily_return, stock):
        rm = stocks_daily_return['sp500'].mean() * 252
        b, a = np.polyfit(stocks_daily_return["sp500"], stocks_daily_return[stock], 1)
        return b, a


    # Calculate beta and alpha
    beta = {}
    alpha = {}

    for i in stocks_daily_return.columns:
        if i != 'Date' and i != 'sp500':
            b, a = calculate_beta(stocks_daily_return, i)
            beta[i] = b
            alpha[i] = a

    print("Beta:", beta)
    print("Alpha:", alpha)

    # Create DataFrame for Beta Values
    beta_df = pd.DataFrame(columns=['stock', 'beta value'])
    beta_df['stock'] = beta.keys()
    beta_df['beta value'] = [str(round(i, 2)) for i in beta.values()]

    # Display Beta Values
    with col1:
        st.markdown("### Calculated Beta Values")
        st.dataframe(beta_df, use_container_width=True)

    # Calculate Returns using CAPM
    rf = 0  # Risk-free rate
    rm = stocks_daily_return['sp500'].mean() * 252

    return_df = pd.DataFrame()
    return_value = []

    for stock, value in beta.items():
        return_value.append(str(round(rf + (value * (rm - rf)), 2)))

    return_df['Stock'] = beta.keys()
    return_df['Return value'] = return_value

    # Display CAPM Returns
    with col2:
        st.markdown("### Calculated Return using CAPM")
        st.dataframe(return_df, use_container_width=True)

except Exception as e:
    st.write("An error occurred:", e)