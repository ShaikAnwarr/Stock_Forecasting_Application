# import plotly.graph_objects as go
# import dateutil
# import pandas as pd
# import datetime

# def plotly_table(dataframe):
#     headerColor = 'grey'
#     rowEvenColor = '#f8fafd'
#     rowOddColor = '#e1efff'
#     fig = go.Figure(data=[go.Table(
#     header=dict(
#         values=["<b><b>"]+["<b>"+str(i)[:10]+"<b>" for i in dataframe.columns],
#         line_colors='#0078ff', fill_color='#0078ff',
#         align='center', font=dict(color='white', size=15),height=35,
#     ),
#     cells=dict(
#         values=[["<b>"+str(i)+"<b>" for i in dataframe.index]]+[dataframe[i] for i in dataframe.columns], fill_color =[[rowOddColor, rowEvenColor]], align='left', line_colors=['white'],font=dict(color=["black"], size=15)
#     ))
#     ])
#     fig.update_layout(height= 400, margin=dict(l=0, r=0, t=0, b=0))
#     return fig

import plotly.graph_objects as go
import pandas_ta as pta
import dateutil
from dateutil.relativedelta import relativedelta
import datetime


def plotly_table(dataframe):
    headerColor = '#0078ff'
    rowEvenColor = '#f8fafd'
    rowOddColor = '#e1efff'

    fig = go.Figure(data=[go.Table(
        header=dict(
            values=["<b>Metric</b>"] + ["<b>" + str(col) + "</b>" for col in dataframe.columns],
            fill_color=headerColor,
            align='center',
            font=dict(color='white', size=15),
            height=35,
        ),
        cells=dict(
            values=[[str(i) for i in dataframe.index]] + [dataframe[col].tolist() for col in dataframe.columns],
            fill_color=[
                           [rowOddColor if i % 2 == 0 else rowEvenColor for i in range(len(dataframe))]
                       ] * (len(dataframe.columns) + 1),  # Ensure all columns are colored correctly
            align='left',
            line_color='white',
            font=dict(color='black', size=15),
        )
    )])

    fig.update_layout(height=400, margin=dict(l=0, r=0, t=0, b=0))
    return fig


def filter_data(dataframe, num_period):
    if num_period == '1mo':
        date = dataframe.index[-1] + relativedelta(months=-1)
    elif num_period == '5d':
        date = dataframe.index[-1] + relativedelta(days=-5)
    elif num_period == '6mo':
        date = dataframe.index[-1] + relativedelta(months=-6)
    elif num_period == '1y':
        date = dataframe.index[-1] + relativedelta(years=-1)
    elif num_period == '5y':
        date = dataframe.index[-1] + relativedelta(years=-5)
    elif num_period == 'ytd':
        date = datetime.datetime(dataframe.index[-1].year, 1, 1).strftime("%Y-%m-%d")
    else:
        date = dataframe.index[0]

    return dataframe.reset_index()[dataframe.reset_index()['Date'] > date]


def close_chart(dataframe, num_period=False):
    if num_period:
        dataframe = filter_data(dataframe, num_period)

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=dataframe['Date'], y=dataframe['Open'],
        mode='lines',
        name='Open',
        line=dict(width=2, color='#5ab7ff')
    ))

    fig.add_trace(go.Scatter(
        x=dataframe['Date'], y=dataframe['Close'],
        mode='lines',
        name='Close',
        line=dict(width=2, color='black')
    ))

    fig.add_trace(go.Scatter(  # Fixed the typo here
        x=dataframe['Date'], y=dataframe['High'],
        mode='lines',
        name='High',
        line=dict(width=2, color='#0078ff')
    ))

    fig.add_trace(go.Scatter(
        x=dataframe['Date'], y=dataframe['Low'],
        mode='lines',
        name='Low',
        line=dict(width=2, color='red')
    ))

    fig.update_xaxes(rangeslider_visible=True)

    fig.update_layout(
        height=500,
        margin=dict(l=0, r=20, t=20, b=0),
        plot_bgcolor='white',  # Fixed typo
        paper_bgcolor='#e1efff',  # Fixed typo
        legend=dict(
            yanchor="top",
            xanchor="right"
        )
    )

    return fig


import plotly.graph_objects as go


def candlestick(dataframe, num_period):
    dataframe = filter_data(dataframe, num_period)  # Ensure this function works correctly

    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=dataframe['Date'],
        open=dataframe['Open'],
        high=dataframe['High'],
        low=dataframe['Low'],
        close=dataframe['Close'],
        name="Candlestick"
    ))

    fig.update_layout(
        showlegend=False,
        height=500,
        margin=dict(l=0, r=20, t=20, b=0),
        plot_bgcolor='white',
        paper_bgcolor='#e1e1ff'  # Fixed typo from `#elefff`
    )

    return fig


def RSI(dataframe, num_period):
    dataframe['RSI'] = pta.rsi(dataframe['Close'])  # Fixed syntax

    dataframe = filter_data(dataframe, num_period)  # Ensure filter_data is defined elsewhere

    fig = go.Figure()

    # RSI line
    fig.add_trace(go.Scatter(
        x=dataframe['Date'],
        y=dataframe['RSI'],
        name='RSI',
        line=dict(width=2, color='orange')  # Fixed line formatting
    ))

    # Overbought line (RSI 70)
    fig.add_trace(go.Scatter(
        x=dataframe['Date'],
        y=[70] * len(dataframe),
        name='Overbought',
        line=dict(width=2, color='red', dash="dash")  # Fixed incorrect syntax
    ))

    # Oversold line (RSI 30)
    fig.add_trace(go.Scatter(
        x=dataframe['Date'],
        y=[30] * len(dataframe),
        fill='tonexty',
        name='Oversold',
        line=dict(width=2, color='#79da84', dash='dash')  # Fixed incorrect syntax
    ))

    fig.update_layout(
        yaxis_range=(0, 100),
        height=200,
        plot_bgcolor='white',
        paper_bgcolor='#e1efff',
        margin=dict(l=0, r=10, t=20, b=20),  # Fixed margin keys
        legend=dict(
            orientation="h",
            yanchor="top",
            y=1.02,
            xanchor="right",
            x=1
        )
    )

    return fig


def Moving_average(dataframe, num_period):
    dataframe['SMA_50'] = pta.sma(dataframe['Close'], 50)  # Fixed syntax

    dataframe = filter_data(dataframe, num_period)  # Ensure filter_data is defined elsewhere

    fig = go.Figure()

    # Open price line
    fig.add_trace(go.Scatter(
        x=dataframe['Date'],
        y=dataframe['Open'],
        mode='lines',
        name='Open',
        line=dict(width=2, color='#5ab7ff')
    ))

    # Close price line
    fig.add_trace(go.Scatter(
        x=dataframe['Date'],
        y=dataframe['Close'],
        mode='lines',
        name='Close',
        line=dict(width=2, color='black')
    ))

    # High price line
    fig.add_trace(go.Scatter(
        x=dataframe['Date'],
        y=dataframe['High'],
        mode='lines',
        name='High',
        line=dict(width=2, color='#0078ff')
    ))

    # Low price line
    fig.add_trace(go.Scatter(
        x=dataframe['Date'],
        y=dataframe['Low'],
        mode='lines',
        name='Low',
        line=dict(width=2, color='red')
    ))

    # 50-day SMA line
    fig.add_trace(go.Scatter(
        x=dataframe['Date'],
        y=dataframe['SMA_50'],
        mode='lines',
        name='SMA 50',
        line=dict(width=2, color='purple')
    ))

    fig.update_xaxes(rangeslider_visible=True)

    fig.update_layout(
        height=500,
        margin=dict(l=0, r=20, t=20, b=20),  # Fixed margin keys
        plot_bgcolor='white',
        paper_bgcolor='#e1efff',
        legend=dict(
            yanchor="top",
            xanchor="right"
        )
    )

    return fig


def MACD(dataframe, num_period):
    macd = pta.macd(dataframe['Close']).iloc[:, 0]  # MACD Line
    macd_signal = pta.macd(dataframe['Close']).iloc[:, 1]  # Signal Line
    macd_hist = pta.macd(dataframe['Close']).iloc[:, 2]  # Histogram

    dataframe['MACD'] = macd
    dataframe['MACD Signal'] = macd_signal
    dataframe['MACD Hist'] = macd_hist

    dataframe = filter_data(dataframe, num_period)  # Ensure filter_data function is defined

    fig = go.Figure()

    # MACD Line
    fig.add_trace(go.Scatter(
        x=dataframe['Date'],
        y=dataframe['MACD'],
        mode='lines',
        name='MACD',
        line=dict(width=2, color='orange')
    ))

    # MACD Signal Line
    fig.add_trace(go.Scatter(
        x=dataframe['Date'],
        y=dataframe['MACD Signal'],
        mode='lines',
        name='MACD Signal',
        line=dict(width=2, color='red', dash='dash')
    ))

    # MACD Histogram
    fig.add_trace(go.Bar(
        x=dataframe['Date'],
        y=dataframe['MACD Hist'],
        name='MACD Histogram',
        marker=dict(
            color=['red' if val < 0 else 'green' for val in macd_hist]
        )
    ))

    fig.update_layout(
        height=200,
        plot_bgcolor='white',
        paper_bgcolor='#e1efff',
        margin=dict(l=0, r=0, t=0, b=20),
        legend=dict(
            orientation="h",
            yanchor="top",
            y=1.02,
            xanchor="right",
            x=1
        )
    )

    return fig


def Moving_average_forecast(forecast):
    fig = go.Figure()

    # Plot historical close prices
    fig.add_trace(go.Scatter(
        x=forecast.index[:-30],
        y=forecast['Close'].iloc[:-30],
        mode='lines',
        name='Close Price',
        line=dict(width=2, color='black')
    ))

    # Plot future forecasted close prices
    fig.add_trace(go.Scatter(
        x=forecast.index[-30:],
        y=forecast['Close'].iloc[-30:],
        mode='lines',
        name='Future Close Price',
        line=dict(width=2, color='red')
    ))

    # Enable range slider
    fig.update_xaxes(rangeslider_visible=True)

    # Customize layout
    fig.update_layout(
        height=500,
        margin=dict(l=0, r=20, t=20, b=20),
        plot_bgcolor='white',
        paper_bgcolor='#E1EFFF',
        legend=dict(yanchor="top", xanchor="right")
    )

    return fig