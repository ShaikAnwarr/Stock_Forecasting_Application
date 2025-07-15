📊 Stock Forecasting Application

  > This repository contains a Streamlit-based stock forecasting application built to analyze, visualize, and predict stock performance. It features stock information, price prediction using ARIMA models, return and risk analysis via CAPM, and technical indicator-based insights for smarter investment decisions.

🚀 Features:

  📌 Stock Information:
  
       > Get an overview of any stock with business summary, sector classification, and key financial metrics.

🔮 Stock Price Prediction:

       > Predicts the next 30 days of closing prices using historical data and ARIMA time series models.

📈 Stock Analysis with Indicators:
  > Visual analysis powered by technical indicators including:

        RSI (Relative Strength Index)

        MACD (Moving Average Convergence Divergence)

        Simple & Exponential Moving Averages

📉 CAPM Calculations:
 > Compute expected return and beta values using the Capital Asset Pricing Model.


🧰 Requirements:
 > To run this application, install the following dependencies:
    pip install streamlit pandas yfinance plotly pandas_ta numpy scikit-learn statsmodels
  Or, use a requirements.txt to streamline setup.

📁 File Structure:

> Trading_App.py – Main Streamlit app entry point that interfaces with users.

> Stock_Prediction.py – Contains stock prediction logic using ARIMA and Linear Regression models.

> Stock_Analysis.py – Provides detailed technical analysis (RSI, MACD, moving averages).

> model_train.py – Utility functions for training and evaluating the stock prediction models.

> plotly_figure.py – Functions to generate and display interactive charts with Plotly.

> CAPM_Return.py – Calculates the expected return using the Capital Asset Pricing Model (CAPM).

> CAPM_Beta.py – Computes the beta value of a stock for CAPM calculations.

> requirements.txt – Python dependencies needed for the application to function.


⚙️ How to Run
 Clone the repository:
   > git clone https://github.com/ShaikAnwarr/Stock_Forecasting_Application.git

Navigate into the project directory:
 > cd Stock_Forecasting_Application

Install dependencies:
 > pip install -r requirements.txt

Start the Streamlit app:
 > streamlit run Trading_App.py

The app will open in your default web browser.

📌 Demo:
![WhatsApp Image 2025-04-23 at 21 46 02_775a4b76](https://github.com/user-attachments/assets/a14d0685-de22-4ea0-abc7-8e09cc3f0ee0)
![WhatsApp Image 2025-04-23 at 21 46 02_9953f9ec](https://github.com/user-attachments/assets/18372e37-aee1-4697-a0a6-00c92ffe308b)
![WhatsApp Image 2025-04-23 at 21 46 03_fb2e0495](https://github.com/user-attachments/assets/608170d6-2c15-4c42-852c-4dfc04db03ae)
![WhatsApp Image 2025-04-23 at 21 46 03_d9ff54cd](https://github.com/user-attachments/assets/a04256e7-151b-4f44-916b-59493198e446)
![WhatsApp Image 2025-04-23 at 21 46 04_e1d010dc](https://github.com/user-attachments/assets/9fa4ee86-4f26-4678-a706-c60d895ea980)
![WhatsApp Image 2025-04-23 at 21 46 05_cf737464](https://github.com/user-attachments/assets/5b54fbea-4ab7-4a62-87a1-c316d8b405e8)
![WhatsApp Image 2025-04-23 at 21 46 06_9d13abb3](https://github.com/user-attachments/assets/f238b400-b963-44b5-9d1d-300cedd3bcb9)
![WhatsApp Image 2025-04-23 at 21 46 05_590ede61](https://github.com/user-attachments/assets/9a1a8ff1-6125-4066-96bf-7e7ab61f7481)

👤 Author:
  Shaik Anwarr
  B.Tech CSE | Android & Python Developer
