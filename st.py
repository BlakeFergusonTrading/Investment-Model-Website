import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime
import matplotlib.pyplot as plt
import time

st.markdown(
    """
    <style>
    div[data-baseweb="button"] button {
        height: 50px;
        line-height: 50px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Define the list of tickers to display in the animation
tickers = ['AAPL', 'MSFT', 'AMZN', 'NVDA', 'GOOG', 'BRK.B', 'META', 'XOM', 'UNH', 'TSLA', 'JNJ', 'JPM', 'V', 'PG', 'MA', 'LLY', 'HD', 'CVX', 'MRK', 'ABBV', 'AVGO', 'PEP', 'KO', 'PFE', 'COST', 'MCD', 'TMO', 'WMT', 'SPY']

ticker_css = """
.ticker {
  position: fixed;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  background: url('https://i.imgur.com/FV41vwb.png') repeat-x;
  animation: ticker 30s linear infinite;
  font-size: 24px;
  color: white;
  text-shadow: 1px 1px black;
}

@keyframes ticker {
  0% {
    background-position: 0 0;
  }
  100% {
    background-position: -2000px 0;
  }
}
"""

# Add the ticker symbols to the CSS class
for ticker in tickers:
    ticker_css = ticker_css.replace('url(', f'url(https://finviz.com/chart.ashx?t={ticker}&ty=c&ta=1&p=d&s=l);')

# Add the CSS class to the page
st.markdown(f"""
<style>
{ticker_css}
</style>
""", unsafe_allow_html=True)

# Add the ticker class to the page container
st.markdown("<div class='ticker'></div>", unsafe_allow_html=True)

st.markdown("<h1 style='text-align: center; color: green; font-size: 80px; text-shadow: 2px 2px #ff0000;'>Investment Model  </h1>", unsafe_allow_html=True)

def load_data(start_date):
    all_tickers = yf.Tickers('')
    all_tickers_list = all_tickers.tickers
    portfolio_data = pd.DataFrame()
    portfolio_assets = []
    for a in all_tickers_list:
        try:
            data = yf.download(a, start=start_date, interval='1d')
        except:
            continue
        if len(data) > 0: # check if data is available
            portfolio_data[a] = data['Adj Close']
            portfolio_data[a + '_volatility'] = data['Adj Close'].pct_change().rolling(21).std() * np.sqrt(252)
            portfolio_assets.append(a)
    portfolio_data = portfolio_data.loc[:, portfolio_data.notna().any()] # Filter out columns with all NaN values
    return portfolio_data, portfolio_assets

def add_stock():
    st.write("")
    st.write("")
    st.write("")
    st.write("<p style='font-weight: bold; font-size: 28px; color: black; text-shadow: 2px 2px #999; '><em>Enter the tickers of new stocks to add to the portfolio (e.g. AAPL, TSLA, MSFT):</p>", unsafe_allow_html=True)
    new_stocks = st.text_input("")
    if new_stocks:
        new_stocks = new_stocks.replace(" ","").split(",")
        new_stocks = list(set(new_stocks)) # remove duplicates
        for stock in new_stocks:
            try:
                stock_info = yf.Ticker(stock).info
                if stock not in portfolio_assets:
                    data = yf.download(stock, start=start_date, interval='1d')
                    if len(data) > 0: # check if data is available
                        portfolio_data[stock] = data['Adj Close']
                        portfolio_data[stock + '_volatility'] = data['Adj Close'].pct_change().rolling(21).std() * np.sqrt(252)
                        portfolio_assets.append(stock)
                        placeholder = st.success('Stock {} added to the portfolio.'.format(stock))
                        time.sleep(1)
                        placeholder.empty()
                else:
                    placeholder = st.error('Invalid ticker symbol {}.'.format(stock))
                    time.sleep(1)
                    placeholder.empty()
            except:
                continue

def select_stocks():
    st.write("")
    st.write("")
    st.write("")
    select_stocks_prompt = ""
    st.write("<p style='font-weight: bold; font-size: 28px; color: black; text-shadow: 2px 2px #999;'><em>Select the stocks to include in the analysis:</p>", unsafe_allow_html=True)
    selected_assets = st.multiselect(select_stocks_prompt, portfolio_assets, default=portfolio_assets)
    if selected_assets:
        if all(elem in portfolio_assets for elem in selected_assets): # check if all selected assets are in portfolio_assets
            selected_data = portfolio_data[selected_assets]
            selected_data = selected_data.dropna() # Drop any rows that contain NaN values
            if not selected_data.empty:
                return selected_data
            else:
                st.warning('No data available for the selected assets. Please try again.')
        else:
            st.warning('Please select only the assets that are in the portfolio.')
    else:
        return portfolio_data

# Allow the user to select the starting date for the data
st.write("")
st.write("")
st.write("<p style='font-weight: bold; font-size: 28px; color: black; text-shadow: 2px 2px #999;'><em>Select the starting date for the data:</p>", unsafe_allow_html=True)
start_date = st.date_input("", datetime(2019, 10, 29))

portfolio_data, portfolio_assets = load_data(start_date)

# Add a button to add new stocks to the portfolio
add_stock()

# Select the stocks to include in the analysis
selected_data = select_stocks()

# Display the portfolio data if assets have been selected
if selected_data.empty:
    st.write("")
else:
    st.line_chart(selected_data / selected_data.iloc[0] * 100)

    portfolio_log_returns = np.log(selected_data / selected_data.shift(1))

    portfolio_returns = []
    portfolio_volatility = []
    portfolios_weights = []
    asset_weights = []

    for x in range(1000):
        portfolio_weights = np.random.random(len(selected_data.columns))
        portfolio_weights /= np.sum(portfolio_weights)
        portfolio_returns.append(np.sum(portfolio_weights * portfolio_log_returns.mean())*250)
        portfolio_volatility.append(np.sqrt(np.dot(portfolio_weights.T,np.dot(portfolio_log_returns.cov()*250, portfolio_weights))))
        asset_weights.append(portfolio_weights)

    portfolio_returns = np.array(portfolio_returns)
    portfolio_volatility = np.array(portfolio_volatility)

    sharpe_ratio = portfolio_returns / portfolio_volatility

    df_structure = {'Return' : portfolio_returns, 'Volatility' : portfolio_volatility, 'Sharpe Ratio': sharpe_ratio}
    for i, asset in enumerate(selected_data.columns):
        df_structure[asset] = [weight[i] for weight in asset_weights]

    portfolios_all_assets = pd.DataFrame(df_structure)
    max_sharpe = portfolios_all_assets[(portfolios_all_assets['Return']/portfolios_all_assets['Volatility'])==((portfolios_all_assets['Return']/portfolios_all_assets['Volatility']).max())]

    st.write("<p style='font-weight: bold; font-size: 28px; color: black; text-shadow: 2px 2px #999;'><em>Portfolio Weights and Return/Risk</p>", unsafe_allow_html=True)
    max_sharpe_perc = max_sharpe.iloc[:, 2:].applymap(lambda x: '{:.2f}%'.format(x * 100).rstrip('.0') if isinstance(x, (int, float)) and 'Ratio' not in str(x) else x)
    max_sharpe_perc['Sharpe Ratio'] = max_sharpe['Sharpe Ratio'].apply(lambda x: '{:.2f}'.format(x))
    max_sharpe_perc.insert(0, 'Return', max_sharpe['Return'].apply(lambda x: '{:.2f}%'.format(x * 100)))
    max_sharpe_perc.insert(1, 'Volatility', max_sharpe['Volatility'].apply(lambda x: '{:.2f}%'.format(x * 100)))
    st.dataframe(max_sharpe_perc)

    st.write("<p style='font-weight: bold; font-size: 28px; color: black; text-shadow: 2px 2px #999;'><em>Markowitz Efficient Frontier</p>", unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize = (7, 5))
    portfolios_all_assets.plot(x = 'Volatility', y = 'Return', kind = 'scatter', ax=ax, facecolor='black', edgecolor='black')
    ax.set_xlabel('Expected Volatility')
    ax.set_ylabel('Expected Return')
    st.pyplot(fig)


    # Calculate correlation matrix and covariance matrix
    selected_data_corr = selected_data.corr()
    selected_data_cov = selected_data.cov()

    # Display correlation matrix
    st.write("<p style='font-weight: bold; font-size: 28px; color: black; text-shadow: 2px 2px #999;'><em>Correlation Matrix</p>", unsafe_allow_html=True)
    st.dataframe(selected_data_corr)

    # Display covariance matrix
    st.write("<p style='font-weight: bold; font-size: 28px; color: black; text-shadow: 2px 2px #999;'><em>Covariance Matrix</p>", unsafe_allow_html=True)
    st.dataframe(selected_data_cov)
