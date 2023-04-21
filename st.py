yfinance
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime
import matplotlib.pyplot as plt
import time

st.set_page_config(page_title='Markowitz Efficient Frontier', layout='wide')

st.markdown("<h1 style='text-align: center; color: green; text-shadow: 1px 1px #fff;'>Markowitz Efficient Frontier</h1>", unsafe_allow_html=True)

st.write('This webpage displays the Markowitz Efficient Frontier for a portfolio of stocks. The efficient frontier shows the set of portfolios that offer the highest expected return for a given level of risk or the lowest risk for a given level of expected return. The portfolio consists of all assets listed on Yahoo Finance. The expected returns and volatilities for each portfolio are calculated using historical data from Yahoo Finance.')

def load_data(start_date):
    all_tickers = yf.Tickers('')
    all_tickers_list = all_tickers.tickers
    portfolio_data = pd.DataFrame()
    portfolio_assets = []
    for a in all_tickers_list:
        data = yf.download(a, start=start_date, interval='1d')
        if len(data) > 0: # check if data is available
            portfolio_data[a] = data['Adj Close']
            portfolio_data[a + '_volatility'] = data['Adj Close'].pct_change().rolling(21).std() * np.sqrt(252)
            portfolio_assets.append(a)
    portfolio_data = portfolio_data.loc[:, portfolio_data.notna().any()] # Filter out columns with all NaN values
    return portfolio_data, portfolio_assets

def add_stock():
    new_stocks = st.text_input('Enter the tickers of new stocks to add to the portfolio (e.g. AAPL, GOOG):')
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
                    placeholder = st.warning('Stock {} is already in the portfolio.'.format(stock))
                    time.sleep(1)
                    placeholder.empty()
            except:
                placeholder = st.error('Invalid ticker symbol {}.'.format(stock))
                time.sleep(1)
                placeholder.empty()

def select_stocks():
    selected_assets = st.multiselect('Select the stocks to include in the analysis:', portfolio_assets, default=None)
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
start_date = st.date_input("Select the starting date for the data:", datetime(2019, 10, 29))

portfolio_data, portfolio_assets = load_data(start_date)

# Add a button to add new stocks to the portfolio
add_stock()

# Select the stocks to include in the analysis
selected_data = select_stocks()

# Display the portfolio data
# Display the portfolio data if assets have been selected
if selected_data.empty:
    st.warning('Please add assets to the portfolio.')
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

    st.write('Portfolio Weights and Return/Risk')
    max_sharpe_perc = max_sharpe.iloc[:, 2:].applymap(lambda x: '{:.2f}%'.format(x * 100).rstrip('.0') if isinstance(x, (int, float)) and 'Ratio' not in str(x) else x)
    max_sharpe_perc['Sharpe Ratio'] = max_sharpe['Sharpe Ratio'].apply(lambda x: '{:.2f}'.format(x))
    max_sharpe_perc.insert(0, 'Return', max_sharpe['Return'].apply(lambda x: '{:.2f}%'.format(x * 100)))
    max_sharpe_perc.insert(1, 'Volatility', max_sharpe['Volatility'].apply(lambda x: '{:.2f}%'.format(x * 100)))
    st.dataframe(max_sharpe_perc)

    fig, ax = plt.subplots(figsize = (7, 5))
    portfolios_all_assets.plot(x = 'Volatility', y = 'Return', kind = 'scatter', ax=ax, facecolor='black', edgecolor='black');
    ax.set_xlabel('Expected Volatility')
    ax.set_ylabel('Expected Return')
    ax.set_title('Markowitz Efficient Frontier')
    st.pyplot(fig)


    # Calculate correlation matrix and covariance matrix
    selected_data_corr = selected_data.corr()
    selected_data_cov = selected_data.cov()

    # Display correlation matrix
    st.write('Correlation Matrix')
    st.dataframe(selected_data_corr)

    # Display covariance matrix
    st.write('Covariance Matrix')
    st.dataframe(selected_data_cov)