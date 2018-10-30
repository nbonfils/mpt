from flask import Flask, render_template, request
app = Flask(__name__)

import quandl
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
    
# setup quandl api
quandl.ApiConfig.api_key = 'ExKfeNE2oKQGyYM_KFk1'

#stocks = ['CNP', 'F', 'WMT', 'GE', 'TSLA']
stocks = ['KO', 'PFE', 'AAPL', 'WTI', 'TSLA']

def get_data():
    # get the market data for specific assets
    data = quandl.get_table('WIKI/PRICES', ticker = stocks,
            qopts = { 'columns' : ['date', 'ticker', 'adj_close'] },
            date = { 'gte' : '2014-1-1', 'lte' : '2017-12-31' },
            paginate=True)
    # clean the data, with date as index and with the tickers as columns
    clean = data.set_index('date')
    return clean.pivot(columns='ticker')

@app.route('/', methods=['GET', 'POST'])
def mpt():
    # if we don't chose any risk we simply display the best sharpe ratio
    chosen_risk = -1
    if request.method == 'POST':
        try:
            chosen_risk = float(request.form['risk'])
        except:
            chosen_risk = -1

    filename = 'data'
    if os.path.isfile(filename):
        table = pd.read_pickle(filename)
    else:
        table = get_data()
        table.to_pickle(filename)

    # calculate daily and  annual returns
    returns_daily = table.pct_change()
    returns_annual = returns_daily.mean() * 251

    # get daily and annual covariance matrix of the returns
    cov_daily = returns_daily.cov()
    cov_annual = cov_daily * 251

    # initialize returns, volatility and stock weights array
    port_returns = []
    port_volatility = []
    sharpe_ratio = []
    stock_weights = []

    # set the number for random portfolio generation
    num_assets = len(stocks)
    num_portfolios = 50000
    # set the seed in order to reproduce the same results
    np.random.seed(999)

    # generate the portfolios
    for p in range(num_portfolios):
        weights = np.random.random(num_assets)
        weights /= np.sum(weights)
        returns = np.dot(weights, returns_annual)
        volatility = np.sqrt(np.dot(weights.T, np.dot(cov_annual, weights)))
        # sharpe ratio with a risk free return rate of 0 assumed here
        sharpe = returns / volatility
        sharpe_ratio.append(sharpe)
        port_returns.append(returns)
        port_volatility.append(volatility)
        stock_weights.append(weights)

    # a dict containing all the returns and risks for the portfolios
    portfolios = {'Returns' : port_returns, 'Volatility' : port_volatility,
            'Sharpe Ratio' : sharpe_ratio}
    for idx, symbol in enumerate(stocks):
        portfolios[symbol + ' Weight'] = [w[idx] for w in stock_weights]

    # create a dataframe containing our portfolio dict
    df = pd.DataFrame(portfolios)
    col_order = ['Returns', 'Volatility', 'Sharpe Ratio'] + [symbol + 
            ' Weight' for symbol in stocks]
    df = df[col_order]
    
    min_risk = df['Volatility'].min()
    max_risk = df['Volatility'].max()
    max_sharpe = df['Sharpe Ratio'].max()
    
    # find the min volatility portfolio and best sharpe ratio one
    min_risk_port = df.loc[df['Volatility'] == min_risk]
    sharpe_port = df.loc[df['Sharpe Ratio'] == max_sharpe]

    delta = 0.005
    if chosen_risk != -1:
        selected_idx = df['Volatility'].between(chosen_risk - delta, chosen_risk + delta, 
                inclusive=False)
        selected_df = df[selected_idx]
        selected_port = selected_df.loc[selected_df['Returns'].idxmax()]
    else:
        selected_port = sharpe_port
        chosen_risk = selected_port['Volatility'].item()

    # plot our efficient frontier
    plt.style.use('seaborn')
    df.plot.scatter(x='Volatility', y='Returns', c='Sharpe Ratio', 
            cmap='RdYlGn', edgecolor='black', figsize=(10,8), grid=True)
    plt.scatter(x=selected_port['Volatility'], y=selected_port['Returns'], 
            c='red', marker='D', s=200)
    plt.xlabel('Volatility (Std. Dev.)')
    plt.ylabel('Expected Returns')
    plt.title('Efficient Frontier')
    plt.savefig('static/plot.png')

    # prepare dict for easier access in the template
    risk = { 'min' : min_risk, 'max' : 0.25, 'chosen' : chosen_risk }
    portfolio = { 'selected' : selected_port, 'min_risk' : min_risk_port,
            'sharpe' : sharpe_port }

    return render_template('index.html', risk=risk, portfolio=portfolio)
