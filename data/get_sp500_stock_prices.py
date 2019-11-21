import bs4 as bs
import pickle
import requests
import yfinance as yf
import numpy as np
import pandas as pd

def retrieve_sp500_tickers_from_wiki():
    """
    The original code is taken from [1] and modified. 
    
    Reference:
        [1] pythonprogramming.net https://pythonprogramming.net/sp500-company-list-python-programming-for-finance/
    """

    resp = requests.get('http://en.wikipedia.org/wiki/List_of_S&P_500_companies')
    print(resp)
    soup = bs.BeautifulSoup(resp.text, 'lxml')
    table = soup.find('table', {'class': 'wikitable sortable'})
    tickers = []
    for row in table.findAll('tr')[1:]:
        ticker = row.findAll('td')[0].text
        gics = row.findAll('td')[3].text
        tickers.append({"name":ticker.strip(), "gics":gics.strip()})
        
    return tickers

def retrieve_stock_prices(tickers_list, start_date, end_date):
    data = pd.DataFrame(columns=[t["name"] for t in tickers_list])
    for ticker in tickers_list:
        data[ticker["name"]] = yf.download(ticker["name"], '2015-1-1', '2019-1-1')["Adj Close"]
    return data

tickers_list = retrieve_sp500_tickers_from_wiki()
stock_prices = retrieve_stock_prices(tickers_list, start_date = "2015-1-1", end_date = "2019-1-1")

ticker_names=pd.DataFrame(tickers_list)
ticker_names = ticker_names.sort_values(by="gics")
ticker_names.to_csv("ticker_names.csv", sep="\t", index = False)


# Compute the logarithmic return
stock_prices = stock_prices[ticker_names["name"].values]
log_return = stock_prices.apply(lambda x : np.log(x))\
    .diff()\
    .drop(stock_prices.head(1).index)\
    .dropna(axis="columns")\
    .dropna(axis="rows")

log_return.to_csv("sp500-log-return.csv", sep="\t", index = False)
