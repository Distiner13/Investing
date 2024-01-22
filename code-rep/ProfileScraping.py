'''
Collects the profiles (not data) of companies from yahoo finance
(Functional, for now)
'''

import requests
import collections
from sklearn.linear_model import LinearRegression
from bs4 import BeautifulSoup
import yfinance as yf
import numpy as np
collections.Callable = collections.abc.Callable
import pandas as pd


def getProfile(symbol):
    headers = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                             "AppleWebKit/605.1.15 (KHTML, like Gecko) "
                             "Version/15.4 Safari/605.1.15"}

    url = "https://finance.yahoo.com/quote/"+symbol+"/profile?p="+ symbol


    response = requests.get(url,headers= headers)


    t = response.text
    info = []

    soup = BeautifulSoup(t, features = "html.parser")
    trs = soup.find("div",{"class":"Mb(25px)"})
    addr = trs.find("p", {"class":"D(ib) W(47.727%) Pend(40px)"})
    si = trs.find_all("span", class_ = "Fw(600)")
    address = addr.text
    for item in si:
        info.append(item.contents)
    sector = info[0][0]
    industry = info[1][0]
    return([sector,industry])


def analyze_data(data, string):
    val = data[string].values
    print('Data Values: ', val)
    #print('Data [string]: ', data[string])
    values = [value for value in val if not pd.isna(value)]
    print('only floats: ', values)
    try:
        last_2_years_positive = values[0] > 0 and values[1] > 0
        l2y = int(last_2_years_positive)
    except:
        l2y = '-'

    try:
        arr = np.array(values)
        x = np.arange(1, len(arr) + 1).reshape(-1, 1)
        y = arr.reshape(-1, 1)
        model = LinearRegression()
        model = LinearRegression().fit(x, y)
        slope = -(model.coef_[0][0])
    except Exception as e:
        print("An exception occurred:", e)
        slope = '-'

    return slope, l2y

def getFinancials(symbol):
    company = yf.Ticker(symbol)

    importantMetrics = ['EBIT']

    income_statement = company.financials
    #print('income_statement: ', income_statement)
    #annual_income_statement = income_statement[income_statement.index.str.endswith('12-31')]
    #print('annual_income_statement: ', annual_income_statement)
    data = income_statement.loc[[importantMetrics[0]]].transpose()
    #print('This is the data we extracted from sheet: ', data)
    '''
    print(importantMetrics[0])
    print(data)
    '''
    slope, last_2_years_positive = analyze_data(data, importantMetrics[0])
    '''
    print(f"Last 2 Years Positive Earnings: {last_2_years_positive}")
    print(f"slope: {slope}")

    print("Income Statement:")
    print(income_statement)
    '''
    return slope, last_2_years_positive










