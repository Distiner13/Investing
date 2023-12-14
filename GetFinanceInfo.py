import requests
import collections
collections.Callable = collections.abc.Callable
from bs4 import BeautifulSoup

def getFinancialInformation(symbol):
    headers = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                             "AppleWebKit/605.1.15 (KHTML, like Gecko) "
                             "Version/15.4 Safari/605.1.15"}

    url = "https://finviz.com/quote.ashx?t="+symbol+"&p=d"


    response = requests.get(url,headers= headers)


    t = response.text

    soup = BeautifulSoup(t, features = "html.parser")
    trs= soup.find_all("td", class_ = "snapshot-td2-cp")
    trs_data = soup.find_all("td", class_ = "snapshot-td2")
    names = []
    values = []
    namVal = {}


    for i in range(len(trs)):
        try:
            name = trs[i].text
            names.append(name)
        except:
            continue

    for i in range(len(trs_data)):
        try:
            value = trs_data[i].text
            values.append(value)
        except:
            continue

    print(names)
    print(values)

#scrape information from this website: https://finance.yahoo.com/quote/AA/profile?p=AA
getFinancialInformation("MMM")








