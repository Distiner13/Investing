
data={"symbol":[],
      "metric":[],
      "value":[]}

tickerSymbols = getCompanyList()
for symbol in tickerSymbols:
    names,values = getFinancialInformation(symbol)

    for i in range(len(names)):
        data["symbol"].append(symbol)
        data["metric"].append(names[i])
        data["value"].append(values[i])