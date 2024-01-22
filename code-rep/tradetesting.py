'''
This was made to test out the technical trading stategy
(Functional, for now)
'''

#%%
import pandas as pd
import numpy as np
#import talib
import yfinance as yf
import pandas as pd
import numpy as np
#from stockstats import StockDataFrame
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import matplotlib.dates as mdates


#%% Position Sizer Class
entry_price = 0
class PositionSizer:
    def __init__(self, risk_per_trade, portfolio_value):
        self.risk_per_trade = risk_per_trade
        self.portfolio_value = portfolio_value

    def calculate_position_size(self, entry_price, stop_loss_price):
        risk_amount = self.portfolio_value * self.risk_per_trade
        dollar_risk_per_share = entry_price - stop_loss_price
        position_size = risk_amount / dollar_risk_per_share
        return position_size


class TradeManager:
    def __init__(self, position_sizer):
        self.position_sizer = position_sizer
        self.open_position = False

    def generate_buy_signal(self, entry_price, stop_loss_price):
        position_size = self.position_sizer.calculate_position_size(entry_price, stop_loss_price)
        self.open_position = True
        return position_size

    def generate_sell_signal(self, exit_price, position_size):
        self.open_position = False
        return position_size * (entry_price - exit_price)  # Calculate profit

class CaseInsensitiveString:
    def __init__(self, value):
        self.value = value
        self.lower_value = value.lower()
        self.upper_value = value.title()

    def __str__(self):
        try:
            return str(self.upper_value)
        except Exception as e:
            return str(self.lower_value)
        
    '''def __repr__(self):
        return self.value

    def __getattr__(self, attr):
        try:
            # Try the attribute with the original value
            return getattr(self.value, attr)
        except AttributeError:
            try:
                # Try the attribute with the lowercase value
                return getattr(self.lower_value, attr)
            except AttributeError:
                # Try the attribute with the uppercase value
                return getattr(self.upper_value, attr)'''

#%% Column Names Initialized
'''
high = CaseInsensitiveString('High')
low = CaseInsensitiveString('Low')
open = CaseInsensitiveString('Open')
close = CaseInsensitiveString('Close')
adj_close = CaseInsensitiveString('adj close')
volume = CaseInsensitiveString('volume')
shortema = CaseInsensitiveString('shortema')
longema = CaseInsensitiveString('longema')
rsi = CaseInsensitiveString('rsi')
bollingerupper = CaseInsensitiveString('bollingerupper')
bollingermiddle = CaseInsensitiveString('bollingermiddle')
bollingerlower = CaseInsensitiveString('bollingerlower')
atr_14 = CaseInsensitiveString('atr_14')
ATR = CaseInsensitiveString('ATR')
SMA200 = CaseInsensitiveString('200SMA')
SMA20 = CaseInsensitiveString('20SMA')
SMA50 = CaseInsensitiveString('50SMA')
Predicted_High = CaseInsensitiveString('Predicted High')
Predicted_Low = CaseInsensitiveString('predicted Low')
'''
open = "1"
high = "2"
low = "3"
close = "4"
adj_close = "5"
volume = "6"
shortema = "7"
longema = "8"
rsi = "9"
bollingerupper = "10"
bollingermiddle = "11"
bollingerlower = "12"
atr_14 = "13"
ATR = "14"
SMA200 = "15"
SMA20 = "16"
SMA50 = "17"
Predicted_High = "18"
Predicted_Low = "19"
#%% Functions
def transform_string(input_string):
    # Convert to all lowercase
    lowercase_version = input_string.lower()
    
    # Uppercase the first letter of each word
    words = lowercase_version.split()
    uppercase_words = [word.capitalize() for word in words]
    uppercase_version = ' '.join(uppercase_words)
    
    return lowercase_version, uppercase_version

def calculate_rsi(data, window):
    delta = data[close].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    avg_gain = gain.rolling(window=window, min_periods=1).mean()
    avg_loss = loss.rolling(window=window, min_periods=1).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_high_bound(data, target_date, days):
# Extract the high prices for the last 50 days
  #print(f'within highbound: {data.columns}')
  t = pd.Timestamp(target_date)
  last_date = pd.Timestamp(end_date)
  high_prices = data.iloc[data.index.get_loc(t)-days:data.index.get_loc(t)][high]


  # Create an array of integers from 1 to 50 (representing the number of days)
  X = np.arange(data.index.get_loc(t)-days, data.index.get_loc(t)).reshape(-1, 1)

  # Initialize a Linear Regression model
  model = LinearRegression()

  # Fit the model to the high prices and days
  model.fit(X, high_prices)

  # Calculate the residuals
  residuals = high_prices - model.predict(X)

  # Find the index of the data point with the maximum positive residual
  index_max_residual = np.argmax(residuals)

  # Calculate a new intercept to make the line touch this point
  new_intercept = high_prices.iloc[index_max_residual] - model.coef_[0] * X[index_max_residual]

  # Update the model with the new intercept
  model.intercept_ = new_intercept

  extended_X = np.arange(data.index.get_loc(t)-days, data.index.get_loc(data.index[-1])).reshape(-1, 1)

  #print(extended_X)
  extended_predictions = model.predict(extended_X)
  #print("here is the extended_predictions", extended_predictions)
  #print(len(extended_predictions))

  # Create a DataFrame for the extended predictions
  data.loc[data.index.get_loc(t)-days:data.index.get_loc(data.index[-1]), [Predicted_High]] = extended_predictions
  #print("Updated DataFrame:")
  #print(data)

  return data

def calculate_low_bound(data, target_date, days):
# Extract the high prices for the last 50 days
  t = pd.Timestamp(target_date)
  last_date = pd.Timestamp(end_date)
  high_prices = data.iloc[data.index.get_loc(t)-days:data.index.get_loc(t)][low]

  # Create an array of integers from 1 to 50 (representing the number of days)
  X = np.arange(data.index.get_loc(t)-days, data.index.get_loc(t)).reshape(-1, 1)

  # Initialize a Linear Regression model
  model = LinearRegression()

  # Fit the model to the high prices and days
  model.fit(X, high_prices)

  # Calculate the residuals
  residuals = high_prices - model.predict(X)

  # Find the index of the data point with the maximum positive residual
  index_max_residual = np.argmin(residuals)

  # Calculate a new intercept to make the line touch this point
  new_intercept = high_prices.iloc[index_max_residual] - model.coef_[0] * X[index_max_residual]

  # Update the model with the new intercept
  model.intercept_ = new_intercept

  extended_X = np.arange(data.index.get_loc(t)-days, data.index.get_loc(data.index[-1])).reshape(-1, 1)

  #print(extended_X)
  extended_predictions = model.predict(extended_X)
  #print("here is the extended_predictions", extended_predictions)
  #print(len(extended_predictions))

  # Create a DataFrame for the extended predictions
  data.loc[data.index.get_loc(t)-days:data.index.get_loc(data.index[-1]), [Predicted_Low]] = extended_predictions
  #print("Updated DataFrame:")
  #print(data)

  return data

def return_high_bound_intercept(data, target_date, days):
# Extract the high prices for the last 50 days
  t = pd.Timestamp(target_date)
  last_date = pd.Timestamp(end_date)
  high_prices = data.iloc[data.index.get_loc(t)-days:data.index.get_loc(t)][high]



  # Create an array of integers from 1 to 50 (representing the number of days)
  X = np.arange(data.index.get_loc(t)-days, data.index.get_loc(t)).reshape(-1, 1)

  # Initialize a Linear Regression model
  model = LinearRegression()

  # Fit the model to the high prices and days
  model.fit(X, high_prices)

  # Calculate the residuals
  residuals = high_prices - model.predict(X)

  # Find the index of the data point with the maximum positive residual
  index_max_residual = np.argmax(residuals)

  # Calculate a new intercept to make the line touch this point
  new_intercept = high_prices.iloc[index_max_residual] - model.coef_[0] * X[index_max_residual]

  # Update the model with the new intercept
  model.intercept_ = new_intercept


  return model.intercept_, model.coef_

def return_low_bound_intercept(data, target_date, days):
# Extract the high prices for the last 50 days
  t = pd.Timestamp(target_date)
  last_date = pd.Timestamp(end_date)
  high_prices = data.iloc[data.index.get_loc(t)-days:data.index.get_loc(t)][low]



  # Create an array of integers from 1 to 50 (representing the number of days)
  X = np.arange(data.index.get_loc(t)-days, data.index.get_loc(t)).reshape(-1, 1)

  # Initialize a Linear Regression model
  model = LinearRegression()

  # Fit the model to the high prices and days
  model.fit(X, high_prices)

  # Calculate the residuals
  residuals = high_prices - model.predict(X)

  # Find the index of the data point with the maximum positive residual
  index_max_residual = np.argmin(residuals)

  # Calculate a new intercept to make the line touch this point
  new_intercept = high_prices.iloc[index_max_residual] - model.coef_[0] * X[index_max_residual]

  # Update the model with the new intercept
  model.intercept_ = new_intercept

  return model.intercept_,model.coef_

#%%Define your trading strategy parameters
#time frames
short_period = 14
long_period = 305
period = 60
#Risk parameters
rsi_overSold = 30
rsi_overBought = 70
bollinger_std = 2
risk_per_trade = 0.02  # 2% risk per trade

Bound_need_to_be_evaluated = True; 
Position = False

# Initialize position and capital
costofTransaction = 10
capital = 100000
position = 0
exit_price = 0
stop_loss_price = 0
position_size = 0
MinimumNumberofShares = 0
highCoef, highInt, lowCoef, lowInt = 0, 0, 0, 0

#%%
# initiate position sizer
position_sizer = PositionSizer(risk_per_trade, capital)
trade_manager = TradeManager(position_sizer)

#%%
# Download historical data using yfinance
symbol = 'AAPL'
start_date = '2021-01-04'
trading_start_date = '2021-04-01'
Newdate = trading_start_date
end_date = '2023-10-04'
Data = yf.download(symbol, start=start_date, end=end_date)
data = Data[Data.index>=trading_start_date]

#print(f'initial: {data.columns}')

new_column_names = {col: str(i + 1) for i, col in enumerate(data.columns)}
# Rename the columns using the mapping
data.rename(columns=new_column_names, inplace=True)

#%%
# Calculate indicators
data[shortema] = data[close].ewm(span=short_period, adjust=False).mean()
data[longema] = data[close].ewm(span=long_period, adjust=False).mean()
data[rsi] = calculate_rsi(data, window=14)
#data[rsi] = talib.rsi(data[close])

data['RollingMean'] = data[close].rolling(window=20).mean()
data['RollingStd'] = data[close].rolling(window=20).std()

data[bollingerupper] = data['RollingMean'] + bollinger_std * data['RollingStd']
data[bollingermiddle] = data['RollingMean']
data[bollingerlower] = data['RollingMean'] - bollinger_std * data['RollingStd']

data.drop(['RollingMean', 'RollingStd'], axis=1, inplace=True)  # Clean up temporary columns
#data[bollingerupper], data[bollingermiddle], data[bollingerlower] = talib.BBANDS(data[close], timeperiod=20, nbdevup=bollinger_std, nbdevdn=bollinger_std)

#stock = StockDataFrame.retype(data)  # Convert DataFrame to StockDataFrame --> 
#print(data.columns)
#print(stock.columns)
#data[ATR] = stock[atr_14]  # ATR calculated using a 14-period window
#data[ATR] = talib.ATR(data[high], data[low], data[close])
data[SMA200] = data[close].rolling(window=200).mean()
data[SMA20] = data[close].rolling(window=20).mean()
data[SMA50] = data[close].rolling(window=50).mean()
#print(data)
 
#print(f'after indicators data: {data.columns}')

new_column_names = {col: str(i + 1) for i, col in enumerate(Data.columns)}
# Rename the columns using the mapping
Data.rename(columns=new_column_names, inplace=True)
#print(f'after indicators DATA: {Data.columns}')

columnsLost = data
#print(f'columns lost: {columnsLost}')
columns_to_append = ['7', '8', '9', '10', '11', '12', '15', '16', '17']
'''columnsLost = pd.DataFrame()
columnsLost = columnsLost.join(data[columns_to_append])

print(f'col lost: {columnsLost.columns}')
'''
#%%New loop
f = True
buy = 30
sell = 31
signaldf = pd.DataFrame(index=data.index)
signaldf[buy] = [False for _ in range(len(data))]
signaldf[sell] = [False for _ in range(len(data))]
#print(signaldf)

#%%
counter = 0
for index, row in data.iterrows(): #row is the date with all info
    print(f'\n\ncounter: {counter}')
    if counter>=period+29:
        Bound_need_to_be_evaluated = True
        f = True
        Newdate = index
    
    counter+=1
    #print(f'current date: {index}')
    if Bound_need_to_be_evaluated == True: 
        data = calculate_high_bound(Data, Newdate, period)
        data = calculate_low_bound(Data, Newdate, period)

        highInt, highCoef = return_high_bound_intercept(Data, Newdate, period)
        lowInt, lowCoef = return_low_bound_intercept(Data, Newdate, period)
        Bound_need_to_be_evaluated = False
        
        #print(f'before JOIN: {data.columns}')
        #columns_to_append = ['7', '8', '9', '10', '11', '12', '15', '16', '17']
        data = data.join(columnsLost[columns_to_append])
        #print(f'after JOIN: {data.columns}')
        data[open] = data[open].shift(-1)
        data[open] = data[open].fillna(0)
        counter = 0
    
    Slope = (highCoef+lowCoef)/2
    if f is True:
        #print(f'Slope: {Slope}')
        f = False
    
    print(f'Slope: {Slope}')

    Tagretprice = 0
    NumberofShares = 0
    #USE data.iloc[index...] to avoid error since row does not contain columns 18,19 which are added here
    #data.loc[index, column_name]
    '''if d is True:
        data[open] = data[open].shift(-1)
        data[open] = data[open].fillna(0)
        d = False'''
    
    print(f'date: {index}, open price {data.loc[index, open]}, close price {data.loc[index, close]}\n\n')
    
    if data.loc[index, open] != 0:
        if Slope > 0:
            if Position is False:
                #print(f'before conditions DATA: {Data.columns}')
                #print(f'before conditions data: {data.columns}')
                #print(f'index and Row: \n{index} \n {row}')
                #print(f'Row: {data.loc[index, SMA20]}')
                #print(f'Row: {data.loc[index, Predicted_Low]}')
                if data.loc[index, SMA20] > data.loc[index, SMA50] or data.loc[index, SMA50] > data.loc[index, SMA200] and data.loc[index, SMA20] > data.loc[index, Predicted_Low]:
                    print('Buy signal generated')
                    signaldf.loc[index, buy] = True
                    entry_price = data.loc[index, open]
                    exit_price = data.loc[index, Predicted_High]
                    if exit_price < entry_price:
                        exit_price = entry_price + abs(entry_price - exit_price)
                    print(f'Buy signal entry price = {entry_price}')
                    print(f'Buy signal exit price = {exit_price}')
                    MinimumNumberofShares = 1 + (int((2*costofTransaction)/abs(exit_price - entry_price)))
                    position_size = max(MinimumNumberofShares, trade_manager.generate_buy_signal(entry_price, stop_loss_price))
                    capital = capital - (position_size * entry_price + costofTransaction)
                    print(f"Buy {position_size} shares at {entry_price} at {index}")
                    stop_loss_price = entry_price - abs((exit_price - entry_price)/2)
                    print(f'Buy signal stoploss price = {stop_loss_price}')
                    Position = True
            if Position is True:
                if data.loc[index, close] >= exit_price or data.loc[index, close] <= stop_loss_price:
                    print('Sell signal generated')
                    signaldf.loc[index, sell] = True
                    print(f'Sell signal exit price <= close price = {exit_price} <= {data.loc[index, close]}')
                    print(f'Sell signal stoploss price = {stop_loss_price}')
                    Sellprice = data.loc[index, open] 
                    Bound_need_to_be_evaluated = True 
                    Newdate = index
                    capital = capital + (position_size * Sellprice - costofTransaction)
                    print(f"Sell {position_size} shares at {Sellprice} at {index}")
                    position_size = 0
                    Position= False
                    f = True
        
        if Slope < 0:
            if Position is False:
                #print(f'before conditions DATA: {Data.columns}')
                #print(f'before conditions data: {data.columns}')
                #print(f'index and Row: \n{index} \n {row}')
                #print(f'Row: {data.loc[index, SMA20]}')
                #print(f'Row: {data.loc[index, Predicted_Low]}')
                '''
                if data.loc[index, close] >= data.loc[index, SMA200] and data.loc[index, close] > data.loc[index, Predicted_High]:
                    print('Buy signal generated')
                    signaldf.loc[index, buy] = True
                    entry_price = data.loc[index, open]
                    exit_price = 1.5*(data.loc[index, Predicted_High]-data.loc[index, SMA200]) + entry_price
                    if exit_price < entry_price:
                        exit_price = entry_price + (entry_price - exit_price)
                    print(f'Buy signal entry price = {entry_price}')
                    print(f'Buy signal exit price = {exit_price}')
                    MinimumNumberofShares = 1 + (int((2*costofTransaction)/(exit_price - entry_price)))
                    position_size = max(MinimumNumberofShares, trade_manager.generate_buy_signal(entry_price, stop_loss_price))
                    capital = capital - (position_size * entry_price + costofTransaction)
                    print(f"Buy {position_size} shares at {entry_price} at {index}")
                    stop_loss_price = data.loc[index, SMA200]
                    print(f'Buy signal stoploss price = {stop_loss_price}')
                    Position = True
                '''
                f = True
            if Position is True:
                #if data.loc[index, close] >= exit_price or data.loc[index, close] <= stop_loss_price:
                print('Sell signal generated')
                signaldf.loc[index, sell] = True
                print(f'Sell signal exit price <= close price: {exit_price} <= {data.loc[index, close]}')
                print(f'Sell signal stoploss price = {stop_loss_price}')
                Sellprice = data.loc[index, open]                    
                Bound_need_to_be_evaluated = True 
                Newdate = index
                capital = capital + (position_size * Sellprice - costofTransaction)
                print(f"Sell {position_size} shares at {Sellprice} at {index}")
                position_size = 0
                Position= False
                f = True
        

if Position is True:
    capital = capital + (position_size * entry_price + costofTransaction)
    print(f"\n\nClosing position at the end of the period. Capital: {capital}\nProfit = {capital-100000}\n\n")
else:
    print(f"\n\nClosing position at the end of the period. Capital: {capital}\nProfit = {capital-100000}\n\n")

#%%PLOT
plt.figure(figsize=(20, 10))  
#plt.gca().xaxis.set_major_locator(plt.MaxNLocator(33))  # Show every month
#plt.gcf().autofmt_xdate()
plt.xlabel('Date')
plt.ylabel('Price ($)')
plt.title(f'Trading Over 2 Years {symbol}')

plt.plot(data.index, data[open], color='black')
plt.plot(data.index, data[Predicted_Low], color='red')
plt.plot(data.index, data[close], color='black')
plt.plot(data.index, data[Predicted_High], color='green')
plt.plot(data.index, data[SMA200], color='blue')
plt.plot(data.index, data[SMA20], color='blue')
plt.plot(data.index, data[SMA50], color='blue')
plt.plot(signaldf.index, signaldf[buy]*10, color='blue')
plt.plot(signaldf.index, signaldf[sell]*5, color='red')

plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=1))
plt.xticks(rotation=45)  # Rotate the x-axis labels for better readability
plt.gca().xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter("%b-%Y"))

plt.legend(["open", "pred_low", "close", "pred_high", "200", "20", "50"], loc ="lower right")
plt.grid(True)
plt.tight_layout()
plt.show()

#%%
#%%