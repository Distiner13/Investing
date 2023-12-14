#%%
import pandas as pd
import numpy as np
#import talib
import yfinance as yf
import pandas as pd
import numpy as np
from stockstats import StockDataFrame
#%%
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


#########################
def calculate_rsi(data, window):
    delta = data['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    avg_gain = gain.rolling(window=window, min_periods=1).mean()
    avg_loss = loss.rolling(window=window, min_periods=1).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi
#%%
# Define your trading strategy parameters
short_period = 14 #use these
long_period = 305
#decide on expected return
minimum_return = 10
rsi_overSold = 35
rsi_overBought = 65
bollinger_std = 2

# Download historical data using yfinance
symbol = 'TTGT'
start_date = '2022-10-01'
end_date = '2023-08-13'
data = yf.download(symbol, start=start_date, end=end_date)
print(data.columns)

# Calculate indicators
data['shortema'] = data['Close'].ewm(span=short_period, adjust=False).mean()
data['longema'] = data['Close'].ewm(span=long_period, adjust=False).mean()
data['rsi'] = calculate_rsi(data, window=14)
#data['rsi'] = talib.rsi(data['close'])

data['RollingMean'] = data['Close'].rolling(window=20).mean()
data['RollingStd'] = data['Close'].rolling(window=20).std()

data['bollingerupper'] = data['RollingMean'] + bollinger_std * data['RollingStd']
data['bollingermiddle'] = data['RollingMean']
data['bollingerlower'] = data['RollingMean'] - bollinger_std * data['RollingStd']

data.drop(['RollingMean', 'RollingStd'], axis=1, inplace=True)  # Clean up temporary columns
'''data['bollingerupper'], data['bollingermiddle'], data['bollingerlower'] = talib.BBANDS(data['close'], timeperiod=20,
                                                                                       nbdevup=bollinger_std,
                                                                                       nbdevdn=bollinger_std)'''

stock = StockDataFrame.retype(data)  # Convert DataFrame to StockDataFrame
data['ATR'] = stock['atr_14']  # ATR calculated using a 14-period window
#data['ATR'] = talib.ATR(data['High'], data['Low'], data['close'])
data["200SMA"] = data["close"].rolling(window=200).mean()
data["20SMA"] = data["close"].rolling(window=20).mean()
data["50SMA"] = data["close"].rolling(window=50).mean()
print(data.columns)
#%%
def calculate_trendlines(date, days=50):
    data = yf.download(symbol, start=start_date, end=date)

    # Calculate the highest high and lowest low for the past 'days' days
    data['High' + str(days)] = data['High'].rolling(window=days).max()
    data['Low' + str(days)] = data['Low'].rolling(window=days).min()

    # Create trendlines for highs and lows
    high_trendline = np.polyfit(range(len(data)), data['High' + str(days)], 1)
    low_trendline = np.polyfit(range(len(data)), data['Low' + str(days)], 1)
    print(high_trendline)
    print(low_trendline)
    return high_trendline, low_trendline

def calculate_price_at_date(date, high_trendline, low_trendline):
    data = yf.download(symbol, start=start_date, end=date)
    print(f'len(data): {len(data)}')
    # Calculate the price at the specific date using the trendlines
    high_price = np.polyval(high_trendline, len(data))
    low_price = np.polyval(low_trendline, len(data))

    return high_price, low_price

date_of_interest = '2023-07-03'
high_trendline, low_trendline = calculate_trendlines(date_of_interest)

# Calculate prices at the specified date
high_price, low_price = calculate_price_at_date(date_of_interest, high_trendline, low_trendline)
print(f'high price: {(high_price)}, low price: {(low_price)}')
#%%
# Initialize position and capital
capital = 100000
position = 0
risk_per_trade = 0.02  # 2% risk per trade

position_sizer = PositionSizer(risk_per_trade, capital)
trade_manager = TradeManager(position_sizer)

exit_price = 0
stop_loss_price = 0
position_size = 0

# Trading loop
for index, row in data.iterrows():
    # Buy Signal
    if row['rsi'] < rsi_overSold and row['close'] < row['bollingerlower'] or row['shortema'] > row['longema']:
        entry_price = row['close']
        stop_loss_price = entry_price * 0.95
        exit_price = entry_price * 1.1
        position_size = trade_manager.generate_buy_signal(entry_price, stop_loss_price)
        capital -= position_size * row['close']
        print(f"Buy {position_size} shares at {row['close']} at {index}")

    # Sell Signal
    elif row['close'] > row['bollingerupper'] or row['shortema'] < row['longema'] or row['close'] >= exit_price or row['close'] <= stop_loss_price:
        profit = trade_manager.generate_sell_signal(exit_price, position_size)
        print("Profit:", profit)
        print(f"Sell {position_size} at {row['close']} at {index}")

    '''
    # Update stop loss based on ATR
    if position > 0:
        stop_loss = row['close'] - row['ATR'] * 2
        if row['low'] <= stop_loss:
            capital = position * stop_loss
            position = 0
            print(f"Stop Loss triggered at {stop_loss}")
    '''

# If there's an open position at the end, close it
if position > 0:
    capital = position * data.iloc[-1]['close']
    print(f"Closing position at the end of the period. Capital: {capital}")

#############################################
# Example usage





'''
# Generate buy signal

exit_price = 55

# Generate sell signal
'''
