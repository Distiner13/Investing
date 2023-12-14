import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from stockstats import StockDataFrame

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

def calculate_rsi(data, window):
    delta = data['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    avg_gain = gain.rolling(window=window, min_periods=1).mean()
    avg_loss = loss.rolling(window=window, min_periods=1).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

###################################################################################
ticker = 'GPRO'
company = yf.download(ticker, start='2019-01-01', end='2023-08-25')
###################################################################################
# Calculate the 10-month SMA
company['10-month SMA'] = company['Close'].rolling(window=10*21).mean()

# Define your trading strategy parameters
short_period = 14 #use these
long_period = 305
#decide on expected return
###################################################################################
minimum_return = 10
rsi_overSold = 35
rsi_overBought = 65
bollinger_std = 2
###################################################################################
# Calculate indicators
company['shortema'] = company['Close'].ewm(span=short_period, adjust=False).mean()
company['longema'] = company['Close'].ewm(span=long_period, adjust=False).mean()
company['rsi'] = calculate_rsi(company, window=14)

company['RollingMean'] = company['Close'].rolling(window=20).mean()
company['RollingStd'] = company['Close'].rolling(window=20).std()

company['bollingerupper'] = company['RollingMean'] + bollinger_std * company['RollingStd']
company['bollingermiddle'] = company['RollingMean']
company['bollingerlower'] = company['RollingMean'] - bollinger_std * company['RollingStd']

company.drop(['RollingMean', 'RollingStd'], axis=1, inplace=True)  # Clean up temporary columns
stock = StockDataFrame.retype(company)  # Convert DataFrame to StockDataFrame
company['ATR'] = stock['atr_14']  # ATR calculated using a 14-period window



# Initialize variables
cash = 100000
position = None  # 'INVESTED' or 'CASH'
shares_bought = 0
equity_curve = [cash]  # Start with initial cash
equity_dates = [company.index[0]]  # Start with initial date
risk_per_trade = 0.1  # 10% risk per trade
position_sizer = PositionSizer(risk_per_trade, cash)
trade_manager = TradeManager(position_sizer)

exit_price = 0
stop_loss_price = 0
position_size = 0

# Iterate through the data
for i in range(len(company)):
    if pd.notna(company['10-month sma'][i]) and pd.notna(company['ATR'][i]) and pd.notna(company['rsi'][i]):
        if company['close'][i] > company['10-month sma'][i] and company['rsi'][i] <= rsi_overSold:
            if position != 'INVESTED':
                position = 'INVESTED'
                price = company['close'][i]
                stop_loss_price = price - (2 * company['ATR'][i])
                shares_bought = abs(int(trade_manager.generate_buy_signal(entry_price, stop_loss_price)))
                #shares_bought = cash / company['close'][i]
                cash -= company['open'][i+1] * shares_bought - minimum_return
                equity_curve.append(cash + shares_bought * company['open'][i+1])
                equity_dates.append(company.index[i])  # Add the date to equity_dates
                print(f"Buy signal on {company.index[i].date()}. Price: {company['open'][i+1]:.2f}. Shares: {shares_bought}")
        elif company['close'][i] <= stop_loss_price or company['close'][i] <= company['10-month sma'][i]:
            if position == 'INVESTED':
                position = 'CASH'
                cash += shares_bought * company['open'][i+1] - minimum_return
                shares_bought = 0
                equity_curve.append(cash)
                equity_dates.append(company.index[i])  # Add the date to equity_dates
                print(f"Sell signal on {company.index[i].date()}. Price: {company['open'][i+1]:.2f}")
                print(f"Total cash: ${cash:.2f}")

# If there are remaining shares at the end, sell them and update equity_curve
if shares_bought > 0:
    cash += shares_bought * company['close'][-1] - minimum_return
    equity_curve.append(cash)
    equity_dates.append(company.index[-1])  # Add the date to equity_dates
    print(f"Selling all remaining shares at the end of simulation. Price: {company['close'][-1]:.2f}")
    print(f"Total cash: ${cash:.2f}")

# Plot equity curve
plt.figure(figsize=(10, 6))
plt.plot(equity_dates, equity_curve, label='Equity Curve', color='blue')
plt.axhline(y=100000, color='r', linestyle='--', label='Initial Capital')
plt.title(f'Equity Curve of {ticker} Stock Trading Strategy')
plt.xlabel('Date')
plt.ylabel('Equity')
plt.legend()
plt.grid()
plt.show()
