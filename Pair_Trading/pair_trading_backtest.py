"""
  Pair trading is also called mean reversion trading. We find two cointegrated assets, normally a
  stock and an ETF index or two stocks in the same industry. We run an cointegration test on the
  historical data. We set the trigger condition for both stocks theoretically these two stocks
  cannot drift away from each other. It's like a drunk man with a dog. The invisible dog leash
  would keep both assets in line when one stock is getting too bullish, we short the bullish one
  and long the bearish one, vice versa after several lags of time, the dog would converge to the
  drunk man. It's when we make profits nevertheless, the backtest is based on historical datasets
  in real stock market, market conditions are dynamic. Two assets may seem cointegrated for the
  past two years. They just drift far away from each other after one company launch a new product
  or whatsoever. I am talking about nvidia and amd, two gpu companies. After bitcoin mining boom
  and machine learning hype, stock price of nvidia went skyrocketing, amd didnt change much on the
  contrary, the cointegrated relationship just broke up. So be extremely cautious with cointegration,
  there is no such thing as riskless statistical arbitrage, always check the cointegration status
  before trading execution.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
import fix_yahoo_finance as yf
from sklearn.model_selection import train_test_split

_UPPER_BOUND_1 = 0.5
_UPPER_BOUND_2 = 0.2
_LOWER_BOUND_2 = -0.2
_LOWER_BOUND_1 = -0.5

# Check cointegration status
def cointegration(data1, data2):
    # Train test split 
    df1, test1, df2, test2 = train_test_split(data1, data2, test_size=0.5, shuffle=False)
    
    train = pd.DataFrame()
    train['asset1'] = df1['Close']
    train['asset2'] = df2['Close']
    
    # This is the part where we test the cointegration. In this case, I use Engle-Granger
    # two-step method which is invented by the mentor of my mentor!!! Generally people use
    # Johanssen test to check the cointegration status. The first step for EG is to run a
    # linear regression on both variables. Next, we do OLS and obtain the residual after
    # that we run unit root test to check the existence of cointegration. If it is stationary,
    # we can determine its a drunk man with a dog. The first step would be adding a constant
    # vector to asset1
    x = sm.add_constant(train['asset1'])
    y = train['asset2']
    model = sm.OLS(y,x).fit()
    resid = model.resid
    
    print(model.summary())
    print('\n', sm.tsa.stattools.adfuller(resid))

    # This phrase is how we set the trigger conditions. First we normalize the residual, we would
    # get a vector that follows standard normal distribution. Generally speaking, most tests use
    # one sigma level as the threshold, two sigma level reaches 95% which is relatively difficult
    # to trigger after normalization, we should obtain a white noise follows N(0,1). We set +-1 as
    # the threshold, eventually we visualize the result
    signals = pd.DataFrame()
    signals['asset1'] = test1['Close']
    signals['asset2'] = test2['Close']
    
    signals['fitted'] = np.mat(sm.add_constant(signals['asset1']))*np.mat(model.params).reshape(2,1) 
    signals['residual'] = signals['asset2'] - signals['fitted']

    resid_hist_mean = np.mean(resid)
    resid_hist_std = np.std(resid)
    
    signals['z'] = (signals['residual'] - resid_hist_mean)/resid_hist_std
    #signals['z'] = (signals['residual'] - np.mean(signals['residual']))/np.std(signals['residual'])
    
    # Use z*0 to get panda series instead of an integer result
    signals['upper_bound1'] = resid_hist_mean + _UPPER_BOUND_1*resid_hist_std
    signals['lower_bound1'] = resid_hist_mean + _LOWER_BOUND_1*resid_hist_std
    signals['upper_bound2'] = resid_hist_mean + _UPPER_BOUND_2*resid_hist_std
    signals['lower_bound2'] = resid_hist_mean + _LOWER_BOUND_2*resid_hist_std
    
    return signals

# The signal generation process is very straight forward, if the normalized residual gets above or
# below threshold, we long the bearish one and short the bullish one, vice versa. I only need to
# generate trading signal of one asset, the other one should be the opposite direction.
def signal_generation(df1, df2, method):
    
    signals = method(df1, df2)
    signals['position'] = 0
    signals['buy'] = 0
    signals['sell'] = 0
    signals['close_position'] = 0
    signals['below_lower_bound1'] = 0
    signals['above_upper_bound1'] = 0 
    
    # Signal to buy asset 2 and sell asset 1
    signals['below_lower_bound1'] = np.where(signals['z'] < signals['lower_bound1'], 1, 0)
    # Signal to sell asset 2 and buy asset 1
    signals['above_lower_bound1'] = np.where(signals['z'] > signals['upper_bound1'], 1, 0)

    for i in range(1, len(signals['position'])):
        if signals.ix[i-1,'position']==0 and signals.ix[i,'below_lower_bound1']==1:
            signals.ix[i,'position'] = 1  # Open position to buy asset 2 and sell asset 1
            signals.ix[i,'buy'] = 1
        elif signals.ix[i-1,'position']==0 and signals.ix[i,'above_lower_bound1']==1:
            signals.ix[i,'position'] = -1  # Open position to sell asset 1 and buy asset 2
            signals.ix[i,'sell'] = 1
        elif signals.ix[i,'z'] < signals.ix[i,'upper_bound2'] and \
             signals.ix[i,'z'] > signals.ix[i,'lower_bound2']:
            signals.ix[i,'position'] = 0  # Close position if residual is between lower_bound2 and upper_bound2
            if signals.ix[i-1,'position']:
                signals.ix[i,'close_position'] = 1  # At time i, we close previous position
        else:
            signals.ix[i,'position'] = signals.ix[i-1,'position']

    return signals

# visualization
def plot(new, ticker1, ticker2):
    
    # z stats figure
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(211)
    ax.plot(new['z'], label='z statistics', c='#e8175d')
    ax.plot(new.loc[new['buy']==1].index, new['z'][new['buy']==1], lw=0, marker='^',
            markersize=8, label='LONG {}'.format(ticker2), c='g', alpha=0.7)
    ax.plot(new.loc[new['sell']==1].index, new['z'][new['sell']==1], lw=0, marker='v',
            markersize=8, label='SHORT {}'.format(ticker2), c='r', alpha=0.7)    
    ax.plot(new.loc[new['close_position']==1].index, new['z'][new['close_position']==1],
            lw=0, marker='o', markersize=8, label='Close position', c='b', alpha=0.7)
    
    #ax.fill_between(new.index, new['z upper limit'], new['z lower limit'],
    #                label='+- %s sigma' % _BOUND, alpha=0.5, color='#f7db4f')
    
    plt.legend(loc='best')
    plt.title('Cointegration Normalized Residual')
    plt.xlabel('Date')
    plt.ylabel('value')
    plt.grid(True)
    
    # positions
    bx = fig.add_subplot(212, sharex=ax)
    bx.plot(new['position'])
    #new['asset1'].plot(label='{}'.format(ticker1))
    #new['asset2'].plot(label='{}'.format(ticker2))
    
    #bx.plot(new.loc[new['buy']==1].index, new['asset2'][new['buy']==1], lw=0, marker='^',
    #        markersize=8, label='LONG {}'.format(ticker2), c='g', alpha=0.7)

    #bx.plot(new.loc[new['positions1']==-1].index, new['asset1'][new['positions1']==-1], lw=0,
    #        marker='v', markersize=8, label='SHORT {}'.format(ticker1), c='r', alpha=0.7)

    #bx.plot(new.loc[new['positions2']==1].index, new['asset2'][new['positions2']==1], lw=0,
    #        marker=2, markersize=12, label='LONG {}'.format(ticker2), c='g', alpha=0.9)

    #bx.plot(new.loc[new['positions2']==-1].index, new['asset2'][new['positions2']==-1], lw=0,
    #        marker=3, markersize=12, label='SHORT {}'.format(ticker2), c='r', alpha=0.9)

    bx.legend(loc='best')
    plt.title('Pair Trading')
    plt.xlabel('Date')
    plt.ylabel('price')
    plt.grid(True)
    plt.show()

def main():
    # the sample i am using are NVDA and AMD from 2012 to 2015
    stdate = '2010-01-01'
    eddate = '2017-12-31'
    ticker1 = 'WMT'
    ticker2 = 'AMZN'  # 'RBC'

    df1 = yf.download(ticker1, start=stdate, end=eddate)
    df2 = yf.download(ticker2, start=stdate, end=eddate)    

    signals = signal_generation(df1, df2, cointegration)
    
    plot(signals, ticker1, ticker2)

# how to calculate stats could be found from my other code called Heikin-Ashi
# https://github.com/tattooday/quant-trading/blob/master/heikin%20ashi%20backtest.py
    
if __name__ == '__main__':
    main()
