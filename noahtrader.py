"""
Functions used during this process of accumulating and modelling the return of investment of assets over some given period. 

For an explanation of the functions used please see the documentation of each. 

In order to quickly execute, just run 

obtain_dataset(), and it should do the rest. 

"""
from datetime import datetime,timedelta
import os
import pandas as pd
import numpy as np
import sys
import seaborn as sns
import matplotlib.pyplot as plt

now = datetime.now()
today  = now.strftime("%Y-%m-%d")
from pandas import date_range

def getStock(symbol,start_date='2019-01-01',end_date=today,source='yahoo'):
    """Obtain the price history of a given stock, over the designated period of observations. 
    
    Parameters
    ----------
    start_date : str
        Starting date of observations. Ex. 1996-03-30
    end_date : str
        Ending date of observations. Ex. 2019-03-30

    symbol : str
        Stock ticker symbol used. Ex. Apple = AAPL
    
    source : str
        Financial data source. Default = 'google', but others like yahoo finance also possible. 
        
    Returns
    -------
    
    panel_data : dataframe
        Pandas dataframe of 
    """
    import pandas_datareader.data as web
    panel_data = web.DataReader(symbol, source, start_date,end_date)
    idx = date_range(start_date, end_date)
    panel_data = panel_data.reindex(idx)
    panel_data.fillna(method='ffill',inplace=True)
    panel_data.fillna(method='bfill',inplace=True)

    #and if the starting date was accidentlay a non-trading day...
#    panel_data.dropna(inplace=True)
    return panel_data




def load_sentiment():
    """Load in the sentiment dataset from sentex.csv, converting it to a datetime index, and use those dates available for the collection of stocks over that period. 
    
    Returns
    -------
    
        sentdex : dataframe
            Dataframe of each stock over the given interval, and it's associated sentiment. 
        
        companies : list 
            List of all companies with recorded sentiments, mostly S&P. 
        
            
        start_date : date
        end_date : date
        
    
    """
    sentdex = pd.read_csv('sentex.csv',index_col='date')
    sentdex.index = pd.to_datetime(sentdex.index)

    companies = sentdex['symbol'].unique()  #these are all the companies I can obtain sentiment for. 

    start_date = sentdex.index.min()  #['date'].min()
    end_date = sentdex.index.max()
    
    try:
        os.chdir(r'/Volumes/NSKDRIVE/stock_plots')
    except:
        print("Volume not connected. Aborting")
        
        sys.exit()
        
    return sentdex,companies,start_date,end_date


sentdex,companies,start_date,end_date = load_sentiment()


def normalize_df(df): 
    from sklearn.preprocessing import MinMaxScaler

    scaler = MinMaxScaler() 
    scaled_values = scaler.fit_transform(df) 
    df.loc[:,:] = scaled_values
    return df


def save_figs(stocks,holding_days=10):
    """Create and save figs for stock CNN, files will be saved in my flash drive, and the title of the image will contain necessary labelling. 
    
    Parameters
    ==========
    
    stocks : dataframe
        Dataset of OHLCAC of all stocks, and then their associated sentiment saved above. 
        
    holding_days : int
        Period of observations, to project a % increase over the next x days. 
    """
    import os
    os.chdir(r'/Volumes/NSKDRIVE/stock_plots')
    sns.set_style('white')
    plt.axis('off')
    my_dpi = 96
    holding_daysreturns =  -stocks['Adj Close'].pct_change(periods = -holding_days).copy(deep=True) # returns 30 days later (if you buy that day.)
    for col in holding_daysreturns.columns:

        good_times = holding_daysreturns[col].loc[holding_daysreturns[col] >= .05] # greater than a 5% increse is pretty damn good.  
        
        bad_times = holding_daysreturns[col].loc[holding_daysreturns[col] <= -.05] # greater than a 5% increse is pretty damn good.  

        normalized_stocks = normalize_df(stocks.copy(deep=True))

        #investment was over a day
        for gtime,btime in zip(good_times.index,bad_times.index):
            plt.figure(figsize=(20/my_dpi, 30/my_dpi), dpi=my_dpi)
            plt.axis('off')
            try:
                sentiment_std = sentdex.loc[(sentdex['symbol'] == col) & 
                        (sentdex.index <= gtime)
                       & (sentdex.index >= gtime - timedelta(days=holding_days))].std().values[0]

                sentiment_avg = sentdex.loc[(sentdex['symbol'] == col) & 
                        (sentdex.index <= gtime)
                       & (sentdex.index >= gtime - timedelta(days=holding_days))].mean().values[0]

                sentiment_avg = round(sentiment_avg,2)
                sentiment_std = round(sentiment_std,2)
            except:
                sentiment_avg = 0 #no news is good news!
                sentiment_std = 0
            #normalized_stocks.loc[gtime - timedelta(days=holding_days):gtime].plot(legend=False)
            normalized_stocks['Adj Close'][col].loc[gtime - timedelta(days=holding_days):gtime].plot(color='k',linewidth=1.5)
            plt.savefig('posReturns/'+col+'for'+ str(holding_days)+'days%'
                        +str(100*round(good_times[gtime], 2))+'on'+str(gtime.year)+'-'
                        +str(gtime.month)+'-'+str(gtime.day)+'SENTAVG='+str(sentiment_avg)+'SENTSTD='+str(sentiment_std)+'.png',dpi=my_dpi)
            


            plt.figure(figsize=(20/my_dpi, 30/my_dpi), dpi=my_dpi)
            plt.axis('off')

            try:
                sentiment_std = sentdex.loc[(sentdex['symbol'] == col) & 
                        (sentdex.index <= btime)
                       & (sentdex.index >= btime - timedelta(days=holding_days))].std().values[0]

                sentiment_avg = sentdex.loc[(sentdex['symbol'] == col) & 
                        (sentdex.index <= btime)
                       & (sentdex.index >= btime - timedelta(days=holding_days))].mean().values[0]

                sentiment_avg = round(sentiment_avg,2)
                sentiment_std = round(sentiment_std,2)

            except:
                sentiment_avg = 0 #no news is good news!
                sentiment_std = 0
      
            normalized_stocks['Adj Close'][col].loc[btime - timedelta(days=holding_days):btime].plot(color='k',linewidth=1.5)

         #   normalized_stocks.loc[btime - timedelta(days=holding_days):btime].plot(legend=False)
            plt.savefig('negReturns/'+col+'for'+ str(holding_days)+'days%'
                       +str(100*round(bad_times[btime], 2))+'on'+str(btime.year)+'-'
                        +str(btime.month)+'-'+str(btime.day)+'SENTAVG='+str(sentiment_avg)+'SENTSTD='+str(sentiment_std)+'.png',dpi=my_dpi)
            
        return 
    
def obtain_dataset(tot=-1):
    """
    Run this function to obtain dataset of all images and associated input/outputs for CNN.
    
    Parameters
    -----------
    
    tot : int 
        Run over tot companies, default is all, or -1. 
    """
    for company in companies[:tot]:
        print(company)
        try:
            stocks = getStock(start_date=start_date,end_date=end_date,symbol=[company])
            save_figs(stocks=stocks)
        except:
            print("Dead link")
    return

            
if __name__ == '__main__':
    obtain_dataset()


