################## Functions for exploratory analysis ##################
#import requests
import pandas as pd
import numpy as np
from dateutil.relativedelta import relativedelta
import matplotlib.pyplot as plt
import os
import datetime
import math
from statsmodels.api import OLS

def sine_scaled(x, b):
    return math.sin((2*math.pi*x)/b)

def cos_translated(x, a=0, b=1): 
    return math.cos(2*math.pi*(x-a)/b)

#get the first of the month as a date 
def first_of_month(dt_): 
    return datetime.date(dt_.year, dt_.month, 1)

#identify the date where df['value'] achieves its maximum 
def where_max(df): 
    val_max = df['value'].max()
    date_max = df.loc[df['value'] == val_max, 'date'].values[0]
    return date_max

def hour_of_year(date_): 
    diff = date_ - datetime.datetime(date_.year, 1, 1, 0, 0)
    return diff.total_seconds() / (60*60)

def modify_data_demand(df): 
    """
    Add various variables to the data set.
    """
    df['period_dt'] = pd.to_datetime(df['period_dt'])
    #df_demand['date'] = pd.to_datetime(df_demand['date']).dt.floor('D')
    df['date'] = df['period_dt'].dt.floor('D')
    df['hour'] = df['period_dt'].dt.floor('h')
    df['month'] = df['date'].dt.month
    df['year'] = df['date'].dt.year
    df['hour_of_day'] = df['hour'].dt.hour
    df['day_of_year'] = df['date'].dt.dayofyear
    df['sine_365'] = df['day_of_year'].apply(lambda x: sine_scaled(x, 365.25))
    return df

def ercot_data_from_csv(path_demand): 
    """
    Load the full demand dataset from retrieve_ercot_data and add various variables. 
    Fill in missing values for the variable 'value' with the most recent non-missing
    value from the same day of the week and hour
    """
    df_demand = pd.read_csv(path_demand)
    df_demand['period_dt'] = pd.to_datetime(df_demand['period_dt'])
    # make sure that data are sorted chronologically
    df_demand = df_demand.sort_values('period_dt').reset_index(drop=True)

    # create variable 'date'
    df_demand = (df_demand 
        .assign(date = lambda x: x['period_dt'].dt.date))
    
    # hour of year
    df_demand['hour_of_year'] = df_demand['period_dt'].apply(hour_of_year)
    df_demand['day_of_year'] = df_demand['period_dt'].dt.day_of_year
    df_demand['day_of_week'] = df_demand['period_dt'].dt.day_of_week
    df_demand['year'] = df_demand['period_dt'].dt.year
    df_demand['hour_of_day'] = df_demand['period_dt'].dt.hour
    df_demand['sine_365'] = df_demand['day_of_year'].apply(lambda x: sine_scaled(x, 365.25))
    
    # fill in NAs with most recent non-NA value for the same hour of the day and day of week
    idx_na = df_demand[df_demand['value'].isna()].index
    for idx in idx_na: 
        na_period = df_demand.loc[idx, 'period'][10:13]
        na_day_of_week = df_demand.loc[idx, 'day_of_week']
        i = idx - 1
        while True: 
            if (df_demand.loc[i, 'period'][10:13] == na_period) and (df_demand.loc[i,'day_of_week'] == na_day_of_week) and (not np.isnan(df_demand.loc[i, 'value'])): 
                df_demand.loc[idx, 'value'] = df_demand.loc[i, 'value']
                break 
            i = i-1

    return df_demand

class DemandData:
    def __init__(self, data_=None, path_demand=None):
        if data_ is None:
            if path_demand is not None:
                data_ = ercot_data_from_csv(path_demand)
            else:
                raise ValueError('One of data_ or path_load must be provided.')
        self.data = data_
    def __call__(self):
        return self.data
    def query(self, query_str, inplace=False):
        data = self.data.query(query_str).reset_index(drop=True)
        if inplace:
            self.data = data
        else:
            return DemandData(data_=data)
    def window_datetime(self, datetime_begin, datetime_end, return_object=False):
        df_window = (
            self.data
                .query('period_dt >= @datetime_begin')
                .query('period_dt <= @datetime_end')
                .reset_index(drop=True)
        )
        if return_object:
            return DemandData(df_window)
        else:
            return df_window
        
def fill_na_168hr(df):
    """
    Replace missing values with value from one week earlier, and create 
    an indicator 'value_was_na' for periods whose values were originally missing.
    Alternative to the methodology in ercot_data()
    """
    df['value_lag168'] = df['value'].shift(168)
    df['value'] = np.where(df['value'].isna(), df['value_lag168'], df['value'])
    df['value_was_na'] = np.where(df['value'].isna(), 1, 0)
    df = df.drop('value_lag168', axis=1)
    return df

def plot_daily_end(df_daily): 
    df_daily_end = df_daily[-100:]
    plt.plot(df_daily_end['date'], 
        df_daily_end['value']/1000)
    plt.xticks(ticks = df_daily_end['date'][(np.arange(df_daily_end.shape[0])%3)==0], 
          labels = df_daily_end['date'][(np.arange(df_daily_end.shape[0])%3)==0].dt.strftime('%m/%d'),
          rotation = 45, 
          size = 8)
    plt.ylabel('GWh')
    plt.grid()
    plt.show()

def plot_monthly_years(df_monthly):
    #monthly plot 
    for year, values in df_monthly.groupby(['year']):
        plt.plot(values['month'],
            values['value']/1E03, 
            label = year)
    plt.xticks(ticks=list(range(1,13)), 
               labels = list(range(1,13)))
    plt.grid(True, color='grey')
    plt.ylabel('GWh / day')
    plt.legend(loc='upper right')
    plt.show()

def plot_monthly_full(df_monthly): 
    df_monthly['date'] = [datetime.date(int(row['year']), int(row['month']), 1) for i, row in df_monthly.iterrows()]
    plt.plot(df_monthly['date'], 
        df_monthly['value']/1E06)
    plt.grid(color='gray', alpha=0.2)
    plt.ylabel('GWh / day')
    plt.show()

def plot_hourly(df_hour): 
    plt.plot(df_hour['hour_of_day'], 
        df_hour['value'])
    plt.show()

#plot of realized vs fitted 
def plot_fitted(df, name_realized, name_fitted): 
    plt.plot(df['date'], 
              df[name_realized], 
              label = 'realized')
    plt.plot(df['date'], 
             df[name_fitted], 
             label = 'fitted')
    plt.legend()
    plt.show()