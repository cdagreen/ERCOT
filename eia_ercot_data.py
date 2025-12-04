import requests
import pandas as pd
import numpy as np
from dateutil.relativedelta import relativedelta
import matplotlib.pyplot as plt
import os
import datetime
from dotenv import load_dotenv

#ercot_string = 'https://api.eia.gov/v2/electricity/rto/region-data/data/?frequency=hourly&data[0]=value&facets[respondent][]=ERCO&end=2015-07-01T00&sort[0][column]=period&sort[0][direction]=desc&offset=0&length=10000&api_key=input_api'

def eia_get(ercot_string_, start_date_, end_date_,  series_type_): 
    ercot_string_ = ercot_string_ \
        .replace('input_start', start_date_) \
        .replace('input_end', end_date_) \
        .replace('input_type', series_type_)
    result = requests.get(ercot_string_)
    return pd.DataFrame(result.json()['response']['data'])

def relativedelta_error_catch(row_): 
    try: 
        d = relativedelta(row_['hour'], row_['hour_lag1'])
    except: 
        d = None 
    return d

def data_quality(return_ = False):
    df_test = df_demand.copy()
    df_test['hour'] = pd.to_datetime(df_test['period'])
    df_test['hour_lag1'] = df_test['hour'].shift(1)
    df_test['diff_hour'] = df_test.apply(relativedelta_error_catch, axis=1)
    df_test['diff_hour_lead1'] = df_test['diff_hour'].shift(-1)
    print(df_test['diff_hour'].describe())
    # check for nulls in 'value'
    print('count of nulls \n', 
          df_test['value'].isnull().sum())
    print('nulls head', 
          df_test[df_test['value'].isnull()].head())
    
    if return_: 
        return df_test

def eia_data_load_main(): 
    start_date = '2015-07-01'
    # yesterday_str = (datetime.date.today() - relativedelta(days = 1)).strftime("%Y-%m-%d")
    today_end_str = datetime.date.today().strftime('%Y-%m-%d') + "T23"
    #series_type 'D': demand
    series_type = 'D'

    # load .env file with API key
    load_dotenv()

    #API key
    eia_api = os.environ['eia_api']

    #ercot_string_orig = 'https://api.eia.gov/v2/electricity/rto/region-data/data/?frequency=hourly&data[0]=value&facets[respondent][]=ERCO&facets[type][]=input_type&start=input_start&end=input_end&api_key=input_api'
    ercot_string = 'https://api.eia.gov/v2/electricity/rto/region-data/data/?frequency=hourly&data[0]=value&facets[respondent][]=ERCO&facets[type][]=input_type&start=input_start&end=input_end&sort[0][column]=period&sort[0][direction]=desc&offset=0&api_key=input_api'
    ercot_string = ercot_string.replace('input_api', eia_api)

    df_list = []
    end_date = today_end_str
    #while (end_date >= start_)

    end_date_date = pd.to_datetime(end_date)
    start_date_iter = pd.to_datetime(start_date)
    df_list = []
    while start_date_iter <= end_date_date: 
        end_date_iter = min(start_date_iter + relativedelta(hours=4999), end_date_date)
    
        df_list.append(eia_get(ercot_string, 
                           start_date_iter.strftime('%Y-%m-%dT%H'),
                           end_date_iter.strftime('%Y-%m-%dT%H'), 
                           series_type))
        start_date_iter = start_date_iter + relativedelta(hours=5000)
        print(start_date_iter)

    # concatenate, drop duplicates, sort 
    # (duplicates can arise due to overlap in dates of consecutive API requests)
    df_demand = (pd.concat(df_list) 
        .drop_duplicates() 
        .sort_values('period') 
        .reset_index(drop = True)
    )
    
    # datetime version of variable 'period'
    df_demand['period_dt'] = pd.to_datetime(df_demand['period'])
    return df_demand

# Function to fill in any missing periods with NAs

def fill_gaps_dt(df_): 
    dt_min = min(df_['period_dt'])
    dt_max = max(df_['period_dt'])

    # full sequence of hours between dt_min and dt_max
    h_diff = (dt_max - dt_min).total_seconds()/3600
    h_diff = int(np.ceil(h_diff))
    dt_seq = [dt_min + relativedelta(hours = h) for h in range(h_diff+1)]
    dt_seq = pd.to_datetime(dt_seq)

    # get any missing hours
    hours_missing = set(dt_seq).difference(set(df_['period_dt']))
    if (len(hours_missing)==0): 
        df_full = df_
        print('No gaps to fill')
    
    else: 
        print('Filling ', len(hours_missing), ' missing periods')

        hours_missing = sorted(hours_missing)

        # complete dataframe by adding dummy rows corresponding to missing hours
        period_char = list(map(lambda x: x.strftime(format = '%Y-%m-%dT%H'), hours_missing))
        df_missing = pd.DataFrame({'period': period_char,
               'respondent': 'ERCO',
               'respondent-name': 'Electric Reliability Council of Texas, Inc.',
               'type': 'D',
               'type-name': 'Demand', 
               'value': None,
               'value-units': 'megawatthours',
               'period_dt': hours_missing})

        # concatenate df_missing with original dataframe
        df_full = pd.concat([df_, df_missing]) \
            .sort_values('period_dt') \
            .reset_index(drop= True)
    
    return df_full

# function to get some summary information on missing values 

def summarize_missing(df_): 
    n_null = df_['value'].isnull().sum()
    print('Number of missing values: ', n_null)

    # get list of episodes of consecutive missing values, 
    # with starting period, ending period, and length in hours
    df_missing = df_[df_['value'].isnull()] \
        .reset_index(drop = True)
    
    episodes = []
    start = df_missing['period_dt'][0]
    last = start
    i = 1
    while (i < df_missing.shape[0]):
        current = df_missing['period_dt'][i]
        if (((current - last).total_seconds()/3600) >= 2): 
            episodes.append([start, last, (last-start).total_seconds()/3600 + 1])
            start = current 
        last = current
        i+= 1
    print('Strings of consecutive missing values: ', episodes)


def time_gaps(dates):
    time_diffs = np.array([(dates[i+1]-dates[i]).total_seconds() / (60**2)  for i in range(len(dates)-1)])
    print(f'minimum time between consecutive periods: {np.min(time_diffs)} hours')
    print(f'maximimum time between consecutive periods: {np.max(time_diffs)} hours')