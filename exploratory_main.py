#import requests
import pandas as pd
import numpy as np
from dateutil.relativedelta import relativedelta
import matplotlib.pyplot as plt
import os
import datetime
#import math
from statsmodels.api import OLS
#from scipy.linalg import solve_banded

import exploratory

# update exploratory functions 
#import importlib
#importlib.reload(exploratory)

# load hourly demand data
path_demand = 'data/demand_full_2025_11_30.csv'
df_demand = exploratory.ercot_data(path_demand)

# add some variables to the data 
# df_demand = pd.read_csv(path_demand)
# df_demand = exploratory.modify_data_demand(df_demand)

# df_demand['value'].isna().value_counts()
# df_demand = exploratory.fill_na_168hr(df_demand)
# df_demand['value'].isna().value_counts()

# df_demand_test = exploratory.ercot_data(path_demand)
# (df_demand['value'] == df_demand_test['value']).value_counts()
# (df_demand['period_dt'] == df_demand_test['period_dt']).value_counts()

# confirm that there are no time gaps in the data
eia_ercot_data.time_gaps(df_demand['period_dt'])

# estimation dataset 
df_est = df_demand.copy().dropna()
df_est['const'] = 1
df_est['cos_365_220'] = df_est['date'].dt.dayofyear.apply(lambda x: exploratory.cos_translated(x, a = 220, b = 365.25))
df_est['cos_hour_10'] = df_est['hour'].dt.hour.apply(lambda x : exploratory.cos_translated(x, a = 10, b = 24))
df_est['trend'] = df_est['hour'].map(lambda x: (x - df_est['hour'][0]).days)

# daily data 
df_daily = (
    df_demand 
        .groupby('date', as_index = False) 
        .aggregate({'value':'sum'})
        .iloc[:-1]
)

# plot the end of the data
exploratory.plot_daily_end(df_daily)

df_daily = (df_daily 
    .assign(first_of_month = lambda x: x['date'].apply(lambda x: datetime.date(x.year, 
                                                                               x.month, 
                                                                               1)))
    .assign(first_of_quarter = lambda x: x['date'].apply(lambda x: datetime.date(x.year, 
                                                                               3*((x.month - 1)//3 + 1), 
                                                                               1)))
)

df_daily = (df_daily
    .assign(month_label = np.where(df_daily['date'] == df_daily['first_of_month'], 
                                   df_daily['date'].apply(lambda x: x.strftime('%Y-%m')), 
                                   ''))
    .assign(quarter_label = np.where(df_daily['date'] == df_daily['first_of_quarter'], 
                                     df_daily['date'].apply(lambda x: x.strftime('%Y-%m')), 
                                    ''))
)

plt.plot(df_daily['date'],
         df_daily['value'] / 1000)
plt.xticks(ticks = df_daily['date'][df_daily['quarter_label']!=''], 
           labels = df_daily['quarter_label'][df_daily['quarter_label']!=''], 
           rotation = 45, 
           fontsize = 8)
plt.grid(color='gray',alpha=0.2)
plt.ylabel('GWh')
plt.title('Daily Demand')
plt.show()


######monthly time series 
df_monthly = (
    df_demand
        .assign(day_of_month = lambda df: df['period_dt'].dt.day)
        .groupby(['year', 'month'], as_index = False) 
        .agg(value = ('value','sum'), 
             days = ('day_of_month', 'max'))
        .iloc[:-1]
        .assign(value = lambda df: df['value'] / df['days'])
)

#plot each year separately 
exploratory.plot_monthly_years(df_monthly)

#plot full time series 
exploratory.plot_monthly_full(df_monthly)

#Hourly seasonality 

df_hour = (
    df_demand 
        .groupby('hour_of_day', as_index = False) 
        .agg({'value':np.mean})
)
df_hour['value'] = df_hour['value']/df_hour['value'].mean()

exploratory.plot_hourly(df_hour)

#Annual Fourier 

fit = OLS(df_est['value'], 
         df_est.loc[:,['const', 'sine_365']], 
        hasconst = True).fit()
fit.summary()

df_est['fittedvalue_annual_sine'] = fit.fittedvalues

plt.plot(df_demand['date'], 
        df_demand['sine_365'])
plt.show()

est_daily = (
    df_est 
        .iloc[18:-1]
        .groupby('date', as_index = False) 
        .agg({
            'value': 'sum', 
            'fittedvalue_annual_sine': 'sum'
        })
)

est_daily[est_daily['fittedvalue'].isnull()]

est_daily[est_daily['fittedvalue']<0.3E06]

#plot realized vs fitted 
exploratory.plot_fitted(
            est_daily, 
            'value', 
            'fittedvalue_annual_sine'
)

plt.plot(
    est_daily['date'], 
    est_daily['value']
)
plt.plot(
    est_daily['date'], 
    est_daily['fittedvalue_annual_sine']
)
plt.show()

#When does demand peak?

df_daily[df_daily['value'] == df_daily['value'].max()]['date'].values[0]
df_daily['year'] = df_daily['date'].dt.year
pd.DataFrame([(year, exploratory.where_max(df)) for year, df in df_daily.groupby('year')])

#exclude winter outliers
pd.DataFrame([(year, exploratory.where_max(df), pd.to_datetime(exploratory.where_max(df)).dayofyear) for year, df in \
              df_daily[df_daily['date'].dt.dayofyear.between(100,300)].groupby('year')])

#peak at August 8th
plt.plot(df_est['date'],
        df_est['cos_365_220'])
plt.plot()
plt.xticks(
    ticks = df_est['date'][df_est['date'].dt.month.isin([1,7]) & (df_est['date'].dt.day == 1)], 
    labels = df_est['date'][df_est['date'].dt.month.isin([1,7]) & (df_est['date'].dt.day == 1)].dt.strftime('%Y-%m'), 
    rotation = 45, 
    size = 9, 
    fontweight = 'light'
)
plt.grid(color='grey', alpha=0.2)
plt.show()

fit = OLS(df_est['value'], 
         df_est[['cos_365_220', 'const']], 
         hasconst = True).fit()
fit.summary()

df_est['fitted_365_220'] = fit.fittedvalues

plt.plot(df_daily['date'], 
        df_daily['value'])
plt.plot(df_est['date'], 
        df_est['fitted_365_220']*24)
plt.show()

## fit with daily and annual Fourier terms 
fit = OLS(df_est['value'],
    df_est[['const', 'cos_365_220', 'cos_hour_10']], 
   hasconst = True).fit()
fit.summary()

plt.plot(df_est['date'], 
         fit.resid)
plt.show()

df_est['resid_cos_year_day'] = fit.resid
df_est['first_of_month'] = [datetime.date(int(row['year']), int(row['month']), 1) for i, row in df_est.iterrows()]

resid_monthly = df_est \
    .groupby('first_of_month', as_index = False) \
    .aggregate({'resid_cos_year_day':sum})

plt.plot(resid_monthly['first_of_month'],
    resid_monthly['resid_cos_year_day'])
plt.show()

#fit with fourier terms and time trend 

fit = OLS(df_est['value'],
    df_est[['const', 'cos_365_220', 'cos_hour_10', 'trend']], 
   hasconst = True).fit()
fit.summary()

plt.plot(df_est['date'], 
         fit.resid)
plt.show()

plt.plot(df_est['date'], 
         df_est['trend'])
plt.show()

#examine residuals 

df_resid = pd.DataFrame({'hour': df_est['hour'], 
             'residuals' : fit.resid})

#monthly pattern of residuals 
df_resid['first_of_month'] = df_resid['hour'].apply(exploratory.first_of_month)

resid_monthly = df_resid \
    .groupby('first_of_month', as_index = False) \
    .aggregate({'residuals':np.mean})

plt.plot(resid_monthly['first_of_month'], 
         resid_monthly['residuals'])
plt.show()

df_resid['hour_of_day'] = df_resid['hour'].dt.hour

resid_hourly = df_resid \
    .groupby('hour_of_day', as_index = False) \
    .aggregate({'residuals' : np.mean})

plt.plot(resid_hourly['hour_of_day'], 
         resid_hourly['residuals'])
plt.show()

#### suppose that demand is symetric around maximum day

# number of days until next august 8th 
# if |day - 220| <= 365.25/2, then |day - 220| 
# else 365.25/2 - |day - 220| 

np.min(np.arange(10), np.zeros(10))

def day_distance(x, a = 220):
    if np.abs(x-220) <= 365.25/2: 
        return np.abs(x-220)
    else: 
        return 365.25 - np.abs(x-220)

df_est['days_from_peak'] =  df_est['date'].dt.dayofyear.apply(day_distance)

plt.plot(df_est['trend'], 
         df_est['days_from_peak'])
plt.show()

df_est.head()



