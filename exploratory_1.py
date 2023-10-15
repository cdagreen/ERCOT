#import requests
import pandas as pd
import numpy as np
from dateutil.relativedelta import relativedelta
import matplotlib.pyplot as plt
import os
import datetime
import math
from statsmodels.api import OLS

os.chdir('ERCOT')
os.getcwd()

#load data 
df_demand = pd.read_csv('data/demand_2023_09_04.csv')

modify_data_demand()

df_demand.head()

df_est = df_demand.dropna()
df_est['const'] = 1
fit = OLS(df_est['value'], 
         df_est.loc[:,['const', 'sine_365']], 
          hasconst = True).fit()
fit.summary()

df_est['fittedvalue'] = fit.fittedvalues
    df = df.dropna()
    df['const'] = 1 
     
df_window = df_demand[df_demand['date'] >= pd.to_datetime('2023-08-01')]
plt.plot(df_window['hour'], 
        df_window['value'])
plt.show()

#daily data 
df_daily = df_demand \
    .groupby('date', as_index = False) \
     .aggregate({'value':np.sum})

plot_daily_end()

#Yearly seasonality
df_monthly = df_demand \
    .groupby(['year', 'month'], as_index = False) \
    .agg({'value': np.mean})

plot_monthly_years()

def plot_monthly_years():
    #monthly plot 
    for year, values in df_monthly.groupby(['year']):
        plt.plot(values['month'],
            values['value'], 
            label = year)
    plt.legend()
    plt.show()

def plot_monthly_full(): 
    df_monthly['date'] = [datetime.date(int(row['year']), int(row['month']), 1) for i, row in df_monthly.iterrows()]
    plt.plot(df_monthly['date'], 
        df_monthly['value'])
    plt.plot()
    plt.show()

plot_monthly_full()

#monthly time series 


#Hourly seasonality 

df_hour = df_demand \
    .groupby('hour_of_day', as_index = False) \
    .agg({'value':np.mean})
df_hour['value'] = df_hour['value']/df_hour['value'].mean()

def plot_hourly(): 
    plt.plot(df_hour['hour_of_day'], 
        df_hour['value'])
    plt.show()

plot_hourly()

#Annual Fourier 
math.pi

plt.plot(df_demand['date'], 
        df_demand['sine_365'])
plt.show()
help(OLS)

est_daily = df_est \
    .groupby('date', as_index = False) \
    .aggregate({'value':np.sum, 
                 'fittedvalue':np.sum})

est_daily[est_daily['fittedvalue'].isnull()]

est_daily[est_daily['fittedvalue']<0.3E06]

plt.plot(est_daily['date'], 
         est_daily['value'], 
         label = 'realized')
plt.plot(est_daily['date'], 
         est_daily['fittedvalue'], 
         label = 'fitted')
plt.legend()
plt.show()

#When does demand peak?
def where_max(df): 
    val_max = df['value'].max()
    date_max = df.loc[df['value']== val_max, 'date'].values[0]
    return date_max

df_daily[df_daily['value']== df_daily['value'].max()]['date'].values[0]
df_daily['year'] = df_daily['date'].dt.year
pd.DataFrame([(year, where_max(df)) for year, df in df_daily.groupby('year')])

#exclude winter outliers
pd.DataFrame([(year, where_max(df), pd.to_datetime(where_max(df)).dayofyear) for year, df in \
              df_daily[df_daily['date'].dt.dayofyear.between(100,300)].groupby('year')])

def cos_translated(x, a = 0, b = 1): 
    return math.cos(2*math.pi*(x-a)/b)


plt.plot(np.arange(1,366), 
        list(map(lambda x: cos_shifted(x, 220, 365), np.arange(1, 366))))
plt.plot()

df_est['cos_365_220'] = df_est['date'].dt.dayofyear.apply(lambda x: cos_translated(x, a = 220, b = 365.25))
plt.plot(df_est['date'],
        df_est['cos_365_220'])
plt.plot()
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
df_est['cos_hour_10'] = df_est['hour'].dt.hour.apply(lambda x : cos_translated(x, a = 10, b = 24))
fit = OLS(df_est['value'],
    df_est[['const', 'cos_365_220', 'cos_hour_10']], 
   hasconst = True).fit()
fit.summary()

fit.resid

plt.plot(df_est['date'], 
         fit.resid)
plt.show()

df_est['resid_cos_year_day'] = fit.resid

df_est['first_of_month'] = [datetime.date(int(row['year']), int(row['month']), 1) for i, row in df_est.iterrows()]

df_est[df_est['period'].isna()]

resid_monthly = df_est \
    .groupby('first_of_month', as_index = False) \
    .aggregate({'resid_cos_year_day':sum})

plt.plot(resid_monthly['first_of_month'],
    resid_monthly['resid_cos_year_day'])
plt.show()

#fit with fourier terms and time trend 
a = df_est['hour'].dt.date - df_est['hour'].dt.date[0]

relativedelta.relativedelta(df_est['hour'].dt.date, df_est['hour'].dt.date[0])

df_est['cos_hour_10'] = df_est['hour'].dt.hour.apply(lambda x : cos_translated(x, a = 10, b = 24))
fit = OLS(df_est['value'],
    df_est[['const', 'cos_365_220', 'cos_hour_10']], 
   hasconst = True).fit()
fit.summary()



