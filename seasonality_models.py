############# Load Packages ###########
import pandas as pd
import numpy as np
from dateutil.relativedelta import relativedelta
import matplotlib.pyplot as plt
import os
import datetime
from math import isnan

import cyclic
import exploratory

# load data
df_demand = exploratory.ercot_data(path_demand = 'data/demand_full_2025_11_30.csv')

# length of seasonal period
n = 365 * 24

### hourly sum of demand
df_hour_sum = (
    df_demand
        .groupby('hour_of_year', as_index = False)
        .aggregate({'value': 'mean'})
)

# remove December 31st for leap years
df_hour_sum = df_hour_sum[df_hour_sum['hour_of_year'] <= n-1]
m = df_hour_sum['value']

######## Fit cyclic spline with penalty for concavity in trend ########
spline = cyclic.CyclicSpline(H=24)
spline.fit(
    df_demand['hour_of_year'], 
    df_demand['value'], 
    c_h = 1, 
    c_d = 300
)
spline.fit_plot()

spline.fit(
    df_demand['hour_of_year'], 
    df_demand['value'], 
    c_h = 1, 
    c_d = 1
)
spline.fit_plot()
spline.fit(
    df_demand['hour_of_year'], 
    df_demand['value'], 
    c_h = 1, 
    c_d = 0
)
spline.fit_plot()
spline.fit(
    df_demand['hour_of_year'], 
    df_demand['value'], 
    c_h = 1, 
    c_d = 1e04
)
spline.fit_plot()


##### Iterative algorithm inspired by the STL algorithm and STR (Dokumentov and Hyndman)
# remove data from February 29th
df_demand_noleap = (
    df_demand
        .query('(period_dt.dt.day!=29) or (period_dt.dt.month!=2)')
        .reset_index(drop=True)
)
# check that the removal was successful
# df_demand.shape[0] - df_demand_noleap.shape[0]
#df_demand_noleap[df_demand_noleap['date'] == datetime.date(2016,2,28)].index
#df_demand_noleap['period_dt'].iloc[5820:5840]
# [d for d in df_demand['period_dt'].values if d not in df_demand_noleap['period_dt'].values]

# define the component models
low_pass = cyclic.MovingAverage(n)
seasonal_spline = cyclic.CyclicSpline(
    H = 24, 
    c_h = 1, 
    c_d = 1e04
)
trend_ma = cyclic.MovingAverage(int(np.round(1.5*n)))

# data as numpy arrays
Y = np.array(df_demand_noleap['value'])
X = np.array(df_demand_noleap['hour_of_year'])


decomp = cyclic.Decomposition(
    trend_ma, 
    seasonal_spline, 
    low_pass
)

seasonals, trend = decomp.fit(
    X,
    Y,
    p = 10
)

idx = (df_demand_noleap['period_dt'].dt.hour % 24 == 10) | (df_demand_noleap['period_dt'].dt.hour % 24 == 22)

for i, seasonal in enumerate(seasonals):
    df_plot = (
         pd.DataFrame({
            'date': df_demand_noleap['date'],
            'seasonal': seasonal
        })
        .groupby('date', as_index=False)
        .agg({'seasonal': 'sum'})
    )
    plt.plot(
        df_plot['date'], 
        df_plot['seasonal'], 
        label = str(i)
    )
plt.legend()
plt.show()
