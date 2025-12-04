############# Load Packages ###########
import pandas as pd
import numpy as np
from dateutil.relativedelta import relativedelta
import matplotlib.pyplot as plt
import os
import datetime
from math import isnan

import cyclic

# load data
df_demand = exploratory.ercot_data(path_demand = 'data/demand_full_2025_11_30.csv')
# number of "seasons"
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

