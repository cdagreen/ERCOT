

############# Load Packages ###########

import requests
import pandas as pd
import numpy as np
from dateutil.relativedelta import relativedelta
import matplotlib.pyplot as plt
import os
import datetime
from scipy.sparse import coo_array, diags
from scipy.linalg import solve_circulant
from math import isnan

os.chdir('/Users/carlgreen/ERCOT')


############ Functions #############

def hour_of_year(date_): 
    diff = date_ - datetime.datetime(date_.year, 1, 1, 0, 0)
    return diff.total_seconds()/(60*60)

def ercot_data(): 
    ## read data 
    df_demand = pd.read_csv('data/demand_2024_05_16.csv')
    df_demand['period_dt'] = pd.to_datetime(df_demand['period_dt'])
    # make sure that data are sorted chronologically
    df_demand = df_demand.sort_values('period_dt').reset_index(drop=True)

    # create variable 'date'
    df_demand = (df_demand 
        .assign(date = lambda x: x['period_dt'].dt.date))
    
    # fill in NAs with most recent non-NA value for the same hour of the day
    idx_na = df_demand[df_demand['value'].isna()].index
    for idx in idx_na: 
        na_period = df_demand.loc[idx, 'period'][10:13]
        i = idx - 1
        while True: 
            if (df_demand.loc[i, 'period'][10:13] == na_period) & (~np.isnan(df_demand.loc[i, 'value'])): 
                df_demand.loc[idx, 'value'] = df_demand.loc[i, 'value']
                break 
            i = i-1

    # hour of year
    df_demand['hour_of_year'] = df_demand['period_dt'].apply(hour_of_year)
    df_demand['day_of_year'] = df_demand['period_dt'].dt.day_of_year
    df_demand['day_of_week'] = df_demand['period_dt'].dt.day_of_week
    df_demand['year'] = df_demand['period_dt'].dt.year
    df_demand['hour_of_day'] = df_demand['period_dt'].dt.hour

    return df_demand
    

#### Daily data
df_demand = ercot_data()

df_demand.iloc[:,4:8]

df_daily = (df_demand
    .groupby('date', as_index = False)
    .aggregate({'value': 'sum'})
)

df_daily['value'].hist(bins=30)
plt.xlabel('Daily Demand (MWh)')
plt.show()

buckets = pd.cut(df_daily['value'], bins = 10)
values = buckets.value_counts()

plt.bar(
    x = [str(interval) for interval in values.index],
    height = values.values
)
plt.xlabel('Value Ranges')
plt.ylabel('Frequency')
plt.title('Bar Plot of Value Ranges')
plt.show()


# Plot daily 
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
         df_daily['value'])
plt.xticks(ticks = df_daily['date'], 
           labels = df_daily['quarter_label'], 
           rotation = 45, 
           fontsize = 8)
plt.show()

# seasonality options 
    # interaction between daily and annual
    # separate hour-of-year effects
        # smoothness of 24-hour differences
            #smoothness within day

# hour-of-year effects
df_hour_mean = (df_demand 
    .groupby('hour_of_year', as_index = False) 
    .aggregate({'value':'mean'})
)

plt.plot(df_hour_mean['hour_of_year'], 
         df_hour_mean['value'])
plt.show()

# Estimate hourly model with penalty for curvature in hourly and daily trends

########## Solve Circulant #########

# load data
df_demand = ercot_data()
# number of "seasons"
n = 365*24

###### penalty for curvature in hourly trend

### hourly sum of demand
df_hour_sum = (df_demand
    .groupby('hour_of_year', as_index = False)
    .aggregate({'value': 'mean'})
)

# remove December 31st for leap years
df_hour_sum = df_hour_sum[df_hour_sum['hour_of_year'] <= n-1]
m = df_hour_sum['value']

a = np.array(np.arange(10))
b = np.array(np.arange(10))

m_pd = (pd.DataFrame({'x':x,
                   'y':y})
        .groupby('x')
        .agg({'y':np.mean})
    )['y']


# compute mean of y by grouped x by 
# first converting to pandas (unaccountably fast)
# returns tuple: 1D arrays of unique values of x, and corresponding mean of y
def group_mean_pd(y, x): 
    m_pd = (pd.DataFrame({'x':x,
                   'y':y})
        .groupby('x', as_index=False)
        .agg({'y':'mean'})
    )
    return (np.array(m_pd['x']), np.array(m_pd['y']))

def group_mean_np(y, x):
    """Compute the mean of y by grouped x
        Parameters
        __________
        y: 1D numpy array
        x: 1D numpy array""" 
    idx = x.argsort()
    unique_ = np.unique(x[idx], return_index = True)
    split_ = np.split(
        y[idx],
        unique_[1][1:]
    )
    m = np.array((list(map(np.mean, split_))))
    return (unique_[0], m)

# dispatcher to compute mean of y by grouped x
def get_group_mean(y, x, method = 'pd'):
     if method == 'pd': 
         return group_mean_pd(y, x)
     elif method == 'np': 
         return group_mean_np(y, x)
     else: 
         raise ValueError('method ' + method + ' not found')


    # H: seasonal length
    # c_h: penalty for concavity in trend of contiguous values (gap 1)
    # c_d: penalty for concavity in seasonal trend (gap H)
class CyclicSpline: 
    def __init__(self, H, c_h = None, c_d = None):
        self.H = H
        self.c_h = c_h
        self.c_d = c_d
    def fit(self, x, y, H = None, c_h = None, c_d = None): 
        if c_h is not None: 
            self.c_h = c_h
        if c_d is not None: 
            self.c_d = c_d
        if H is not None: 
            self.H = H
        self.y = y
        self.x = x
        # compute the mean of y by grouped x
        x_unique, m = get_group_mean(y, x)
        # terms in D'D related to penalties
        dd_pen = [1, -4, 6, -4, 1]
        n = len(m)
        penalty_hour = coo_array((dd_pen, ([n-2, n-1, 0, 1, 2], np.tile(0,5))), shape = (n,1))
        penalty_day = coo_array((dd_pen, ([n-2*self.H, n-self.H, 0, self.H, 2*self.H], np.tile(0,5))), shape = (n,1))

        xx_diag = coo_array(([1], ([0], [0])), shape = (n,1))

        # compile all coefficients 
        c = (c_h*penalty_hour + c_d*penalty_day + xx_diag).toarray().reshape(-1)

        x = solve_circulant(c, m)
        self.m = m
        self.fitted = x
        self.x_unique = x_unique
        return x
    def fit_plot(self, downsample = True, alpha = 0.6): 
    # demand appears to have trough at 10:00 and peak at 22:00 (UTC)
        if downsample == True:
            idx = (self.x_unique % 24 == 10) | (self.x_unique % 24 == 22)
        else: 
            idx = np.tile(True, len(self.x_unique))
        plt.plot(
            self.x_unique[idx], 
            self.m[idx], 
            alpha = alpha
        )
        plt.plot(
            self.x_unique[idx], 
            self.fitted[idx], 
            alpha = alpha
        )
        plt.grid()
        plt.xlabel('hour of year')
        plt.ylabel('megawatthours')
        plt.show()
    def predict(self, xnew): 
        if 'x_unique' not in dir(self): 
            raise Exception('Fitted values not found')
        fitted_dict = dict(zip(self.x_unique, self.fitted))
        return np.array([fitted_dict[xx] for xx in xnew])


ypred = predict(spline1, spline1.x)


spline1.predict(spline1.x)

def predicted_plot(ypred): 
    plt.plot(
        df_demand['period_dt'], 
        df_demand['value'], 
        alpha = 0.6
    )
    plt.plot(
        df_demand['period_dt'], 
        ypred, 
        alpha = 0.6
    )
    plt.ylabel('megawatthours')
    plt.show()

predicted_plot(ypred)



fitted_dict = dict(zip(spline1.x_unique, spline1.fitted))
np.array(list(map(fitted_dict, spline1.x)))

np.array([fitted_dict[xx] for xx in spline1.x])

spline1.x_unique            

spline1 = CyclicSpline(H = 24)
spline1.fit(
    df_demand['hour_of_year'], 
    df_demand['value'], 
    c_h = 1, 
    c_d = 300
)

spline1.fit_plot()

plot_fit(spline1, downsample=True)

df_daily_mean = (df_demand
    .groupby('date', as_index = False)
    .agg({'value':'mean'})
)

df_hour_means = (pd.merge(
        df_demand, 
        df_daily_mean, 
        how = 'outer', 
        on = 'date', 
        suffixes = ['', '_mean']
    )
    .assign(value_demean = lambda x: x['value']-x['value_mean'])
    .groupby('hour_of_day', as_index = False)
    .agg({'value_demean':'mean'})
)

df_demand.columns



x = np.array(df_demand['hour_of_year'])
x.shape

spline1.fitted

spline1.m

a = np.arange(10)


def grid_search(estimator, m, grid_,  test_size, size_type = 'fraction'):
    if size_type == 'fraction':
        if ((test_size <=0) | (test_size >= 1)): 
            raise ValueError("test_size must be in (0,1)")
        test_size = np.floor(test_size*len(m))
        if test_size == 0: 
            raise ValueError("test_size = 0")
    m_train = m[:-test_size]
    m_test = m[test_size:]


spline1.fit(m,  )

df_demand['year'].unique()

xx1 = cyclic_spline(m, 24, 1, 100, 365*24)
xx2 = cyclic_spline(m, 24, 1, 1000, 365*24)

a = 24*50
plt.plot(df_hour_mean['hour_of_year'][0:a], 
         df_hour_mean['value'][0:a])
plt.plot(df_hour_sum['hour_of_year'][0:a], 
         xx1[0:a])
plt.show()

plt.plot(df_hour_sum['hour_of_year'][0:a], 
         xx1[0:a])
plt.plot(df_hour_sum['hour_of_year'][0:a], 
         xx2[0:a])
plt.show()

# terms in D'D related to penalties
dd_pen = [1, -4, 6, -4, 1]
penalty_hour = coo_array((dd_pen, ([n-2, n-1, 0, 1, 2], np.tile(0,5))), shape = (n,1))

H = 24
penalty_day = coo_array((dd_pen, ([n-2*H, n-H, 0, H, 2*H], np.tile(0,5))), shape = (n,1))

# number of years
Y = 1
xx_diag = coo_array(([Y], ([0], [0])), shape = (n,1))

# c_hour and c_day are penalties for hourly and daily trend concavity
c_hour = 1
c_day = 1000

# compile all coefficients 
#c = (c_hour*penalty_hour + c_day*penalty_day + xx_diag).toarray().reshape(-1)
c = (c_hour*penalty_hour + c_day*penalty_day+ xx_diag).toarray().reshape(-1)

### hourly sum of demand
df_hour_sum = (df_demand
    .groupby('hour_of_year', as_index = False)
    .aggregate({'value': 'mean'})
)

# remove December 31st for leap years
df_hour_sum = df_hour_sum[df_hour_sum['hour_of_year'] <= n-1]
b = df_hour_sum['value']

# estimate day coefficients
x = solve_circulant(c, b)

a = 24*90
plt.plot(df_hour['hour_of_year'][0:a], 
         df_hour['value'][0:a])
plt.plot(df_hour_sum['hour_of_year'][0:a], 
         x[0:a])
plt.show()

########## empirical Bayes ########

# variance of day means 

df_hour_sum = (df_demand
    .groupby('hour_of_year', as_index = False)
    .aggregate({'value': 'mean'})
)

df_demand = ercot_data()

df_demand.head()

df_demand = df_demand.sort_values('period_dt').reset_index(drop=True)

# hourly curvature
df_demand['value_lag1'] = df_demand['value'].shift(1)

# subtract mean for the day of week
df_day_means = (df_demand 
    .groupby('day_of_week', as_index=False)
    .aggregate({'value':'mean'})
    .rename(columns={'value': 'day_of_week_mean'})
)

df_test = (df_demand   
    .join(df_day_means, 
          how = 'left', 
          on = 'day_of_week', 
          lsuffix = '_1',
          rsuffix = '_r')
)
df_day_means

df_demand = (pd.merge(df_demand, 
         df_day_means, 
         how = 'left', 
         on = 'day_of_week') 
    .sort_values('period_dt')
    .reset_index(drop = True) 
    .assign(value_demean = lambda x: x['value']- x['day_of_week_mean'])
)

df_demand.head()

##### empirical curvature
# hourly
v_0 = df_demand['value'][:-2].reset_index(drop=True)
v_1 = df_demand['value'][1:-1].reset_index(drop=True)
v_2 = df_demand['value'][2:].reset_index(drop=True)

# daily
d = 24
v_0 = df_demand['value'][:-(2*d)].reset_index(drop=True)
v_1 = df_demand['value'][d:-d].reset_index(drop=True)
v_2 = df_demand['value'][(2*d):].reset_index(drop=True)


calc_1 = np.mean(((v_1 - v_0) - (v_2 - v_1))**2)
calc_1

var_day = calc_1

# somewhat different from D matrix before: this time dimension T
T = df_demand.shape[0]
DD = diags([1, -2, 1], [0, 1, 2], shape = [T-2, T])
DTD = np.transpose(DD).dot(DD) 
y = df_demand['value'].values.reshape(-1,1)
np.transpose(y).dot(DTD)

d = 24
DD = diags([1, -2, 1], [0, d, 2*d], shape = [T-2*d, T])
DTD = np.transpose(DD).dot(DD) 
y = df_demand['value'].values.reshape(-1,1)

calc_2 = DTD.dot(y).transpose().dot(y)/(T-2*d)

###### one method: covariance, across years, of curvature for each hour of year

df_demand['value_lag1'] = df_demand['value'].shift(1)
df_demand['value_lead1'] = df_demand['value'].shift(-1)
df_demand[['value', 'value_lag1', 'value_lead1']].head()

d = 24
df_demand['value_lag1'] = df_demand['value_demean'].shift(d)
df_demand['value_lead1'] = df_demand['value_demean'].shift(-d)

df_demand = (df_demand
    .assign(curvature = lambda x: x['value_lead1'] - 2*x['value_demean'] + x['value_lag1'])
    .assign(curvature_demean = lambda x: x['curvature'] - x['curvature'].mean())
)

temp = (df_demand[d:-d]   
    .groupby('hour_of_year')
    .agg({'curvature_demean': lambda x: (x.sum()**2 - np.sum(x**2))/2})
)

(temp['curvature_demean'] > 0).value_counts()

# number of products
nn = (df_demand[1:-1]
    .groupby('hour_of_year')
    .size()
)
n = np.sum(nn*(nn-1)/2)

# covariance across years
cov_hour = np.sum(temp['curvature_demean'])/n

cov_day = np.sum(temp['curvature_demean'])/n

# set penalty equal to the ratio of the variance to the covariance
var_hour/cov_hour

var_day/cov_day

cov_day



df_demand[df_demand['value_lag1'].apply(isnan) | df_demand['value_lead1'].apply(isnan)]
df_demand.iloc[[1,-1]]

df_demand['day_of_year_check'] = 
df_demand['period_dt'].dt.day_of_year

def day_of_year(date_): 
    diff = date_ - datetime.datetime(date_.year, 1, 1, 0, 0)
    return diff.days + 1

date_ = df_demand['period_dt'][0]

day_of_year(date_)

df_demand['day_of_year'] = df_demand['period_dt'].dt.day_of_year
df_demand['day_of_year_check'] = df_demand['period_dt'].apply(day_of_year)


(df_demand['day_of_year'] == df_demand['day_of_year_check']).value_counts()

sorted(aa)

a = np.array([[1,4],[3,1]])

a[0,1]

dict_mean = {1:'a', 2:'b'}

def group_means(y, x): 
    """
    Input: vectors y and x
    Return: dictionary containing mean of y for each value of x
    """
    dict_mean = {}
    for i in range(len(y)):
        xx = x[i]
        if xx in dict_mean.keys(): 
            dict_mean[xx]['n'] += 1
            dict_mean[xx]['mean'] = (dict_mean[xx]['mean']*(dict_mean[xx]['n'] -1) + y[i])/dict_mean[xx]['n']
        else: 
            dict_mean[xx] = {}
            dict_mean[xx]['n'] = 1
            dict_mean[xx]['mean'] = y[i]
    return dict_mean

# test group_means()
a = 5
b = 10
def test_group_means(a, b):
    x = np.tile(np.arange(a), b)
    y = 2*x + np.random.randn(a*b)
    return group_means(y, x)

test_group_means(5, int(1E08))


np.tile([1,2], 10)

np.tile(np.arange(5), 10)
