############# Load Packages ###########
import pandas as pd
import numpy as np
from dateutil.relativedelta import relativedelta
import matplotlib.pyplot as plt
import os
import datetime
from math import isnan
#from statsmodels.tsa.statespace.exponential_smoothing import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
import random
#from scipy.ndimage import uniform_filter1d
from zoneinfo import ZoneInfo

# custom modules
import cyclic
import exploratory
import backtesting

# load hourly electricity demand data
df_demand = exploratory.DemandData(path_demand = 'data/demand_full_2025_11_30.csv')

# length of seasonal period
n = 365 * 24

##### Iterative algorithm inspired by the STL algorithm and STR (Dokumentov and Hyndman)
# remove data from December 31st of leap years
df_demand_noleap = (
    df_demand
        .query('(period_dt.dt.year % 4 != 0) or (period_dt.dt.month != 12) or (period_dt.dt.day != 31)')
)
# check that the removal was successful
# df_demand.shape[0] - df_demand_noleap.shape[0]
#df_demand_noleap[df_demand_noleap['date'] == datetime.date(2016,2,28)].index
#df_demand_noleap['period_dt'].iloc[5820:5840]
# [d for d in df_demand['period_dt'].values if d not in df_demand_noleap['period_dt'].values]

###### Data as numpy arrays ######
X = df_demand_noleap()['hour_of_year'].values
Y = df_demand_noleap()['value'].values

######### Model definition ######### 
# define the component models
model_decomp = cyclic.DecompositionCyclicSpline(H=24, c_h=1, c_d=1e04, n=n)
model_arima = cyclic.TimeSeriesStatsmodels(ARIMA, order=(1,1,0), trend='t')
# combined model
decomp_trend = cyclic.DecompositionTrend(model_decomp, model_arima)

###### Model estimation ######
decomp_trend_results = decomp_trend.fit(X, Y)
decomp_trend_results.fit_trend.results_sm.summary()
decomp_trend_results.fit_seasonal.fit_seasonal.spline_
decomp_trend_results.fit_seasonal.fit_seasonal.x_unique
decomp_trend_results.fit_trend.index_end
decomp_trend_results.fit_seasonal.fit_seasonal.x

####### Model Forecasts #######
predicted, predicted_seasonal, predicted_trend = decomp_trend_results.predict(180*24)
plt.plot(
    np.arange(90*24),
    Y[-(90*24):]
)
plt.plot(
    np.arange(90*24,270*24), 
    predicted
)
plt.show()

######### Backtesting #########
dt_min = df_demand_noleap()['period_dt'].min().to_pydatetime()
periods_est = [(dt_min, datetime.datetime(y,1,1, tzinfo=ZoneInfo('America/Chicago'))) for y in range(2019,2026)]
periods_eval = [(datetime.datetime(y,1,1, tzinfo=ZoneInfo('America/Chicago')),datetime.datetime(y+1,1,1,tzinfo=ZoneInfo('America/Chicago')))  for y in range(2019,2026)]
eval_functions = {
    'MAE': backtesting.mae, 
    'RMSE': backtesting.rmse, 
    'Rsquared': backtesting.r2
}    
backtester = backtesting.Backtest(eval_functions, depvar='value', indepvars='hour_of_year')
results= backtester.backtests(decomp_trend, df_demand_noleap, 'value', periods_est, periods_eval)
# error statistics
backtester.backtest_evaluation()

# backtest plots
for backtest in backtester.results:
    plt.plot(backtest[2], label='realized')
    plt.plot(backtest[1][0], label='forecast')
    plt.title('Estimation through ' + backtest[0].strftime('%Y-%m-%d'))
    plt.legend()
    plt.xlabel('hour of year')
    plt.ylabel('megawatthours')
    plt.show()

######################## Prophet (benchmark) ########################
# format data to Prophet's requirements
df_prophet = (
    df_demand_noleap()
        .loc[:,['period_dt','value']]
        .rename(columns={'period_dt': 'ds', 
                         'value': 'y'})
        .assign(ds = lambda df: df['ds'].dt.tz_convert('UTC').dt.tz_localize(None))
)
m = Prophet()

m.seasonality_prior_scale # prior for seasonality variance
m.seasonality_mode

# adjust the prior variance of the seasonal component
m.fit(df_prophet)
df_future = m.make_future_dataframe(periods=365)
df_forecast = m.predict(df_future)
fig = m.plot(df_forecast)
fig.show()
backtester_prophet = backtesting.BacktestProphet(eval_functions)
results_prophet = backtester_prophet.backtests(df_demand_noleap, periods_est, periods_eval)
backtester_prophet.backtest_evaluation()

# adjust the prior variance of the seasonal component
var_seasonal = 1E-03
m = Prophet(seasonality_prior_scale=var_seasonal)
m.fit(df_prophet)
df_future = m.make_future_dataframe(periods=365)
df_forecast = m.predict(df_future)
fig = m.plot(df_forecast)
fig.show()
backtester_prophet = backtesting.BacktestProphet(eval_functions, seasonality_prior_scale=var_seasonal)
results_prophet = backtester_prophet.backtests(df_demand_noleap, periods_est, periods_eval)
backtester_prophet.backtest_evaluation()

# backtest plots
for backtest in backtester_prophet.results:
    plt.plot(backtest[2], label='realized')
    plt.plot(backtest[1], label='forecast')
    plt.title('Estimation through ' + backtest[0].strftime('%Y-%m-%d'))
    plt.legend()
    plt.xlabel('hour of year')
    plt.ylabel('megawatthours')
    plt.show()

#################### Empirical Bayes ####################       
df_demand['value_lag1'] = df_demand['value'].shift(1)
df_demand['value_lead1'] = df_demand['value'].shift(-1)
df_demand[['value','value_lag1','value_lead1']].head()

d = 24
df_demand['value_lag1'] = df_demand['value_demean'].shift(d)
df_demand['value_lead1'] = df_demand['value_demean'].shift(-d)

df_demand = (
    df_demand
        .assign(curvature = lambda x: x['value_lead1'] - 2*x['value_demean'] + x['value_lag1'])
        .assign(curvature_demean = lambda x: x['curvature'] - x['curvature'].mean())
)

temp = (
    df_demand[d:-d]   
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

cov_days

def curvature_variance(Y, n) -> float:
    """
    Function to estimate the variance of the curvature (adjacent units, no stride) of the seasonal component. 

    Parameters
    ----------
        Y (np.array): array of outcomes
        n (int): length of seasonal cycle

    Returns 
    ----------
        float: estimate of variance
    """
    # vector of curvatures
    #K = Y[2:] - 2*Y[1:-1] + Y[:-2]
    K = Y
    T = K.shape[0]

    # number of observations in each group j
        # since the data are in chronological order, the first T%n periods occur one more time than the others
    nj = np.concatenate((
        np.repeat(int(np.ceil(T/n)),  T%n),
        np.repeat(int(np.floor(T/n)), n-T%n)
    ))

    # group sums
    A = np.concatenate((K, np.zeros(n*int(np.ceil(T/n))-T))).reshape((int(np.ceil(T/n)),n)).sum(axis=0)
    # sum of squares between groups
    SSB = ((A**2) / nj).sum() - T*K.mean()**2
    MSB = SSB / (n-1)
    # sum of squares within groups 
    SSW = (K**2).sum() - ((A**2) / nj).sum()
    MSW = SSW / (T - n)
    lam = (T - (nj**2).sum() / T) / (n -1)
    var_a = (MSB - MSW) / lam
    return var_a

def seasonal_variance(Y):
    for i in range(12, Y.shape[0]-12):
        Y - 


######################## Test seasonal variance estimation ########################
n = 365*24

def seasonal_mean(t, n1, n2, a1, a2) -> float:
    return a1*np.sin(2*np.pi*t/n1) + a2*np.sin(2*np.pi*t/n2)

class SeasonalSampler:
    def __init__(self, n1, n2, a1, a2, b, s, seed=None):
        self.n1 = n1
        self.n2 = n2
        self.a1 = a1
        self.a2 = a2
        self.b = b
        self.s = s
        self.seed = seed
    def seasonal_mean(self,t):
        return self.a1*np.sin(2*np.pi*t/self.n1) + self.a2*np.sin(2*np.pi*t/self.n2)
    def error_process(self, T, seed=None):
        u0 = 0
        uu = []
        if seed is not None:
            random.seed(seed)
        elif self.seed is not None:
            random.seed(self.seed)
        for _ in range(T):
            u = self.b*u0 + random.gauss(mu=0, sigma=self.s)
            uu.append(u)
            u0 = u
        return np.array(uu)
    def sample(self,T):
        return self.seasonal_mean(np.arange(T) % self.n1) + self.error_process(T)
    def true_variance_curvature(self):
        y = self.seasonal_mean(np.arange(self.n1))
        return np.var(y[2:] - 2*y[1:-1] + y[:-2])
    def true_variance_seasonal(self):
        return np.var(self.seasonal_mean(np.arange(self.n1)))
    def true_variance_seasonal_1(self):
        return np.var(self.a1*np.sin(2*np.pi*np.arange(self.n1)/self.n1))
    def true_variance_seasonal_2(self):
        return np.var(self.a2*np.sin(2*np.pi*np.arange(self.n2)/self.n2))
    
sampler = SeasonalSampler(n1=365*24, n2=24, a1=2, a2=1, b=0.8, s=0.5)


sample = sampler.sample(365*24*10+ 100*24)

plt.plot(sample)
plt.show()

sampler.true_variance_curvature()
sampler.true_variance_seasonal()
curvature_variance(sample, 365*24)

variances = []
for _ in range(10000):
    sample = sampler.sample(365*24*10+ 100*24)
    variances.append(curvature_variance(sample, 365*24))

np.mean(variances)
np.std(variances)

### variance of hour-of-day effects
# the effect of daylight savings time should be negligible since only correlations at 24-hour lags are considered
kernel_24 = np.concat((np.ones(12), np.zeros(1), np.ones(11)))/23
b = (Y.shape[0] - 23) % 24
Yf = (Y[12:-11] - np.convolve(Y, kernel_24, mode='valid'))[:-b].reshape((-1, 24))
Cov = (Yf[1:] * Yf[:-1]).mean(axis=0)

### variance within hour of day, across days

# here may be worth accounting for daylight savings time?
    # same one-lag covariance, this year versus last year? to capture drift?

X.max()
X.min()

8760 
kernel_annual = np.concat((np.ones(int(8760/2))/8759, np.zeros(1), np.ones(int(8760/2)-1)/8759))
b = (Y.shape[0] - 8760 + 1) % 8760
Yf = (Y[int(8760/2):-(int(8760/2)-1)] - np.convolve(Y, kernel_annual, mode='valid'))[:-b].reshape((-1,8760))
Cov = (Yf[1:] * Yf[:-1]).mean(axis=0)

def cov_adjacent(Y, n, k=2):
    n1 = int(np.ceil(n/2))
    n2 = n1 - 1
    b = (Y.shape[0] - n + 1) % n
    kernel = np.ones(n) / n
    #kernel = np.concat((np.ones(n1)/(n-1), np.zeros(1), np.ones(n2)/(n-1)))
    Yf = (Y[n1:-n2] - np.convolve(Y, kernel, mode='valid'))[:-b].reshape((-1,n))
    Cov = (Yf[2:] * Yf[:-2]).mean()
    return Cov

cov_adjacent(Y, 8760)
cov_adjacent(Y, 24)

sampler.true_variance_seasonal()

sampler = SeasonalSampler(n1=365*24, n2=24, a1=0, a2=1, b=0.5, s=0.5)
sample = sampler.sample(365*24*10000+ 100*24)
sampler.true_variance_seasonal_1()
cov_adjacent(sample, 365*24)
sampler.true_variance_seasonal_2()

cov_adjacent(sample, 24)
