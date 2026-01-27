############# Load Packages ###########
import pandas as pd
import numpy as np
from dateutil.relativedelta import relativedelta
import matplotlib.pyplot as plt
import os
import datetime
from math import isnan
from statsmodels.tsa.statespace.exponential_smoothing import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA

import cyclic
import exploratory

# load hourly electricity demand data
df_demand = exploratory.DemandData(path_demand = 'data/demand_full_2025_11_30.csv')

# length of seasonal period
n = 365 * 24

### hourly sum of demand
df_hour_sum = (
    df_demand()
        .groupby('hour_of_year', as_index = False)
        .aggregate({'value': 'mean'})
)

# remove December 31st for leap years
df_hour_sum = df_hour_sum[df_hour_sum['hour_of_year'] <= n-1].copy()
m = df_hour_sum['value']

######## Fit cyclic spline with penalty for concavity in trend ########
spline = cyclic.CyclicSpline(H=24)
spline.fit(
    df_demand()['hour_of_year'], 
    df_demand()['value'], 
    c_h = 1, 
    c_d = 300
)
spline.fit_plot()

spline.fit(
    df_demand()['hour_of_year'], 
    df_demand()['value'], 
    c_h = 1, 
    c_d = 1
)
spline.fit_plot()
spline.fit(
    df_demand()['hour_of_year'], 
    df_demand['value'], 
    c_h = 1, 
    c_d = 0
)
spline.fit_plot()
spline.fit(
    df_demand()['hour_of_year'], 
    df_demand()['value'], 
    c_h = 1, 
    c_d = 1e04
)
spline.fit_plot()

##### Iterative algorithm inspired by the STL algorithm and STR (Dokumentov and Hyndman)
# remove data from December 31st of leap years
df_demand_noleap = (
    df_demand()
        .query('(period_dt.dt.year % 4 != 0) or (period_dt.dt.month != 12) or (period_dt.dt.day != 31)')
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

decomp_results = decomp.fit(
    X,
    Y,
    p = 10
)

decomp_results.fit_seasonal.x_unique

decomp_results.fit_seasonal.plot_fit()

idx = (df_demand_noleap['period_dt'].dt.hour % 24 == 10) | (df_demand_noleap['period_dt'].dt.hour % 24 == 22)

for i, seasonal in enumerate(seasonals):
    df_plot = (
         pd.DataFrame({
            'date': df_demand_noleap['date'][-n:],
            'seasonal': seasonal[-n:]
        })
        .groupby('date', as_index=False)
        .agg({'seasonal': 'sum'})
    )
    plt.plot(
        df_plot['date'][-n:], 
        df_plot['seasonal'][-n:], 
        label = str(i)
    )
plt.legend()
plt.show()

residuals = Y - decomp_results.fitted_values_seasonal
df_residuals = pd.DataFrame({
    'date': df_demand_noleap['date'], 
    'residuals': residuals 
})
df_residuals_daily = ( 
    df_residuals
        .groupby('date', as_index=False)
        .agg({'residuals': 'sum'})
        .iloc[:-1,]
)

plt.plot(residuals)
plt.show()

plt.plot(
    df_residuals_daily['date'], 
    df_residuals_daily['residuals']
)
plt.show()

model_ets = ExponentialSmoothing(
    endog = residuals, 
    trend = True
)
results_ets = model_ets.fit()
results_ets.summary()

fcast_ets = results_ets.predict(start=Y.shape[0], end=Y.shape[0]+n)

plt.plot(fcast_ets)
plt.show()

fcast_ets_insample = results_ets.predict(start=0, end=Y.shape[0])
plt.plot(fcast_ets_insample)
plt.show()

# AR1 on first-differenced data
model_arima = ARIMA(
    endog = residuals, 
    order = (1, 1, 0),
    trend = 't'
)
results_arima = model_arima.fit()
results_arima.summary()
fcast_arima = results_arima.predict(start=Y.shape[0], end=Y.shape[0]+240)
plt.plot(fcast_arima)
plt.show()

class SeasonalTrend:
    def __init__(self, model_seasonal, model_trend):
        self.model_seasonal = model_seasonal
        self.model_trend = model_trend
    def fit(self, df):
        
def create_statsmodels(model_function, **kwargs):
    def model_(endog):
        return model_function(endog, **kwargs)
    return model_

arima = ARIMA()
    
model_arima = cyclic.TimeSeriesStatsmodels(ARIMA, order=(1,1,0), trend='t')
arima_result = model_arima.fit(X)
arima_result.summary()

arima_result.predict(h=30)

seasonal_spline = cyclic.CyclicSpline(
    H = 24, 
    c_h = 1, 
    c_d = 1e04
)
decomp = cyclic.DecompositionCyclicSpline(H=24, c_h=1, c_d=1e04, n=n)
decomp_results = decomp.fit(
    X,
    Y,
    p = 10
)

decomp_results.fit_seasonal.spline_

model_decomp = cyclic.DecompositionCyclicSpline(H=24, c_h=1, c_d=1e04, n=n)
model_arima = cyclic.TimeSeriesStatsmodels(ARIMA, order=(1,1,0), trend='t')

decomp_trend = cyclic.DecompositionTrend(model_decomp, model_arima)
decomp_trend_results = decomp_trend.fit(X, Y, p=10)
decomp_trend_results.fit_trend.results_sm.summary()
decomp_trend_results.fit_seasonal.fit_seasonal.spline_
decomp_trend_results.fit_seasonal.fit_seasonal.x_unique
decomp_trend_results.fit_trend.index_end
decomp_trend_results.fit_seasonal.fit_seasonal.x

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




model_arima = ARIMA(
    endog = X, 
    order = (1, 1, 0),
    trend = 't'
)
arima_result = model_arima.fit()
arima_result.summary()

arima_result.predict(start=X.shape[0]+1, end=X.shape[0]+30)

class Backtest:
    def __init__(self, eval_functions):
        self.eval_functions = eval_functions
    def backtest(self, model_, df, target, period_est, period_eval):
        df_est = df.window_datetime(period_est[0], period_est[1])
        fit = model_.fit(df_est)
        df_eval = df.window_datetime(period_eval[0], period_eval[1])
        forecast = fit.predict(df_eval.shape[0])
        error_dict = {}
        for name, f in self.eval_functions.items():
            error_dict[name] = f(df_eval[target], forecast)
        return [forecast, error_dict]
    def backtests(self, model_, df, target, periods_est, periods_eval):
        backtest_results = [[period_est[0]] + backtest(model_, df, target, period_est, period_eval) for period_est, period_eval in zip(periods_est, periods_eval)]
        self.results = backtest_results
        return backtest_results
    def backtest_evaluation(self):
        return pd.concat([pd.DataFrame(backtest[2]) for backtest in self.results], axis=0)
    def backtest_evaluation_mean(self):
        df_errors = self.backtest_evaluation()
        return df_errors.mean(axis=0)
    
df_demand_noleap.dtypes
df_demand_noleap['period_dt'].min()

dt_min = df_demand_noleap['period_dt'].min()
periods_est = [(dt_min, datetime.datetime(y,1,1)) for y in range(2017,2026)]
periods_eval = [(datetime.datetime(y,1,1),datetime.datetime(y+1,1,1))  for y in range(2017,2026)]
    
backtester = Backtest()
    

def backtest(results, df, target, period_est, period_eval, eval_functions):
    df_est = df.window_datetime(period_est[0], period_est[1])
    fit = results.fit(df_est)
    df_eval = df.window_datetime(period_eval[0], period_eval[1])
    forecast = fit.predict(df_eval.shape[0])
    error_dict = {}
    for name, f in eval_functions.items():
        error_dict[name] = f(df_eval[target], forecast)
    return (forecast, error_dict)

def backtests(results, df, target, periods_est, periods_eval, eval_functions):
    
    for period_est, period_eval in zip(periods_est, periods_eval):
        backtest(results, df, target, period_est, period_eval, eval_functions)

df_demand.window_datetime(datetime.datetime(2018,1,1), datetime.datetime(2020,1,1))

backtest()

backtest




help(results_ets.predict)

dir(results_ets)