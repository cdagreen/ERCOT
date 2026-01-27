import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import coo_array, diags
from scipy.linalg import solve_circulant
import warnings

# compute mean of y by grouped x by 
# first converting to pandas (unaccountably fast)
# returns tuple: 1D arrays of unique values of x, and corresponding mean of y
def group_mean_pd(y, x):
    """Compute the mean of y by grouped x
        Parameters
        __________
        y: 1D numpy array
        x: 1D numpy array""" 
    m_pd = (
        pd.DataFrame({
            'x': x,
            'y': y
        })
        .groupby('x', as_index=False)
        .agg({'y': 'mean'})
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
def get_group_mean(y, x, method='pd'):
     """
     Dispatcher to compute the mean of y by grouped x
     Parameters
    __________
    y: 1D numpy array
    x: 1D numpy array
    method: string, 'pd' to use pandas handler, 'np' to use numpy handler
     """
     if method == 'pd': 
         return group_mean_pd(y, x)
     elif method == 'np': 
         return group_mean_np(y, x)
     else: 
         raise ValueError(f'method {method} not found')
     
class CyclicSpline:
    """
    Class to estimate a smoothing spline model for seasonality in time series with one seasonal cycle. 
    One parameter for each period (e.g. hour of year for hourly data), with penalties for concavity in the 
    trend between contiguous values and another for concavity in the seasonal trend. 

    For simplicity, the estimation procedure first computes the mean of the series for each period. This 
    simplifies the computations but comes at the cost of accuracy if some periods occur more often than others in the data

    parameters:
        H (int): length of seasonal period
        c_h (float): penalty for concavity in contiguous values (i.e. between immediate neighbors)
        c_d (float): penalty for concavity in seasonal trend (i.e. from one point in current cycle to same point in the next, at offset of H)
    """
    def __init__(self, H, c_h=None, c_d=None):
        self.H = H
        self.c_h = c_h
        self.c_d = c_d
    def fit(self, x, y, H=None, c_h=None, c_d=None): 
        self.x = x
        self.y = y
        if c_h is not None:
            if self.c_h is not None:
                warnings.warn(f'CyclicSpline object already has attribute c_h with value {self.c_h}. This will be overwritten to {c_h}')
            self.c_h = c_h
        if c_d is not None:
            if self.c_d is not None:
                warnings.warn(f'CyclicSpline object already has attribute c_d with value {self.c_d}. This will be overwritten to {c_d}')
            self.c_d = c_d
        if H is not None: 
            self.H = H

        # compute the mean of y by grouped x
        x_unique, m = get_group_mean(y, x)
        # terms in D'D related to penalties
        dd_pen = [1, -4, 6, -4, 1]
        n = len(m)
        penalty_hour = coo_array((dd_pen, ([n-2, n-1, 0, 1, 2], np.tile(0,5))), shape = (n,1))
        penalty_day = coo_array((dd_pen, ([n-2*self.H, n-self.H, 0, self.H, 2*self.H], np.tile(0,5))), shape = (n,1))

        xx_diag = coo_array(([1], ([0], [0])), shape = (n,1))

        # compile all coefficients 
        c = (self.c_h*penalty_hour + self.c_d*penalty_day + xx_diag).toarray().reshape(-1)

        spline_ = solve_circulant(c, m)
        fitted_dict = dict(zip(x_unique, spline_))
        fitted_values = np.array([fitted_dict[xx] for xx in x])
        return CyclicSplineResults(x_unique, spline_, x, y, m, fitted_values)
    
class CyclicSplineResults:
    def __init__(self, x_unique, spline_, x, y, y_mean, fitted_values):
        self.x_unique = x_unique
        self.spline_ = spline_
        self.x = x
        self.y = y
        self.y_mean = y_mean
        self.fitted_values = fitted_values
    def predict(self, xnew):
        fitted_dict = dict(zip(self.x_unique, self.spline_))
        return np.array([fitted_dict[xn] for xn in xnew])
    def plot_fit(self, downsample=True, alpha=0.6):
        # demand appears to have trough at 10:00 and peak at 22:00 (UTC)
        if downsample:
            idx = (self.x_unique % 24 == 10) | (self.x_unique % 24 == 22)
        else: 
            idx = np.tile(True, len(self.x_unique))
        plt.plot(
            self.x_unique[idx], 
            self.y_mean[idx], 
            alpha = alpha
        )
        plt.plot(
            self.x_unique[idx], 
            self.spline_[idx], 
            alpha = alpha
        )
        plt.grid()
        plt.xlabel('hour of year')
        plt.ylabel('megawatthours')
        plt.show()
    
class MovingAverage:
    def __init__(self, k):
        self.k = k
        #self.n_l = n_l
    def __call__(self, Y):
        if isinstance(Y, (pd.Series, pd.DataFrame)):
            Y = np.array(Y).reshape(-1)         
        m = np.empty(Y.shape[0])
        m[0:(self.k // 2 + 1)] = np.nan
        m[-((self.k+1) // 2):] = np.nan
        for i in range((self.k//2 + 1), Y.shape[0]-(self.k+1) // 2):
            m[i] = Y[i:(i+self.k)].mean()
        return m

class Decomposition:
    """
    Class to perform a seasonal decomposition with an iterative method

    Arguments:
        trend (function or callable object): model for the trend component
        seasonal (object with method .fit()): model for the seasonal component
        low_pass (function or callable object): function for removing trend from seasonal (is this necessary? STL is different)
        p (int): number of iterations
    """
    def __init__(self, trend, seasonal, low_pass, p):
        self.trend = trend
        self.seasonal = seasonal
        self.low_pass = low_pass
        self.center_mask = None
        self.p = p
    def iteration(self, X, Y, Yt=None):
        if Yt is None:
            Rl = Y - self.low_pass(Y)
        else:
            Rl = Y - Yt - self.low_pass(Y-Yt)
        if self.center_mask is None:
            self.center_mask = ~np.isnan(Rl)
        results_seasonal = self.seasonal.fit(X[self.center_mask], Rl[self.center_mask])
        Ys = results_seasonal.predict(X)
        Rs = Y - Ys
        Yt = self.trend(Rs)
        return Ys, Yt, results_seasonal
    def fit(self, X, Y):
        seasonals = []
        Yt = np.zeros_like(Y)
        for _ in range(self.p):
            (Ys, Yt, results_seasonal) = self.iteration(X, Y, Yt)
            seasonals.append(Ys)
        return DecompositionResults(results_seasonal, Ys, Yt, X[-1], seasonals)

class DecompositionResults:
    def __init__(self, fit_seasonal,  fitted_values_seasonal, fitted_values_trend, last_season, seasonal_iterations=None):
        self.fit_seasonal = fit_seasonal
        self.fitted_values_seasonal = fitted_values_seasonal
        self.fitted_values_trend = fitted_values_trend
        self.last_season = last_season
        self.seasonal_iterations = seasonal_iterations
    def predict_seasonal(self, X):
        return self.fit_seasonal.predict(X)
    

class DecompositionCyclicSpline(Decomposition):
    def __init__(self, H, c_h, c_d, n, p):
        trend_ma = MovingAverage(int(np.round(1.5*n)))
        spline = CyclicSpline(H, c_h, c_d, p)
        low_pass = MovingAverage(n)
        super().__init__(trend_ma, spline, low_pass)
    
class DecompositionTrend:
    """
    Model combining seasonal decomposition with a trend model for the residuals (seasonally adjusted series)
    
    Parameters
    ________________________________________
        model_seasonal: seasonal decomposition model object
        model_trend: model object for the seasonally-adjusted series

    """
    def __init__(self, model_seasonal, model_trend):
        self.model_seasonal = model_seasonal
        self.model_trend = model_trend
    def fit(self, X, Y, p):
        results_seasonal = self.model_seasonal.fit(X, Y, p)
        Yr = Y - results_seasonal.fitted_values_seasonal
        results_trend = self.model_trend.fit(Yr)
        return DecompositionTrendResults(results_seasonal, results_trend)

class TimeSeriesStatsmodels:
    def __init__(self, model_class, **kwargs):
        self.model_class = model_class
        self.kwargs = kwargs
    def fit(self, X):
        fit_result = self.model_class(
            endog = X, 
            **self.kwargs
        ).fit()
        index_end = X.shape[0]
        return TimeSeriesStatsmodelsResults(fit_result, index_end)
    
class TimeSeriesStatsmodelsResults:
    def __init__(self, results_sm, index_end):
        self.results_sm = results_sm
        self.index_end = index_end
    def predict(self, h):
        return self.results_sm.predict(start=self.index_end+1, end=self.index_end+h)
    
class DecompositionTrendResults:
    def __init__(self, fit_seasonal, fit_trend):
        self.fit_seasonal = fit_seasonal
        self.fit_trend = fit_trend
    def predict(self, h):
        x0 = self.fit_seasonal.last_season
        n = len(self.fit_seasonal.fit_seasonal.x_unique)
        x = (x0 + range(1,h+1)) % n
        predicted_seasonal = self.fit_seasonal.predict_seasonal(x)
        predicted_trend = self.fit_trend.predict(h)
        predicted = predicted_seasonal + predicted_trend
        return predicted, predicted_seasonal, predicted_trend
    


