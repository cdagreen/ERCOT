import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import coo_array, diags
from scipy.linalg import solve_circulant

# compute mean of y by grouped x by 
# first converting to pandas (unaccountably fast)
# returns tuple: 1D arrays of unique values of x, and corresponding mean of y
def group_mean_pd(y, x): 
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
        self.x = x
        self.y = y
        if c_h is not None: 
            self.c_h = c_h
        if c_d is not None: 
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