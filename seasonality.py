from scipy.linalg import solveh_banded
from scipy.sparse import coo_array
from scipy.linalg import solve_circulant
from scipy.fft import fft, ifft
import numpy as np

# (X^T X + D^T D)^-1 X^T y 
#let s be the variable that defines the season 
#X^T X is a diagonal matrix, where the ith element of the diagonal is 
# the number of occurrences of the corresponding period 

#given vector of seasons s = [1, 2, 3, ... S]
XX = 

df_daily['day_of_year'] = df_daily['date'].dt.dayofyear

df_daily.head()

season_count = np.zeros(366)
len(season_count)
s = df_daily['day_of_year']
for ss in s: 
    season_count[ss-1] += 1 

D = 

np.ones(10)

#assume S >= 5 
ab = np.array([0, 0] np.ones(S-2)])

S = 366

AB = np.array((np.concatenate((np.zeros(2), np.ones(S-2))), 
              np.concatenate(np.array([0, -2]), -4*np.ones(S-3), np.array([-2])), 
              np.concatenate(np.array([1, 5]), 6*np.ones(S-4), np.array([1, 5]))), 
              
              np.concatenate((np.zeros(2), np.ones(S-2))))

A.shape

season_count


ss

np.array([1, 2])

np.concatenate((np.array([1,2]), np.array([1, 1])))

########## Solve Circulant #########
n = 365
k = 100
y = np.random.standard_normal(n*k)

# matrix D^TD in sparse format

row = np.repeat(range(n), 5)
col = (row + (n-2) + np.tile(range(5), n)) % n 
data = np.tile([1, -4, n + 6, -4, 1], n)
A = coo_array((data, (row, col)), shape=(n, n))

col

col_daily = 


b = np.sum(np.reshape(y, (k,n), order = 'C'), axis = 0)

c = coo_array((data[0:5], (col[0:5], np.tile(0,5))), shape = (n,1)).toarray().reshape(-1)
solve_circulant(c, b)



ifft(fft(b)/(fft(c) + 


