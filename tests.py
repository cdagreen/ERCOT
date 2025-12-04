
# check that np_group_mean() works

m = np_group_mean(
        df_demand['value'], 
        df_demand['hour_of_year'])

m = group_mean_loop(
        df_demand['value'], 
        df_demand['hour_of_year'])

m = group_mean_dict(
        df_demand['value'], 
        df_demand['hour_of_year'])

m_pd = (df_demand
    .groupby('hour_of_year', as_index = False)
    .aggregate({'value':np.mean})
    .reset_index(drop=True)
)

np.max(np.abs(pd.Series(m) - m_pd['value']))


def group_mean_loop(y, x):
    return [np.sum(y[x==xx]) for xx in np.unique(x)]

def group_mean_dict(y, x): 
    dict_ = {}
    for x_, y_ in zip(x,y): 
        if dict_.get(x_) is None: 
            dict_[x_] = (1, y_)
        else: 
            dict_[x_] = (dict_[x_][0]+1, 
                         (dict_[x_][0]*dict_[x_][1] + y_)/(dict_[x_][0] +1))
    _, m_ = list(zip(*dict_.values()))
    xx = np.array(list(dict_.keys()))
    idx = xx.argsort()
    return (xx[idx], np.array(m_)[idx])

list(m.values())

k_ = np.array(list(m.keys()))

_, m_ = list(zip(*m.values()))

list(zip(m))

###### time the group mean ######

import time

np.random.rand(10)

n = int(1E7)
x = np.random.choice(range(100), n)
y = 2*x + np.random.rand(n)

t1 = time.time()
m = np_group_mean(y, x)
t2 = time.time()
t2-t1

t1 = time.time()
m = group_mean_loop(y, x)
t2 = time.time()
t2-t1

t1 = time.time()
unique_ = np.unique(x)
t2 = time.time()
t2 - t1

t1 = time.time()
unique_ = set(x)
t2 = time.time()
t2 - t1

t1 = time.time()
m = group_mean_dict(y, x)
t2 = time.time()
t2-t1

t1 = time.time()
idx = x.argsort()
t2 = time.time()
t2 - t1

# Pandas (for comparison)
df_ = pd.DataFrame({'y':y, 'x':x})

t1 = time.time()
m_pd = (df_ 
    .groupby('x', as_index = False)
    .agg({'y':np.mean}))
t2 = time.time()
t2 - t1

t0 = time.time()
df_test = df_.sort_values('x')
t1 = time.time()
t1 - t0