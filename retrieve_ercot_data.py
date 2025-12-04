import eia_ercot_data
import pandas as pd

import eia_ercot_data

# read data up to current date
df_demand = eia_ercot_data.eia_data_load_main()

# add dummy rows corresponding to missing periods
df_demand = eia_ercot_data.fill_gaps_dt(df_demand)

# summary information on missing values
eia_ercot_data.summarize_missing(df_demand)

date_str = datetime.datetime.now().strftime('%Y_%m_%d')
df_demand.to_csv('data/demand_' + date_str + '.csv')

help(df_demand.ffill)

# # read old data (longer history)
df_demand_old = pd.read_csv('data/demand_2023_09_04.csv')
df_demand_old = df_demand_old.drop('index', axis=1)
df_demand_old['period_dt'] = pd.to_datetime(df_demand_old['period'])

# # fill any gaps in old data
df_demand_old = eia_ercot_data.fill_gaps_dt(df_demand_old)

df_demand_new = (
    df_demand
        .query('period_dt > datetime.datetime(2023,9,4)')
        .reset_index(drop=True)
)

df_demand_full = pd.concat(
    [df_demand_old, 
     df_demand_new], 
    axis=0
)

date_str = datetime.datetime.now().strftime('%Y_%m_%d')
df_demand_full.to_csv('data/demand_full_' + date_str + '.csv')

# check for gaps
time_diffs = np.array([(df_demand_full['period_dt'].iloc[i+1]-df_demand_full['period_dt'].iloc[i]).total_seconds() / (60**2)  for i in range(df_demand_full.shape[0]-1)])
np.min(time_diffs)
np.max(time_diffs)
np.argmax(time_diffs)

# # fill any gaps in old data
# df_demand_old = fill_gaps_dt(df_demand_old)

# summarize_missing(df_demand_old)