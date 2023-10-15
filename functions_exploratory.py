################## Functions for exploratory analysis ##################

def sine_scaled(x, b):
    return math.sin((2*math.pi*x)/b)

def modify_data_demand(): 
    #df = pd.read_csv('data/demand_2023_09_04.csv')
    df_demand['date'] = pd.to_datetime(df_demand['date']).dt.floor('D')
    df_demand['hour'] = pd.to_datetime(df_demand['hour'])
    df_demand['month'] = df_demand['date'].dt.month
    df_demand['year'] = df_demand['date'].dt.year
    df_demand['hour_of_day'] = df_demand['hour'].dt.hour
    df_demand['day_of_year'] = df_demand['date'].dt.dayofyear
    df_demand['sine_365'] = df_demand['day_of_year'].apply(lambda x: sine_scaled(x, 365.25))


def plot_daily_end(): 
    df_daily_end = df_daily[df_daily['date'].dt.date >= datetime.date(2023,7,15)][0:-1]
    plt.plot(df_daily_end['date'], 
        df_daily_end['value'])
    plt.xticks(ticks = df_daily_end['date'], 
          labels = df_daily_end['date'].dt.strftime('%m/%d'),
          rotation = 45, 
          size = 8)
    plt.show()