import pandas as pd
import numpy as np
from prophet import Prophet

class Backtest:
    def __init__(self, eval_functions, depvar, indepvars):
        self.eval_functions = eval_functions
        self.depvar = depvar
        self.indepvars = indepvars
    def backtest(self, model_, df, target, period_est, period_eval):
        df_est = df.window_datetime(period_est[0], period_est[1])
        X = df_est[self.indepvars].values
        Y = df_est[self.depvar].values
        fit = model_.fit(X, Y)
        df_eval = df.window_datetime(period_eval[0], period_eval[1])
        forecast = fit.predict(df_eval.shape[0])
        error_dict = {}
        for name, f in self.eval_functions.items():
            error_dict[name] = f(df_eval[target].values, forecast[0])
        return [forecast, df_eval[target].values, error_dict]
    def backtests(self, model_, df, target, periods_est, periods_eval):
        backtest_results = [[period_est[1]] + self.backtest(model_, df, target, period_est, period_eval) for period_est, period_eval in zip(periods_est, periods_eval)]
        self.results = backtest_results
        return backtest_results
    def backtest_evaluation(self):
        return pd.concat([pd.DataFrame(backtest[3], index=[0]) for backtest in self.results], axis=0)
    def backtest_evaluation_mean(self):
        df_errors = self.backtest_evaluation()
        return df_errors.mean(axis=0)
    
class BacktestProphet:
    def __init__(self, eval_functions, depvar='value', datevar='period_dt', **kwargs):
        self.eval_functions = eval_functions
        self.depvar = depvar
        self.datevar= datevar
        self.params = kwargs
        print(kwargs)
    def backtest(self, df, period_est, period_eval):
        df_est = df.window_datetime(period_est[0], period_est[1])
        df_eval = df.window_datetime(period_eval[0], period_eval[1])
        df_est = (
            df_est
                .loc[:,[self.depvar,self.datevar]]
                .rename(columns={self.depvar: 'y', self.datevar: 'ds'})
        )
        if df_est['ds'].dt.tz is not None:
            df_est['ds'] = df_est['ds'].dt.tz_convert('UTC').dt.tz_localize(None)
        model_ = Prophet(**self.params)
        model_.fit(df_est)
        df_ds_predict = model_.make_future_dataframe(periods=365)
        df_predict = model_.predict(df_ds_predict)
        forecast = df_predict.tail(df_eval.shape[0])['yhat'].values
        fig = model_.plot(df_predict)
        error_dict = {}
        for name, f in self.eval_functions.items():
            error_dict[name] = f(df_eval[self.depvar].values, forecast)
        return [forecast, df_eval[self.depvar].values, error_dict, fig]
    def backtests(self, df, periods_est, periods_eval):
        backtest_results = [[period_est[1]] + self.backtest(df, period_est, period_eval) for period_est, period_eval in zip(periods_est, periods_eval)]
        self.results = backtest_results
        return backtest_results    
    def backtest_evaluation(self):
        return pd.concat([pd.DataFrame(backtest[3], index=[0]) for backtest in self.results], axis=0)
    def backtest_evaluation_mean(self):
        df_errors = self.backtest_evaluation()
        return df_errors.mean(axis=0)
    
def mae(y, ypred):
    return np.mean(np.abs(y-ypred))

def rmse(y, ypred):
    return np.sqrt(np.mean((y-ypred)**2))

def r2(y, ypred):
    return 1 - np.sum((y-ypred)**2)/np.sum((y-np.mean(y))**2)