import numpy as np
import pandas as pd
from statsmodels.tsa.ar_model import AutoReg as AR
import pandas_datareader as web
import datetime as dt
from scipy.stats import t
import collections


def main():

    START = dt.datetime(1960, 1, 1)
    END = dt.datetime(2022, 8, 1)

    # US inflation time series
    cpi = web.DataReader(["CPIAUCSL"], "fred", START, END)

    # Ensure stationarity
    cpi = np.log(cpi).diff(1)
    cpi.dropna(inplace=True)

    forecast_A = []
    forecast_B = []
    h = 1  # Horizon of prediction
    num_oos = 5  # Testset length

    # Recursivly estimate the model and predict the next observation
    for i in range(-num_oos, 0, h):

        # Fit the model
        model_A = AR(cpi[:i], lags=1).fit()
        model_B = AR(cpi[:i], lags=3).fit()

        # Predict the next value
        pred_A = model_A.forecast(1)
        pred_B = model_B.forecast(1)

        forecast_A.append(pred_A)
        forecast_B.append(pred_B)

    # Testset
    oos_set = cpi.iloc[-num_oos:].squeeze().to_list()

    results = dm_test(oos_set, forecast_A, forecast_B, h=1, harvey_adj=True)
    print(results)


def dm_test(real_values, pred1, pred2, h=1, harvey_adj=True):

    e1_lst = []
    e2_lst = []
    d_lst = []

    real_values = pd.Series(real_values).apply(lambda x: float(x)).tolist()
    pred1 = pd.Series(pred1).apply(lambda x: float(x)).tolist()
    pred2 = pd.Series(pred2).apply(lambda x: float(x)).tolist()

    # Length of forecasts
    T = float(len(real_values))

    # Construct loss differential according to error criterion (MSE)
    for real, p1, p2 in zip(real_values, pred1, pred2):
        e1_lst.append((real - p1)**2)
        e2_lst.append((real - p2)**2)
    for e1, e2 in zip(e1_lst, e2_lst):
        d_lst.append(e1 - e2)

    # Mean of loss differential
    mean_d = pd.Series(d_lst).mean()

    # Calculate autocovariance
    def autocovariance(Xi, N, k, Xs):
        autoCov = 0
        T = float(N)
        for i in np.arange(0, N-k):
            autoCov += ((Xi[i+k])-Xs)*(Xi[i]-Xs)
        return (1/(T))*autoCov

    # Calculate the denominator of DM stat
    gamma = []
    for lag in range(0, h):
        gamma.append(autocovariance(d_lst, len(d_lst), lag, mean_d))  # 0, 1, 2
    V_d = (gamma[0] + 2*sum(gamma[1:]))/T

    # Calculate DM stat
    DM_stat = V_d**(-0.5)*mean_d

    # Calculate and apply Harvey adjustement
    # It applies a correction for small sample
    if harvey_adj is True:
        harvey_adj = ((T+1-2*h+h*(h-1)/T)/T)**(0.5)
        DM_stat = harvey_adj*DM_stat

    # Calculate p-value
    p_value = 2*t.cdf(-abs(DM_stat), df=T - 1)

    dm_return = collections.namedtuple('dm_return', 'DM p_value')
    result = dm_return(DM=DM_stat, p_value=p_value)

    return result


if __name__ == "__main__":
    main()
