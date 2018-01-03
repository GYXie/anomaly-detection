# -*- encoding=utf-8 -*-
import os
from math import sqrt

import matplotlib.pyplot as plt
import numpy as np
from numpy import array
from pandas import Series
from scipy.optimize import fmin_l_bfgs_b


# references
# 1. http://blog.csdn.net/u010665216/article/details/78051192
# 2. https://www.otexts.org/fpp/7/5
# 3. https://www.usenix.org/legacy/events/lisa00/full_papers/brutlag/brutlag_html/
# 4. http://www.itl.nist.gov/div898/handbook/pmc/section4/pmc435.htm
# 5. http://www.itl.nist.gov/div898/handbook/pmc/section4/pmc436.htm

class FittedResult:
    fitted = None
    level = None
    trend = None
    residuals = None
    d = None
    season = None
    SSE = None
    MSE = None
    RMSE = None
    anoms = None

    def __str__(self):
        return str(self.__dict__)


def holt_winters(x,
                 m,
                 alpha=None,  # level
                 beta=None,  # trend
                 gamma=None,  # seasonal component
                 seasonal='additive',
                 delta=2.5,  #
                 verbose=False
                 ):
    """

    :param x:
    :param m:
    :param alpha:
    :param beta:
    :param gamma:
    :param seasonal: 'additive' or 'multiplicative'
    :param verbose:
    :return:
    """
    lenx = len(x)
    ## initialize l0, b0, s0
    l_start = np.mean(x[0:m])
    b_start = (np.mean(x[m:2 * m]) - l_start) / m
    if seasonal == 'additive':
        s_start = x[0:m] - l_start
    else:
        s_start = x[0:m] / l_start

    # initialise array of l, b, s
    level = np.zeros(lenx)
    trend = np.zeros(lenx)
    season = np.zeros(lenx)
    xfit = np.zeros(lenx)
    residuals = np.zeros(lenx)
    d = np.zeros(lenx)
    SSE = 0

    lastlevel = level0 = l_start
    lasttrend = trend0 = b_start
    season0 = s_start
    if verbose:
        print(l_start, b_start, s_start)

    for i in range(lenx):
        if i > 0:
            # definel l(t-1)
            lastlevel = level[i - 1]
            # define b(t-1)
            lasttrend = trend[i - 1]
        # define s(t-m)
        if i >= m:
            lastseason = season[i - m]
        else:
            lastseason = season0[i]

        # forecast for this period i
        if seasonal == 'additive':
            xhat = lastlevel + lasttrend + lastseason
        else:
            xhat = (lastlevel + lasttrend) * lastseason
        xfit[i] = xhat
        res = x[i] - xhat
        residuals[i] = res
        SSE = SSE + res * res

        # calculate weighted average absolute deviation
        if i >= m:
            d[i] = gamma * abs(res) + (1 - gamma) * d[i - m]
        else:
            d[i] = abs(res)

        if verbose:
            print(i, res, x[i], xhat, lastlevel, lasttrend, lastseason)

        # calculate level[i]
        if seasonal == "additive":
            level[i] = alpha * (x[i] - lastseason) + (1 - alpha) * (lastlevel + lasttrend)
        else:
            level[i] = alpha * (x[i] / lastseason) + (1 - alpha) * (lastlevel + lasttrend)

        # calculate trend[i]
        trend[i] = beta * (level[i] - lastlevel) + (1 - beta) * lasttrend

        # calculate season[i]
        if seasonal == 'additive':
            season[i] = gamma * (x[i] - lastlevel - lasttrend) + (1 - gamma) * lastseason
        else:
            season[i] = gamma * (x[i] / (lastlevel + lasttrend)) + (1 - gamma) * lastseason

    lower = xfit - delta * d
    upper = xfit + delta * d
    anoms = np.where((x < lower) | (x > upper))[0]
    result = FittedResult()
    result.SSE = SSE
    MSE = SSE / lenx
    result.MSE = MSE
    result.RMSE = sqrt(MSE)
    result.fitted = xfit
    result.residuals = residuals
    result.d = d
    result.level = level
    result.trend = trend
    result.season = season
    result.anoms = anoms
    return result


def sse(params, *args):
    alpha, beta, gamma = params
    x, seasonal, m = args
    return holt_winters(x, m, alpha, beta, gamma, seasonal).SSE


def holt_winters_auto(x, m):
    initial_values = array([0.0, 1.0, 0.0])
    boundaries = [(0, 1), (0, 1), (0, 1)]
    seasonal = 'additive'
    parameters = fmin_l_bfgs_b(sse, x0=initial_values, args=(x, seasonal, m), bounds=boundaries, approx_grad=True)
    alpha, beta, gamma = parameters[0]
    result = holt_winters(x, m, alpha, beta, gamma, verbose=False)
    return result

