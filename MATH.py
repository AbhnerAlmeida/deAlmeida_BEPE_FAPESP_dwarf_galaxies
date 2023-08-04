#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 30 2022
@author: Abhner P. de Almeida (abhner.almeida AAT usp.br)
"""

####################################################################################################
####################################################################################################

import numpy as np

####################################################################################################
####################################################################################################

def split_quantiles(x, y, total_bins = 20, quantile = 0.95):
    '''
    Compute  quantiles for a distribution
    Parameters
    ----------
    x : x data. array with float
    y : y data. array with float
    total_bins : bins to split the x range. Default 20
    quantile : superior limit for the quantile request. Default 0.95

    Returns
    -------
    Arrays with the values in the range of x, and the median, the inferior quantile and 
    the superior quantile for y.
    -------
    Author: Abhner P. de Almeida (abhner.almeida AAT usp.br)
    '''
    
    x = np.array([value for value in x])
    y = np.array([value for value in y])

    #Remove NaN and inf
    argNaN = np.argwhere(np.isnan(x)).transpose()[0]

    y = np.delete(y, argNaN)
    x = np.delete(x, argNaN)
    
    argInf = np.argwhere(np.isinf(x)).transpose()[0]
    y = np.delete(y, argInf)
    x = np.delete(x, argInf)

    #Sort the array
    xargs = np.argsort(x)
    x = x[xargs]
    y = y[xargs]

    #Compute the xRange
    
    if min(x) !=0 and min(x)*max(x) > 0:
        xRange = np.geomspace(min(x), max(x), total_bins + 1)
    else:
        xRange = np.linspace(min(x), max(x), total_bins + 1)

    yquantile95 = np.array([])
    yquantile5 = np.array([])
    ymedian = np.array([])
    xRangefinal = np.array([])

    #Compute the quantiles
    for k, value in enumerate(xRange):
        
        if value == xRange[-2]:
            array = y[np.where(x >= xRange[k - 1])]
        elif k == len(xRange) - 1:
            continue
        else:
            array = y[np.where((x >= value) & (x <xRange[k + 1]))]
        if len(array) < 5:
            continue
        ymedian = np.append(ymedian, np.median(array))
        yquantile95 = np.append(yquantile95, np.quantile(array, quantile))
        yquantile5 = np.append(yquantile5, np.quantile(array, 1-quantile))
        xRangefinal = np.append(xRangefinal, value)

    array = y[np.where(x >= xRange[-2])]
    if len(array) >= 5:
        ymedian = np.append(ymedian, np.median(array))
        yquantile95 = np.append(yquantile95, np.quantile(array, quantile))
        yquantile5 = np.append(yquantile5, np.quantile(array, 1-quantile))
        xRangefinal = np.append(xRangefinal, xRange[-1])
        
    return xRangefinal, ymedian, yquantile95, yquantile5

####################################################################################################
####################################################################################################

def bootstrap(x):
    '''
    make bootstrap
    Parameters
    ----------
    x : x data. array with float
    Returns
    -------
    bootstrap sample
    -------
    Author: Abhner P. de Almeida (abhner.almeida AAT usp.br)
    '''
    values = np.random.choice(x,replace=True,size=len(x))
    return values


####################################################################################################
####################################################################################################

def boostrap_func(x,func=np.median,num_boots=1000):
    """bootstrap measure of standard deviation of function of 1D data set
    arguments:
        x: data (1d numpy array)
        func: function (default np.median)
        num_trials: number of bootstrap samples (defult 1000)
    returns: standard deviation of function
    requirements: numpy, bootstrap [in this file]
    author: Gary Mamon (gam AAT iap.fr)"""

    val = np.zeros(num_boots)
    # loop over bootstrap samples
    x = x[~np.isnan(x)]
    if len(x) > 1:
        for n in range(num_boots):
            # function for the given bootstrap sample
            values = bootstrap(x)
            val[n] = func(values)
        return np.std(val)
    return x

####################################################################################################
####################################################################################################

def BinomialError(N,n,Wilson_nsigma=1.65,Wilson_center_to_zero=False):
    """Binomial fractions
    Arguments:
        N: total number of points in bin (numpy array)
        n: number of interesting points in bin (numpy array of same length as N)
        Wilson_nsigma: 1 for 1 sigma, 1.65 for 95% upper limit
        Wilson_center_to_zero: center points to 0 for Wilson
        recommend default values for 1 sigma on regular points and 1.65 sigma 
                (95% confidence upper [n=0] or lower [n=N] limits)
    Author: Gary Mamon (gam AAT iap.fr)"""
    if not isinstance(N,np.ndarray):
        raise ValueError("N must be a numpy array")
    if not isinstance(n,np.ndarray):
        raise ValueError("n must be a numpy array")
    if len(n) != len(N):
        raise ValueError("N and n must have same length")
    N_tmp = np.where(N==0,1,N)
    p_orig = np.where(N==0,-1,n/N_tmp)
    error_p_orig = np.where(N==0,-1,np.sqrt(p_orig*(1-p_orig)/N))
    Wilson_nsigma2 = Wilson_nsigma*Wilson_nsigma
    if np.max(Wilson_nsigma) > 0:
        if Wilson_center_to_zero:
            p = np.select([n==0,n==N],[0,1],p_orig)
        else:
            p = np.where(((n==0) | (n==N)),(n+0.5*Wilson_nsigma2)/(N+Wilson_nsigma2),p_orig)
        error_p = np.where(((n==0) | (n==N)),Wilson_nsigma*np.sqrt(n*(1-n/N)+0.25*Wilson_nsigma2)/(N+Wilson_nsigma2),
                              error_p_orig)
    else:
        p = p_orig
        error_p = error_p_orig
    return p, error_p