#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 13 15:20:16 2022

@author: gam
"""

import numpy as np
import pandas as pd
import matplotlib as mpl
from matplotlib import pyplot as plt
import os
import glob
home_dir = os.getenv("HOME") + '/'
import emcee

def ReadCosmoMCchains(file_prefix,chain_root_dir='NOSAVE/COSMOMC/chains/',
                      chain_dir='NGC6121',
                      expand=True,burnin=None,chains=None,mix=False,verbose=0):
    """Read CosmoMCMC chains 
    
    arguments:
        file_prefix (without _n.txt[.gz])
        chain directory (relative to home directory) 
            [default 'NOSAVE/COSMOMC/chains/WINGS']
        expand: expand chains by number of passes (1st column) [default True]
        burnin: number of burnin elements per chain 
            [default None for 2000*N_free-parameters]
        mix: mix chains (default False)
        chains: chain number (None for all) [default None]
        verbose: verbosity (0 for no messages) [default 0]
        
    returns: list of chains (list of numpy arrays)
    
    author: Gary Mamon (gam AAT iap.fr)
    """
        
    # chain files
    path = home_dir + chain_root_dir + chain_dir + '/' + file_prefix + '_[0-9].txt*'
    if verbose >= 1:
        print("ReadCosmoMCchains: path=",path)
    files = np.sort(glob.glob(path))
    if chains is not None:
        files = files[chains-1]
    if verbose >= 1:
        print("files=",files)
    # loop over chains (i.e. over files)        
    tabs = []
    for i, file in enumerate(files):
        # read file (.txt or .txt.gz)
        tab = np.loadtxt(file)
        if verbose >= 1:
            print("chain shape = ",i+1,tab.shape,"...")
        if not expand:
            if verbose >= 2:
                print(tab[0:3])
            if mix:
                tabs.extend(tab)
            else:
                tabs.append(tab)
        else:
            if verbose >= 1:
                if verbose >= 2:
                    print(tab[0:1])
            # expand file by number of passes
            tabExpand = ExpandChain(tab,burnin=burnin,verbose=verbose)
            if verbose >= 1:
                print("expanded")
                if verbose >= 2:
                    print(tabExpand[0:5])
            if mix:
                tabs.extend(tabExpand)
            else:
                tabs.append(tabExpand)
    # params = ['nll','norm','darkscale','darkpar2','ltracermasstot',
    #           'ltracerradius_E','ltracerradius_S0','ltracerradius_S',
    #           'ltracermass_E','ltracermass_S0','ltracermass_S',
    #           'tracerpar2_E','tracerpar2_S0','tracerpar2_S',
    #           'fractracer_E','fractracer_S0','fractracer_S',
    #           'lanis0_E','lanis0_S0','lanis0_S',
    #           'lanisinf_E','lanisinf_S0','lanisinf_S',
    #           'lanisradius_E','lanisradius_S0','lanisradius_S',
    #           'lbhmass','lBilop']
    # if not expand:
    #     params.insert(0,'wt')
              
    if mix:
        return np.array(tabs)
    else:
        return tabs

def ExpandChain(tab,burnin=None,verbose=0):
    """Expand CosmoMC chain (using 1st column for number of passes)
    
    Arguments:
        tab: chain (np.array shape [Nelements,Nparams+2], 2 for Npasses, lnL)
        burnin: number of burnin elements (None for 2000*N_free-parameters)
            [default None]
        verbose: verbosity (0 for none) [default 0]
        
    Returns: expanded chain without number of passes (numpy array)
    
    Author: Gary Mamon (gam AAT iap.fr)"""
    
    # prepare output table
    if verbose > 0:
        print("entering ExpandChain ...")
    Nout = np.sum(tab[:,0]).astype(int)
    tabout = np.zeros((Nout,tab.shape[1]-1))
    
    # burnin estimation: 2000*(number of free parameters)
    if (burnin is None) or (burnin==0):
        # number of free parameters
        free = 0
        # loop over all (free and fixed) parameters
        for j in range(2,tab.shape[-1]): # start at 2 to skip Npasses, lnL
            if np.std(tab[:,j]) > 0.0001:
                free += 1
        burnin = int(2000*free)
        if verbose >= 1:
            print("burnin=",burnin)
            
    # loop over chain elements
    jj = 0
    for i, line in enumerate(tab[:,:]):
        # number of passes
        num = line[0].astype(int)
        
        # loop over number of passes
        for j in range(num):
            tabout[jj] = line[1:] # 1 is to remove Npasses column
            jj += 1
            
    # return post-burnin elements
    return np.array(tabout[burnin:])
        
def AutoCorrWINGS3(tabs=None,models=['6','7','7c','12','12E','15','15E'],
                   verbose=0):
    """Autocorrelation `time' of each free parameter of WINGS chains
    Arguments: 
        list of chains
        models (list)
        verbose (0 for none, default 0)
    Returns nothing (prints out results)
    Author: Gary Mamon (gam AAT iap.fr)
    """
    # read list of runs
    df = pd.read_csv(home_dir + 'PAPERS/MAMPOSST/WINGS/prefixes2_WINGS3.txt',sep=' ')
    # convert list of runs to dictionary
    model_dict = dict(df.itertuples(False,None))
    # dictionary of requested models
    model_dict2 = {model:model_dict[model] for model in models}
    # parameters
    params = {1:'norm',2:'darkscale',3:'darkpar2',
              5:'ltr_E',6:'ltr_S0',7:'ltr_S',
              17:'lA0_E',18:'lA0_S0',19:'lA0_S',
              20:'lAinf_E',21:'lAinf_S0',22:'lAinf_S',
              23:'rA_E',24:'rA_S0',25:'rA_S'}
    
    # loop over models
    for i, model in enumerate(model_dict2):
        print("\nmodel",model)
        
        # free parameters for model
        if model == '6':
            freeparams = [1,2,3,5,6,7,17,18,19,20,21,22]
        elif model in ['7','7c']:
            freeparams = [1,2,3,5,6,7,17,18,19,20,21,22,23,24,25]
        elif model in ['12','12E']:
            freeparams = [1,2,5,6,7,17,18,19,20,21,22,23,24,25]
        else:
            freeparams = [1,2,5,6,7,17,18,19,20,21,22]
            
        # read and expand chains
        fileprefix = model_dict2[model]
        if verbose >= 1:
            print("file prefix = ", fileprefix)
        tabs = ReadCosmoMCchains(model_dict2[model],verbose=verbose)
        if verbose >= 1:
            print("len tabs = ",len(tabs))
            print("shape tabs[0] = ",tabs[0].shape)
        
        # flatten chains
        Ntot = 0
        for i in range(len(tabs)):
            Ntot += np.sum(len(tabs[i]))
        tabs2 = np.zeros((Ntot,tabs[0].shape[1]))
        for j in range(tabs[0].shape[1]):
            N1 = 0
            for i in range(len(tabs)):
                tab = tabs[i]
                # print("shape tab = ",tab.shape)
                tabs2[N1:N1+len(tab),j] = tab[:,j]
                N1 += len(tab)
        
        # loop over free parameters
        for j in freeparams:
            # autocorrelation `time'
            tau = emcee.autocorr.integrated_time(tabs2[:,j],tol=50,quiet=True)
            # print
            print(params[j],'%.1f'%tau[0])

def paramsMean(chainstab,param):
    params = {1:'norm',2:'darkscale',3:'darkpar2',
              5:'ltr_E',6:'ltr_S0',7:'ltr_S',
              17:'lA0_E',18:'lA0_S0',19:'lA0_S',
              20:'lAinf_E',21:'lAinf_S0',22:'lAinf_S',
              23:'rA_E',24:'rA_S0',25:'rA_S'}
    params_inv = {v: k for k, v in params.items()}
    rvir = 10**chainstab[:,params_inv['norm']]
    cdark = 10**chainstab[:,params_inv['darkscale']]
    rsdark = rvir/cdark
    betasym0E = chainstab[:,params_inv['lA0_E']]
    betasym0S0 = chainstab[:,params_inv['lA0_S0']]
    betasym0S = chainstab[:,params_inv['lA0_S']]
    # these betasyms in chains are from -1 to 1
    betasyminfE = chainstab[:,params_inv['lAinf_E']]
    betasyminfS0 = chainstab[:,params_inv['lAinf_S0']] 
    betasyminfS = chainstab[:,params_inv['lAinf_S']]
    beta0E = 2*betasym0E/(1+betasym0E)
    beta0S0 = 2*betasym0S0/(1+betasym0S0)
    beta0S = 2*betasym0S/(1+betasym0S)
    betainfE = 2*betasyminfE/(1+betasyminfE)
    betainfS0 = 2*betasyminfS0/(1+betasyminfS0)
    betainfS = 2*betasyminfS/(1+betasyminfS)
    rbetaE = 10**chainstab[:,params_inv['rA_E']]
    rbetaS0 = 10**chainstab[:,params_inv['rA_S0']]
    rbetaS = 10**chainstab[:,params_inv['rA_S']]
    rtrE = 10**chainstab[:,params_inv['ltr_E']]
    rtrS0 = 10**chainstab[:,params_inv['ltr_S0']]
    rtrS = 10**chainstab[:,params_inv['ltr_S']]
    ctrE = rvir/rtrE
    ctrS0 = rvir/rtrS0
    ctrS = rvir/rtrS
    rtroverrvirE = 1/ctrE
    rtroverrvirS0 = 1/ctrS0
    rtroverrvirS = 1/ctrS
    if param == 'rbeta/rvir':
        return [(rbetaE/rvir).mean(),(rbetaS0/rvir).mean(),
                (rbetaS/rvir).mean()]
    elif param == 'rbeta/rtr':
        return [(rbetaE/rtrE).mean(),(rbetaS0/rtrS0).mean(),
                (rbetaS/rtrS).mean()]
    elif param == 'ctr':
        return [ctrE.mean(),ctrS0.mean(),ctrS.mean()]
    elif param == 'rtr/rvir':
        return [rtroverrvirE.mean(),rtroverrvirS0.mean(),rtroverrvirS.mean()]
    elif param == 'rbeta25':
        # return radius of beta=0.25
        rbeta25E = (1/4-beta0E)/(betainfE-1/4)*rbetaE
        rbeta25S0= (1/4-beta0S0)/(betainfS0-1/4)*rbetaS0
        rbeta25S= (1/4-beta0S)/(betainfS-1/4)*rbetaS
        return [(rbeta25E/rvir).mean(), (rbeta25S0/rvir).mean(), 
                (rbeta25S/rvir).mean()]
    elif param == 'beta0': 
        return [beta0E.mean(),beta0S0.mean(),beta0S.mean()]
    elif param == 'betainf':
        return [betainfE.mean(),betainfS0.mean(),betainfS.mean()]
    elif param == 'betavir':
        betavirE = beta0E + (betainfE-beta0E)*rvir/(rvir+rbetaE)
        betavirS0 = beta0S0 + (betainfS0-beta0S0)*rvir/(rvir+rbetaS0)
        betavirS = beta0S + (betainfS-beta0S)*rvir/(rvir+rbetaS)
        return [betavirE.mean(),betavirS0.mean(),betavirS.mean()]
    elif param == 'betasymvir':
        betavirE = beta0E + (betainfE-beta0E)*rvir/(rvir+rbetaE)
        betavirS0 = beta0S0 + (betainfS0-beta0S0)*rvir/(rvir+rbetaS0)
        betavirS = beta0S + (betainfS-beta0S)*rvir/(rvir+rbetaS)
        # new definition
        betasymvirE = 2*betavirE/(2-betavirE)
        betasymvirS0 = 2*betavirS0/(2-betavirS0)
        betasymvirS = 2*betavirS/(2-betavirS)
        return [betasymvirE.mean(),betasymvirS0.mean(),betasymvirS.mean()]
    elif param == 'lA0':
        betasmeanE = betasym0E.mean()
        betasmeanS0 = betasym0S0.mean()
        betasmeanS = betasym0S.mean()
        # these betasyms go from -1 to 1
        betameanE = 2*betasmeanE/(1+betasmeanE)
        betameanS0 = 2*betasmeanS0/(1+betasmeanS0)
        betameanS = 2*betasmeanS/(1+betasmeanS)
        return[[betasmeanE,betameanE],[betasmeanS0,betameanS0],
               [betasmeanS,betameanS]]
    elif param == 'lAinf':
        betasmeanE = betasyminfE.mean()
        betasmeanS0 = betasyminfS0.mean()
        betasmeanS = betasyminfS.mean()
        # these betasyms go from -1 to 1
        betameanE = 2*betasmeanE/(1+betasmeanE)
        betameanS0 = 2*betasmeanS0/(1+betasmeanS0)
        betameanS = 2*betasmeanS/(1+betasmeanS)
        return[[betasmeanE,betameanE],[betasmeanS0,betameanS0],
               [betasmeanS,betameanS]]
    elif param == 'darkpar2':
        return chainstab[:,params_inv['darkpar2']].mean()
    else:
        return chainstab[:,params_inv[param]].mean()
    
def paramsMLE(chainstab,param):
    params = {1:'norm',2:'darkscale',3:'darkpar2',
              5:'ltr_E',6:'ltr_S0',7:'ltr_S',
              17:'lA0_E',18:'lA0_S0',19:'lA0_S',
              20:'lAinf_E',21:'lAinf_S0',22:'lAinf_S',
              23:'rA_E',24:'rA_S0',25:'rA_S'}
    params_inv = {v: k for k, v in params.items()}
    chainMLE = chainstab[chainstab[:,0].argsort()][0]
    rvir = chainMLE[params_inv['norm']]
    betasym0E = chainMLE[params_inv['lA0_E']]
    betasym0S0 = chainMLE[params_inv['lA0_S0']]
    betasym0S = chainMLE[params_inv['lA0_S']]
    # these betasyms in chains are from -1 to 1
    betasyminfE = chainMLE[params_inv['lAinf_E']]
    betasyminfS0 = chainMLE[params_inv['lAinf_S0']] 
    betasyminfS = chainMLE[params_inv['lAinf_S']]
    beta0E = 2*betasym0E/(1+betasym0E)
    beta0S0 = 2*betasym0S0/(1+betasym0S0)
    beta0S = 2*betasym0S/(1+betasym0S)
    betainfE = 2*betasyminfE/(1+betasyminfE)
    betainfS0 = 2*betasyminfS0/(1+betasyminfS0)
    betainfS = 2*betasyminfS/(1+betasyminfS)
    rbetaE = 10**chainMLE[params_inv['rA_E']]
    rbetaS0 = 10**chainMLE[params_inv['rA_S0']]
    rbetaS = 10**chainMLE[params_inv['rA_S']]
    if param == 'rbeta/rvir':
        return [(rbetaE/rvir).mean(),(rbetaS0/rvir).mean(),
                (rbetaS/rvir).mean()]
    elif param == 'rbeta25':
        # return radius of beta=0.25
        rbeta25E = (1/4-beta0E)/(betainfE-1/4)*rbetaE
        rbeta25S0= (1/4-beta0S0)/(betainfS0-1/4)*rbetaS0
        rbeta25S= (1/4-beta0S)/(betainfS-1/4)*rbetaS
        return [(rbeta25E/rvir).mean(), (rbeta25S0/rvir).mean(), 
                (rbeta25S/rvir).mean()]
    elif param == 'beta0': 
        return [beta0E.mean(),beta0S0.mean(),beta0S.mean()]
    elif param == 'betainf':
        return [betainfE.mean(),betainfS0.mean(),betainfS.mean()]
    elif param == 'betavir':
        betavirE = beta0E + (betainfE-beta0E)*rvir/(rvir+rbetaE)
        betavirS0 = beta0S0 + (betainfS0-beta0S0)*rvir/(rvir+rbetaS0)
        betavirS = beta0S + (betainfS-beta0S)*rvir/(rvir+rbetaS)
        return [betavirE.mean(),betavirS0.mean(),betavirS.mean()]
    elif param == 'betasymvir':
        betavirE = beta0E + (betainfE-beta0E)*rvir/(rvir+rbetaE)
        betavirS0 = beta0S0 + (betainfS0-beta0S0)*rvir/(rvir+rbetaS0)
        betavirS = beta0S + (betainfS-beta0S)*rvir/(rvir+rbetaS)
        # new definition
        betasymvirE = 2*betavirE/(2-betavirE)
        betasymvirS0 = 2*betavirS0/(2-betavirS0)
        betasymvirS = 2*betavirS/(2-betavirS)
        return [betasymvirE.mean(),betasymvirS0.mean(),betasymvirS.mean()]
    elif param == 'lA0':
        betasmeanE = betasym0E.mean()
        betasmeanS0 = betasym0S0.mean()
        betasmeanS = betasym0S.mean()
        # these betasyms go from -1 to 1
        betameanE = 2*betasmeanE/(1+betasmeanE)
        betameanS0 = 2*betasmeanS0/(1+betasmeanS0)
        betameanS = 2*betasmeanS/(1+betasmeanS)
        return[[betasmeanE,betameanE],[betasmeanS0,betameanS0],
               [betasmeanS,betameanS]]
    elif param == 'lAinf':
        betasmeanE = betasyminfE.mean()
        betasmeanS0 = betasyminfS0.mean()
        betasmeanS = betasyminfS.mean()
        # these betasyms go from -1 to 1
        betameanE = 2*betasmeanE/(1+betasmeanE)
        betameanS0 = 2*betasmeanS0/(1+betasmeanS0)
        betameanS = 2*betasmeanS/(1+betasmeanS)
        return[[betasmeanE,betameanE],[betasmeanS0,betameanS0],
               [betasmeanS,betameanS]]
    elif param == 'ctr':
        return 10**(chainMLE[params_inv['norm']]-chainMLE[params_inv['ltr_E']])
    elif param == 'darkpar2':
        return chainMLE[params_inv['darkpar2']]
    else:
        return chainMLE[params_inv[param]]
    
def plotChains(chains,params=['norm','darkscale','lAinf_E','lAinf_S0','lAinf_S']):
    param_dict = {1:'norm',2:'darkscale',3:'darkpar2',
              5:'ltr_E',6:'ltr_S0',7:'ltr_S',
              17:'lA0_E',18:'lA0_S0',19:'lA0_S',
              20:'lAinf_E',21:'lAinf_S0',22:'lAinf_S',
              23:'rA_E',24:'rA_S0',25:'rA_S'}
    params_inv = {v: k for k, v in param_dict.items()}
    print("params=",params)
    print("params_inv=",params_inv)
    param_nums = [params_inv[p] for p in params]
    print("param_nums=",param_nums,"type=",type(param_nums))
    print("len param_nums=",len(param_nums))
    npars = len(params)
    formatter_limits = mpl.rcParams['axes.formatter.limits']
    mpl.rcParams['axes.formatter.limits']= [-8,8]
    fig, axes = plt.subplots(npars, figsize=(10, 7), sharex=True)
    for i, num in enumerate(param_nums):
        ax = axes[i]
        for j in range(6):
            ax.plot(chains[j][:,num], 'k', alpha=0.3)
        ax.set_ylabel(params[i].replace('_','\_'))
        if i == len(param_nums)-1:
            ax.set_xlabel('chain element')
    mpl.rcParams['axes.formatter.limits']= formatter_limits
