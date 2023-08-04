#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 23 21:36:01 2023
@author: Abhner P. de Almeida (abhner.almeida AAT usp.br)
"""

####################################################################################################
####################################################################################################

import numpy as np
import pandas as pd
import MATH as mh


####################################################################################################
####################################################################################################

def extractDF(Name, PATH='./data/'):
    '''
    Extract a specific dataframe
    Parameters
    ----------
    Name : file name. str
    PATH : path for the file. Default: './data/'. str

    Returns
    -------
    Requested dataframe
    -------
    Author: Abhner P. de Almeida (abhner.almeida AAT usp.br)
    '''
    
    df = pd.read_pickle(PATH+Name+".pkl")

    return df

####################################################################################################
####################################################################################################


def makeDF(sample, case, IDs=None, PATH='./data/', dfName=None, SampleName='Sample', verbose=False):
    '''
    Make the parameter evolution dataframe for a specific sample
    Parameters
    ----------
    sample : sample name. str
    case : parameter name. str
    IDs : for random cases the subhalo IDs. Default: None, can be a int or array with int
    PATH : path for the file. Default: './data/'. str
    dfName : for the complete dataframe sample to select the subhalos. Default: None. str
    SampleName : the sample name if using a specific dataframe sample. Default: Sample. str
 
    Returns
    -------
    Requested dataframe
    -------
    Author: Abhner P. de Almeida (abhner.almeida AAT usp.br)
    '''

    if verbose:
        print('\n Taking df for the EVOLUTION of '+case +' ...')

    #verify the parameter df
    try:
        dfEvolution = extractDF(case, PATH=PATH+'Evolution/')
    except:
        if verbose:
            print('You don\'t have the DF for this case')
        return
	
    if verbose:
        print('Restricting  df for the '+sample+' SAMPLE ...')

    #take subhalo for the sample
    
    dSample = extractWithCondition(
                sample, PATH=PATH, dfName=dfName, SampleName=SampleName)
            
    #restricting dfEolution only for subhalos in dfSample
    
    if verbose:
        print('Take the '+case+' EVOLUTION for the '+sample+' SAMPLE  ...')

    if type(IDs) == int or IDs is None:
        keys = dSample.SubfindID

        try:
            df = dfEvolution[keys.astype(str)].copy()
        except:
            df = pd.DataFrame()
            for key in keys:
                try:
                    df[str(key)] = dfEvolution[str(key)].copy()
                except:
                    continue
    else:
        keys = IDs
        try:
            df = dfEvolution[keys.astype(str)].copy()
        except:
            df = pd.DataFrame()
            for key in keys:
                try:
                    df[str(key)] = dfEvolution[str(key)].copy()
                except:
                    continue

    return df

####################################################################################################
####################################################################################################

def extractWithCondition(sample, PATH='./data/', dfName=None, SampleName='Sample'):
    '''
    Extract a subhalo sample
    Parameters
    ----------
    sample : sample name. str
    case : parameter name. str
    PATH : path for the file. Default: './data/'. str
    dfName : for the complete dataframe sample to select the subhalos. Default: None. str
    SampleName : the sample name if using a specific dataframe sample. Default: Sample. str
 
    Returns
    -------
    Requested dataframe
    -------
    Author: Abhner P. de Almeida (abhner.almeida AAT usp.br)
    '''
    
    NamesSample = []
    NameCheck = True

    #if you have the data frame for a specific sample
    if dfName is None:

        for NameClass in ['Compact', 'Diffuse', 'Normal']:
            if NameClass in sample:
                df = extractDF(NameClass, PATH=PATH + 'Samples/')
                NameCheck = False
        if NameCheck:
            df = extractDF('Compact', PATH=PATH + 'Samples/')
            df2 = extractDF('Normal', PATH=PATH + 'Samples/')
            df3 = extractDF('Diffuse', PATH=PATH + 'Samples/')
            df = df.append(df2)
            df = df.append(df3)

        # Restricting the sample
        
        for NameSample in ['Central', 'Satellite', 'Isolated', 'Backsplash', 'FirstInfall']:
            if NameSample in sample and not 'All' in sample:
                df = df.loc[df.Sample == NameSample]
        for NameSample in ['CentralAll', 'SatelliteAll']:

            if NameSample in sample:
                if 'Central' in NameSample:
                    df = df.loc[df.SampleAll == 'Central']
                elif 'Satellite' in NameSample:
                    df = df.loc[df.SampleAll == 'Satellite']

        if 'LosesBH' in sample:
            df = df.loc[df.BH == 'LosesBH']
        elif 'nBH' in sample:
            df = df.loc[df.BH == 'nBH']

        elif 'BH' in sample and not 'LosesBH' in sample and not 'nBH' in sample:
            df = df.loc[df.BH == 'BH']

        for NameSample in ['MW', 'LowerMW', 'LargerMW']:
            if NameSample in sample:
                df = df.loc[df.MWLike == NameSample]

        for NameSample in ['Inner', 'Between', 'Outer']:
            if NameSample in sample:
                df = df.loc[df.GroupRegion == NameSample]

        for BORN in ['Young', 'Intermediare', 'Old']:
            if BORN in sample:
                df = df.loc[df.Age == BORN]

        for BORN in ['radBefore', 'radAfter']:
            if BORN in sample:
                df = df.loc[df.SizeEvolution == BORN]

        for BORN in ['True', 'False']:
            if BORN in sample:
                df = df.loc[df.SizeExtreme == BORN]

    #if you use the dataframe with all the subhalos

    elif dfName is not None:
        df = extractDF(dfName, PATH=PATH + 'Samples/')

        # Restricting the sample if sample is not 'All'
        if sample != 'All':
            for name in ['Compact', 'Diffuse', 'Normal',  'ControlSample', 'HigherSize', 'LowerSize', 'HigherSigma', 'LowerSigma', '1SigmaLower', '2SigmaLower', '3SigmaLower', '1SigmaHigher', '2SigmaHigher', '3SigmaHigher']:
                if name in sample:
                    df = df.loc[df[SampleName] == name]

            for name in ['Satellite', 'Central']:
                if name in sample:
                    try:
                        df = df.loc[df.CentSat == name]
                    except:
                        df = df.loc[df.CentralSatellite == name]

            for name in ['Young', 'Old']:
                if name in sample and not ('YoungBH' in sample):
                    df = df.loc[df.Born == name]
            for name in ['EarlierInfall', 'IntermediateInfall', 'RecentInfall']:
                if name in sample:
                    df = df.loc[df.InfallTime == name]
            for name in ['WithoutBH', 'LosesBH', 'BH']:
                if name in sample and not ('OldBH' in sample or  'IntermediateBH' in sample or 'YoungBH' in sample):
                    df = df.loc[df.BHSample == name]
                    break
            for name in ['OldBH', 'IntermediateBH', 'YoungBH', 'WithoutBH']:
                if name in sample:
                    df = df.loc[df.BHBorn == name]

    return df


