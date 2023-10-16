#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 20 09:19:41 2023
@author: Abhner P. de Almeida (abhner.almeida AAT usp.br)
"""

####################################################################################################
####################################################################################################

import numpy as np
import pandas as pd
import scipy as sp
import matplotlib.pyplot as plt
import os

from scipy import stats
from matplotlib.ticker import FixedLocator, FixedFormatter
from matplotlib.ticker import FuncFormatter
from matplotlib.offsetbox import AnchoredText
from matplotlib.lines import Line2D
from scipy.interpolate import interp1d


# local
import MATH
import WorkSample

plt.style.use('abhner.mplstyle')


####################################################################################################
####################################################################################################
def PlotHist(names, columns, rows, Type='z0', snap=99, density=False,  mean=False, legend=False, 
             LegendNames=None, title=False, median=False, medianPlot=False, xlabelintext=False,
             alphaShade=0.3,  linewidth=1.8, fonttext=18, fontlegend=18, fontTitle=28, fontlabel=24,  
             nboots=100, framealpha = 0.95, ColumnPlot=True, NormCount=False, savepath='fig/PlotHist', 
             savefigname='fig', yscale='linear', dfName='Sample', SampleName='Samples', loc='best', 
             limaixsy=False, liminvalue=[0], limax=[1], bins=10, seed=16010504):
    '''
    Plot Hist
    Parameters
    ----------
    names : sample names. array with str
    columns : specific set in the sample / or different param to plot in each column. array with str
    rows : specific set in the sample / or different param to plot in each row. array with str
    Type : plot type. Default: 'z0' since the plot is for a specific snapshot. str
    snap : specific snapshot. Default: 99. int in [0, 99]
    density : to plot a PDF. Default: False. bool
    legend : to make the legend. Default: False. bool
    LegendNames : name in the legend. Default: None. None or array with str
    title : to make the title. Default: False. array with str
    mean : to show the mean line. Default: False. bool
    median : to show the median line. Default: False. bool
    medianPlot : to show the median error region. Default: False. bool
    xlabelintext : to show the x param as a text. Default: False. bool
    alphaShade : alpha value for the median error region. Default: 0.3. float
    linewidth : linewidth for histogram. Default: 0.3. float
    fonttext : fontsize for text. Default: 18. float
    fontlegend : fontsize for legend. Default: 18. float
    fontTitle : fontsize for Title. Default: 28. float
    fontlabel : fontsize for labels. Default: 24. float
    nboots  : fboots for compute median error. Default: 100. int
    framealpha  : alpha for legend. Default: 0.95. float
    ColumnPlot : to make a column plot. Default: True. bool
    NormCount : normalized histogram. Default: False. bool
    savepath : save path. Default: 'fig/PlotHist'. str
    savefigname : fig name. Default: 'fig'. str
    yscale  : scale for y axis. Default: 'linear'. str
    dfName : for the complete dataframe sample to select the subhalos. Default: None. str
    SampleName : the sample name if using a specific dataframe sample. Default: Sample. str
    loc : loc legend. Default: 'best'. str
    limaixsy : limit y axis. Default: False. bool
    liminvalue : min limit y axis. Default:[0]. array with float
    limax : max limit y axis. Default:[1]. array with float
    bins : bins for the histogram. Default:10. int
    seed : seed if you use some random choice. Default: 16010504.
    
    Returns
    -------
    Requested Histogram
    -------
    Author: Abhner P. de Almeida (abhner.almeida AAT usp.br)
    '''
    np.random.seed(seed)

    # Verify NameParameters
    if type(columns) is not list and type(columns) is not np.ndarray:
        columns = [columns]

    if type(rows) is not list and type(rows) is not np.ndarray:
        rows = [rows]

    if ColumnPlot:
        if columns == 'Snap':
            columns = snap
            datas = makedata(names, columns, rows, 'Snap',
                             snap=snap, dfName=dfName, SampleName=SampleName)
        else:
            datas = makedata(names, columns, rows, Type,
                             snap=snap, dfName=dfName, SampleName=SampleName)
    else:
        if columns == 'Snap':
            columns = snap
            datas = makedata(names, rows, columns, 'Snap',
                             snap=snap, dfName=dfName, SampleName=SampleName)
        else:
            datas = makedata(names, rows, columns, Type,
                             snap=snap, dfName=dfName, SampleName=SampleName)

    dfTime = WorkSample.extractDF('SNAPS_TIME')

    # Define axes
    plt.rcParams.update({'figure.figsize': (6*len(columns), 4*len(rows))})
    fig = plt.figure()
    gs = fig.add_gridspec(len(rows), len(columns), hspace=0, wspace=0)
    axs = gs.subplots(sharex='col', sharey='row')

    # Verify axs shape
    if type(axs) is not list and type(axs) is not np.ndarray:
        axs = [axs]
    if type(axs[0]) is not np.ndarray:
        axs = np.array([axs])
        if len(columns) == 1:
            axs = axs.T

    for i, row in enumerate(rows):

        for j, column in enumerate(columns):

            if ColumnPlot:
                titlename = column
                param = row
                data = datas[i][j]
            else:
                titlename = row
                param = column
                data = datas[j][i]

            for l, values in enumerate(data):

                values = np.array([value for value in values])
                values = values[~np.isnan(values)]
                values = values[~np.isinf(values)]
                if len(values) == 0:
                    continue

                if density:

                    kde = stats.gaussian_kde(values)
                    xx = np.linspace(min(values), max(values), 1000)
                    factor = 1
                    axs[i][j].plot(xx, factor*kde(xx), color=colors.get(names[l], 'black'),
                                   ls=lines.get(names[l], 'solid'), linewidth=linewidth, dash_capstyle = capstyles.get(names[l], 'projecting'))

                else:
                    if NormCount:
                        hist, bin_edges = np.histogram(
                            values, bins=bins, density=density)
                        axs[i][j].step(bin_edges[:-1], hist/sum(hist), color=colors.get(
                            names[l], 'black'), ls=lines.get(names[l], 'solid'))
                    else:
                        axs[i][j].hist(values, bins=bins, color=colors.get(names[l], 'black'), alpha=1, histtype='step', ls=lines.get(
                            names[l], 'solid'), density=density,  linewidth=linewidth)

                if mean:
                    print(names[l] + ': '+str(np.nanmean(values)))

                    axs[i][j].axvline(np.nanmean(values), color=colors.get(
                        names[l], 'black'), ls=lines.get(names[l], 'solid'), linewidth=linewidth)

                if median or medianPlot:
                    print(names[l] + ': '+str(np.nanmedian(values)))

                    #axs[i][j].axvline(np.nanmedian(values), color =  colors.get(names[l], 'black'), ls =  lines.get(names[l], 'solid'), linewidth = linewidth)
                    axs[i][j].arrow(np.nanmedian(values), 2, 0, -1.05, color=colors.get(
                        names[l], 'black'), ls=lines.get(names[l], 'solid'), linewidth=linewidth, head_width=0.1)

                    if medianPlot:
                        xerr = MATH.boostrap_func(values, num_boots=nboots)
                        xerr = np.std(values)
                        axs[i][j].axvspan(np.nanmedian(values) - xerr, np.nanmedian(values) + xerr, color=colors.get(names[l], 'black'),
                                          ls=lines.get(names[l], 'solid'), linewidth=linewidth, alpha=alphaShade)

            # Plot details

            axs[i][j].grid(True, color="grey",  which="major", linestyle="-.")
            axs[i][j].set_yscale(yscale)
            axs[i][j].tick_params(labelsize=0.85*fontlabel)

            if yscale == 'log':
                axs[i][j].yaxis.set_major_formatter(
                    FuncFormatter(format_func_loglog))

            if j == 0 and i == 0:
                if legend:
                    if len(LegendNames) >= 1:
                        custom_lines, label, ncol, mult = Legend(
                            LegendNames[0])
                        axs[i][j].legend(
                            custom_lines, label, ncol=ncol, loc=loc, fontsize=mult*fontlegend, framealpha = framealpha)

            if limaixsy:
                axs[i][j].set_ylim(liminvalue[i], limax[i])

            if j == 0:
                if density:
                    axs[i][j].set_ylabel('Density', fontsize=fontlabel)
                else:
                    if NormCount:
                        axs[i][j].set_ylabel(
                            'Normalized Count', fontsize=fontlabel)
                    else:
                        axs[i][j].set_ylabel('Count', fontsize=fontlabel)
                axs[i][j].tick_params(axis='y', labelsize=0.85*fontlabel)

            if j == len(columns) - 1:
                if xlabelintext:

                    Afont = {'color':  'black',
                             'size': fonttext,
                             }
                    if type(xlabelintext) is not bool:
                        anchored_text = AnchoredText(
                            titles[xlabelintext[i]], loc='upper right', prop=Afont)
                    else:
                        anchored_text = AnchoredText(
                            texts[param], loc='upper right', prop=Afont)
                    axs[i][j].add_artist(anchored_text)

            if i == 0:
                if columns == 'Snap':
                    axs[i][j].set_title(
                        r'z = %.1f' % dfTime.z.loc[dfTime.Snap == titlename].values[0], fontsize=fontTitle)
                if title:
                    axs[i][j].set_title(titles.get(
                        titlename), fontsize=fontTitle)
                if 'Gyr' in labels.get(param, 'None') and not 'Gyr^' in labels.get(param, 'None'):
                    axs[i][j].tick_params(bottom=True, top=False)
                    ax2label = axs[i][j].secondary_xaxis('top')

                    zticks = np.array([0., 0.2, 0.5, 1., 2., 5., 20.])
                    zlabels = np.array(
                        ['0', '0.2', '0.5', '1', '2', '5', '20'])
                    zticks_Age = np.array(
                        [13.803, 11.323, 8.587, 5.878, 3.285, 1.2, 0])

                    zticks = zticks.tolist()
                    zticks_Age = zticks_Age.tolist()

                    x_locator = FixedLocator(zticks_Age)
                    x_formatter = FixedFormatter(zlabels)
                    ax2label.xaxis.set_major_locator(x_locator)
                    ax2label.xaxis.set_major_formatter(x_formatter)
                    ax2label.set_xlabel(r"z")
                    axs[i][j].tick_params(labelsize=0.85*fontlabel)

            if i == len(rows) - 1:
                if limin.get(param) is not None:
                    axs[i][j].set_xlim(limin.get(param), limmax.get(param))
                if scales.get(param) is not None:
                    axs[i][j].set_xscale(scales.get(param, 'linear'))

                if len(row) > 1:
                    if 'Gyr' in labels.get(param, 'None') and not 'Gyr^' in labels.get(param, 'None'):
                        axs[i][j].set_xlabel(r'Gyr', fontsize=fontlabel)

                    elif 'Stellar' in labels.get(param, 'None'):
                        axs[i][j].set_xlabel(
                            r'$\log M [\mathrm{M_\odot}]$', fontsize=fontlabel)
                    else:
                        axs[i][j].set_xlabel(
                            labels.get(param), fontsize=fontlabel)
                    axs[i][j].tick_params(labelsize=0.85*fontlabel)

                else:
                    axs[i][j].set_xlabel(labels.get(param), fontsize=fontlabel)
                    axs[i][j].tick_params(axis='x', labelsize=0.85*fontlabel)

                if 'Gyr' in labels.get(param, 'None') and not 'Gyr^' in labels.get(param, 'None'):
                    axs[i][j].set_xlim(-0.9, 14.5)
                    axs[i][j].set_xticks([0, 2, 4, 6, 8, 10, 12, 14])
                    axs[i][j].set_xticklabels(
                        ['', '2', '4', '6', '8', '10', '12', '14'])

    savefig(savepath, savefigname)

    return

####################################################################################################
####################################################################################################

def PlotMedianEvolution(names, columns, rows, Type='Evolution', Xparam=['Time'], title=False, xlabelintext=False, lineparams=False, legendColumn = False,
                        alphaShade=0.3,  linewidth=0.7, framealpha = 0.95, fontTitle=28,  fontlabel=26,  fontlegend=20,  nboots=100,  ColumnPlot=True, limaxis=False,
                        savepath='fig/PlotMedianEvolution', loctext = ['best'], savefigname='fig', dfName='Sample', SampleName='Samples', legend=False, LegendNames='None',  loc=['best'],
                        bins=10, seed=16010504):
    '''
    Plot Median Evolution or Co-Evolution
    Parameters
    ----------
    names : sample names. array with str
    columns : specific set in the sample / or different param to plot in each column. array with str
    rows : specific set in the sample / or different param to plot in each row. array with str
    Type : plot type. Default: 'Evolution'. str 'Evolution' or 'CoEvolution.
    Xparam : x param for Co-evolution. Default: ['Time']. array with str
    title : to make the title. Default: False. False or array with str
    limaxis : if you use lineparams, to make the labels. Default:False.bool
    lineparams : if we want different params as different lines. Default: False. bool
    legendColumn : if the legend is in the column 0 in each row. Default: False. bool
    loctext : loc for text. Default:['best']. array with str
    The rest is the same as the previous functions
    Returns
    -------
    Requested Evolution or Co-Evolution plot
    -------
    Author: Abhner P. de Almeida (abhner.almeida AAT usp.br)
    '''

    np.random.seed(seed)

    dfTime = WorkSample.extractDF('SNAPS_TIME')
    snapsTime = np.array([88, 81, 64, 51, 37, 24])
    # Verify NameParameters
    if type(columns) is not list and type(columns) is not np.ndarray:
        columns = [columns]

    if type(rows) is not list and type(rows) is not np.ndarray:
        rows = [rows]

    if Type == 'Evolution':
        if ColumnPlot:
            if lineparams:
                datasAll = []
                dataserrAll = []
                for row in rows:
                    datas, dataserr = makedataevolution(
                        names, columns, row, Type, SampleName=SampleName, dfName = dfName, nboots=nboots)
                    datasAll.append(datas)
                    dataserrAll.append(dataserr)
            else:
                datas, dataserr = makedataevolution(
                    names, columns, rows, Type, SampleName=SampleName, dfName = dfName, nboots=nboots)
            time = dfTime.Age.values

        else:
            if lineparams:
                datas, dataserr = makedataevolution(
                    names, rows[0], columns, Type, SampleName=SampleName, dfName = dfName, nboots=nboots)
            else:
                datas, dataserr = makedataevolution(
                    names, rows, columns, Type, SampleName=SampleName, dfName = dfName, nboots=nboots)
            time = dfTime.Age.values
            if lineparams:
                rows = [rows]

    elif Type == 'CoEvolution':
        if ColumnPlot:
            if lineparams:
                datasX, datasXerr = makedataevolution(
                    names, columns, Xparam, Type, SampleName=SampleName, dfName = dfName, nboots=nboots)
                datasY, datasYerr = makedataevolution(
                    names, columns, rows[0], Type, SampleName=SampleName, dfName = dfName, nboots=nboots)

            else:
                datasX, datasXerr = makedataevolution(
                    names, columns, Xparam, Type, SampleName=SampleName, dfName = dfName, nboots=nboots)
                datasY, datasYerr = makedataevolution(
                    names, columns, rows, Type, SampleName=SampleName, dfName = dfName, nboots=nboots)

        else:
            datasY, datasYerr = makedataevolution(
                names, columns, rows, Type, SampleName=SampleName, dfName = dfName, nboots=nboots)
            datasX, datasXerr = makedataevolution(
                names, columns, Xparam, Type, SampleName=SampleName, dfName = dfName, nboots=nboots)
            columns = Xparam

    # Define axes
   
    if len(columns) == len(rows):
        plt.rcParams.update({'figure.figsize': (6*len(columns), 6*len(rows))})
    else:
        plt.rcParams.update({'figure.figsize': (6*len(columns), 4*len(rows))})
    fig = plt.figure()
    gs = fig.add_gridspec(len(rows), len(columns), hspace=0, wspace=0)
    axs = gs.subplots(sharex='col', sharey='row')

    # Verify axs shape
    if type(axs) is not list and type(axs) is not np.ndarray:
        axs = [axs]
    if type(axs[0]) is not np.ndarray:
        axs = np.array([axs])
        if len(columns) == 1:
            axs = axs.T

    for i, row in enumerate(rows):

        for j, column in enumerate(columns):

            if not lineparams:

                if Type == 'Evolution':
                    if ColumnPlot:
                        param = row
                        data = datas[i][j]
                        dataerr = dataserr[i][j]
                    else:
                        param = column
                        data = datas[j][i]
                        dataerr = dataserr[j][i]
                elif Type == 'CoEvolution':
                    param = row
                    if ColumnPlot:
                        xparam = Xparam[i]
                        dataX = datasX[0][j]
                        data = datasY[i][j]

                    else:
                        xparam = Xparam[i]
                        dataX = datasX[j][0]
                        data = datasY[i][0]

                for l, values in enumerate(data):

                    values = np.array([value for value in values])

                    if Type == 'Evolution':
                        err = np.array([value for value in dataerr[l]])
                        if ('Compact' in names[l] and not 'Quantile'  in names[l]) or 'ControlSample' in names[l] or ('Diffuse' in names[l] and not 'Quantile'  in names[l]):
                            axs[i][j].plot(time[~np.isnan(values)], values[~np.isnan(
                            values)], color=colors[names[l] + 'Median'], ls=lines[names[l]],
                                lw=1.5*linesthicker.get(names[l], linewidth), dash_capstyle = capstyles.get(names[l], 'projecting'))
                        else:
                            axs[i][j].plot(time[~np.isnan(values)], values[~np.isnan(
                            values)], color=colors[names[l] ], ls=lines[names[l]], lw=1.5*linesthicker.get(names[l], linewidth), dash_capstyle = capstyles.get(names[l], 'projecting'))
         
                        axs[i][j].fill_between(
                            time, values - err, values + err, color=colors[names[l]], ls=lines[names[l]], alpha=alphaShade)

                    elif Type == 'CoEvolution':

                        x = np.array([value for value in dataX[l]])
                        colorSnap = np.array(
                            ['magenta', 'blue', 'cyan', 'lime', 'darkorange', 'red'])
                        if ('Compact' in names[l] and not 'Quantile'  in names[l]) or 'ControlSample' in names[l] or ('Diffuse' in names[l] and not 'Quantile'  in names[l]):
                            
                            axs[i][j].plot(x, values, color=colors.get(names[l]  + 'Median', 'black'), ls=lines.get(
                                names[l], 'dashed'), lw= 1.5* linesthicker.get(names[l], linewidth), dash_capstyle = capstyles.get(names[l], 'projecting'))
                        else:
                            axs[i][j].plot(x, values, color=colors.get(names[l]  , 'black'), ls=lines.get(
                                names[l], 'dashed'), lw= 1.5* linesthicker.get(names[l], linewidth), dash_capstyle = capstyles.get(names[l], 'projecting'))

                        axs[i][j].scatter(x[99-snapsTime], values[99-snapsTime], color=colorSnap,
                                          lw= 2 *linewidth, marker='d',  edgecolors=colors.get(names[l]), s=80, alpha=0.9)
                        axs[i][j].scatter(x[0], values[0], color='black', lw= 2 * linewidth, marker='o',  edgecolors=colors.get(
                            names[l]), s=100 , alpha=0.9)

            if lineparams:
                for m, xparam in enumerate(Xparam):
                    for k, paramname in enumerate(row):
                        if Type == 'Evolution':
                            if ColumnPlot:
                                param = paramname
                                data = datasAll[i][k][j]
                                dataerr = dataserrAll[i][k][j]
                            else:
                                param = column
                                data = datasAll[i][j][k]
                                dataerr = dataserrAll[i][j][k]
                        elif Type == 'CoEvolution':
    
                            if ColumnPlot:
                                xparam = Xparam[i]
                                param = paramname
                                dataX = datasX[m][j]
                                data = datasY[k][j]
    
                            else:
                                param = column
                                param = paramname
                                dataX = datasX[i][j]
                                data = datasY[k][j]
    
                        for l, values in enumerate(data):
    
                            values = np.array([value for value in values])
    
                            if Type == 'Evolution':

                                err = np.array([value for value in dataerr[l]])
                                try:
                                    axs[i][j].plot(
                                        time, values, color=colors[names[l] + 'Median'], ls=lines.get(paramname, '--'), dash_capstyle = capstyles.get(paramname, 'projecting'),
                                        lw=1.5*linesthicker.get(paramname, linewidth))
                                    axs[i][j].fill_between(
                                        time, values - err, values + err, color=colors[names[l]], ls=lines.get(names[l], '--'), alpha=alphaShade)
                                except:
                                    axs[i][j].plot(
                                        time, values, color=colors[column + 'Median'], ls=lines.get(paramname, '--'), lw=1.5*linesthicker.get(paramname, linewidth),
                                        dash_capstyle = capstyles.get(paramname, 'projecting'))
                                    axs[i][j].fill_between(
                                        time, values - err, values + err, color=colors[column], ls=lines.get(names[l], '--'), alpha=alphaShade)
    
                            elif Type == 'CoEvolution':
                                x = np.array([value for value in dataX[l]])
                                colorSnap = np.array(
                                    ['magenta', 'blue', 'cyan', 'lime', 'darkorange', 'red'])
    
                                if len(Xparam) > 1:
                                    axs[i][j].plot(x, values, color=colors.get(names[l] + Xparam[m] , 'black'), ls=lines.get(
                                            paramname, 'dashed'), lw= 1.5* linesthicker.get(paramname, linewidth), dash_capstyle = capstyles.get(paramname, 'projecting'))

                                    axs[i][j].scatter(x[99-snapsTime], values[99-snapsTime], color=colorSnap,
                                                      lw= 2 *linewidth, marker='d',  edgecolors=colors.get(names[l] + Xparam[m]), s=80, alpha=0.9)
                                    axs[i][j].scatter(x[0], values[0], color='black', lw= 2 * linewidth, marker='o',  edgecolors=colors.get(
                                        paramname), s=100 , alpha=0.9)
                                else:
                                    if ('Compact' in names[l] and not 'Quantile'  in names[l]) or 'ControlSample' in names[l]:
            
                                        axs[i][j].plot(x, values, color=colors.get(names[l]  + 'Median', 'black'), ls=lines.get(
                                            paramname, 'dashed'), lw= 1.5* linesthicker.get(paramname, linewidth), dash_capstyle = capstyles.get(paramname, 'projecting'))
                                    else:
                                        axs[i][j].plot(x, values, color=colors.get(names[l]  , 'black'), ls=lines.get(
                                            paramname, 'dashed'), lw= 1.5* linesthicker.get(paramname, linewidth), dash_capstyle = capstyles.get(paramname, 'projecting'))
            
                                    axs[i][j].scatter(x[99-snapsTime], values[99-snapsTime], color=colorSnap,
                                                      lw= 2 *linewidth, marker='d',  edgecolors=colors.get(paramname + 'Median'), s=80, alpha=0.9)
                                    axs[i][j].scatter(x[0], values[0], color='black', lw= 2 * linewidth, marker='o',  edgecolors=colors.get(
                                        paramname), s=100 , alpha=0.9)

    

            # Plot details

            axs[i][j].grid(True, color="grey",  which="major", linestyle="dotted")

            if limin.get(param) != None:
                axs[i][j].set_ylim(limin.get(param), limmax.get(param))
            if scales.get(param) != None:
                axs[i][j].set_yscale(scales.get(param))
            if scales.get(param) == 'log':
                axs[i][j].yaxis.set_major_formatter(
                    FuncFormatter(format_func_loglog))

            if j == 0 and i == 0:
                if legend:
                    if len(LegendNames) >= 1 and len( LegendNames[0])  > 0:
                        custom_lines, label, ncol, mult = Legend(
                            LegendNames[0])
                        axs[i][j].legend(
                            custom_lines, label, ncol=ncol, loc=loc[i], fontsize=mult*fontlegend, framealpha = framealpha)

            if len(columns) > 1 and not legendColumn:
                if j == 1 and i == 0:
                    if legend:
                        if len(LegendNames) > 1:
                            custom_lines, label, ncol, mult = Legend(
                                LegendNames[1])
                            axs[i][j].legend(
                                custom_lines, label, ncol=ncol, loc=loc[i], fontsize=mult*fontlegend, framealpha = framealpha)
                            
                if j == 2 and i == 0:
                    if legend:
                        if len(LegendNames) > 2:
                            custom_lines, label, ncol, mult = Legend(
                                LegendNames[2])
                            axs[i][j].legend(
                                custom_lines, label, ncol=ncol, loc=loc[i], fontsize=mult*fontlegend, framealpha = framealpha)
                
            elif legendColumn:
                if j == 0 and i == 1:
                    if legend:
                        if len(LegendNames) > 1:
                            custom_lines, label, ncol, mult = Legend(
                                LegendNames[1])
                            axs[i][j].legend(
                                custom_lines, label, ncol=ncol, loc=loc[i], fontsize=mult*fontlegend, framealpha = framealpha)
                if j == 0 and i == 2:
                    if legend:
                        if len(LegendNames) > 2:
                            custom_lines, label, ncol, mult = Legend(
                                LegendNames[2])
                            axs[i][j].legend(
                                custom_lines, label, ncol=ncol, loc=loc[i], fontsize=mult*fontlegend, framealpha = framealpha)
                    

            if j == 0:

                if xlabelintext:
                    axs[i][j].set_ylabel(
                        labelsequal.get(param), fontsize=fontlabel)
                    axs[i][j].tick_params(axis='y', labelsize=0.85*fontlabel)

                else:
                    axs[i][j].set_ylabel(labels.get(param), fontsize=fontlabel)
                    axs[i][j].tick_params(axis='y', labelsize=0.85*fontlabel)

            if j == len(columns) - 1:
                if xlabelintext and not limaxis and len(rows) > 1:
                    Afont = {'color':  'black',
                             'size': fontlabel,
                             }
                    anchored_text = AnchoredText(
                        texts.get(param), loc='upper right', prop=Afont)
                    axs[i][j].add_artist(anchored_text)

            if xlabelintext and limaxis and len(rows) > 1 and len(LegendNames) == 1:
                Afont = {'color':  'black',
                         'size': fontlabel,
                         }
                anchored_text = AnchoredText(
                    texts[param], loc=loctext[i], prop=Afont)
                axs[i][j].add_artist(anchored_text)

            if i == 0:

                if title:
                    axs[i][j].set_title(titles.get(
                        title[j], title[j]), fontsize=fontTitle)

                if Type == 'Evolution':

                    axs[i][j].tick_params(
                        bottom=True, top=False, labelsize=0.85*fontlabel)
                    ax2label = axs[i][j].secondary_xaxis('top')

                    zticks = np.array([0., 0.2, 0.5, 1., 2., 5., 20.])
                    zlabels = np.array(
                        ['0', '0.2', '0.5', '1', '2', '5', '20'])
                    zticks_Age = np.array(
                        [13.803, 11.323, 8.587, 5.878, 3.285, 1.2, 0])

                    zticks = zticks.tolist()
                    zticks_Age = zticks_Age.tolist()

                    x_locator = FixedLocator(zticks_Age)
                    x_formatter = FixedFormatter(zlabels)
                    ax2label.xaxis.set_major_locator(x_locator)
                    ax2label.xaxis.set_major_formatter(x_formatter)
                    ax2label.set_xlabel(r"z", fontsize=fontlabel)
                    ax2label.tick_params(labelsize=0.85*fontlabel)

            if i == len(rows) - 1:
                if Type == 'Evolution':

                    axs[i][j].set_xlabel(r't [Gyr]', fontsize=fontlabel)
                    axs[i][j].set_xticks([0, 2, 4, 6, 8, 10, 12, 14])
                    axs[i][j].set_xticklabels(
                        ['', '2', '4', '6', '8', '10', '12', '14'])
                    axs[i][j].tick_params(axis='x', labelsize=0.85*fontlabel)

                elif Type == 'CoEvolution':
                    axs[i][j].set_xscale(scales.get(xparam, 'linear'))
                    if scales.get(xparam) == 'log':
                        axs[i][j].xaxis.set_major_formatter(
                            FuncFormatter(format_func_loglog))
                    axs[i][j].set_xlabel(labels.get(xparam, 'None'), fontsize=fontlabel)
                    axs[i][j].tick_params(axis='x', labelsize=0.85*fontlabel)

    savefig(savepath, savefigname)

    return

####################################################################################################
####################################################################################################

def PlotCustom(names, columns, rows, dataMarker=None, dataLine=None, IDs=None, textinplot=False, Type='Evolution', Xparam='Time', title=False, xlabelintext=False, lineparams=False,
               alphaShade=0.3,  linewidth=0.95, framealpha = 0.95, fontTitle=28, fontlabel=24, fontlegend=18, fonttext=14,  nboots=100,  ColumnPlot=True, limaxis=False,
               savepath='fig/PlotCustom', QuantileError=True, savefigname='fig', loctext = 'center left', dfName ='Sample', SampleName='Samples', legend=False, LegendNames='None',  loc='best',
               bins=10, seed=16010504):
    '''
    Plot Custom plot
    Parameters
    ----------
    names : sample names. array with str
    columns : specific set in the sample / or different param to plot in each column. array with str
    rows : specific set in the sample / or different param to plot in each row. array with str
    dataMarker : provide the parameter to set the marker. Default: None. None or str 'NumMergersTotal'
    dataLine : provide the parameter to set the line thickeness. Default: None. None or str 'SubhaloBHMass'
    IDs: IDs for selected subhalos. Default: None. None or array with int
    textinplot : text to put in plot. Default: False. False or array wit str
    QuantileError : to plot the quantile error region. Default: True. bool
    The rest is the same as the previous functions
    Returns
    -------
    Requested Evolution or Co-Evolution plot
    -------
    Author: Abhner P. de Almeida (abhner.almeida AAT usp.br)
    '''

    np.random.seed(seed)

    dfTime = WorkSample.extractDF('SNAPS_TIME')
    snapsTime = np.array([88, 81, 64, 51, 37, 24])
    # Verify NameParameters
    if type(columns) is not list and type(columns) is not np.ndarray:
        columns = [columns]

    if type(rows) is not list and type(rows) is not np.ndarray:
        rows = [rows]

    if Type == 'Evolution':
        if ColumnPlot:
            if lineparams:
                datas, dataserr = makedataevolution(
                    names, columns, rows[0], Type, SampleName=SampleName, dfName = dfName, IDs=IDs, nboots=nboots)
            else:
                datas, dataserr = makedataevolution(
                    names, columns, rows, Type, SampleName=SampleName, dfName = dfName,  IDs=IDs, nboots=nboots)
            time = dfTime.Age.values
            if dataMarker is not None:
                if 'Merger' in dataMarker:
                    datasMarker, dataMarkererr = makedataevolution(names, columns, [
                                                                   'NumMergersTotal', 'NumMinorMergersTotal', 'NumMajorMergersTotal'], Type,  IDs=IDs, SampleName=SampleName, dfName = dfName, nboots=nboots)
                else:
                    datasMarker, dataMarkererr = makedataevolution(names, columns, [
                                                                   dataMarker], Type,  IDs=IDs, SampleName=SampleName, dfName = dfName, nboots=nboots)
            if dataLine is not None:
                datasLine, dataLineererr = makedataevolution(names, columns, [
                                                             dataLine], Type,  IDs=IDs, SampleName=SampleName, dfName = dfName, nboots=nboots)

        else:
            if lineparams:
                datas, dataserr = makedataevolution(
                    names, rows[0], columns, Type, SampleName=SampleName,dfName = dfName, nboots=nboots)
            else:
                datas, dataserr = makedataevolution(
                    names, rows, columns, Type, SampleName=SampleName, dfName = dfName, nboots=nboots)
            time = dfTime.Age.values
            if dataMarker is not None:
                if 'Merger' in dataMarker:
                    datasMarker, dataMarkererr = makedataevolution(names, rows,  [
                                                                   'NumMergersTotal', 'NumMinorMergersTotal', 'NumMajorMergersTotal'], Type, SampleName=SampleName, dfName = dfName, nboots=nboots)
                else:
                    datasMarker, dataMarkererr = makedataevolution(
                        names, rows, [dataMarker], Type, SampleName=SampleName, dfName = dfName, nboots=nboots)
            if dataLine is not None:
                datasLine, dataLineererr = makedataevolution(
                    names, rows, [dataLine],  Type, SampleName=SampleName, dfName = dfName, nboots=nboots)

            if lineparams:
                rows = [rows]

    elif Type == 'CoEvolution':
        if ColumnPlot:
            datasX, datasXerr = makedataevolution(
                names, columns, [Xparam], Type, SampleName=SampleName, dfName = dfName, nboots=nboots)
            datasY, datasYerr = makedataevolution(
                names, columns, rows, Type, SampleName=SampleName, dfName = dfName, nboots=nboots)
            if dataMarker is not None:
                if 'Merger' in dataLine:
                    datasMarker, dataMarkererr = makedataevolution(names,  columns, [
                                                                   'NumMergersTotal', 'NumMinorMergersTotal', 'NumMajorMergersTotal'],  Type, SampleName=SampleName, dfName = dfName, nboots=nboots)
                else:
                    datasMarker, dataMarkererr = makedataevolution(
                        names, columns, [dataMarker], Type, SampleName=SampleName, dfName = dfName, nboots=nboots)
            if dataLine is not None:
                datasLine, dataLineererr = makedataevolution(
                    names, columns, [dataLine],  Type, SampleName=SampleName, dfName = dfName, nboots=nboots)

        else:
            datasY, datasYerr = makedataevolution(
                names, columns, rows, Type, SampleName=SampleName, dfName = dfName, nboots=nboots)
            datasX, datasXerr = makedataevolution(
                names, columns, Xparam, Type, SampleName=SampleName, dfName = dfName, nboots=nboots)
            columns = Xparam
            if dataMarker is not None:
                if 'Merger' in dataMarker:
                    datasMarker, dataMarkererr = makedataevolution(
                        names,  ['NumMergersTotal', 'NumMinorMergersTotal', 'NumMajorMergersTotal'],  columns, Type, SampleName=SampleName, dfName = dfName, nboots=nboots)
                else:
                    datasMarker, dataMarkererr = makedataevolution(
                        names, [dataMarker], columns, Type, SampleName=SampleName, dfName = dfName, nboots=nboots)
            if dataLine is not None:
                datasLine, dataLineererr = makedataevolution(
                    names, [dataLine], columns,  Type, SampleName=SampleName, dfName = dfName, nboots=nboots)

    # Define axes
    if len(columns) > 2:
        plt.rcParams.update({'figure.figsize': (6*len(columns), 6*len(rows))})
    else:
        plt.rcParams.update({'figure.figsize': (6*len(columns), 4*len(rows))})

    fig = plt.figure()
    gs = fig.add_gridspec(len(rows), len(columns), hspace=0, wspace=0)
    axs = gs.subplots(sharex='col', sharey='row')

    # Verify axs shape
    if type(axs) is not list and type(axs) is not np.ndarray:
        axs = [axs]
    if type(axs[0]) is not np.ndarray:
        axs = np.array([axs])
        if len(columns) == 1:
            axs = axs.T

    for i, row in enumerate(rows):

        for j, column in enumerate(columns):

            if not lineparams:

                if Type == 'Evolution':
                    if ColumnPlot:
                        param = row
                        data = datas[i][j]
                        dataerr = dataserr[i][j]
                        if dataMarker is not None:
                            if 'Merger' in dataMarker:
                                datamarkerTotvalues = datasMarker[0][j]
                                datamarkervalues = datasMarker[1][j]
                                dataMarkervalues = datasMarker[2][j]
                            else:
                                datamarkervalues = datasMarker[0][j]

                        if dataLine is not None:
                            datalinevalues = datasLine[0][j]

                    else:
                        param = column
                        data = datas[j][i]
                        dataerr = dataserr[j][i]
                        if dataMarker is not None:
                            if 'Merger' in dataMarker:
                                datamarkerTotvalues = datasMarker[0][i]
                                datamarkervalues = datasMarker[1][i]
                                dataMarkervalues = datasMarker[2][i]
                            else:
                                datamarkervalues = datasMarker[0][i]
                        if dataLine is not None:
                            datalinevalues = datasLine[0][i]

                elif Type == 'CoEvolution':
                    param = row
                    if ColumnPlot:
                        xparam = Xparam
                        dataX = datasX[0][j]
                        data = datasY[i][j]
                        if dataMarker is not None:
                            if 'Merger' in dataMarker:
                                datamarkerTotvalues = datasMarker[0][j]
                                datamarkervalues = datasMarker[1][j]
                                dataMarkervalues = datasMarker[2][j]
                            else:
                                datamarkervalues = datasMarker[0][j]
                        if dataLine is not None:
                            datalinevalues = datasLine[0][j]

                    else:
                        xparam = Xparam[j]
                        dataX = datasX[j][0]
                        data = datasY[i][0]
                        if dataMarker is not None:
                            if 'Merger' in dataMarker:
                                datamarkerTotvalues = datasMarker[j][0]
                                datamarkervalues = datasMarker[1][j]
                                dataMarkervalues = datasMarker[2][2]
                            else:
                                datamarkervalues = datasMarker[0][j]
                        if dataLine is not None:
                            datalinevalues = datasLine[0][j]

                for l, values in enumerate(data):

                    values = np.array([value for value in values])

                    if Type == 'Evolution':
                        err = np.array([value for value in dataerr[l]])
                        axs[i][j].plot(time[~np.isnan(values)], values[~np.isnan(
                            values)], color=colors[names[l]], ls=lines[names[l]], lw=linesthicker.get(names[l], linewidth), dash_capstyle = capstyles.get(names[l], 'projecting'))

                        if dataLine is not None:
                            linevalues = np.array(
                                [value for value in datalinevalues[l]])

                            axs[i][j].plot(time[(~np.isinf(linevalues)) & (~np.isnan(linevalues)) & (~np.isnan(values))], values[(~np.isinf(linevalues)) & (
                                ~np.isnan(linevalues)) & (~np.isnan(values))], color=colors[names[l]], ls=lines[names[l]], alpha=0.7, lw=5, dash_capstyle = capstyles.get(names[l], 'projecting'))

                        if QuantileError:
                            axs[i][j].fill_between(time[~np.isnan(values)], values[~np.isnan(values)] - err[~np.isnan(values)], values[~np.isnan(
                                values)] + err[~np.isnan(values)], color=colors[names[l]], ls=lines[names[l]], alpha=alphaShade)

                        if dataMarker is not None:
                            markervalues = np.array(
                                [value for value in datamarkervalues[l]])

                            if 'Merger' in dataMarker:
                                mergerTot = np.array(
                                    [value for value in datamarkerTotvalues[l]])
                                MarkerTotvalues = np.array(
                                    [value for value in datamarkerTotvalues[l]])

                                mergernumber = np.array(
                                    [value for value in datamarkervalues[l]])

                                Mergernumber = np.array(
                                    [value for value in dataMarkervalues[l]])
                                Markervalues = np.array(
                                    [value for value in datamarkervalues[l]])

                                mergernumber = np.flip(mergernumber)
                                Mergernumber = np.flip(Mergernumber)
                                mergerTot = np.flip(mergerTot)

                                for nmergerindex, nmerger in enumerate(mergernumber):

                                    if nmergerindex == 0:
                                        markervalues[nmergerindex] = 0
                                        continue
                                    else:
                                        if np.isnan(nmerger):
                                            markervalues[nmergerindex] = 0
                                        else:
                                            if np.isnan(mergernumber[nmergerindex - 1]):
                                                markervalues[nmergerindex] = int(
                                                    nmerger)
                                            else:
                                                markervalues[nmergerindex] = int(
                                                    nmerger) - int(mergernumber[nmergerindex - 1])

                                for nmergerindex, nmerger in enumerate(Mergernumber):

                                    if nmergerindex == 0:
                                        Markervalues[nmergerindex] = 0
                                        continue
                                    else:
                                        if np.isnan(nmerger):
                                            Markervalues[nmergerindex] = 0
                                        else:
                                            if np.isnan(mergernumber[nmergerindex - 1]):
                                                Markervalues[nmergerindex] = int(
                                                    nmerger)
                                            else:
                                                Markervalues[nmergerindex] = int(
                                                    nmerger) - int(Mergernumber[nmergerindex - 1])

                                for nmergerindex, nmerger in enumerate(mergerTot):

                                    if nmergerindex == 0:
                                        MarkerTotvalues[nmergerindex] = 0
                                        continue
                                    else:
                                        if np.isnan(nmerger):
                                            MarkerTotvalues[nmergerindex] = 0
                                        else:
                                            if np.isnan(mergernumber[nmergerindex - 1]):
                                                MarkerTotvalues[nmergerindex] = int(
                                                    nmerger)
                                            else:
                                                MarkerTotvalues[nmergerindex] = int(
                                                    nmerger) - int(mergerTot[nmergerindex - 1])
                                MarkerTotvalues = MarkerTotvalues - Markervalues - markervalues
                                Markervalues = np.flip(Markervalues)
                                markervalues = np.flip(markervalues)
                                MarkerTotvalues = np.flip(MarkerTotvalues)

                            axs[i][j].scatter(time[(Markervalues > 0) & (~np.isnan(values))], values[(Markervalues > 0) & (
                                ~np.isnan(values))], color=colors[names[l]], lw=1., marker='o',  edgecolors='black', s=130, alpha=0.8)
                            axs[i][j].scatter(time[(markervalues > 0) & (~np.isnan(values))], values[(markervalues > 0) & (
                                ~np.isnan(values))], color=colors[names[l]], lw=1., marker='s',  edgecolors='black', s=110, alpha=0.8)
                            axs[i][j].scatter(time[(MarkerTotvalues > 0) & (~np.isnan(values))], values[(MarkerTotvalues > 0) & (
                                ~np.isnan(values))], color=colors[names[l]], lw=1., marker='p',  edgecolors='black', s=110, alpha=0.8)

                    elif Type == 'CoEvolution':
                        x = np.array([value for value in dataX[l]])
                        colorSnap = np.array(
                            ['magenta', 'blue', 'cyan', 'lime', 'darkorange', 'red'])
                        axs[i][j].scatter(x[99-snapsTime], values[99-snapsTime], color=colorSnap,
                                          lw=1., marker='d',  edgecolors=colors.get(names[l]), s=30, alpha=0.9)
                        axs[i][j].scatter(x[0], values[0], color='black', lw=1., marker='o',  edgecolors=colors.get(
                            names[l]), s=30, alpha=0.9)
                        axs[i][j].plot(x, values, color=colors.get(
                            names[l], 'black'), ls=lines.get(names[l], 'dashed'))

            if lineparams:
                for k, paramname in enumerate(row):
                    if Type == 'Evolution':
                        if ColumnPlot:
                            param = paramname
                            data = datas[k][j]
                            dataerr = dataserr[k][j]
                        else:
                            param = column
                            data = datas[j][k]
                            dataerr = dataserr[j][k]
                    elif Type == 'CoEvolution':

                        if ColumnPlot:
                            xparam = Xparam
                            param = paramname
                            dataX = datasX[0][j]
                            data = datasY[k][j]

                        else:
                            param = column
                            xparam = Xparam[i]
                            param = paramname
                            dataX = datasX[i][j]
                            data = datasY[k][j]

                    for l, values in enumerate(data):

                        values = np.array([value for value in values])

                        if Type == 'Evolution':
                            err = np.array([value for value in dataerr[l]])
                            axs[i][j].plot(
                                time, values, color=colors[names[l]], ls=lines.get(paramname, '--'))
                            axs[i][j].fill_between(
                                time, values - err, values + err, color=colors[names[l]], ls=lines.get(paramname, '--'), alpha=alphaShade)

                        elif Type == 'CoEvolution':
                            x = np.array([value for value in dataX[l]])
                            axs[i][j].plot(
                                x, values, color=colors[names[l]], ls=lines.get(paramname, '--'))

            # Plot details

            axs[i][j].grid(True, color="grey",  which="major", linestyle="dotted")

            if limin.get(param) != None:
                axs[i][j].set_ylim(limin.get(param), limmax.get(param))
            if scales.get(param) != None:
                axs[i][j].set_yscale(scales.get(param))
            if scales.get(param) == 'log':
                axs[i][j].yaxis.set_major_formatter(
                    FuncFormatter(format_func_loglog))

            if j == 0 and i == 0:
                if legend:
                    if len(LegendNames) >= 1:
                        custom_lines, label, ncol, mult = Legend(
                            LegendNames[0])
                        axs[i][j].legend(
                            custom_lines, label, ncol=ncol, loc=loc, fontsize=mult*fontlegend, framealpha = framealpha)
            
            if len(columns) == 1:
                if j == 0 and i == 1:
                    if legend:
                        if len(LegendNames) > 1:
                            custom_lines, label, ncol, mult = Legend(
                                LegendNames[1])
                            axs[i][j].legend(
                                custom_lines, label, ncol=ncol, loc=loc, fontsize=mult*fontlegend, framealpha = framealpha)
            if j == 1 and i == 0:
                if legend:
                    if len(LegendNames) > 1:
                        custom_lines, label, ncol, mult = Legend(
                            LegendNames[1])
                        axs[i][j].legend(
                            custom_lines, label, ncol=ncol, loc=loc, fontsize=mult*fontlegend, framealpha = framealpha)

            if j == 2 and i == 0:
                if legend:
                    if len(LegendNames) > 2:
                        custom_lines, label, ncol, mult = Legend(
                            LegendNames[2])
                        axs[i][j].legend(
                            custom_lines, label, ncol=ncol, loc=loc, fontsize=mult*fontlegend, framealpha = framealpha)

            if j == 3 and i == 0:
                if legend:
                    if len(LegendNames) > 3:
                        custom_lines, label, ncol, mult = Legend(
                            LegendNames[3])
                        axs[i][j].legend(
                            custom_lines, label, ncol=ncol, loc=loc, fontsize=mult*fontlegend, framealpha = framealpha)

            if j == 0:

                if xlabelintext:
                    axs[i][j].set_ylabel(
                        labelsequal.get(param), fontsize=fontlabel)
                    axs[i][j].tick_params(labelsize=0.85*fontlabel)

                else:
                    axs[i][j].set_ylabel(labels.get(param), fontsize=fontlabel)
                    axs[i][j].tick_params(labelsize=0.85*fontlabel)

            if j == len(columns) - 1:
                if xlabelintext and not limaxis and len(rows) > 1:
                    Afont = {'color':  'black',
                             'size': fontlabel,
                             }
                    anchored_text = AnchoredText(
                        texts.get(param), loc='upper right', prop=Afont)
                    axs[i][j].add_artist(anchored_text)

            if xlabelintext and limaxis and len(rows) > 1:
                Afont = {'color':  'black',
                         'size': fontlabel,
                         }
                anchored_text = AnchoredText(
                    texts[param], loc='upper left', prop=Afont)
                axs[i][j].add_artist(anchored_text)

            if textinplot:
                Afont = {'color':  'black',
                         'size': fonttext,
                         }
                anchored_text = AnchoredText(titles.get(
                    textinplot[i], textinplot[i]), loc=loctext[i], prop=Afont)
                axs[i][j].add_artist(anchored_text)

            if i == 0:

                if title:
                    axs[i][j].set_title(titles.get(
                        title[j], title[j]), fontsize=fontTitle)

                if Type == 'Evolution':

                    axs[i][j].tick_params(
                        bottom=True, top=False, labelsize=0.85*fontlabel)
                    ax2label = axs[i][j].secondary_xaxis('top')

                    zticks = np.array([0., 0.2, 0.5, 1., 2., 5., 20.])
                    zlabels = np.array(
                        ['0', '0.2', '0.5', '1', '2', '5', '20'])
                    zticks_Age = np.array(
                        [13.803, 11.323, 8.587, 5.878, 3.285, 1.2, 0])

                    zticks = zticks.tolist()
                    zticks_Age = zticks_Age.tolist()

                    x_locator = FixedLocator(zticks_Age)
                    x_formatter = FixedFormatter(zlabels)
                    ax2label.xaxis.set_major_locator(x_locator)
                    ax2label.xaxis.set_major_formatter(x_formatter)
                    ax2label.set_xlabel(r"z", fontsize=fontlabel)
                    ax2label.set_xlabel(r"z", fontsize=fontlabel)
                    ax2label.tick_params(labelsize=0.85*fontlabel)

            if i == len(rows) - 1:
                if Type == 'Evolution':

                    axs[i][j].set_xlabel(r't [Gyr]', fontsize=fontlabel)
                    axs[i][j].set_xticks([0, 2, 4, 6, 8, 10, 12, 14])
                    axs[i][j].set_xticklabels(
                        ['', '2', '4', '6', '8', '10', '12', '14'])
                    axs[i][j].tick_params(labelsize=0.85*fontlabel)

                elif Type == 'CoEvolution':
                    axs[i][j].set_xscale(scales.get(xparam, 'linear'))
                    if scales.get(xparam) == 'log':
                        axs[i][j].xaxis.set_major_formatter(
                            FuncFormatter(format_func_loglog))
                    axs[i][j].set_xlabel(labels.get(
                        xparam, 'None'), fontsize=fontlabel)
                    axs[i][j].tick_params(labelsize=0.85*fontlabel)
            axs[i][j].tick_params(labelsize=0.85*fontlabel)

    savefig(savepath, savefigname)

    return

####################################################################################################
####################################################################################################

def PlotID(columns, rows, IDs, dataMarker=None, dataLine=None, Type='Evolution', Xparam='Time', title=False, xlabelintext=False, lineparams=False,
           alphaShade=0.3,  linewidth=0.5, fontTitle=28, fontlabel=24, fontlegend=18,  nboots=100,  ColumnPlot=True, limaxis=False,
           savepath='fig/PlotCustom', QuantileError=True, savefigname='fig', dfName='Sample', SampleName='Samples', legend=False, LegendNames='None',  loc='best',
           bins=10, seed=16010504):
    '''
    Plot teh evolution for random sample
    Parameters
    ----------
    columns : specific set in the sample / or different param to plot in each column. array with str
    rows : specific set in the sample / or different param to plot in each row. array with str
    IDs: IDs for selected subhalos. 
    The rest is the same as the previous functions
    Returns
    -------
    Requested Evolution or Co-Evolution plot
    -------
    Author: Abhner P. de Almeida (abhner.almeida AAT usp.br)
    '''

    np.random.seed(seed)

    dfTime = WorkSample.extractDF('SNAPS_TIME')
    snapsTime = np.array([88, 81, 64, 51, 37, 24])
    # Verify NameParameters
    if type(columns) is not list and type(columns) is not np.ndarray:
        columns = [columns]

    if type(rows) is not list and type(rows) is not np.ndarray:
        rows = [rows]

    # Define axes
    plt.rcParams.update({'figure.figsize': (6*len(columns), 4*len(rows))})
    fig = plt.figure()
    gs = fig.add_gridspec(len(rows), len(columns), hspace=0, wspace=0)
    axs = gs.subplots(sharex='col', sharey='row')

    # Verify axs shape
    if type(axs) is not list and type(axs) is not np.ndarray:
        axs = [axs]
    if type(axs[0]) is not np.ndarray:
        axs = np.array([axs])
        if len(columns) == 1:
            axs = axs.T

    time = dfTime.Age.values

    for i, row in enumerate(rows):

        for j, column in enumerate(columns):

            data = WorkSample.makeDF(
                column, row, SampleName=SampleName, dfName = dfName, IDs=IDs[j])
            if Type == 'CoEvolution':
                dataX = WorkSample.makeDF(
                    column, Xparam, SampleName=SampleName, dfName = dfName, IDs=IDs[j])

            if dataLine is not None:
                datalinevalues = WorkSample.makeDF(
                    column, dataLine, SampleName=SampleName, dfName = dfName, IDs=IDs[j])
            if dataMarker is not None:
                datamarkervalues = WorkSample.makeDF(
                    column, dataMarker, SampleName=SampleName, dfName = dfName, IDs=IDs[j])
                if 'Merger' in dataMarker:
                    datamarkerTotvalues = WorkSample.makeDF(
                        column, 'NumMergersTotal', SampleName=SampleName, dfName = dfName, IDs=IDs[j])
                    dataMarkervalues = WorkSample.makeDF(
                        column, 'NumMajorMergersTotal', SampleName=SampleName, dfName = dfName, IDs=IDs[j])
                    datamarkervalues = WorkSample.makeDF(
                        column, 'NumMinorMergersTotal', SampleName=SampleName, dfName = dfName, IDs=IDs[j])

            for l, IDvalue in enumerate(IDs[j]):
                values = np.array(
                    [value for value in data[str(IDvalue)].values])
                if len(values.shape) > 1:
                    values = values.T[0]

                if Type == 'Evolution':
                    axs[i][j].plot(time, values, color=colors.get(
                        str(l), 'black'),  ls=lines[column], lw=0.4)

                    if dataLine is not None:
                        linevalues = np.array(
                            [value for value in datalinevalues[str(IDvalue)].values])
                        if len(linevalues.shape) > 1:
                            linevalues = linevalues.T[0]
                            linevalues = np.array(
                                [value for value in linevalues])
                        axs[i][j].plot(time[(~np.isinf(linevalues)) & (~np.isnan(linevalues))], values[(~np.isinf(
                            linevalues)) & (~np.isnan(linevalues))], color=colors.get(str(l), 'black'), ls=lines[column], lw=1.8)

                    if dataMarker is not None:
                        markervalues = np.array(
                            [value for value in datamarkervalues[str(IDvalue)].values])
                        if len(markervalues.shape) > 1:
                            markervalues = markervalues.T[0]
                            markervalues = np.array(
                                [value for value in markervalues])

                        if 'Merger' in dataMarker:
                            mergerTot = np.array(
                                [value for value in datamarkerTotvalues[str(IDvalue)].values])
                            if len(mergerTot.shape) > 1:
                                mergerTot = mergerTot.T[0]
                                mergerTot = np.array(
                                    [value for value in mergerTot])

                            MarkerTotvalues = np.array(
                                [value for value in datamarkerTotvalues[str(IDvalue)].values])
                            if len(MarkerTotvalues.shape) > 1:
                                MarkerTotvalues = MarkerTotvalues.T[0]
                                MarkerTotvalues = np.array(
                                    [value for value in MarkerTotvalues])

                            mergernumber = np.array(
                                [value for value in datamarkervalues[str(IDvalue)].values])
                            if len(mergernumber.shape) > 1:
                                mergernumber = mergernumber.T[0]
                                mergernumber = np.array(
                                    [value for value in mergernumber])

                            Mergernumber = np.array(
                                [value for value in dataMarkervalues[str(IDvalue)].values])
                            if len(Mergernumber.shape) > 1:
                                Mergernumber = Mergernumber.T[0]
                                Mergernumber = np.array(
                                    [value for value in Mergernumber])

                            Markervalues = np.array(
                                [value for value in datamarkervalues[str(IDvalue)].values])
                            if len(Markervalues.shape) > 1:
                                Markervalues = Markervalues.T[0]
                                Markervalues = np.array(
                                    [value for value in Markervalues])

                            mergernumber = np.flip(mergernumber)
                            Mergernumber = np.flip(Mergernumber)
                            mergerTot = np.flip(mergerTot)

                            for nmergerindex, nmerger in enumerate(mergernumber):

                                if nmergerindex == 0:
                                    markervalues[nmergerindex] = 0
                                    continue
                                else:
                                    if np.isnan(nmerger):
                                        markervalues[nmergerindex] = 0
                                    else:
                                        if np.isnan(mergernumber[nmergerindex - 1]):
                                            markervalues[nmergerindex] = int(
                                                nmerger)
                                        else:
                                            markervalues[nmergerindex] = int(
                                                nmerger) - int(mergernumber[nmergerindex - 1])

                            for nmergerindex, nmerger in enumerate(Mergernumber):

                                if nmergerindex == 0:
                                    Markervalues[nmergerindex] = 0
                                    continue
                                else:
                                    if np.isnan(nmerger):
                                        Markervalues[nmergerindex] = 0
                                    else:
                                        if np.isnan(mergernumber[nmergerindex - 1]):
                                            Markervalues[nmergerindex] = int(
                                                nmerger)
                                        else:
                                            Markervalues[nmergerindex] = int(
                                                nmerger) - int(Mergernumber[nmergerindex - 1])

                            for nmergerindex, nmerger in enumerate(mergerTot):

                                if nmergerindex == 0:
                                    MarkerTotvalues[nmergerindex] = 0
                                    continue
                                else:
                                    if np.isnan(nmerger):
                                        MarkerTotvalues[nmergerindex] = 0
                                    else:
                                        if np.isnan(mergernumber[nmergerindex - 1]):
                                            MarkerTotvalues[nmergerindex] = int(
                                                nmerger)
                                        else:
                                            MarkerTotvalues[nmergerindex] = int(
                                                nmerger) - int(mergerTot[nmergerindex - 1])
                            MarkerTotvalues = MarkerTotvalues - Markervalues - markervalues
                            Markervalues = np.flip(Markervalues)
                            markervalues = np.flip(markervalues)
                            MarkerTotvalues = np.flip(MarkerTotvalues)

                        axs[i][j].scatter(time[(Markervalues > 0)], values[(Markervalues > 0)], color=colors.get(
                            str(l), 'black'), lw=1., marker='o',  edgecolors='black', s=90, alpha=0.5)
                        axs[i][j].scatter(time[(markervalues > 0)], values[(markervalues > 0)], color=colors.get(
                            str(l), 'black'), lw=1., marker='s',  edgecolors='black', s=45, alpha=0.5)
                        axs[i][j].scatter(time[(MarkerTotvalues > 0)], values[(MarkerTotvalues > 0)], color=colors.get(
                            str(l), 'black'), lw=1., marker='p',  edgecolors='black', s=30, alpha=0.5)

                elif Type == 'CoEvolution':
                    x = dataX[str(IDvalue)].values
                    if len(x.shape) > 1:
                        x = np.array([value for value in x.T[0]])
                    else:
                        x = np.array([value for value in x])
                    colorSnap = np.array(
                        ['magenta', 'blue', 'cyan', 'lime', 'darkorange', 'red'])
                    axs[i][j].scatter(x[99-snapsTime], values[99-snapsTime], color=colorSnap,
                                      lw=1., marker='d',  edgecolors=colors.get(column), s=40, alpha=0.9)
                    axs[i][j].scatter(x[0], values[0], color='black', lw=1.,
                                      marker='o',  edgecolors=colors.get(column), s=40, alpha=0.9)
                    axs[i][j].plot(x, values, color=colors.get(
                        str(l), 'black'), ls=lines.get(column, 'dashed'))

                    if dataLine is not None:
                        linevalues = np.array(
                            [value for value in datalinevalues[str(IDvalue)].values])
                        if len(linevalues.shape) > 1:
                            linevalues = linevalues.T[0]
                            linevalues = np.array(
                                [value for value in linevalues])
                        axs[i][j].plot(x[(~np.isinf(linevalues)) & (~np.isnan(linevalues))], values[(~np.isinf(linevalues)) & (
                            ~np.isnan(linevalues))], color=colors.get(str(l), 'black'), ls=lines[column], lw=3.)

                    if dataMarker is not None:
                        markervalues = np.array(
                            [value for value in datamarkervalues[str(IDvalue)].values])
                        if len(markervalues.shape) > 1:
                            markervalues = markervalues.T[0]
                            markervalues = np.array(
                                [value for value in markervalues])

                        if 'Merger' in dataMarker:
                            mergerTot = np.array(
                                [value for value in datamarkerTotvalues[str(IDvalue)].values])
                            if len(mergerTot.shape) > 1:
                                mergerTot = mergerTot.T[0]
                                mergerTot = np.array(
                                    [value for value in mergerTot])

                            MarkerTotvalues = np.array(
                                [value for value in datamarkerTotvalues[str(IDvalue)].values])
                            if len(MarkerTotvalues.shape) > 1:
                                MarkerTotvalues = MarkerTotvalues.T[0]
                                MarkerTotvalues = np.array(
                                    [value for value in MarkerTotvalues])

                            mergernumber = np.array(
                                [value for value in datamarkervalues[str(IDvalue)].values])
                            if len(mergernumber.shape) > 1:
                                mergernumber = mergernumber.T[0]
                                mergernumber = np.array(
                                    [value for value in mergernumber])

                            Mergernumber = np.array(
                                [value for value in dataMarkervalues[str(IDvalue)].values])
                            if len(Mergernumber.shape) > 1:
                                Mergernumber = Mergernumber.T[0]
                                Mergernumber = np.array(
                                    [value for value in Mergernumber])

                            Markervalues = np.array(
                                [value for value in datamarkervalues[str(IDvalue)].values])
                            if len(Markervalues.shape) > 1:
                                Markervalues = Markervalues.T[0]
                                Markervalues = np.array(
                                    [value for value in Markervalues])

                            mergernumber = np.flip(mergernumber)
                            Mergernumber = np.flip(Mergernumber)
                            mergerTot = np.flip(mergerTot)

                            for nmergerindex, nmerger in enumerate(mergernumber):

                                if nmergerindex == 0:
                                    markervalues[nmergerindex] = 0
                                    continue
                                else:
                                    if np.isnan(nmerger):
                                        markervalues[nmergerindex] = 0
                                    else:
                                        if np.isnan(mergernumber[nmergerindex - 1]):
                                            markervalues[nmergerindex] = int(
                                                nmerger)
                                        else:
                                            markervalues[nmergerindex] = int(
                                                nmerger) - int(mergernumber[nmergerindex - 1])

                            for nmergerindex, nmerger in enumerate(Mergernumber):

                                if nmergerindex == 0:
                                    Markervalues[nmergerindex] = 0
                                    continue
                                else:
                                    if np.isnan(nmerger):
                                        Markervalues[nmergerindex] = 0
                                    else:
                                        if np.isnan(mergernumber[nmergerindex - 1]):
                                            Markervalues[nmergerindex] = int(
                                                nmerger)
                                        else:
                                            Markervalues[nmergerindex] = int(
                                                nmerger) - int(Mergernumber[nmergerindex - 1])

                            for nmergerindex, nmerger in enumerate(mergerTot):

                                if nmergerindex == 0:
                                    MarkerTotvalues[nmergerindex] = 0
                                    continue
                                else:
                                    if np.isnan(nmerger):
                                        MarkerTotvalues[nmergerindex] = 0
                                    else:
                                        if np.isnan(mergernumber[nmergerindex - 1]):
                                            MarkerTotvalues[nmergerindex] = int(
                                                nmerger)
                                        else:
                                            MarkerTotvalues[nmergerindex] = int(
                                                nmerger) - int(mergerTot[nmergerindex - 1])
                            MarkerTotvalues = MarkerTotvalues - Markervalues - markervalues
                            Markervalues = np.flip(Markervalues)
                            markervalues = np.flip(markervalues)
                            MarkerTotvalues = np.flip(MarkerTotvalues)

                        axs[i][j].scatter(x[(Markervalues > 0)], values[(Markervalues > 0)], color=colors.get(
                            str(l), 'black'), lw=1., marker='o',  edgecolors='black', s=75, alpha=0.5)
                        axs[i][j].scatter(x[(markervalues > 0)], values[(markervalues > 0)], color=colors.get(
                            str(l), 'black'), lw=1., marker='o',  edgecolors='black', s=30, alpha=0.5)
                        axs[i][j].scatter(x[(MarkerTotvalues > 0)], values[(MarkerTotvalues > 0)], color=colors.get(
                            str(l), 'black'), lw=1., marker='o',  edgecolors='black', s=15, alpha=0.5)

            # Plot details

            axs[i][j].grid(True, color="grey",  which="major", linestyle="-.")
            axs[i][j].tick_params(axis='y', labelsize=0.85*fontlabel)
            axs[i][j].tick_params(axis='x', labelsize=0.85*fontlabel)
	    
            if limin.get(row) != None:
                axs[i][j].set_ylim(limin.get(row), limmax.get(row))
            if scales.get(row) != None:
                axs[i][j].set_yscale(scales.get(row))
            if scales.get(row) == 'log':
                axs[i][j].yaxis.set_major_formatter(
                    FuncFormatter(format_func_loglog))

            if j == 0 and i == 0:
                if legend:
                    if len(LegendNames) >= 1:
                        custom_lines, label, ncol, mult = Legend(
                            LegendNames[0])
                        axs[i][j].legend(
                            custom_lines, label, ncol=ncol, loc=loc, fontsize=mult*fontlegend)

            if j == 1 and i == 0:
                if legend:
                    if len(LegendNames) > 1:
                        custom_lines, label, ncol, mult = Legend(
                            LegendNames[1])
                        axs[i][j].legend(
                            custom_lines, label, ncol=ncol, loc=loc, fontsize=mult*fontlegend)

            if j == 0:

                if xlabelintext:
                    axs[i][j].set_ylabel(
                        labelsequal.get(row), fontsize=fontlabel)

                else:
                    axs[i][j].set_ylabel(labels.get(row), fontsize=fontlabel)

            if j == len(columns) - 1:
                if xlabelintext and not limaxis and len(rows) > 1:
                    Afont = {'color':  'black',
                             'size': fontlabel,
                             }
                    anchored_text = AnchoredText(
                        texts.get(row), loc='upper right', prop=Afont)
                    axs[i][j].add_artist(anchored_text)

            if xlabelintext and limaxis and len(rows) > 1:
                Afont = {'color':  'black',
                         'size': fontlabel,
                         }
                anchored_text = AnchoredText(
                    texts[row], loc='upper left', prop=Afont)
                axs[i][j].add_artist(anchored_text)

            if i == 0:

                if title:
                    axs[i][j].set_title(titles.get(
                        title[j], title[j]), fontsize=fontTitle)

                if Type == 'Evolution':

                    axs[i][j].tick_params(bottom=True, top=False)
                    ax2label = axs[i][j].secondary_xaxis('top')

                    zticks = np.array([0., 0.2, 0.5, 1., 2., 5., 20.])
                    zlabels = np.array(
                        ['0', '0.2', '0.5', '1', '2', '5', '20'])
                    zticks_Age = np.array(
                        [13.803, 11.323, 8.587, 5.878, 3.285, 1.2, 0])

                    zticks = zticks.tolist()
                    zticks_Age = zticks_Age.tolist()

                    x_locator = FixedLocator(zticks_Age)
                    x_formatter = FixedFormatter(zlabels)
                    ax2label.xaxis.set_major_locator(x_locator)
                    ax2label.xaxis.set_major_formatter(x_formatter)
                    ax2label.set_xlabel(r"z")
                    ax2label.tick_params(labelsize=0.85*fontlabel)

            if i == len(rows) - 1:
                if Type == 'Evolution':

                    axs[i][j].set_xlabel(r't [Gyr]', fontsize=fontlabel)
                    axs[i][j].set_xticks([0, 2, 4, 6, 8, 10, 12, 14])
                    axs[i][j].set_xticklabels(
                        ['', '2', '4', '6', '8', '10', '12', '14'])

                elif Type == 'CoEvolution':
                    axs[i][j].set_xscale(scales.get(row, 'linear'))
                    if scales.get(row) == 'log':
                        axs[i][j].xaxis.set_major_formatter(
                            FuncFormatter(format_func_loglog))
                    axs[i][j].set_xlabel(labels.get(
                        row, 'None'), fontsize=fontlabel)

    savefig(savepath, savefigname)

    return


####################################################################################################
####################################################################################################

def PlotScatter(names, columns, ParamX, ParamsY,  Type='z0', snap=[99], title=False, medianBins=False, medianAll=False, xlabelintext=False, All=None,
                alphaScater=0.5, alphaAll=0.2,  alphaShade=0.3,  linewidth=1.2, fontTitle=28, fontlabel=26,  fontlegend=20, nboots=100, Nlim=100,  medianSample=False, MarkerSizes = None,
                m='o', markersize=30, quantile=0.95, legend=False, LegendNames=None, framealpha = 0.85, q= 0.95,
                savepath='fig/PlotScatter', savefigname='fig', dfName='Sample', SampleName='Name', 
                columnspacing = 0.5, handletextpad = -0.5, labelspacing = 0.3,
                bins=10, seed=16010504):
    '''
    Plot evolution for random sample
    Parameters
    ----------
    columns : specific set in the sample / or different param to plot in each column. array with str
    ParamX : specific the x param. str
    ParamsY : specific the y param. array with str
    medianBins : to plot the median of the x - y relation. Default: False. bool
    medianSample : to plot the median of all subhalos excepet 'Compacts'. Default: False. bool
    medianAll : to plot the median for each name in names of the x - y relation. Default: False. bool
    All : to plot all subhalos. Default: None. None or str name for dataframe with all subhalos. 
    alphaScater : alpha for scatter plot. Default: 0.5. float
    alphaAll : alpha for scatter plot of all subhalos. Default: 0.2. float
    m : marker. Default: 'o'. str
    markersize : marker size. Default: 30. float
    quantile : median quantile to plot. Default: 0.95. float
    The rest is the same as the previous functions
    Returns
    -------
    Requested Evolution or Co-Evolution plot
    -------
    Author: Abhner P. de Almeida (abhner.almeida AAT usp.br)
    '''

    np.random.seed(seed)

    # Verify NameParameters
    if type(columns) is not list and type(columns) is not np.ndarray:
        columns = [columns]

    if type(ParamsY) is not list and type(ParamsY) is not np.ndarray:
        ParamsY = [ParamsY]

    if type(ParamX) is not list and type(ParamX) is not np.ndarray:
        ParamsX = [ParamX]

    while len(ParamsX) != len(ParamsY):
        ParamsX.append(ParamX)
    if columns == ['Snap']:
        columns = snap
        dataX = makedata(names, columns, ParamsX, 'Snap',
                         snap=snap, SampleName=SampleName, dfName = dfName)
        dataY = makedata(names, columns, ParamsY, 'Snap',
                         snap=snap, SampleName=SampleName, dfName = dfName)

    else:
        dataX = makedata(names, columns, ParamsX, Type,
                         snap=snap, SampleName=SampleName, dfName = dfName)
        dataY = makedata(names, columns, ParamsY, Type,
                         snap=snap, SampleName=SampleName, dfName = dfName)
        if MarkerSizes != None:
            dataMarker = makedata(names, columns, MarkerSizes, Type,
                             snap=snap, SampleName=SampleName, dfName = dfName)


    dfTime = WorkSample.extractDF('SNAPS_TIME')
    All = WorkSample.extractDF('Samples/All')
    All = All.loc[(All.SubhaloMassInRadType4.between(7.5, 10))]


    # Define axes
    if len(snap) > 1:
        columns = np.full(len(snap), 'Snap')
    plt.rcParams.update({'figure.figsize': (6*len(columns), 6*len(ParamsY))})
    fig = plt.figure()
    gs = fig.add_gridspec(len(ParamsY), len(columns), hspace=0, wspace=0)
    axs = gs.subplots(sharex='col', sharey='row')

    # Verify axs shape
    if type(axs) is not list and type(axs) is not np.ndarray:
        axs = [axs]
    if type(axs[0]) is not np.ndarray:
        axs = np.array([axs])
        if len(columns) == 1:
            axs = axs.T

    for i, param in enumerate(ParamsY):

        for j, titlename in enumerate(columns):

            if All is not None:
                xAll = All[ParamX]
                yAll = All[param]

                axs[i][j].scatter(xAll, yAll, color=colors['All'],
                                  edgecolor=colors['All'], alpha=1., marker='.', s=10)

            for l, values in enumerate(dataY[i][j]):
                                

                if MarkerSizes != None:
                    Markers = dataMarker[0][0][l]
                    
                    axs[i][j].scatter(dataX[i][j][l][Markers <= 1], values[Markers <= 1], color=colors.get(
                            names[l] + 'ScatterPlot'), edgecolor='black', alpha=alphaScater, lw = linesthicker.get(names[l] + 'ScatterPlot'), 
                            marker=markers.get(names[l]), s=20)
                    axs[i][j].scatter(dataX[i][j][l][Markers == 2], values[Markers == 2], color=colors.get(
                            names[l] + 'ScatterPlot'), edgecolor='black', alpha=alphaScater, lw = linesthicker.get(names[l] + 'ScatterPlot'), 
                            marker=markers.get(names[l]), s=45)
                    axs[i][j].scatter(dataX[i][j][l][Markers >= 3], values[Markers >= 3], color=colors.get(
                            names[l] + 'ScatterPlot'), edgecolor='black', alpha=alphaScater, lw = linesthicker.get(names[l] + 'ScatterPlot'), 
                            marker=markers.get(names[l]), s=120)


                else:
                    axs[i][j].scatter(dataX[i][j][l], values, color=colors.get(
                            names[l] + 'ScatterPlot'), edgecolor='black', alpha=alphaScater, lw = linesthicker.get(names[l] + 'ScatterPlot'), marker=markers.get(names[l]), s=5*msize.get(names[l]+'ScatterPlot'))

                if medianBins:
                    xmeanfinal, ymedian, yquantile95, yquantile5 = MATH.split_quantiles(
                        dataX[i][j][l], values, total_bins=bins, quantile=quantile)

                    axs[i][j].errorbar(xmeanfinal, ymedian, yerr=(ymedian - yquantile5, yquantile95 - ymedian),
                                       ls='None', markeredgecolor='black', elinewidth=2, ms=10, fmt='s', c=colors[names[l]])

                elif medianAll:
                    xmeanfinal, ymedian, yquantile95, yquantile5 = MATH.split_quantiles(
                        dataX[i][j][l], values, total_bins=bins)
                    axs[i][j].plot(xmeanfinal, ymedian, color=colors.get(
                        names[l]), ls=lines.get(names[l]), linewidth=linewidth)
                    axs[i][j].fill_between(xmeanfinal, yquantile5,  yquantile95, color=colors.get(
                        names[l]), alpha=alphaShade)

            if medianSample:
                
                dfSample = WorkSample.extractDF('Sample')
                SampleControl = dfSample.loc[(dfSample.SubhaloMassInRadType4.between(8.4, 9.2)) & (dfSample.SubhaloHalfmassRadType4 >= -0.35) & (dfSample.Flags == 1)]
                sizesControl = np.array([10**values for values in SampleControl.SubhaloHalfmassRadType4.values])
                MassesControl = np.array([10**values for values in SampleControl.SubhaloMassInRadType4.values])


                xmeanfinal, ymedian, yquantile95, yquantile5 = MATH.split_quantiles(
                    MassesControl ,sizesControl, total_bins=20, quantile=q) 
                 
                xmeanfinal = np.log10(xmeanfinal)
                ymedian = np.log10(ymedian)
                yquantile95 = np.log10(yquantile95)
                yquantile5 = np.log10(yquantile5)
                
                axs[i][j].plot(xmeanfinal, ymedian, color='tab:red',
                               ls=lines.get(names[l]), linewidth=linewidth)
                axs[i][j].fill_between(
                    xmeanfinal, yquantile5,  yquantile95, color='tab:red', alpha=alphaShade)

            # Plot details

            axs[i][j].grid(True, color="grey",  which="major", linestyle="-.")
            if param == 'Group_M_Crit200':
                if 'Central' in titlename:
                    axs[i][j].set_ylim(limin.get(param+ 'Central'), limmax.get(param+ 'Central'))
            else:
                    axs[i][j].set_ylim(limin.get(param+ 'Scatter'), limmax.get(param+ 'Scatter'))

            axs[i][j].set_yscale(scales.get(param, 'linear'))

            if scales.get(param) == 'log':
                axs[i][j].yaxis.set_major_formatter(
                    FuncFormatter(format_func_loglog))

            if  ParamX == 'SubhaloMassInRadType4':
                axs[i][j].axvline(
                    8.4, ls='--', color='tab:red', linewidth=linewidth)
                axs[i][j].axvline(
                    9.2, ls='--', color='tab:red', linewidth=linewidth)

            if j == 0 and i == 0:
                if legend:
                    if len(LegendNames) >= 1:
                        custom_lines, label, ncol, mult = Legend(
                            LegendNames[0])
                        axs[i][j].legend(
                            custom_lines, label, ncol=ncol, fontsize=mult*fontlegend, framealpha=framealpha, 
                            columnspacing = columnspacing, handletextpad = handletextpad, labelspacing = labelspacing)
                        
            if j == 1 and i == 0:
                if legend:
                    if len(LegendNames) > 1:
                        custom_lines, label, ncol, mult = Legend(
                            LegendNames[1])
                        axs[i][j].legend(
                            custom_lines, label, ncol=ncol, fontsize=mult*fontlegend, 
                            columnspacing = columnspacing, handletextpad = handletextpad, labelspacing = labelspacing)


            if j == 0:
                axs[i][j].set_ylabel(labels.get(param), fontsize=fontlabel)
                axs[i][j].tick_params(axis='y', labelsize=0.85*fontlabel)

            if j == len(columns) - 1:
                if xlabelintext:
                    Afont = {'color':  'black',
                             'size': fontlabel,
                             }
                    anchored_text = AnchoredText(
                        texts.get(param), loc='upper right', prop=Afont)
                    axs[i][j].add_artist(anchored_text)

            if i == 0:
                if columns[j] == 'Snap':
                    axs[i][j].set_title(
                        r'$z = %.1f$' % dfTime.z.loc[dfTime.Snap == snap[j]].values[0], fontsize=fontTitle)
                if title:
                    axs[i][j].set_title(titles.get(
                        title[j], title[j]), fontsize=fontTitle)

            if i == len(ParamsY) - 1:
                axs[i][j].set_xlabel(labels.get(ParamX), fontsize=fontlabel)

                axs[i][j].set_xscale(scales.get(ParamX, 'linear'))
                if scales.get(ParamX) == 'log':
                    axs[i][j].xaxis.set_major_formatter(
                        FuncFormatter(format_func_loglog))
                if param == 'Group_M_Crit200':
                    if 'Central' in titlename:
                        axs[i][j].set_xlim(limin.get(ParamX+ 'Central'), limmax.get(ParamX+ 'Central'))
                elif ParamX == 'SubhaloMassInRadType4':
                        axs[i][j].set_xlim(8.2, 9.4)
                else:
                       
                    axs[i][j].set_xlim(limin.get(ParamX+ 'Scatter'), limmax.get(ParamX+ 'Scatter'))
                axs[i][j].tick_params(axis='x', labelsize=0.85*fontlabel)

    savefig(savepath, savefigname)

    return

####################################################################################################
####################################################################################################

def PlotProfile(IDs, names, columns, rows, PartTypes,  ParamX='rad', Condition='All', cumulative=False, title=False, xlabelintext=False, 
                framealpha = 0.95, linewidth=1.2, fontTitle=28, fontlabel=24,  fontlegend=18, nboots=100, Nlim=100,  
                quantile=0.95, rmaxlim=120, norm=False, legend=False, LegendNames=None, line=False, 
                columnspacing = 0.7, handletextpad = 0.4, labelspacing = 0.3, handlelength = 1.0,
                savepath='fig/PlotProfile', savefigname='fig', dfName='Sample', SampleName='Samples',  loc='best',
                nbins=25, seed=16010504, TRANSPARENT = False):
    '''
    Plot radial profiles
    Parameters
    ----------
    IDs: IDs for selected subhalos. 
    names : sample names. array with str
    rows : specific set in the sample / or different param to plot in each row. array with str
    PartTpes : particle types for each param. array with str
    ParamX : specific the x param. Default: 'rad'. str
    Condition : specific the particle condition. Default: 'Original'. str
    cumulative : make cumulative plot. Default: False. bool
    rmaxlim : max rad. Default: 120. float
    norm : to normalized with the max value. Default: False. bool
    line: line for half rads. Default: False. bool
    The rest is the same as the previous functions
    Returns
    -------
    Requested Evolution or Co-Evolution plot''
    -------
    Author: Abhner P. de Almeida (abhner.almeida AAT usp.br)
    '''

    np.random.seed(seed)

    # Verify NameParameters
    if type(columns) is not list and type(columns) is not np.ndarray:
        columns = [columns]

    if type(rows) is not list and type(rows) is not np.ndarray:
        rows = [rows]

    dfTime = WorkSample.extractDF('SNAPS_TIME')

    # Define axes
    plt.rcParams.update({'figure.figsize': (6*len(columns), 6*len(rows))})
    fig = plt.figure()
    gs = fig.add_gridspec(len(rows), len(columns), hspace=0, wspace=0)
    axs = gs.subplots(sharex='col', sharey='row')

    # Verify axs shape
    if type(axs) is not list and type(axs) is not np.ndarray:
        axs = [axs]
    if type(axs[0]) is not np.ndarray:
        axs = np.array([axs])
        if len(columns) == 1:
            axs = axs.T

    dFHalfStar = WorkSample.extractDF(
        'SubhaloHalfmassRadType4', PATH='./data/Evolution/')
    dFHalfGasRad = WorkSample.extractDF(
        'SubhaloHalfmassRadType0', PATH='./data/Evolution/')
    dfGasMass = WorkSample.extractDF(
        'SubhaloMassType0', PATH='./data/Evolution/')

    for i, row in enumerate(rows):
        if PartTypes[i] == 'PartType4':
            dFHalfRad = WorkSample.extractDF(
                'SubhaloHalfmassRadType4', PATH='./data/Evolution/')
        if PartTypes[i] == 'PartType0' or PartTypes[i] == 'gas':
            dFHalfRad = WorkSample.extractDF(
                'SubhaloHalfmassRadType0', PATH='./data/Evolution/')
        if PartTypes[i] == 'PartType1' or PartTypes[i] == 'DM':
            dFHalfRad = WorkSample.extractDF(
                'SubhaloHalfmassRadType1', PATH='./data/Evolution/')
            
        if 'Galactic' in Condition:
            dFHalfRadGas = WorkSample.extractDF(
                'SubhaloHalfmassRadType0', PATH='./data/Evolution/')

        for l, ID in enumerate(IDs):
            Rads = np.array([])
           

            for j, snap in enumerate(columns):

                RadStars = np.array([])
                RadGas = np.array([])
                GasMass = np.array([])

                for idValue in ID:
                    HalfRad = dFHalfStar[str(
                        idValue)].loc[dFHalfStar.Snap == snap].values[0]
                    HalfGasRad = dFHalfGasRad[str(
                        idValue)].loc[dFHalfGasRad.Snap == snap].values[0]
                    GasMassType = dfGasMass[str(
                        idValue)].loc[dfGasMass.Snap == snap].values[0]
                    RadStars = np.append(RadStars, HalfRad)
                    RadGas = np.append(RadGas, HalfGasRad)
                    GasMass = np.append(GasMass, GasMassType)

                Rads = np.array([])

                for idValue in ID:
                    if 'Galactic' in Condition :
                        HalfRadGas = dFHalfRadGas[str(idValue)].loc[dFHalfRadGas.Snap == snap].values[0]
                        Rads = np.append(Rads, 10**HalfRadGas)
                        
                    else:
                        HalfRad = dFHalfRad[str(
                            idValue)].loc[dFHalfRad.Snap == snap].values[0]
                        Rads = np.append(Rads, 10**HalfRad)

                if PartTypes[i] == 'PartType4':
                    rmin = np.median(Rads)/5
                    rmax = np.median(Rads)*150

                if (PartTypes[i] == 'PartType0' or PartTypes[i] == 'gas') :
                    rmin = np.median(Rads)/300
                    rmax = np.median(Rads)*7

                if (PartTypes[i] == 'PartType1' or PartTypes[i] == 'DM') :
                    rmin = np.median(Rads)/300
                    rmax = np.median(Rads)*7
                    
                if rmax > rmaxlim or rmax == 0.:
                    rmax = rmaxlim

                if PartTypes[i] == 'PartType0' and rmin < 0.07:
                    rmin = 0.07

                if PartTypes[i] == 'PartType0' and rmin < 0.3:
                    rmin = 0.3

                if PartTypes[i] == 'PartType4' and rmin < 0.1:
                    rmin = 0.1
                    
                
                if np.median(Rads) == 0:
                    continue
                    
                try:
                    if row != 'sSFR' and row != 'joverR' and row != 'DensityGasOverR2' and row != 'DensityStarOverR2' and row != 'GFM_Metallicity_Zodot':

                        if 'Central' in names[l]:
                            df = pd.read_pickle('./data/' + 'Profiles/'+ Condition + '/'+ row+'/'+PartTypes[i]+'/'+str(columns[j])+'/'+names[l] + Condition +'.pkl')
                        elif 'Satellite' in names[l]:
                            df = pd.read_pickle('./data/' + 'Profiles/'+ Condition + '/'+row+'/'+PartTypes[i]+'/'+str(columns[j])+'/'+names[l] + Condition  +'.pkl')
                        
                        else:
                            df = pd.read_pickle('./data/' + 'Profiles/'+ Condition + '/'+row+'/'+PartTypes[i]+'/'+str(columns[j])+'/'+names[l] + Condition  +'.pkl')
                        
                        rad = df.Rads.values
                        ymedian = df.ymedians.values
                        yerr = df.yerrs.values
                    
                    elif row == 'sSFR':

                            
                        df = pd.read_pickle('./data/' + 'Profiles/'+ Condition + '/'+'SFR'+'/'+PartTypes[i]+'/'+str(columns[j])+'/'+names[l]+ Condition + '.pkl')
                        radSFR = df.Rads.values
                        ymedianSFR = df.ymedians.values
                        yerrSFR = df.yerrs.values

                        df = pd.read_pickle('./data/' + 'Profiles/'+ Condition + '/'+'Mstellar'+'/PartType4/'
                                            +str(columns[j])+'/'+names[l]+ Condition + '.pkl')
                        radMstellar = df.Rads.values
                        ymedianMstellar = df.ymedians.values

                        new_y = sp.interpolate.interp1d(
                            radMstellar, ymedianMstellar, kind='cubic', fill_value='extrapolate')(radSFR)
                        rad = radSFR
                        ymedian = ymedianSFR / new_y
                        yerr = yerrSFR / new_y
                        
                    elif row == 'GFM_Metallicity_Zodot':

                            
                        df = pd.read_pickle('./data/' + 'Profiles/'+ Condition + '/'+'GFM_Metallicity'+'/'+PartTypes[i]+'/'+str(columns[j])+'/'+names[l]+ Condition + '.pkl')
                        rad = df.Rads.values
                        ymedian = df.ymedians.values / 0.0127
                        yerr = df.yerrs.values  / 0.0127

                      

                    elif row == 'joverR':
                        
                            
                        df = pd.read_pickle('./data/' + 'Profiles/'+ Condition + '/'+'j'+
                                            '/'+PartTypes[i]+'/'+str(columns[j])+'/'+names[l]+ Condition + '.pkl')
                        rad = df.Rads.values
                        ymedian = df.ymedians.values
                        yerr = df.yerrs.values

                        rad = rad
                        ymedian = ymedian / rad
                        yerr = yerr / rad

                    elif row == 'DensityGasOverR2':
                        
         
                            
                        df = pd.read_pickle('./data/' + 'Profiles/'+ Condition + '/'+'DensityGas'+
                                            '/'+PartTypes[i]+'/'+str(columns[j])+'/'+names[l]+ Condition + '.pkl')
                        rad = df.Rads.values
                        ymedian = df.ymedians.values
                        yerr = df.yerrs.values

                        rad = rad
                        ymedian = ymedian * rad**2
                        yerr = yerr * rad**2
                        
                    elif row == 'DensityStarOverR2':
         
                            
                        df = pd.read_pickle('./data/' + 'Profiles/'+ Condition + '/'+'DensityStar'+
                                            '/'+PartTypes[i]+'/'+str(columns[j])+'/'+names[l]+ Condition + '.pkl')
                        rad = df.Rads.values
                        ymedian = df.ymedians.values
                        yerr = df.yerrs.values

                        rad = rad
                        ymedian = ymedian * rad**2
                        yerr = yerr * rad**2
                except:
                    continue
                
                    
                argnan = np.argwhere(~np.isnan(rad)).T[0]
                rad = rad[argnan]
                ymedian = ymedian[argnan]
                yerr = yerr[argnan]

                argnan = np.argwhere(~np.isnan(ymedian)).T[0]
                rad = rad[argnan]
                ymedian = ymedian[argnan]
                yerr = yerr[argnan]

                argnan = np.argwhere(~np.isnan(yerr)).T[0]
                rad = rad[argnan]
                ymedian = ymedian[argnan]
                yerr = yerr[argnan]
                
                
                if cumulative and row in ['Mstellar', 'Mgas']:
                    ymedian = np.cumsum(ymedian)
                    if len(ymedian) == 0:
                        yerr = np.ones(len(rad)) * np.nan
                        ymedian = np.ones(len(rad)) * np.nan
                    else:
                        if norm:

                            yerr = yerr / np.max(ymedian)
                            ymedian = ymedian / np.max(ymedian)

                if len(rad) <= 2:
                    axs[i][j].plot(rad, ymedian, color=colors.get(names[l] ), ls=lines.get(
                        names[l] ), lw= 2.5*linesthicker.get(names[l], linewidth), dash_capstyle = capstyles.get(names[l], 'projecting'))
                   
                else:
                    
                    Func = interp1d(rad, ymedian, kind='linear', fill_value = 'extrapolate')
                    x = np.geomspace(0.1, 70, 100)
    
                    axs[i][j].plot(x, Func(x), color=colors.get(names[l] ), ls=lines.get(
                        names[l] ), lw= 2.5*linesthicker.get(names[l], linewidth), dash_capstyle = capstyles.get(names[l], 'projecting'))
                   
         
                    if line and i == len(rows) - 1:
                        axs[i][j].arrow(2*10**np.nanmedian(RadStars), 1e2, 0, -1e2, color=colors.get(
                            names[l] , 'black'), ls=lines.get(names[l] + 'Profile', 'solid'), linewidth= 3 * linewidth, head_width=0.)
                        
                        #axs[i][j].axvline(,
                                          #ls='--', color=colors.get(names[l]))
                        #axs[i][j].axvline(10**np.nanmedian(RadGas),
                                          #ls='--', color=colors.get(names[l]))

                # Plot details
                axs[i][j].grid(True, color="grey",
                               which="major")
                if cumulative and row in ['Mstellar', 'Mgas']:
                    if norm:
                        axs[i][j].set_ylim(
                            limin.get(row + 'Norm'), limmax.get(row + 'Norm'))
                    else:
                        axs[i][j].set_ylim(
                            limin.get(row + 'Cum'), limmax.get(row + 'Cum'))
                else:
                    axs[i][j].set_ylim(limin.get(row), limmax.get(row))
                axs[i][j].set_xlim(limin.get('rad'), limmax.get('rad'))
                
                if legend:
                    if len(LegendNames.shape) > 1:
                        if i == 0:
                            if legend:
                                if j <= len(LegendNames[0]) - 1:
                                    if LegendNames[0][j] == 'AnyNone':
                                        None
                                    else:
                                        custom_lines, label, ncol, mult = Legend(
                                            LegendNames[0][j], mult = 3)
                                        axs[i][j].legend(
                                            custom_lines, label, ncol=ncol, loc=loc, fontsize=mult*fontlegend, framealpha = framealpha, 
                                            columnspacing = columnspacing, handlelength = handlelength, handletextpad = handletextpad, labelspacing = labelspacing)
    
                        if i == 1:
                           if legend and  len(LegendNames) > 1:
                               if j <= len(LegendNames[1]) - 1:
                                   if LegendNames[1][j] == 'AnyNone':
                                       None
                                   else:
                                       
                                       custom_lines, label, ncol, mult = Legend(
                                           LegendNames[1][j], mult = 3)
                                       axs[i][j].legend(
                                           custom_lines, label, ncol=ncol, loc=loc, fontsize=mult*fontlegend, framealpha = framealpha, 
                                           columnspacing = columnspacing, handlelength = handlelength, handletextpad = handletextpad, labelspacing = labelspacing)
                                    
                                    
                    else:
    
                        if j == 0 and i == 0:
                            if legend:
                                if len(LegendNames) >= 1:
                                    custom_lines, label, ncol, mult = Legend(
                                        LegendNames[0])
                                    axs[i][j].legend(
                                        custom_lines, label, ncol=ncol, loc=loc, fontsize=mult*fontlegend, framealpha = framealpha, 
                                        columnspacing = columnspacing, handlelength = handlelength, handletextpad = handletextpad, labelspacing = labelspacing)
        
                        if j == 1 and i == 0:
                            if legend:
                                if len(LegendNames) > 1:
                                    custom_lines, label, ncol, mult = Legend(
                                        LegendNames[1])
                                    axs[i][j].legend(
                                        custom_lines, label, ncol=ncol, loc=loc, fontsize=mult*fontlegend, framealpha = framealpha, 
                                        columnspacing = columnspacing, handlelength = handlelength,  handletextpad = handletextpad, labelspacing = labelspacing)
        
                        if j == 2 and i == 0:
                            if legend:
                                if len(LegendNames) > 2:
                                    custom_lines, label, ncol, mult = Legend(
                                        LegendNames[2])
                                    axs[i][j].legend(
                                        custom_lines, label, ncol=ncol, loc=loc, fontsize=mult*fontlegend, framealpha = framealpha, 
                                        columnspacing = columnspacing, handletextpad = handletextpad, labelspacing = labelspacing)
        
                        if j == 3 and i == 0:
                            if legend:
                                if len(LegendNames) > 3:
                                    custom_lines, label, ncol, mult = Legend(
                                        LegendNames[3])
                                    axs[i][j].legend(
                                        custom_lines, label, ncol=ncol, loc=loc, fontsize=mult*fontlegend, framealpha = framealpha, 
                                        columnspacing = columnspacing, handletextpad = handletextpad, labelspacing = labelspacing)

                if j == 0:
                    if cumulative and row in ['Mstellar', 'Mgas']:
                        if norm:
                            axs[i][j].set_yscale(
                                scales.get(row + 'Norm', 'linear'))

                            if scales.get(row + 'Norm') == 'log':
                                axs[i][j].yaxis.set_major_formatter(
                                    FuncFormatter(format_func_loglog))
                            axs[i][j].set_ylabel(labels.get(
                                row + 'Norm'), fontsize=fontlabel)
                        else:

                            axs[i][j].set_yscale(
                                scales.get(row + 'Cum', 'linear'))
                            if scales.get(row + 'Cum') == 'log':
                                axs[i][j].yaxis.set_major_formatter(
                                    FuncFormatter(format_func_loglog))
                            axs[i][j].set_ylabel(labels.get(
                                row + 'Cum'), fontsize=fontlabel)

                    else:
                        axs[i][j].set_yscale(scales.get(row, 'linear'))
                        if scales.get(row) == 'log':
                            axs[i][j].yaxis.set_major_formatter(
                                FuncFormatter(format_func_loglog))
                        axs[i][j].set_ylabel(
                            labels.get(row), fontsize=fontlabel)

                if j == len(columns) - 1:
                    if xlabelintext:
                        Afont = {'color':  'black',
                                 'size': fontlabel,
                                 }
                        anchored_text = AnchoredText(
                            texts.get(row), loc='upper right', prop=Afont)
                        axs[i][j].add_artist(anchored_text)

                if i == 0:
                    axs[i][j].set_title(
                        r'$z = %.1f$' % dfTime.z.loc[dfTime.Snap == snap].values[0], fontsize=fontTitle)

                if i == len(rows) - 1:
                    axs[i][j].set_xscale(scales.get(ParamX, 'linear'))
                    if scales.get(ParamX) == 'log':
                        axs[i][j].xaxis.set_major_formatter(
                            FuncFormatter(format_func_loglog))
                    axs[i][j].set_xlabel(labels.get(
                        ParamX), fontsize=fontlabel)

                axs[i][j].tick_params(labelsize=0.85*fontlabel)

    savefig(savepath, savefigname)

    return

####################################################################################################
####################################################################################################

def savefig(savepath, savefigname):
    '''
    save figures
    Parameters
    ----------
    savepath : save path. 
    savefigname : fig name.
    -------
    Author: Abhner P. de Almeida (abhner.almeida AAT usp.br)
    '''

    try:
        plt.savefig('./'+savepath+'/'+savefigname +
                    '.pdf', bbox_inches='tight')

    except:
        path = './'
        directories = savepath.split('/')
        for name in directories:
            path = os.path.join(path, name)
            if not os.path.isdir(path):
                os.mkdir(path)

        plt.savefig('./'+savepath+'/'+savefigname +
                    '.pdf', bbox_inches='tight')

####################################################################################################
####################################################################################################

def Legend(names,  mult = 2):
    '''
    make the legend
    Parameters
    ----------
    names : name for the legend. 
    Returns
    -------
    lines, labels, number of columns and fontsize multiplicative factor for the legend
    -------
    Author: Abhner P. de Almeida (abhner.almeida AAT usp.br)
    '''
    
    custom_lines = []
    label = []
    for name in names:
        if 'Scatter' in name:
            name = name.replace('Scatter', '')
            print(name, colors[name + 'Scatter'], markers.get(name + 'Scatter'), msize.get(name + 'Legend'))
            if 'Bad' in name or 'True' in name:
                custom_lines.append(
                    Line2D([0], [0], color=colors[name + 'Scatter'], linestyle = '', lw=0.5, marker=markers.get(name + 'Scatter'),markeredgewidth = 0.7, markersize = msize.get(name + 'Legend'), markeredgecolor = 'black'))
                
            else:
                custom_lines.append(
                    Line2D([0], [0], color=colors[name + 'Scatter'], lw=0, marker=markers.get(name + 'Scatter'), markersize = msize.get(name+ 'Legend'), markeredgecolor = 'k'))
            label.append(titles.get(name, name))

        elif name == 'None':
            custom_lines.append(Line2D([0], [0], lw=0))
            label.append('')
        
        elif name == 'AnyNone':
            continue
   
        elif name in ['Centrals:', 'Satellites:']:
            custom_lines.append(Line2D([0], [0], lw=0))
            label.append(name)
        elif name in ['True', 'False']:
            custom_lines.append(Line2D([0], [0], color=colors.get(
                name, 'black'), ls=lines.get(name, 'solid'), lw=mult* linesthicker.get(name, 1), dash_capstyle = capstyles.get(name, 'projecting')))
            label.append(titles.get(name, name))

        else:
            custom_lines.append(Line2D([0], [0], color=colors.get(
                name, 'black'), ls=lines.get(name, 'solid'), lw=mult * linesthicker.get(name, 1), dash_capstyle = capstyles.get(name, 'projecting')))
            label.append(titles.get(name, name))

    if len(names) < 4:
        ncol = 1
        mult = 0.6
    elif len(names) <= 8:
        ncol = 2
        mult = 0.6
    else:
        ncol = 3
        mult = 0.5
    return custom_lines, label, ncol, mult


####################################################################################################
####################################################################################################

def makedata(names, columns, row, Type, snap=99, dfName='Sample', SampleName='Samples'):
    '''
    make data frame to plot
    Parameters
    ----------
    names : sample names. array with str
    columns : specific set in the sample / or different param to plot in each column. array with str
    rows : specific set in the sample / or different param to plot in each row. array with str
    Type : plot type. Can be 'z0', 'Snap' or 'Sample'.
    The rest is the same as the previous functions
    Returns
    -------
    Reuested data frame
    -------
    Author: Abhner P. de Almeida (abhner.almeida AAT usp.br)
    '''
    
    data = []
    
    if names[0] == 'All':
        All = WorkSample.extractDF('Samples/All')
        All = All.loc[(All.SubhaloMassInRadType4.between(7.5, 10))]

    if Type == 'Snap':
        if len(columns) == 1:
            nameSec = columns[0]
        else:
            nameSec = ''
        columns = snap
        
    for i, param in enumerate(row):
        dataSplit = []

        for j, split in enumerate(columns):
            dataName = []
            for k, name in enumerate(names):
                if name == 'All':
                    dataName.append(All[param].values)
                elif Type == 'z0':

                    values = WorkSample.extractWithCondition(name + split, dfName=dfName, SampleName=SampleName)
                    dataName.append(values[param].values)
                elif Type == 'Snap':
                    values = WorkSample.makeDF(name + nameSec, param, dfName=dfName, SampleName=SampleName)
                    dataName.append(values.iloc[99 - split].values)
                elif Type == 'Sample':
                    try:
                        values = WorkSample.makeDF(
                            name + split, param, dfName=dfName, SampleName=SampleName)
                        dataName.append(values.iloc[99 - snap[0]].values)
                    
                    except:
                        values = WorkSample.extractWithCondition(
                            name + split, dfName=dfName, SampleName=SampleName)
                        dataName.append(values[param].values)
                    
            dataSplit.append(dataName)

        data.append(dataSplit)


    return data

####################################################################################################
####################################################################################################

def makedataevolution(names, columns, row, Type, IDs=None, func=np.nanmedian, nboots=100, dfName='Sample', SampleName='Samples'):
    '''
    make evolution data frame to plot
    Parameters
    ----------
    names : sample names. array with str
    columns : specific set in the sample / or different param to plot in each column. array with str
    rows : specific set in the sample / or different param to plot in each row. array with str
    Type : plot type. Can be 'z0', 'Snap' or 'Sample'.
    func : can be 'np.mean', 'np.median' or analogues.
    The rest is the same as the previous functions
    Returns
    -------
    Requested data frame
    -------
    Author: Abhner P. de Almeida (abhner.almeida AAT usp.br)
    '''
    data = []
    dataerr = []
    for i, param in enumerate(row):
        dataSplit = []
        dataSpliterr = []
        for j, split in enumerate(columns):
            dataName = []
            dataNameerr = []
            for k, name in enumerate(names):
                values = WorkSample.makeDF(
                    name + split, param, dfName=dfName, SampleName=SampleName, IDs=IDs)

                y = []
                yerr = []
                for snap in np.arange(100):
                    ySnap = values.iloc[snap].values
                    ySnap = np.array([value for value in ySnap])

                    if len(ySnap[~np.isnan(ySnap)]) <= 5:
                        y.append(np.nan)
                        yerr.append(np.nan)
                    else:
                        if 'Num' in param or param == 'r_over_R_Crit200':
                            yerrSnap = MATH.boostrap_func(
                                ySnap, func=np.nanmean, num_boots=nboots)
                            y.append(np.nanmean(ySnap))
                            yerr.append(yerrSnap)
                        else:
                            yerrSnap = MATH.boostrap_func(
                                ySnap, func=func, num_boots=nboots)
                            y.append(func(ySnap))
                            yerr.append(yerrSnap)

                dataName.append(y)
                dataNameerr.append(yerr)

            dataSplit.append(dataName)
            dataSpliterr.append(dataNameerr)

        data.append(dataSplit)
        dataerr.append(dataSpliterr)

    return data, dataerr
    
####################################################################################################
####################################################################################################

def format_func_loglog(value, tick_number):
    '''
    change label in log plots
    Parameters
    ----------
    value : label values.
    tick_number : number of tickers
    Returns
    -------
    Requested label
    -------
    Author: Abhner P. de Almeida (abhner.almeida AAT usp.br)
    '''
    
    if value == 0:
        return str(0)
    sign = value/abs(value)
    N = int(np.round(np.log10(abs(value))))

    if abs(N) < 2:
        string = 10**N
        if sign*string >= 1:
            return str(int(sign*string))
        else:
            return str(sign*string)
    elif abs(N) >= 2:
        N = N*sign
        label = ('$10^{%4.0f}$ ' % N)
        return label


####################################################################################################
####################################################################################################

def binomialerrorplot(x,N,n, ax, color='k',marker='o',mec='k',capsize=0,capthick=1,
                       markersize=30,
                       label=None,scale='linear',eyFactor=None,verbose=0):
    """binomial plot
    arguments:
        x: x array
        N: array of total points
        n: array of good points
        color: face color of points, color of error bars, and color of upper and lower limits
        marker: marker shape
        markersize: markersize
        mec: color surrounding points
        capsize: size of error bar edges and of error bar arrow heads
        label: label for legend
        scale: scale of y axis: 'linear' or 'log'
        eyFactor: length of upper and lower limit symbols (number of dex for scale='log')
        
    author: Gary Mamon (gam AAT iap.fr)
        """
        
    print("markersize=",markersize)
    # restrict to bins with points
    condGood = N>0
    
    x = x[condGood]
    N = N[condGood]
    n = n[condGood]
    
    # statistics on bins with points
    p, ep = MATH.BinomialError(N,n)
    condUpper = n==0
    condLower = n==N
    condPoints = np.logical_not(np.logical_or((condUpper),(condLower)))
    if verbose > 0:
        print(" x   fraction    err(frac)  type")
        print(np.transpose([x,p,ep,condPoints]))
    # points with error bars
    ax.errorbar(x[condPoints],p[condPoints],ep[condPoints],
                 marker=marker,mec=mec,mfc=color,ecolor=color,ls='none',
                 ms=markersize,
                 capsize=capsize,capthick=capthick,label=label)
    
    # points with upper limits
    if scale=='log':
        if eyFactor is None:
            eyFactor = 0.5
        ep = p*(1-10**(-eyFactor))
    else:
        if eyFactor is None:
            eyFactor = 0.075
        ep = 0*ep + eyFactor
    # ensure that upper limit arrows do not extend past y=0
    # apply /2 fudge factor since capsize adds to length of arrow and is only known in pixels
    ep = np.where(p-ep<0,p/2,ep)

    if verbose >= 1:
        print("ep=",ep)
    uplims = np.full(len(x[condUpper]),True)
    ax.errorbar(x[condUpper],p[condUpper],ep[condUpper],
                 marker=',',mec=mec,mfc=color,ecolor=color,ls='none',
                 capsize=capsize,capthick=capthick,uplims=uplims)
    if verbose > 0:
        print("points with upper limits:")
        print()
    
    
    # points with lower limits
    if scale == 'log':
        ep = p*(10**eyFactor-1)
    # ensure that arrows do not extend past y=1
    # apply /2 fudge factor since capsize adds to length of arrow and is only known in pixels
    ep = np.where(p+ep>1,(1-p)/2,ep)
    lolims = np.full(len(x[condLower]),True)
    ax.errorbar(x[condLower],p[condLower],ep[condLower],
                 marker=',',mec=mec,mfc=color,ecolor=color,ls='none',
                 capsize=capsize,capthick=capthick,lolims=lolims)

    ax.set_ylim(-0.1, 1)
    
    if scale =='symlog':
	    ax.set_yscale('symlog')
	    ax.set_yticks([-1e-1, 0, 0.1, 0.2, 0.4, 0.6, 0.8, 1])
	    ax.set_yticklabels(['-0.1', '0', '0.1', '0.2', '0.4', '0.6', '0.8', '1'])
    
    if scale=='log':
        ax.set_yscale('log')

####################################################################################################
####################################################################################################

# dic for labels, scales ...


labels = {  # Masses
    'SubhaloMassInRadType0': r'$\log(M_{\mathrm{gas}}/\mathrm{M_\odot})$',
    'SubhaloMassInHalfRadType5': r'$\log(M_{\star, r_{1/2}}/\mathrm{M_\odot})$',

    'SubhaloMassInRadType1': r'$\log(M_{\mathrm{DM}}/\mathrm{M_\odot})$',
    'SubhaloMassInRadType4': r'$\log(M_{\star} / \mathrm{M_\odot})$',
    'SubhaloMass': r'$\log M \, \, [\mathrm{M_\odot}]$',
    'SubhaloMassType0': r'$\log M_{\mathrm{gas, all}} \, \, [\mathrm{M_\odot}]$',
    'SubhaloMassType1': r'$\log M_{\mathrm{DM, all}} \, \, [\mathrm{M_\odot}]$',
    'SubhaloMassType4': r'$\log M_{\star\mathrm{, all}} \, \, [\mathrm{M_\odot}]$',
    'StellarMassInSitu': r'$\log M_{\mathrm{in-stiu}} \, \, [\mathrm{M_\odot}]$',
    'StellarMassExSitu': r'$\log M_{\mathrm{ex-situ}} \, \, [\mathrm{M_\odot}]$',
    'SubhaloBHMass': r'BH occupation',
    'StellarMassFromCompletedMergers': r'$\log M_{\mathrm{mergers}} \, \, [\mathrm{M_\odot}]$',
    'StellarMassFromCompletedMergersMajor': r'$\log M_{\mathrm{M mergers}} \, \, [\mathrm{M_\odot}]$',
    'StellarMassFromCompletedMergersMinor': r'$\log M_{\mathrm{m mergers}} \, \, [\mathrm{M_\odot}]$',
    'StellarMassFromCompletedMergersMinorAll': r'$\log M_{\mathrm{m mergers}} \, \, [\mathrm{M_\odot}]$',
    'Group_M_Crit200':  r'$\log M_{200} \, \, [\mathrm{M_\odot}]$',
    'Group_M_Crit200_WithoutCorrection':  r'$\log M_{200} \, \, [\mathrm{M_\odot}]$',
    'MassType0Normalize': r'$M_\mathrm{gas} / M_{\mathrm{gas,}  z = 0}$',
    'MassType1Normalize':  r'$M_\mathrm{DM}  / M_{\mathrm{DM, }z = 0}$',
    'fDMTot':  r'$M_\mathrm{DM}  / M$',
    'MassType4Normalize':  r'$M_\star / M_{\star, z = 0}$',
    'HalfRadNormalized': r'$r_{1/2} / r_{1/2, z = 0}$',
    'MassExNormalize':  r'$M_{\mathrm{ex-situ}} / M_{\mathrm{ex-situ}, z = 0}$',
    'MassInNormalize':  r'$M_{\mathrm{in-situ}}/ M_{\mathrm{in-situ}, z = 0}$',
    'MassExNormalizeAll':  r'$M_{\mathrm{ex-situ}} / M_{\star, z = 0}$',
    'MassInNormalizeAll':  r'$M_{\mathrm{in-situ}}/ M_{\star, z = 0}$',
    'FracMassStarEx': r'$M_{\mathrm{ex-situ}}/ M_{\star}$',
    'DeltaMassType0Normalize': r'$d\ln M_\mathrm{gas} / d \ln t \, \, [\mathrm{M_\odot \; Gyr^{-1}}]$',
    'DeltaMassType1Normalize': r'$d\ln M_\mathrm{DM} / d \ln t \, \, [\mathrm{M_\odot \; Gyr^{-1}}]$',
    'DeltaMassType4Normalize': r'$d\ln M_\star / d \ln t \, \, [\mathrm{M_\odot \; Gyr^{-1}}]$',
    'StellarMassFromFlybys': r'$\log M_{\mathrm{fly-bys}} \, \, [\mathrm{M_\odot}]$',
    
    'MstarOverM200': r'$ M_{\star} / M_{200}$',
    'NewLambda': r'$\lambda$',

    # SFR
    'SubhalosSFRinHalfRad': r'$\log \mathrm{sSFR}_{r_{1/2}} \, \, [\mathrm{yr^{-1}}]$',
    'SubhalosSFRinRad': r'$\log \mathrm{sSFR} \, \, [\mathrm{M_\odot \; yr^{-1}}]$',
    'SubhalosSFRwithinHalfandRad': r'$\log \mathrm{sSFR}_{r_{1/2} < r < 2r_{1/2}} \, \, [\mathrm{M_\odot \; yr^{-1}}]$',
    'SubhalosSFRwithinRadandAll': r'$\log \mathrm{sSFR}_{r > 2r_{1/2}} \, \, [\mathrm{M_\odot \; yr^{-1}}]$',
    'sSFR': r'$\mathrm{sSFR} (r)\, \, [\mathrm{yr^{-1}}]$',
    'GFM_Metallicity_Zodot': r'$Z/Z_\odot$',
    'sSFRE': r'$\mathrm{SFR} (r) / M_{\mathrm{H}}\, \, [\mathrm{yr^{-1}}]$',

    'SFR': r'$\mathrm{SFR} \, \, [\mathrm{M_\odot \; yr^{-1}}]$',

    # Rads
    'rad': r'$r \, \, [\mathrm{kpc}]$',
    'SubhaloHalfmassRadType0': r'$\log(r_{1/2, \mathrm{gas}}/\mathrm{kpc})$',
    'SubhaloHalfmassRadType1': r'$\log(r_{1/2, \mathrm{DM}}/\mathrm{kpc})$',
    'SubhaloHalfmassRadType4': r'$\log(r_{1/2, \star}/\mathrm{kpc})$',
    
    'RandomRhalfi': r'$\log(r_{1/2 - i^, 2\mathrm{D} }/\mathrm{kpc})$',


    'r_over_R_Crit200': r'$R/R_{200}$',
    'r_over_R_Crit200_WithoutCorrection': r'$R/R_{200}$',
    'Pericenter': r'$R_\mathrm{per}/R_{200}$',
    'AngMomentum': r'$J_{\mathrm{gas}} \, \, [\mathrm{M_\odot kpc^2 km^{-2}}]$',
    'j': r'$j_{\mathrm{gas}} \, \, [\mathrm{kpc \; km  \; s^{-1}}]$',
    'joverR': r'$j_{\mathrm{gas}} (r) / r  \; \, \, [\mathrm{km  \; s^{-1}}]$',
    'DensityStar': r'$\rho_{\star} (r)  \; \, \, [\mathrm{M_\odot  \; kpc^{-3}}]$',
    'DensityGas': r'$\rho_{\mathrm{gas}} (r)  \; \, \, [\mathrm{M_\odot  \; kpc^{-3}}]$',
    
    'DensityStarOverR2': r'$\rho_{\star} (r)  r^2 \; \, \, [\mathrm{M_\odot  \; kpc^{-1}}]$',
    'DensityGasOverR2':  r'$\rho_{\mathrm{gas}} (r)  r^2 \; \, \, [\mathrm{M_\odot  \; kpc^{-1}}]$',

    'vOvervvirl': r'$v / v_\mathrm{vir}$',

    
    # Group
    'LhardXrayGroupBH': r'$\log L_{\mathrm{_{\scriptsize  2-10 keV_{_{BH \; in \; Group}}}}}$ \, \, [erg s$^{-1}$]',

    # Mergers
    'NumMergersTotal': r'$N$ mergers',
    'NumMinorMergersTotal':  r'$N$ Minor Mergers',
    'NumMajorMergersTotal': r'$N$ Major Mergers',
    
    'MergerTotalRate': r'$N / \mathrm{Gyr}$',
    'rOverR200Born': r'$(R/R_{200})_\mathrm{birth}$',
    'MinorMergerTotalRate': r'$N_{\mathrm{intermediate \; merger}} / \mathrm{Gyr}$',
    'MajorMergerTotalRate': r'$N_{\mathrm{major \; merger}}  / \mathrm{Gyr}$',

    'MedianWithoutMerger': 'Time between \n Merger [Gyr]',
    'MedianWithoutMajorMerger': 'Time between \n Major \n  Merger [Gyr]',
    'MedianWithoutMinorMerger': 'Time between \n Intermediate \n Merger [Gyr]',
    
    'MeanWithoutMerger': 'Time between \n Merger [Gyr]',
    'MeanWithoutMajorMerger': 'Time between \n Major Merger \ n[Gyr]',
    'MeanWithoutMinorMerger': 'Time between \n Intermediate \n Merger [Gyr]',

    # Others
    'Time Infall': r'Time infall [Gyr]',
    'StartBH': r'Seed BH \, \, [Gyr]',
    'LastMerger': 'Last Minor \n  Merger [Gyr]',
    'LastMajorMerger': 'Last Major \n Merger [Gyr]',
    'LastMinorMerger': 'Last intermediate \n Merger [Gyr]',
    'zMajorMerger': '$z$ of last  merger',
    'zMinorMerger': '$z$ of last  merger',
    'zMerger': '$z$ of last merger',
    
    
    
    'zMedianMajor': 'Median $z$' ,
    'zMedianMerger': 'Median $z$' ,
    'zMedianMinor': 'Median $z$' ,
    
    
    'MeanRedshift': r'Mean Merger Time [Gyr]',
    'SatelliteCount': r'$N_\mathrm{satellites}$',
    'Npericenter': r'$N_\mathrm{pericenter \; passages}$',
    'LastPericenter': r'Last $R_\mathrm{per}$ passage [Gyr]',
    'GroupNsubs': r'Number of satellites',
    'GroupNSubs': r'Number of satellites',
    'NumSatCent': r'Number of satellites',
    'SubhaloSpin': r'$\log j \, \, [\mathrm{kpc \; km \; s^{-1}}]$',

    # Profiles
    'Age': r'Mean Age$_\star$ [Gyr]',
    'RadVelocity': r'$v_\mathrm{r, gas} (r) \, \, [\mathrm{km \, s}^{-1}]$',
    'Mstellar': r'$ M_{\star}  \, \, [\mathrm{M_\odot}]$',
    'MstellarCum': r'$ M_{\star} (<r) \, \, [\mathrm{M_\odot}]$',
    'MstellarNorm': r'$ M_{\star} (<r) / M_{\star, \mathrm{tot}}$',
    'Mgas': r'$ M_{\mathrm{gas}}  \, \, [\mathrm{M_\odot}]$',
    'MgasCum': r'$ M_{\mathrm{gas}} (<r) \, \, [\mathrm{M_\odot}]$',
    'MgasNorm': r'$ M_{\mathrm{gas}} (<r) / M_{\star, \mathrm{tot}}$',

    'MassTensorEigenVals': r'$\mu_1 / \sqrt{\mu_2 \mu_3}$',
    'MassTensorEigenValsNew': r'$\mu_1 / \mu_3$',

    
    'AngMomMerger': r'$j_\mathrm{{merger}} / j_{\mathrm{max}}$',

    'zInfall': r'$z$ of first infall',
    'zInfallMinPericenter': r'$z$ at lowest pericenter',

    'zFirstPericenter': '$z$ of first \n pericenter',

    'ExMassEvolution':   r'$\log(M_{\mathrm{ex-situ}}/\mathrm{M}_\odot)$',
    'ExMassType0Evolution':    r'$\log(M_{\mathrm{gas, \;ex-situ}}/\mathrm{M}_\odot)$',
    'ExMassType1Evolution':  r'$\log(M_{\mathrm{DM, \;ex-situ}}/\mathrm{M}_\odot)$',
    'ExMassType4Evolution':   r'$\log(M_{\star, \mathrm{ex-situ}}/\mathrm{M}_\odot)$',
    'ExSubhaloMassType0':    r'$\log(M_{\mathrm{gas, \;ex-situ}}/\mathrm{M}_\odot)$',
    'ExSubhaloMassType1':  r'$\log(M_{\mathrm{DM, \;ex-situ}}/\mathrm{M}_\odot)$',
    'ExSubhaloMassType4':   r'$\log(M_{\star, \mathrm{ex-situ}}/\mathrm{M}_\odot)$',
    'rToRNearYoung': '$d_{\mathrm{NNB}}$ [kpc]',

    'DMFracBorn': r'$M_{\mathrm{DM, \; birth}}/M_{\mathrm{birth}}$ ',
    
    'NumPericenter': r'Passages through the pericenter',

    
    'GasFracBorn': r'$M_{\mathrm{gas, \; birth}}/M_{\mathrm{birth}}$ ',
    'StarFracBorn': r'$M_{\mathrm{\star, \; birth}}/M_{\mathrm{birth}}$ ',
    'GasFrac': r'$M_{\mathrm{gas}}/M$ ',
    'StarFrac':  r'$M_{\mathrm{\star}}/M$ ',

    'zBorn': r'$z_{\mathrm{birth}}$',
    'tsincebirth': r'$t - t_\mathrm{birth} \; [\mathrm{Gyr}]$',
    
    'jProfile': r'$j_{\mathrm{gas}} \, \, [\mathrm{kpc \; km  \; s^{-1}}]$',
    'logjProfile': r'$\log (j_{\mathrm{gas}} / \, \, [\mathrm{kpc \; km  \; s^{-1}}])$',
    'M200Normalized': r'$M_{200} / M_{200, \; z = 0}$',
    
    'DMFracMax': r'$M_{\mathrm{DM, \;}z = 0} / M_{\mathrm{DM, \, max}}$',

    'GasMassNormalized': r'$M_{\mathrm{gas}} / M_{\mathrm{gas, \, max}}$',

    'DMMassNormalized': r'$M_{\mathrm{DM}} / M_{\mathrm{DM, \, max}}$',
    'tlookInfallMinPericenter': r'$t_{\mathrm{look}}$ at lowest pericenter [Gyr]',

    'StarMassNormalized': r'$M/ M_{\mathrm{\star , \, max}}$',

    'DMFracMaxTot': r'$M_{\mathrm{DM, \;}z = 0} / M_{\mathrm{max}}$',
    'rOverR200Min': r'$(R/R_{200})_\mathrm{min}$',
    'rOverR200MinTrue': r'$(R/R_{200})_\mathrm{min}$',

    'DMFrac': r'$M_{\mathrm{DM, \;}z = 0} / M_{z = 0}$',
    'FracGasMass': 'Gas fraction',
    'FracDMMass': 'DM fraction',
    'FracStarMass': 'Stellar fraction',
    
    'RhalfOverMax': r'$r_{1/2}/ r_{1/2,\mathrm{\, max}}$',
    'StarFracMax': r'$M_{\star, \, z = 0} / M_{\star, \mathrm{max}}$',
    'SubhaloStarMetallicityHalfRad_z0': r'$Z / Z_\odot$',
    'SubhaloStarMetallicity_z0': r'$Z / Z_\odot$',

    'ProjectedTo3D': '',
    
    None: 'Any'
}

texts = {  # Masses
    'SubhaloMassInRadType0': r'$\log M_{\mathrm{gas}} \, \, [\mathrm{M_\odot}]$',
    'SubhaloMassInRadType1': r'$\log M_{\mathrm{DM}} \, \, [\mathrm{M_\odot}]$',
    'SubhaloMassInRadType4': r'$\log M_{\star} \, \, [\mathrm{M_\odot}]$',
    'SubhaloMass': r'$\log M \, \, [\mathrm{M_\odot}]$',
    'SubhaloMassType0': r'$\log M_{\mathrm{gas, all}} \, \, [\mathrm{M_\odot}]$',
    'SubhaloMassType1': r'$\log M_{\mathrm{DM, all}} \, \, [\mathrm{M_\odot}]$',
    'SubhaloMassType4': r'$\log M_{\star, all} \, \, [\mathrm{M_\odot}]$',
    'SubhaloBHMass': r'$\log M_{\mathrm{BH}} \, \, [\mathrm{M_\odot}]$',
    'StellarMassInSitu': r'$\log M_{\mathrm{in-stiu}} \, \, [\mathrm{M_\odot}]$',
    'StellarMassExSitu': r'$\log M_{\mathrm{ex-situ}} \, \, [\mathrm{M_\odot}]$',
    'StellarMassFromCompletedMergers': r'Mergers',
    'StellarMassFromCompletedMergersMajor': r'Major Mergers',
    'StellarMassFromCompletedMergersMinor': r'Minor Mergers',
    'StellarMassFromCompletedMergersMinorAll': r'All Minor Mergers',
    'StellarMassExSitu': r'$\log M_{\mathrm{ex-situ}} \, \, [\mathrm{M_\odot}]$',
    'Group_M_Crit200':  r'$\log M_{200} \, \, \, [\mathrm{M_\odot}]$',
    'Group_M_Crit200_WithoutCorrection':  r'$\log M_{200} \, \, \, [\mathrm{M_\odot}]$',
    'MassType0Normalize': r'$M / M_{z = 0}$',
    'MassType1Normalize':  r'$M / M_{z = 0}$',
    'MassType4Normalize':  r'$M / M_{z = 0}$',
    'HalfRadNormalized': r'$r_{1/2} / r_{1/2, z = 0}$',

    'MassExNormalize':  r'$M_\star / M_{\star, z = 0}$',
    'MassInNormalize':  r'$M_\star / M_{\star, z = 0}$',
    'MassExNormalizeAll':  r'$M_{\mathrm{ex-situ}} / M_{\star, z = 0}$',
    'MassInNormalizeAll':  r'$M_{\mathrm{in-situ}}/ M_{\star, z = 0}$',
    'FracMassStarEx': r'$M_{\mathrm{ex-situ}}/ M_{\star}$',
    'DeltaMassType0Normalize': r'$d\ln M / d \ln t',
    'DeltaMassType1Normalize': r'$d\ln M / d \ln t',
    'DeltaMassType4Normalize': r'$d\ln M / d \ln t',
    'rOverR200Born': r'$(R/R_{200})_\mathrm{birth}$',


    # SFR
    'SubhalosSFRinHalfRad': r'$\log sSFR_{r_{1/2}} \, \, [\mathrm{M_\odot \; yr^{-1}}]$',
    'SubhalosSFRinRad': r'$\log sSFR \, \, [\mathrm{M_\odot \; yr^{-1}}]$',
    'SubhalosSFR': r'$\log sSFR_{\mathrm{all}}` \, \, [\mathrm{M_\odot \; yr^{-1}}]$',
    'SubhalosSFRwithinHalfandRad': r'$\log sSFR_{r_{1/2} < r < 2r_{1/2}} \, \, [\mathrm{M_\odot \; yr^{-1}}]$',
    'SubhalosSFRwithinRadandAll': r'$\log sSFR_{r > 2r_{1/2}} \, \, [\mathrm{M_\odot \; yr^{-1}}]$',

    'vOvervvirl': r'$v / v_\mathrm{vir}$',


    'ExMassEvolution':   r'$\log(M_{\mathrm{ex-situ}}/\mathrm{M}_\odot)$',
    'ExMassType0Evolution':    r'$\log(M_{\mathrm{gas, \;ex-situ}}/\mathrm{M}_\odot)$',
    'ExMassType1Evolution':  r'$\log(M_{\mathrm{DM, \;ex-situ}}/\mathrm{M}_\odot)$',
    'ExMassType4Evolution':   r'$\log(M_{\star, \mathrm{ex-situ}}/mathrm{M}_\odot)$',
    
    'ExSubhaloMassType0':    r'$\log(M_{\mathrm{gas, \;ex-situ}}/\mathrm{M}_\odot)$',
    'ExSubhaloMassType1':  r'$\log(M_{\mathrm{DM, \;ex-situ}}/\mathrm{M}_\odot)$',
    'ExSubhaloMassType2':   r'$\log(M_{\star, \mathrm{ex-situ}}/mathrm{M}_\odot)$',


    # Rads
    'SubhaloHalfmassRadType0': r'$\log(r_{1/2, \mathrm{gas}}/\mathrm{kpc})$',
    'SubhaloHalfmassRadType1': r'$\log(r_{1/2, \mathrm{DM}}/\mathrm{kpc})$',
    'SubhaloHalfmassRadType4': r'$\log(r_{1/2, \star}/\mathrm{kpc})$',
    
    'RandomRhalfi': r'$\log(r_{1/2, 2D i}/\mathrm{kpc})$',


    'r_over_R_Crit200': r'$R/R_{200}$',
    'r_over_R_Crit200_WithoutCorrection': r'$R/R_{200}$',
    'currentRoverR200': r'current $R/R_{200}$',
    'Pericenter': r'$R_\mathrm{per}/R_{200}$',

    # Group
    'LhardXrayGroupBH': r'$\log L_{\mathrm{_{\scriptsize  2-10 keV_{_{BH \; in \; Group}}}}}$ \, \, [erg s$^{-1}$]',

    # Mergers
    'NumMergersTotal': r'$N$ mergers',
    'NumMinorMergersTotal':  r'$N$ Minor Mergers',
    'NumMajorMergersTotal': r'$N$ Major Mergers',
    
    'MergerTotalRate': r'$N / \mathrm{Gyr}$',
    'MinorMergerTotalRate': r'$N_{\mathrm{intermediate merger}} / \mathrm{Gyr}$',
    'MajorMergerTotalRate': r'$N_{\mathrm{major merger}}  / \mathrm{Gyr}$',

    # Others
    'TimeInfall': r'Time infall',
    'StartBH': r'Seed BH',
    'LastMerger': r'Last Minor Merger',
    'LastMajorMerger': r'Last Major Merger',
    'LastMinorMerger': r'Last Intermediate Merger',
    'zMajorMerger': 'Major mergers',
    'zMinorMerger': 'Intermediate mergers',
    'zMerger': '$z$ of last \n minor \n merger',

    'SatelliteCount': r'Number of satellites',
    'Npericenter': r'Pericenter Passages',
    'LastPericenter': r'Last pericenter passage',
    'GroupNsubs': r'Number of satellites',
    'GroupNSubs': r'Number of satellites',
    'NumSatCent': r'Number of satellites',
    'AngMomMerger': r'$j_\mathrm{{merger}} / j_{\mathrm{max}}$',


    # Profiles
    'Age': r'Mean Age [Gyr]',
    'RadVelocity': r'$v_\mathrm{r} (r) \, \, [\mathrm{km \, s}^{-1}]$',

    'MassTensorEigenVals': r'$M_1 / \sqrt{M_2 M_3}$',
    'MassTensorEigenValsNew': r'$\mu_1 / \mu_3$',

    'rToRNearYoung': r'$r_{\mathrm{Nearest \;  at \;  the \;  birth}} $ [kpc]',

    'zInfall': r'$z$ of first infall',
    'zInfallMinPericenter': r'$z$ at lowest pericenter [Gyr]',
    'tlookInfallMinPericenter': r'$t_{\mathrm{look}}$ at lowest pericenter',


    'zFirstPericenter': r'$z$ of first pericenter',
    'tsincebirth': r'$t - t_\mathrm{birth}$',
    'M200Normalized': r'$M_{200} / M_{200, \; z = 0}$',
    
    'zMedianMajor': 'Major mergers' ,
    'zMedianMerger': 'All mergers',
    'zMedianMinor': 'Intermediate mergers',



    None: 'Any'
}

labelsequal = {  # Masses
    'SubhaloMassInRadType0': r'$\log(M/\mathrm{M}_\odot)$',
    'SubhaloMassInRadType1':  r'$\log(M/\mathrm{M}_\odot)$',
    'SubhaloMassInRadType4':  r'$\log(M/\mathrm{M}_\odot)$',
    'SubhaloMass':  r'$\log(M/\mathrm{M}_\odot)$',
    'SubhaloMassType0': r'$\log(M/\mathrm{M}_\odot)$',
    'SubhaloMassType1': r'$\log(M/\mathrm{M}_\odot)$',
    'SubhaloMassType4': r'$\log(M/\mathrm{M}_\odot)$',
    'StellarMassInSitu': r'$\log(M_\star/\mathrm{M}_\odot)$',
    'StellarMassExSitu': r'$\log(M_\star/\mathrm{M}_\odot)$',
    'SubhaloBHMass': r'$\log(M/\mathrm{M}_\odot)$',
    'StellarMassFromCompletedMergers': r'$\log(M/\mathrm{M}_\odot)$',
    'StellarMassFromCompletedMergersMajor': r'$\log(M/\mathrm{M}_\odot)$',
    'StellarMassFromCompletedMergersMinor': r'$\log(M/\mathrm{M}_\odot)$',
    'StellarMassFromCompletedMergersMinorAll': r'$\log(M_{\mathrm{ex-situ}}/\mathrm{M}_\odot)$',
    'Group_M_Crit200': r'$\log(M/\mathrm{M}_\odot)$',
    'Group_M_Crit200_WithoutCorrection':  r'$\log(M/\mathrm{M}_\odot)$',
    'MassType0Normalize': r'$M / M_{z = 0}$',
    'MassType1Normalize':  r'$M / M_{z = 0}$',
    'MassType4Normalize':  r'$M / M_{z = 0}$',
    'HalfRadNormalized': r'$r_{1/2} / r_{1/2, z = 0}$',

    'MassExNormalizeAll':  r'Normalized $M_{\mathrm{ex-situ}}$',
    'MassInNormalizeAll':  r'$M_{\mathrm{in-situ}}/ M_{z = 0}$',
    'MassExNormalize':  r'Normalized $M_{\mathrm{ex-situ}}$',
    'MassInNormalize':  r'$M_\star / M_{\star, z = 0}$',
    'FracMassStarEx': r'Normalized $M_{\mathrm{ex-situ}}$',
    'DeltaMassType0Normalize': r'$d\ln M / d \ln t \, \, [\mathrm{M_odot \; Gyr^{-1}}]$',
    'DeltaMassType1Normalize': r'$d\ln M / d \ln t \, \, [\mathrm{M_odot \; Gyr^{-1}}]$',
    'DeltaMassType4Normalize': r'$d\ln M / d \ln t \, \, [\mathrm{M_odot \; Gyr^{-1}}]$',
    'rOverR200Born': r'$(R/R_{200})_\mathrm{birth}$',

    # SFR
    'SubhalosSFRinHalfRad': r'$\log \mathrm{sSFR} \, \, [\mathrm{yr^{-1}}]$',
    'SubhalosSFRinRad': r'$\log \mathrm{sSFR} \, \, [\mathrm{yr^{-1}}]$',
    'SubhalosSFR': r'$\log \mathrm{sSFR} \, \, [\mathrm{yr^{-1}}]$',
    'SubhalosSFRwithinHalfandRad': r'$\log \mathrm{sSFR} \, \, [\mathrm{yr^{-1}}]$',
    'SubhalosSFRwithinRadandAll': r'$\log \mathrm{sSFR} \, \, [\mathrm{yr^{-1}}]$',

    'vOvervvirl': r'$v / v_\mathrm{vir}$',

    'ExMassEvolution':   r'$\log(M_{\mathrm{ex-situ}}/\mathrm{M}_\odot)$',
    'ExMassType0Evolution':    r'$\log(M_{\mathrm{ex-situ}}/\mathrm{M}_\odot)$',
    'ExMassType1Evolution':  r'$\log(M_{\mathrm{ex-situ}}/\mathrm{M}_\odot)$',
    'ExMassType4Evolution':   r'$\log(M_{\mathrm{ex-situ}}/\mathrm{M}_\odot)$',
    
    'ExSubhaloMassType0':    r'$\log(M_{\mathrm{ex-situ}}/\mathrm{M}_\odot)$',
    'ExSubhaloMassType1':  r'$\log(M_{\mathrm{ex-situ}}/\mathrm{M}_\odot)$',
    'ExSubhaloMassType4':   r'$\log(M_{\mathrm{ex-situ}}/\mathrm{M}_\odot)$',
    
    
    # Rads
    
    'SubhaloHalfmassRadType0': r'$\log(r_{1/2}/\mathrm{kpc})$',
    'SubhaloHalfmassRadType1': r'$\log(r_{1/2}/\mathrm{kpc})$',
    'SubhaloHalfmassRadType4': r'$\log(r_{1/2}/\mathrm{kpc})$',
    'RandomRhalfi': r'$\log(r_{1/2, 2D i}/\mathrm{kpc})$',

    'r_over_R_Crit200': r'$R/R_{200}$',
    'r_over_R_Crit200_WithoutCorrection': r'$R/R_{200}$',
    'Pericenter': r'$R_\mathrm{per}/R_{200}$',

    # Group
    'LhardXrayGroupBH': r'$\log L_{\mathrm{_{\scriptsize  2-10 keV_{_{BH \; in \; Group}}}}}$ \, \, [erg s$^{-1}$]',

    # Mergers
    'NumMergersTotal': r'$N$',
    'NumMinorMergersTotal': r'$N$',
    'NumMajorMergersTotal': r'$N$',
    'AngMomMerger': r'$j_\mathrm{{merger}} / j_{\mathrm{max}}$',

    'MergerTotalRate': r'$N / \mathrm{Gyr}$',
    'MinorMergerTotalRate': r'$N / \mathrm{Gyr}$',
    'MajorMergerTotalRate': r'$N / \mathrm{Gyr}$',
    
    # Others
    'TimeInfall': r'Time infall',
    'StartBH': r'Time seed BH',
    'LastMerger': 'Time of  last \n merger [Gyr]',
    'LastMajorMerger': 'Time of last \n merger  [Gyr]',
    'LastMinorMerger': 'Time of last \n merger  [Gyr]',
       'zMajorMerger': '$z$ of last \n merger',
       'zMinorMerger': '$z$ of last \n merger',
       'zMerger': '$z$ of last \n merger',

    'SatelliteCount': r'N',
    'Npericenter': r'Pericenter Passages',
    'LastPericenter': r'Last pericenter passage',
    'GroupNsubs': r'Number of',
    'GroupNSubs': r'Number of',
    'NumSatCent': r'Number of satellites',


    # Profiles
    'Age': r'Mean Age [Gyr]',
    'RadVelocity': r'$v_\mathrm{r} (r) \, \, [\mathrm{km  \,s}^{-1}]$',

    'MassTensorEigenVals': r'$M_1 / \sqrt{M_2 M_3}$',
    'rToRNearYoung': r'$r_{\mathrm{Nearest \;  at \;  the \;  birth}}$ [kpc]',

    'zInfall': r'$z$ of first infall',
    'zInfallMinPericenter': r'$z$ at lowest pericenter',

    'zFirstPericenter': r'$z$ of first pericenter',
    'tsincebirth': r'$t - t_\mathrm{birth}$',

    'GasMassNormalized': r'$M / M_{\mathrm{max}}$',

    'DMMassNormalized': r'$M / M_{\mathrm{max}}$',

    'StarMassNormalized': r'$M / M_{\mathrm{max}}$',
    'StarFracMax': r'$M / M_{\mathrm{max}}$',
    
    
    None: 'Any'
}

scales = {  # Masses
    'SubhaloMassInRadType0': 'linear',
    'SubhaloMassInRadType1': 'linear',
    'SubhaloMassInRadType4': 'linear',
    'SubhaloMass': 'linear',
    'SubhaloMassType0': 'linear',
    'SubhaloMassType1': 'linear',
    'SubhaloMassType4': 'linear',
    'SubhaloBHMass': 'linear',
    'DMFracBorn': 'log',
    'GasFracBorn': 'linear',
    'StarFracBorn': 'linear',
    'GasFrac':  'linear',
    'StarFrac':  'linear',
    'MstarOverM200': 'log',

    'zBorn': 'log',
    'StellarMassInSitu': 'linear',
    'StellarMassExSitu': 'linear',
    'StellarMassFromFlybys': 'linear',
    'StellarMassFromCompletedMergers': 'linear',
    'StellarMassFromCompletedMergersMajor': 'linear',
    'StellarMassFromCompletedMergersMinor': 'linear',
    'StellarMassFromCompletedMergersMinorAll': 'linear',
    'StellarMassExSitu': 'linear',
    'MassExNormalize':  'log',
    'MassInNormalize':  'log',
    'MassExNormalizeAll':  'log',
    'FracMassStarEx': 'log',
    'MassInNormalizeAll':  'log',
    'Group_M_Crit200':  'linear',
    'Group_M_Crit200_WithoutCorrection':  'linear',
    'MassType0Normalize': 'log',
    'MassType1Normalize': 'log',
    'MassType4Normalize': 'log',
    'HalfRadNormalized': 'log',

    'DeltaMassType0Normalize': 'linear',
    'DeltaMassType1Normalize': 'linear',
    'DeltaMassType4Normalize': 'linear',

    # SFR
    'SubhalosSFRinHalfRad': 'linear',
    'SubhalosSFRinRad': 'linear',
    'SubhalosSFR': 'linear',
    'SubhalosSFRwithinHalfandRad': 'linear',
    'SubhalosSFRwithinRadandAll': 'linear',

    'vOvervvirl': 'linear',

    'rOverR200Born': 'log',
    
    'zMedianMajor': 'log' ,
    'zMedianMerger': 'log' ,
    'zMedianMinor': 'log' ,
    
    'zMajorMerger': 'linear' ,
    'zMinorMerger': 'linear' ,
    'zMeger': 'log' ,

    # Rads
    'SubhaloHalfmassRadType0': 'linear',
    'SubhaloHalfmassRadType1': 'linear',
    'SubhaloHalfmassRadType4': 'linear',
    'r_over_R_Crit200': 'log',
    'r_over_R_Crit200_WithoutCorrection': 'log',
    'currentRoverR200': 'log',
    'Pericenter': 'linear',

    # Group
    'LhardXrayGroupBH': 'linear',
    'jProfile': 'log',
    # Mergers
    'NumMergersTotal': 'linear',
    'NumMinorMergersTotal': 'linear',
    'NumMajorMergersTotal': 'linear',
    
    'MergerTotalRate': 'log',
    'MinorMergerTotalRate': 'log',
    'MajorMergerTotalRate': 'log',



    # Others
    'TimeInfall': 'linear',
    'StartBH': 'linear',
    'SatelliteCount': 'linear',
    'Npericenter': 'linear',
    'LastPericenter': 'linear',
    'LastMerger': 'linear',
    'LastMajorMerger': 'linear',
    'LastMinorMerger': 'linear',
    'dlnMdlT': 'log',
    'deltaM200': 'log',
    'deltaMbaryon': 'log',
    'GroupNsubs': 'log',
    'GroupNSubs': 'linear',
    'NumSatCent': 'log',
    'M200Normalized': 'log',


    # Profiles
    'Age': 'linear',
    'sSFR': 'log',
    'GFM_Metallicity_Zodot': 'log',

    'sSFRE': 'log',

    'RadVelocity': 'linear',
    'rad': 'log',
    'SFR': 'log',
    'Mstellar': 'log',
    'MstellarNorm': 'linear',
    'MstellarCum': 'log',
    'j': 'log',
    'joverR': 'linear',
    'DensityStar': 'log',
    'DensityGas': 'log',
    'DensityStarOverR2': 'log',
    'DensityGasOverR2': 'log',
    'Mgas': 'log',
    'MgasCum': 'log',
    'MgasNorm': 'linear',
    'Potential': 'symlog',
    'rOverR200Min': 'log',
    'rOverR200MinTrue': 'log',

    'DMFracMaxTot': 'log',

  'AngMomMedian': 'log',
  'AngMomMax': 'log',
  
    'rToRNearYoung': 'log',
         'MassTensorEigenVals': 'linear',
    'MassTensorEigenValsNew': 'linear',
    
    'GasMassNormalized': 'linear',

    'DMMassNormalized': 'log',

    'StarMassNormalized': 'log',
    
    'zInfallMinPericenter': 'log',

    
    'FracGasMass':'log',
    'FracDMMass': 'log',
    'FracStarMass': 'log',
    'SubhaloStarMetallicity_z0': 'log',

    #'zBorn': 'linear',

    None: 'linear'

}

markers = { # Compact
          'Normal': 'o', 
          'NormalScatter': 'o',  
          'NormalScatterPlot': 'o',     
          'NormalTrueScatterPlot': 'o',  
          'NormalFalseScatterPlot': 'o',  
          # MBC
          'MBC': 'o', 
          'MBCScatter': 'o', 
          'MBCScatterPlot': 'o',               
          'MBCTrueScatterPlot': 'o',  
          'MBCFalseScatterPlot': 'o',  

            # Diffuse
            'Diffuse': 'o', 
            'DiffuseScatter': 'o', 
            'DiffuseScatterPlot': 'o',               
            'DiffuseTrueScatterPlot': 'o',  
            'DiffuseFalseScatterPlot': 'o',  

          
          # SBC
          'SBC': 'o',
          'SBCOld': 'o',
          'SBCOldScatter': 'o',        
          'SBCOldScatterPlot': 'o',        

          'SBCScatter': 'o',        
          'SBCTrueScatterPlot': 'o',  
          'SBCFalseScatterPlot':'o',  
          'SBCYoungScatter': 'o',
          'SBCYoungScatterPlot': 'o',
          'GAMAScatter': '*',
          'GAMARdeScatter': '*',

          'SBCYoung': 'o',
          
          # Compact
          'Compact': 'o',
          'CompactScatter': 'o',        
          'CompactTrueScatterPlot': 'o',  
          'CompactFalseScatterPlot':'o',  
          'CompactYoungScatter': 'o',
          'CompactYoungScatterPlot': 'o',

          'CompactYoung': 'o',
          
          'TDGCentral': 'o',
          'TDGSatellite': 'o',
          
          'TDGCentralScatterPlot': 'o',
          'TDGSatelliteScatterPlot': 'o',
          
          #TNG
          'TNGrage': '.',
          'TNGrageScatter': '.', 
          'SBCGama': 'o',
          'SBCGamaScatter': 'o', 
          
          'MBCGama': 'D',
          'MBCGamaScatter': 'D', 
          
          'NormalGama': '^',
          'NormalGamaScatter': '^', 
          
          
          #Badfçag
          'BadFlag': 'x',
          'BadFlagScatter': 'x',
          'BadFlagScatterPlot': 'x',
          
          #Others
          'Selected': 'o',
          'SelectedScatter': 'o',
          'SelectedSatelliteScatter': 'o',
          'SelectedSatelliteScatterPlot': 'o',
          'SelectedSatellite': 'o',
          'MajorScatter': 'o',
          'IntermediateScatter': 's',
          'MinorScatter': '^',
          'Major': 'o',
          'Intermediate': 's',
          'Minor': '^',
          
          'TrueScatter': 'o',
          'FalseScatter': 'o',

         
    None: 'linear'

}

colors = {  # Compact
          'Normal': 'darkorange', 
          'NormalCentral': 'darkorange',
          'NormalSatellite': 'darkorange',
          'NormalSatelliteProfile': 'darkorange',

          'NormalTrue': 'darkorange',
          'NormalFalse': 'darkorange',
          'NormalCentralError': 'tab:orange', 
          'NormalSatelliteError': 'tab:orange', 
          'NormalSatelliteFalseError': 'tab:orange', 

          'NormalScatter': 'darkorange',  
          'NormalScatterPlot': 'darkorange', 
          'NormalTrueScatterPlot': 'darkorange',
          'NormalFalseScatterPlot':'darkorange',
          'NormalLegend': 'darkorange', 
          'NormalError': 'tab:orange', 
          'NormalWBH': 'darkorange',
            'NormalSatelliteTrue': 'darkorange',
            'NormalTrueScatterPlot': 'darkorange',
            'NormalSatelliteFalse': 'darkorange',

          
          # MBC
          'MBC': 'royalblue', 
          'MBCCentral': 'royalblue',
          'MBCSatellite': 'royalblue',
          'MBCTrue': 'royalblue',
          'MBCFalse': 'royalblue',
          'MBCSatelliteFalse': 'royalblue',
          'MBCSatelliteTrue': 'royalblue',
          'MBCTrueScatterPlot': 'royalblue',
          'MBCFalseScatterPlot':'royalblue',
          'MBCSatelliteError': 'tab:blue',
          'MBCSatelliteFalseError': 'tab:blue',

          'MBCCentralError': 'tab:blue',
          'MBCError': 'tab:blue',
          'MBCLegend': 'royalblue', 
          'MBCScatter': 'royalblue', 
          'MBCScatterPlot': 'royalblue', 
          
          'MBCWBH': 'royalblue',
          
          
          # Diffuse
          'Diffuse': 'royalblue', 
          'DiffuseCentral': 'royalblue',
          'DiffuseSatellite': 'royalblue',
          'DiffuseTrue': 'royalblue',
          'DiffuseFalse': 'royalblue',
          'DiffuseSatelliteFalse': 'royalblue',
          'DiffuseSatelliteTrue': 'royalblue',
          'DiffuseTrueScatterPlot': 'royalblue',
          'DiffuseFalseScatterPlot':'royalblue',
          'DiffuseSatelliteError': 'tab:blue',
          'DiffuseCentralError': 'tab:blue',
          'DiffuseError': 'tab:blue',
          'DiffuseLegend': 'royalblue', 
          'DiffuseScatter': 'royalblue', 
          'DiffuseScatterPlot': 'royalblue', 
          'DiffuseBH': 'royalblue',
          'DiffuseWBH': 'royalblue',
          
          
          # SBC
          'SBC': 'forestgreen',
          
          'SBCOld': 'forestgreen',
          'SBCOldScatter': 'forestgreen',    
          'SBCOldScatterPlot': 'forestgreen',    

          'SBCCentral': 'forestgreen',
          'SBCSatellite': 'forestgreen',
          
          'SBCTrue': 'forestgreen',
          'SBCFalse': 'forestgreen',
          'SBCSatelliteFalse': 'forestgreen',
          'SBCSatelliteTrue': 'forestgreen',
          
          'SBCTrueScatterPlot': 'forestgreen',
          'SBCFalseScatterPlot':'forestgreen',
          
          'SBCWBH': 'royalblue',

           'SBCBH': 'forestgreen',
           'MBCBH': 'royalblue',
           'NormalBH': 'darkorange',
           
           'SBCWithoutBH': 'forestgreen',
           'MBCWithoutBH': 'royalblue',
           'NormalWithoutBH': 'darkorange',
           
           'SBCBHError': 'tab:green',
           'MBCBHError': 'tab:blue',
           'NormalBHError': 'tab:orange',
           
           'SBCWithoutBHError': 'tab:green',
           'MBCWithoutBHError': 'tab:blue',
           'NormalWithoutBHError': 'tab:orange',
          
          'SBCCentralError': 'tab:green',
          'SBCSatelliteError': 'tab:green',
          'SBCSatelliteFalseError': 'tab:green',

          'SBCScatter': 'forestgreen',
          'SBCScatterPlot': 'forestgreen',
          'SBCError': 'tab:green',
          'SBCLegend': 'forestgreen',
          'SBCScatter': 'forestgreen', 
          'GAMAScatter': 'black',
          'GAMARdeScatter': 'red',

          
          'SBCYoungScatter': 'lime',
          'SBCYoungScatterPlot': 'lime',

          'SBCYoung': 'lime',
          
          # Compact
          'Compact': 'forestgreen',
          'CompactCentral': 'forestgreen',
          'CompactSatellite': 'forestgreen',
          'CompactTrue': 'forestgreen',
          'CompactFalse': 'forestgreen',
          'CompactSatelliteFalse': 'forestgreen',
          'CompactSatelliteTrue': 'forestgreen',
          
          'CompactTrueScatterPlot': 'forestgreen',
          'CompactFalseScatterPlot':'forestgreen',
          
          'CompactBH': 'royalblue',
          'CompactWBH': 'royalblue',


          
          'CompactCentralError': 'tab:green',
          'CompactSatelliteError': 'tab:green',
          'CompactScatter': 'forestgreen',
          'CompactScatterPlot': 'forestgreen',
          'CompactError': 'tab:green',
          'CompactLegend': 'forestgreen',
          'CompactScatter': 'forestgreen', 
          'CompactYoungScatter': 'lime',
          'CompactYoungScatterPlot': 'lime',

          'CompactYoung': 'lime',
          
          'TDGCentral': 'lime',
          'TDGSatellite': 'lime',
          'TDGCentralScatterPlot': 'lime',
          'TDGSatelliteScatterPlot': 'lime',
          
          #TNG
          'All': 'grey',
          'TNGrage': 'grey',
          'TNGrageScatter': 'grey', 
          
          'TNGGama': 'black',
          'TNGGamaScatter': 'black', 
          
          'SBCGama': 'black',
          'SBCGamaScatter': 'black', 
          
          'MBCGama': 'black',
          'MBCGamaScatter': 'black', 
          
          'NormalGama': 'black',
          'NormalGamaScatter': 'black', 
          
          #Badfçag
          'BadFlag': 'black',
          'BadFlagScatter': 'black',
          'BadFlagScatterPlot': 'black',
          
          #Others
          'Selected': 'none',
          'SelectedScatter': 'none',
          'SelectedSatelliteScatter': 'none',
          'SelectedSatelliteScatterPlot': 'none',
          'SelectedSatellite': 'none',
          'GMM': 'red',
          'BH': 'red',
          'WithoutBH': 'blue',
          'BHMedian': 'black',
          'WithoutBHMedian': 'black',
          
          'BHProfile': 'black',
          'WBH': 'black',
          
          
          'MajorScatter': 'black',
          'IntermediateScatter': 'black',
          'MinorScatter': 'black',
          'Major': 'black',
          'Intermediate': 'black',
          'Minor': 'black',
                 
          'True':'red',
          'TrueCompare':'red',

          'HigherFivePericenterlowest':'blue',

          'False': 'blue',
          'SatelliteTrue':'red',
          'SatelliteFalse':  'blue',
          'TrueScatter': 'none',
          'FalseScatter': 'black',
          
          'TrueLegend': 'none',
          'FalseLegend': 'black',
          'TrueProfile':'black',
          'FalseProfile': 'black',
        #Random
        
        '0': 'orangered',
        '1': 'chocolate',
        '2': 'salmon',
        '3': 'crimson',
        '4': 'darkviolet',
        '5': 'royalblue',
        '6': 'green',
        '7':  'lime',
        '8': 'brown',
        '9': 'lawngreen',
        '10': 'dodgerblue',
        '11': 'steelblue',
        '12':  'indigo',
        '13': 'magenta',
        '14': 'gold',
    
        None: 'black'

}

capstyles = {  # Compact
    'CompactCentral':  'round',
    'ControlSampleCentral':  'round',
    'ControlSampleScatter': 'round',

    #'SBCCentral':  'round',
    #'Central':  'round',
    #'Centrals': 'round',
    'DiffuseCentral': 'round',
    #'MBCCentral': 'round',
    #'NormalCentral': 'round',

    'SubhaloMassInRadType1':  'round',
    'SubhaloMassType1': 'round',
    'MassType1Normalize': 'round',
    'DeltaMassType1Normalize': 'round',
    'SubhaloHalfmassRadType1': 'round',
    'SubhalosSFRwithinRadandAll': 'round',
    'LastMinorMerger': 'round',
    'StellarMassFromCompletedMergersMinorAll': 'round',
    

    'DMMassNormalized': 'round',
 
    'SBCFalse': 'round',
    'CompactCentral':  'round',

    'CompactBH': 'round',
    'CompactFalse': 'round',


    'MBCFalse': 'round',
    
    
    'NormalFalse': 'round',
    'ExMassType1Evolution': 'round',
    'SBCBH': 'round',
    'MBCBH': 'round',
    'NormalBH': 'round',


    None: 'solid',

}

lines = {  # Compact
    'Compact': 'solid',
    'CompactnBHOld': 'solid',
    'CompactQuantile': 'solid',
    'CompactQuantileOld': 'solid',
    'CompactYoung': 'solid',
    'CompactBHOld': (0, (6, 5)),
    'CompactBH': 'dashed',
    'CompactWithoutBH': 'solid',
    'CompactLosesBH':  (0, (1, 10)),
    'CompactnBHYoung': (0, (1, 2)),

    'CompactCentral':  (0,(0.1,2)),
    'CompactSecondary': 'solid',
    'CompactMain': 'solid',
    # SBC
    'GMM': 'solid',
    'Selected': 'solid',
    'SelectedSatellite': 'solid',
    'SelectedSatelliteScatter': 'solid',

      'SBC': 'solid',
      'SBCOld': 'solid',
      'SBCOldScatterPlot': 'solid',

      'SBCnBHOld': 'solid',
      'SBCQuantile': 'solid',
      'SBCQuantileOld': 'solid',
      'SBCYoung': 'solid',
      'SBCBHOld': (0, (6, 5)),
      'SBCLosesBH':  (0, (1, 10)),
      'SBCnBHYoung': (0, (1, 2)),
      'SBCSatellite': (0, (11, 4)),
      'SBCSatelliteFalse': 'solid',

      'SBCCentral':  'solid',
      
      'SBCEarlierInfall': (0, (10, 8)),
      'SBCRecentInfall': (0,(0.1,2)),
      
      'SBCTrue': (0, (10, 8)),
      'SBCFalse': 'solid',
      'SBCSatelliteTrue': (0, (10, 8)),
      'SBCSatelliteTrueAll': (0, (10, 8)),
      
      'Compact': 'solid',
      'CompactnBHOld': 'solid',
      'CompactQuantile': 'solid',
      'CompactQuantileOld': 'solid',
      'CompactYoung': 'solid',
      'CompactBHOld': (0, (6, 5)),
      'CompactLosesBH':  (0, (1, 10)),
      'CompactnBHYoung': (0, (1, 2)),
      'CompactSatellite': (0, (11, 4)),
      'CompactCentral':  (0,(0.1,2)),
      
      'CompactEarlierInfall': (0, (10, 8)),
      'CompactRecentInfall': (0,(0.1,2)),
      
      'CompactTrue': (0, (10, 8)),
      'CompactFalse': 'solid',
      'CompactSatelliteTrue': (0, (10, 8)),
      'CompactSatelliteTrueAll': (0, (10, 8)),


      'MBCTrue':(0, (10, 8)),
      'MBCFalse': 'solid',
      
      'True':(0, (10, 8)),
      'TrueCompare':'solid',

      'HigherFivePericenterlowest':'solid',

      'False': 'solid',
      'SatelliteTrue':(0, (10, 8)),
      'SatelliteFalse': 'solid',
      'TrueProfile':(0, (10, 8)),
      'FalseProfile': 'solid',
      'NormalTrue': (0, (10, 8)),
      'NormalFalse': 'solid',
      
      'MBCEarlierInfall':(0, (10, 8)),
      'MBCRecentInfall': (0,(0.1,2)),
      
      'NormalEarlierInfall': (0, (10, 8)),
      'NormalRecentInfall': (0,(0.1,2)),
      
      'EarlierInfall': (0, (10, 8)),
      'RecentInfall': (0,(0.1,2)),
      
      
      'SBCBH': 'solid',
      'NormalBH': 'solid',
      
      'SBCWithoutBH': (0, (11, 4)),
      'NormalWithoutBH':(0, (11, 4)),
      
      'BHProfile': (0,(0.1,2)),
      'WBH': (0, (5, 2)),


    # Normal
    'ControlSample': 'solid',
    'ControlSamplenBHOld': 'solid',
    'ControlSampleBHOld': (0, (6, 5)),
    'ControlSampleLosesBHOld': (0, (5, 2, 1, 2)),
    'ControlSampleSatellite': (0, (10, 8)),
    'ControlSampleCentral':  (0,(0.1,2)),
    'ControlSampleBH': 'dashed',
    'ControlSampleWithoutBH': 'solid',
    'ControlSampleScatter': 'solid',

    
    'Normal': 'solid',
    'NormalnBHOld': 'solid',
    'NormalBHOld': (0, (6, 5)),
    'NormalLosesBHOld': (0, (5, 2, 1, 2)),
    'NormalSatellite': (0, (11, 4)),
    'NormalSatelliteFalse': 'solid',


    'NormalSatelliteProfile': 'solid',

    'NormalCentral':  'solid',
    
    'TDGCentral': 'solid',
    'TDGSatellite': 'solid',
    
    'Median': 'solid',
    'MediannBHOld': 'solid',
    'MedianBHOld': (0, (6, 5)),
    'MedianLosesBHOld': (0, (5, 2, 1, 2)),
    'MedianSatellite': (0, (10, 8)),
    'MedianCentral':  (0,(0.1,2)),
    'MedianBH': 'dashed',
    'MedianWithoutBH': 'solid',
    
    'NormalSatelliteProfile':  'solid',
    'SBCSatelliteProfile':  'solid',
    'MBCSatelliteProfile':  'solid',
    'NormalCentralProfile':  'solid',
    'SBCCentralProfile':  'solid',
    'MBCCentralProfile':  'solid',

    'SBCSatelliteTrueProfile':  (0, (11, 4)),
    'SBCSatelliteFalseProfile':  'solid',
    
    'CompactSatelliteProfile':  'solid',
    'CompactCentralProfile':  'solid',
    'MBCCentralProfile':  'solid',

    'CompactSatelliteTrueProfile':  (0, (11, 4)),
    'CompactSatelliteFalseProfile':  'solid',
    
    'MBCSatelliteTrueProfile':  (0, (11, 4)),
    'MBCSatelliteTrue':  (0, (11, 4)),
    'MBCSatelliteFalse':  'solid',


    'MBCSatelliteFalseProfile':  'solid',
    
    'NormalSatelliteTrueProfile':  (0, (11, 4)),
    'NormalSatelliteTrue':  (0, (11, 4)),
    'NormalSatelliteFalseProfile':  'solid',
    

    'BH': 'dashed',
    'WithoutBH': 'solid',
    
    
    'BHMedian': 'solid',
    'WithoutBHMedian':(0, (11, 4)),
    
    # MBC
    'MBC': 'solid',
    'MBCnBHOld': 'solid',
    'MBCBHOld': (0, (6, 5)),
    'MBCLosesBHOld': (0, (5, 2, 1, 2)),
    'MBCSatellite': (0, (11, 4)),
    'MBCCentral': 'solid',
    'MBCMedian': 'solid',
    'MBCBH': 'solid',
    'MBCWithoutBH': (0, (11, 4)),
    
    # Diffuse
    'Diffuse': 'solid',
    'DiffusenBHOld': 'solid',
    'DiffuseBHOld': (0, (6, 5)),
    'DiffuseLosesBHOld': (0, (5, 2, 1, 2)),
    'DiffuseSatellite': (0, (11, 4)),
    'DiffuseCentral': (0,(0.1,2)),
    'DiffuseMedian': 'solid',

    # Diffuse
    'Diffuse': 'solid',
    'DiffusenBHOld': 'solid',
    'DiffuseBHOld': (0, (6, 5)),
    'DiffuseLosesBHOld': (0, (5, 2, 1, 2)),
    'DiffuseCentral': (0,(0.1,2)),
    'DiffuseMedian': 'solid',



    # Sigmas
    '1SigmaLower': 'solid',
    '2SigmaLower': 'solid',
    '3SigmaLower': 'solid',
    '3SigmaLowerStrange': 'dashed',
    '1SigmaHigher': 'solid',
    '2SigmaHigher': 'solid',
    '3SigmaHigher': 'solid',


    'All': 'solid',
    'TNGrage': 'solid',
    'Satellite': (0, (5, 2)),
    'Satellites': (0, (5, 2)),

    'Central':  'solid',
    'Centrals':'solid',



    # Masses
    'SubhaloMassInRadType0': (0, (10, 8)),
    'SubhaloMassInRadType1':  (0,(0.1,2)),
    'SubhaloMassInRadType4':  'solid',
    'SubhaloMassType0': (0, (10, 8)),
    'SubhaloMassType1': (0,(0.1,2)),
    'SubhaloMassType4': 'solid',
    
    'GasMassNormalized':  (0, (10, 8)),

    'DMMassNormalized': (0,(0.1,2)),

    'StarMassNormalized': 'solid',
    
    
    'SubhaloHalfmassRadType0': (0, (10, 8)),
    'SubhaloHalfmassRadType1': (0,(0.1,2)),
    'SubhaloHalfmassRadType4': 'solid',
    
    'MassType0Normalize': (0, (10, 8)),
    'MassType1Normalize': (0,(0.1,2)),
    'MassType4Normalize': 'solid',
    'DeltaMassType0Normalize': (0, (10, 8)),
    'DeltaMassType1Normalize': (0,(0.1,2)),
    'DeltaMassType4Normalize': 'solid',
    'StellarMassInSitu': 'solid',
    'StellarMassExSitu': (0, (10, 8)),
    'MassExNormalize':  'solid',
    'MassInNormalize':  (0, (10, 8)),
    'MassExNormalizeAll':  (0, (10, 8)),
    'MassInNormalizeAll':  (0, (10, 8)),
    
    'ExMassEvolution':  'solid',
    'ExMassType0Evolution':  (0, (10, 8)),
    'ExMassType1Evolution':  (0,(0.1,2)),
    'ExMassType4Evolution':  'solid',

    
    'ExSubhaloMassType0':  (0, (10, 8)),
    'ExSubhaloMassType1':  (0,(0.1,2)),
    'ExSubhaloMassType4':  'solid',

    # sSFR
    'SubhalosSFRinHalfRad': 'solid',
    'SubhalosSFRinRad': 'solid',
    'SubhalosSFRwithinHalfandRad': (0, (10, 8)),
    'SubhalosSFRwithinRadandAll': (0,(0.1,2)),
    'LastMerger': (0, (10, 8)), 
    'LastMajorMerger': 'solid',
    'LastMinorMerger': (0,(0.1,2)),


    'MergerTotalRate':  'solid',
    'MinorMergerTotalRate': (0, (10, 8)),
    'MajorMergerTotalRate':(0,(0.1,2)),
    
    'StellarMassFromFlybys':  (0, (10, 8)), 
    'StellarMassFromCompletedMergersMajor':'solid',
        'StellarMassFromCompletedMergersMinorAll':(0,(0.1,2)),
        
    'r_over_R_Crit200': 'solid',


    None: 'solid',

}

linesthicker = {
    'Normal': 1.1, 
    'NormalBH': 1.1, 
    'NormalWithoutBH': 1.1, 

    'NormalCentral':  1.1,
    'NormalSatellite': 1.1,
    'NormalSatelliteFalse': 1.1,

    'NormalLegend': 0.8, 
    'NormalScatter': 0.5, 
    'NormalScatterPlot': 0.8, 
    'NormalTrueScatterPlot': 0.8, 
    'NormalFalseScatterPlot': 0,
    
    # MBC
    'MBC': 1.1, 
    'MBCCentral':  1.1,
    'MBCSatellite': 1.1,
    'MBCSatelliteFalse': 1.1,
    
    'MBCBH': 1.1, 
    'MBCWithoutBH': 1.1,

    'MBCLegend': 0.8, 
    'MBCScatter': 0.5, 
    'MBCScatterPlot': 0.8, 
    'MBCTrueScatterPlot': 0.8, 
    'MBCFalseScatterPlot': 0,
    
    # Diffuse
    'Diffuse': 1.1, 
    'DiffuseCentral': 1.7,
    'DiffuseSatellite': 1.1,
    'DiffuseLegend': 0.8, 
    'DiffuseScatter': 0.5, 
    'DiffuseScatterPlot': 0.8, 
    'DiffuseTrueScatterPlot': 0.8, 
    'DiffuseFalseScatterPlot': 0,
    
    # SBC
    'SBC': 1.1, 
    'SBCOld': 1.1,
    'SBCOldScatterPlot': 1.1,

    'SBCBH': 1.1, 
    'SBCWithoutBH': 1.1,
    'SBCCentral':  1.1,
    'SBCSatellite': 1.1,
    'SBCSatelliteFalse': 1.1,

    'SBCScatter': 0.8, 
    'GAMAcatter': 0.8, 
    'GAMARdecatter': 0.8, 


    'SBCTrueScatterPlot': 0.8, 
    'SBCFalseScatterPlot': 0,
    'SBCLegend': 0.8, 
    'SBCYoungScatter': 0.5, 
    'SBCYoungScatterPlot': 0.8, 

    'SBCYoung': 1.1, 
    'SBCYoungLegend': 0.8, 
    
    # Compact
    'Compact': 1.1, 
    'CompactCentral': 2.4,
    'CompactSatellite': 2,
    'CompactScatter': 0.8, 
    'CompactTrueScatterPlot': 0.8, 
    'CompactFalseScatterPlot': 0,
    'CompactLegend': 0.8, 
    'CompactYoungScatter': 0.5, 
    'CompactYoungScatterPlot': 0.8, 

    'CompactYoung': 1.1, 
    'CompactYoungLegend': 0.8, 
    
    'TDGCentral': 1.1, 
    'TDGSatellite': 1.1, 
    
    #TNG
    'TNGrage': 1.5,
    'TNGrageScatter': 0.5, 
    
    #Badfçag
    'BadFlag': 2,
    'BadFlagLegend': 3,
    'BadFlagScatter': 1.2,
    'BadFlagScatterPlot': 2,
    
    #Others
    'Selected': 0.8,
    'SelectedScatter': 0.8,
    'SelectedSatelliteScatter': 0.2, 
    'SelectedSatelliteScatterPlot': 0.8,
    'SelectedSatellite': 0.8,


    
    #Others
    'CentralLegend': 0.8,
    'Central': 0.8,
    'Centrals': 0.8,
    
    'BHProfile':  2.,
    'WBH':  2.,

    'SubhaloMassInRadType1': 1.8,
    'SubhaloMassType1': 1.8,
    'DMMassNormalized': 1.8,

    'SubhaloHalfmassRadType1': 1.8,
    'MassType1Normalize': 1.8,
    'ExMassType1Evolution': 1.8,
    'ExSubhaloMassType1': 1.8,

    'DeltaMassType1Normalize': 1.8,
    'SubhalosSFRwithinRadandAll': 1.8,
    'LastMinorMerger': 1.8,
    'StellarMassFromCompletedMergersMinorAll': 1.8,
    

}

msize = {
    'Normal': 10, 
    'NormalLegend': 5, 
    'NormalScatterLegend': 8,

    'NormalScatter': 10, 
    'NormalScatterPlot': 15, 
    'NormalTrueScatterPlot': 15,
    'NormalFalseScatterPlot': 15, 


    
    # MBC
    'MBC': 10, 
    'MBCLegend': 8, 
    'MBCScatter': 22, 
    'MBCScatterPlot': 22, 
    'MBCTrueScatterPlot': 20,
    'MBCFalseScatterPlot': 20,
    
    'MBCScatterLegend': 8,

    
    # MBC
    'Diffuse': 10, 
    'DiffuseLegend': 8, 
    'DiffuseScatterLegend': 8,

    'DiffuseScatter': 20, 
    'DiffuseScatterPlot': 20, 
    'DiffuseTrueScatterPlot': 20,
    'DiffuseFalseScatterPlot': 20,
    
    
    # SBC
    'SBC': 10,
    'SBCScatter': 22,
    'SBCOld': 10,
    'SBCOldScatter': 22,
    'SBCOldScatterPlot': 22,

    'GAMALegend': 8,

    'GAMAScatter': 20,
    'GAMAScatterPlot': 20,
    'GAMAScatterLegend': 20,
    
    'GAMARdeLegend': 8,

    'GAMARdeScatter': 20,
    'GAMARdeScatterPlot': 20,
    'GAMARdeScatterLegend': 20,

    'SBCScatterPlot': 22,
    'SBCTrueScatterPlot': 20,
    'SBCFalseScatterPlot': 20,

    'SBCLegend': 8,
    'SBCScatterLegend': 8,
    'SBCYoungScatter': 35,
    'SBCYoungScatterPlot': 20,

    'SBCYoung': 20, 
    'SBCYoungLegend': 9, 
    
    # Compact
    'Compact': 10,
    'CompactScatter': 20,
    'CompactScatterPlot': 20,
    'CompactTrueScatterPlot': 20,
    'CompactFalseScatterPlot': 20,

    'CompactLegend': 8,
    'CompactYoungScatter': 35,
    'CompactYoungScatterPlot': 20,

    'CompactYoung': 20, 
    'CompactYoungLegend': 9, 
    
    'TDGCentral': 20,
    'TDGSatellite': 20,
    
    'TDGCentralScatterPlot': 20,
    'TDGSatelliteScatterPlot': 20,
    
    #TNG
    'TNGrage': 20,
    'TNGrageScatter': 20, 
    'TNGrageScatterLegend': 7,
    'TNGrageLegend': 7,
    'TNGGama': 20,
    'TNGGamaScatter': 20, 
    'TNGGamaScatterLegend': 7,
    
    'SBCGama': 20,
    'SBCGamaScatter': 7,
    
    'MBCGama': 20,
    'MBCGamaScatter': 7,
    
    'NormalGama': 20,
    'NormalGamaScatter': 7,
    
    'TNGGamaLegend': 7,
    
    #Badfçag
    'BadFlag': 15,
    'BadFlagLegend': 7,
    'BadFlagScatter': 25,
    'BadFlagScatterPlot': 15,
    
    #Others
    'Selected': 6,
     'SelectedLegend': 5,
    'SelectedSatelliteScatter': 20,
    'SelectedSatelliteScatterPlot': 10,
    'SelectedSatellite': 10,
    'MajorScatterLegend': 10,
    'IntermediateScatterLegend': 10,
    'MinorScatterLegend': 10,
    'MajorLegend': 10,
    'IntermediateLegend': 10,
    'MinorLegend': 10,
    
    'TrueLegend': 12,
    'FalseLegend': 12,

  
    }


titles = {  # Classes
     # Compact
    'Normal': 'Normals', 
    'NormalScatter': 'Normals',  
    
    # MBC
    'MBC': r'$\mathrm{Compacts_{MB}}$', 
    'MBCScatter': r'$\mathrm{Compacts_{MB}}$',
    
    # SBC
    'Diffuse': r'$\mathrm{Diffuses}$',
    'SBC': r'$\mathrm{Compacts_{SB}}$',
    'SBCScatter': r'$\mathrm{Compacts_{SB}}$',
    'GAMAScatter': r'$\mathrm{GAMA}$',
    'GAMARdeScatter': r'$\mathrm{GAMA}$',
    'GAMARde': r'$\mathrm{GAMA}$',
    'GAMARdeLegend': r'$\mathrm{GAMA}$',
    'GAMARdeScatterLegend': r'$\mathrm{GAMA}$',


    'SBCYoungScatter': 'Young \n Compacts$_\mathrm{SB}$',
    'SBCYoung': 'Young \n Compacts$_\mathrm{SB}$',

   # SBC
   'Diffuse': r'$\mathrm{Diffuses}$',
   'Compact': r'$\mathrm{Compacts}$',
   'CompactScatter': r'$\mathrm{Compacts}$',
   'CompactYoungScatter': 'Young \n Compacts',
   'CompactYoung': 'Young \n Compacts',

    
    #TNG
    'TNGrage': 'All\n galaxies',
    'TNGrageScatter': 'All\n galaxies',
    'TNGGama': 'TNG50',
    'TNGGamaScatter': 'TNG50',
    
    'SBCGama': r'$\mathrm{Compacts_{SB}}$',
    'SBCGamaScatter': r'$\mathrm{Compacts_{SB}}$',
    
    'MBCGama':r'$\mathrm{Compacts_{MB}}$',
    'MBCGamaScatter': r'$\mathrm{Compacts_{MB}}$',
    
    'NormalGama': 'Normals', 
    'NormalGamaScatter': 'Normals', 
    
    
    #Badfçag
    'BadFlag': 'Bad flags',
    
    #Others
    'Selected': 'Selected',
    'SelectedScatter': 'Selected',
    'SelectedSatellite': 'Satellites',
    
    'True':r'$f_{\mathrm{DM}} \leq 0.7$',
    'TrueCompare':r'$f_{\mathrm{DM}} \leq 0.7$',

    'HigherFivePericenterlowest':'$N_\mathrm{pericenter} \geq 3$ \n and\n $(R/R_{200})_\mathrm{min} < 0.2$',

    'False': r'$f_{\mathrm{DM}} > 0.7$',
    'SatelliteTrue':r'Satellites $f_{\mathrm{DM}} \leq 0.7$',
    'SatelliteFalse': r'Satellites $f_{\mathrm{DM}} > 0.7$',
    'TrueProfile':r'$f_{\mathrm{DM}} \leq 0.7$',
    'FalseProfile': r'$f_{\mathrm{DM}} > 0.7$',
    
    
    
      'EarlierInfall': 'Earlier Infall',
      'RecentInfall': 'Recent Infall',


    'CompactSubhaloHalfmassRadType4': 'Compact, $r_{1/2}$',
    'CompactSubhaloHalfmassRadType0': 'Compact, $r_{1/2, \mathrm{gas}}$',
    'ControlSampleSubhaloHalfmassRadType4': 'Control \n Sample, $r_{1/2}$',
    'ControlSampleSubhaloHalfmassRadType0':  'Control \n Sample, $r_{1/2, \mathrm{gas}}$',


    # Sigmas
    '1SigmaLower': '$5^\mathrm{th} - 32^\mathrm{th}$',
    '2SigmaLower': '$1^\mathrm{st} - 5^\mathrm{th}$',
    '3SigmaLower': '$< 1^\mathrm{st}$',
    '1SigmaHigher': '$68^\mathrm{th} - 95^\mathrm{th}$',
    '2SigmaHigher': '$95^\mathrm{th} - 99^\mathrm{th}$',
    '3SigmaHigher': '$> 99^\mathrm{th}$',


    # Status
    'BH': 'With  BH',
    'nBH': 'Without  BH',
    'BHMedian': 'With  BH',
    'WithoutBHMedian': 'Without  BH',

    'WBH': 'Without BH',
    'WithoutBH': 'Without  BH',
    'LosesBH': 'Loses BH',

          

    # Masses
    'SubhaloMassInRadType0': 'Gas',
    'SubhaloMassInRadType1':  'DM',
    'SubhaloMassType0': 'Gas',
    'SubhaloMassType1': 'DM',
    'SubhaloMassType4': 'Stellar',
    'MassType0Normalize': 'Gas',
    'MassType1Normalize': 'DM',
    'MassType4Normalize': 'Stellar',
    'SubhaloMassType4': 'Stellar',
    'SubhaloHalfmassRadType0': 'Gas',
    'SubhaloHalfmassRadType1': 'DM',
    'SubhaloHalfmassRadType4': 'Stellar',
    'DeltaMassType0Normalize': 'Gas',
    'DeltaMassType1Normalize': 'DM',
    'DeltaMassType4Normalize': 'Stellar',
    
    'ExMassEvolution': 'Total',
    'ExMassType0Evolution': 'Gas',
    'ExMassType1Evolution': 'DM',
    'ExMassType4Evolution': 'Stellar',

    # sSFR
    'SubhalosSFRinHalfRad': 'sSFR within $r_{1/2}$',
    'SubhalosSFRinRad': 'sSFR in $2r_{1/2}$',
    'SubhalosSFRwithinHalfandRad': 'sSFR within \n $r_{1/2} < r < 2r_{1/2}$',
    'SubhalosSFRwithinRadandAll': 'sSFR in \n $r > 2r_{1/2}$',
    'StellarMassInSitu': 'In-situ',
    'StellarMassExSitu': 'Ex-situ',
    'MassExNormalize':  'By \n $M_{\mathrm{ex - situ}, z = 0}$',
    'MassInNormalize':  'In-situ',
    'MassExNormalizeAll':  'By $M_{\star, z = 0}$',
    'MassInNormalizeAll':  'By $M_{\star, z = 0}$',
    'LastMerger': 'Last Minor \n Merger',
    'LastMinorMerger': 'Last intermediate \n Merger',
    'LastMajorMerger': 'Last Major \n Merger',
 
          
   'StellarMassFromFlybys': 'Fly-bys',
   'StellarMassFromCompletedMergersMajor': 'Major mergers',
   'StellarMassFromCompletedMergersMinorAll': 'Minor + \n intermediate \n mergers',
   
   'MajorScatter': 'Major Merger',
   'IntermediateScatter': 'Intermediate Merger',
   'MinorScatter': 'Minor Merger',
   'Major': 'Major Merger',
   'Intermediate': 'Intermediate Merger',
   'Minor': 'Minor Merger',
          
}


limmax = {  # Masses
    'StellarMassFromCompletedMergers': 10.3,
    'StellarMassFromCompletedMergersMajor': 7.7,
    'StellarMassFromCompletedMergersMinor': 7.7,
    'StellarMassFromCompletedMergersMinorAll': 7.2,
    'StellarMassExSitu': 8.3,
    # Rads
    'r_over_R_Crit200': 50.,
    #'r_over_R_Crit200_WithoutCorrection': 20.,
    'r_over_R_Crit200Scatter': 50,

    'MassType0Normalize': 1500.,
    'MassType1Normalize':  90,
    'MassType4Normalize':  4.,
    'HalfRadNormalized': 7.,
    'MassExNormalize':  1.3,
    'MassInNormalize': 1.2,
    'MassExNormalizeAll': 0.06,
    'MassExNormalizeAllScatter': 0.7,
    'FracMassStarEx': 0.12,
    'FracMassStarExScatter': 0.12,

    'SubhaloMass': 12,
    'SubhaloMassInRadType4': 9.3,
    'SubhaloMassInRadType4Scatter': 10.,

    #'SubhaloHalfmassRadType4': 1.6,
    #'SubhaloMassInRadType4': 9.8,
    'GroupNSubs': 42,

    'LastMajorMerger': 10,
    'LastMinorMerger': 10,
    'LastMerger': 10,

    'vOvervvirl': 1,
    # 'StellarMassExSitu': 8.5,
    #'Group_M_Crit200_Central': 11.5,
    #'Group_M_Crit200': 10.98,

    # Profiles
    'Age': 14.2,
    'RadVelocity': 20,
    'rad': 100,
    'sSFR': 2e-9,
    'GFM_Metallicity_Zodot': 3.9,

    'sSFRE': 2e-5,

    'SFR':  0.085,
    'Mstellar': 1e8,
    'MstellarCum': 1e10,
    'MstellarNorm': 1.1,
    'NewLambda': 0.5,

    'Mgas': 1e8,
    'MgasCum':  1e10,
    'MgasNorm': 1.1,

    'j': 3e3,
    'joverR': 78,
    'Potential': -5e3,

    'SubhalosSFRinHalfRad': -7.9,
    'SubhalosSFRinRad': -7.9,
    'SubhalosSFRwithinHalfandRad': -7.7,
    'SubhalosSFRwithinRadandAll': -7.7,

    'MassTensorEigenVals': 1.1,
    'MassTensorEigenValsNew': 1.1,

    'DensityStar': 7e9,
    'DensityGas': 5e8,
    
    'DensityStarOverR2':  5e8,
    'DensityGasOverR2': 5e7,
    'MinorMergerTotalRate': 2,
    'MajorMergerTotalRate': 2,
    'zBorn': 22,
    'DMFracBorn': 1.02,
    'DMFracBornScatter': 1.02,
    'DMFrac': 1.02,
    'DMFracScatter': 1.02,


    'GasFracBorn': 1.02,
    'StarFracBorn': 1.02,
    
    'GasFrac': 1.02,
    'StarFrac': 1.02,

    'MstarOverM200Scatter': 0.9,

    'rOverR200Born': 300,
    
    'jProfile': 1.5e3,
    'M200Normalized': 1.05,
    'DMFracMaxTot': 1.02,
    'DMFracMaxTotScatter': 1.02,
    'DMFracMaxScatter': 1.02,

    
    'GasMassNormalizedScatter': 1.1,

    'DMMassNormalizedScatter': 1.5,

    'StarMassNormalizedScatter': 1.1,
    
    'FracGasMassScatter': 0.2,
    'FracDMMassScatter': 1.1,
    'FracStarMassScatter': 1.1,
    
    'GasMassNormalized': 1.1,

    'DMMassNormalized': 1.1,

    'StarMassNormalized': 1.1,
    
    'FracDMMass': 1.1,
    
    'SubhaloHalfmassRadType4Pericenter': -0.5,

    'GasMassNormalizedPericenter': 0.008,

    'DMMassNormalizedPericenter': 0.008,
    
    'StarMassNormalizedPericenter': 0.14,
    
    'zMedianMajor': 22,
    'zMedianMerger': 22,
    'zMedianMinor': 22,
    
    'zMajorMerger': 22,
    'zMinorMerger': 22,
    'zMeger': 22,
    'zInfallMinPericenterScatter': 5,
    'rOverR200MinScatter': 3,
    'rOverR200MinTrueScatter': 3,

}


limin = {  # Masses
    'StellarMassFromCompletedMergers': 4.3,
    'StellarMassFromCompletedMergersMajor': 4.3,
    'StellarMassFromCompletedMergersMinor': 4.3,
    'StellarMassFromCompletedMergersMinorAll': 4.1,
    'SubhaloHalfmassRadType4': -0.7,
    'SubhaloHalfmassRadType4Scatter': -0.9,
    'SubhaloMassInRadType4': 8.3,
    #'StellarMassExSitu': 4.5,
    'MassType0Normalize': 0.01,
    'MassType1Normalize':   0.01,
    'MassType4Normalize':  0.009,
    'MassInNormalize':  0.002,
    'MassExNormalize':  0.005,
    'MassExNormalizeAll': 0.0008,
    'MassExNormalizeAllScatter': 0.0008,
    'SubhaloMass': 7.8,
    'MstellarCum': 2e6,
    #'SubhaloMassInRadType4': 7.8,
    'vOvervvirl': -1,
    'GroupNSubs': 1.1,
    'NewLambda': 0.0,
    'Group_M_Crit200_Central': 9.5,
    'Group_M_Crit200': 9.2,
    'FracMassStarEx': 0.005,
    'FracMassStarExScatter': 0.005,

    # Rads
    'r_over_R_Crit200': 0.05,
    'r_over_R_Crit200Scatter': 0.005,

    #'r_over_R_Crit200_WithoutCorrection': 0.15,

    'LastMajorMerger': 0.4,
    'LastMinorMerger': 0.4,
    'LastMerger': 0.4,
    # 'StellarMassExSitu': 4,

    # Profiles
    'Age': 0.2,
    'RadVelocity': -80,
    'rad': 0.101,
    'sSFR': 5e-12,
    'GFM_Metallicity_Zodot': 5e-2,

    'sSFRE': 1e-14,

    'SFR':  1e-6,
    'Mstellar': 1e5,

    'MstellarNorm': 0.005,

    'Mgas': 1e5,
    'MgasCum':  2e6,
    'MgasNorm':  0.005,

    'j': 9,
    'joverR': 0,
    'Potential': -2e5,

    'SubhalosSFRinHalfRad': -11.,
    'SubhalosSFRinHalfRadScatter': -14.,

    'SubhalosSFRinRad': -11.,
    'SubhalosSFRwithinHalfandRad': -11.,
    'SubhalosSFRwithinRadandAll': -11.,

    'MassTensorEigenVals': 0.2,
    'MassTensorEigenValsNew': 0.2,

    
    'DensityStar': 1e3,
    'DensityGas': 1e3,
    'DensityStarOverR2': 1e3,
    'DensityGasOverR2': 5e4,

    'MinorMergerTotalRate': 0.001,
    'MajorMergerTotalRate': 0.001,
    'zBorn' : 0.005,
    'DMFrac': 0.0001,
    'DMFracBorn': 0.0001,
    'DMFracBornScatter': 0.0001,
    'DMFracScatter': 0.0001,

    'GasFracBorn': 0.0001,
    'StarFracBorn': 0.0001,
    'GasFrac': 0.0001,
    'StarFrac': 0.0001,

    'MstarOverM200Scatter': 0.0005,
    'rOverR200Born': 0.1,
    'HalfRadNormalized': 0.05,
    
    'jProfile': 8e1,
    'M200Normalized': 0.09,
    'DMFracMaxTot': 1e-4,
    'DMFracMaxTotScatter':  1e-4,
    
    'DMFracMaxScatter': -0.02,

    
    'GasMassNormalizedScatter': -0.1,

    'DMMassNormalizedScatter': 0.0007,

    'StarMassNormalizedScatter': 0.05,
    
    'GasMassNormalized': 0.0005,

    'DMMassNormalized': 0.0005,

    'StarMassNormalized': 0.005,
    
    

    'SubhaloHalfmassRadType4Pericenter': 0.2,

    'GasMassNormalizedPericenter': 0.011,

    'DMMassNormalizedPericenter': 0.011,

    'StarMassNormalizedPericenter': 0.05,
    
    
    'FracGasMassScatter': 0.0009,
    'FracDMMassScatter': 0.09,
    'FracStarMassScatter': 0.0009,
    
    'FracDMMass': 0.09,
    
    'zMedianMajor': 0.05,
    'zMedianMerger': 0.05,
    'zMedianMinor': 0.05,
    
    'zMajorMerger': 0.05,
    'zMinorMerger': 0.05,
    'zMeger': 0.05,
    'zInfallMinPericenterScatter': 0.005,
    'rOverR200MinScatter': 0.009,
    'rOverR200MinTrueScatter': 0.009,


}
