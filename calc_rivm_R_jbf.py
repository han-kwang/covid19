#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Replicating RIVM R method, JBF method by @MrOoijer (Jan van Rongen)

(JBF = Jan Boerenfluitjes)

Original R implementation https://github.com/MrOoijer/Covid-Rt .

Created on Sun Jul 25 14:48:01 2021

@hk_nien on Twitter
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import casus_analysis as ca
import nlcovidstats as nlcs
from tools import set_xaxis_dateformat


def gauss_smooth(data, n=11, sigma=1.67):
    """Apply gaussian smoothing kernel on data array.

    Parameters:

    - n: window size (should be odd)
    - sigma: standard deviation of the window.

    Return:

    - smoothed data, same size as input.
    """
    assert n%2 == 1
    m = (n-1) // 2
    xs = np.arange(-m, m+1)
    kernel = np.exp(-0.5/sigma**2 * xs**2)
    kernel /= kernel.sum()

    # pad input data
    k = m//2 + 1
    data_padded = np.concatenate([
        np.full(m, data[:k].mean()),
        data,
        np.full(m, data[-k:].mean())
        ])
    smooth = np.convolve(data_padded, kernel, mode='same')[m:-m]
    assert smooth.shape == data.shape
    return smooth


if __name__ == '__main__':

    if 0:
        # Run this manually, interactively to refresh data.
        # (Slow, don't do this automatically.)
        ca.create_merged_summary_csv()

    yesterday = (pd.to_datetime('now') - pd.Timedelta(1, 'd')).strftime('%Y-%m-%d')
    cdf = ca.load_merged_summary(yesterday, '2099-01-01')
    fdate = cdf.iloc[-1]['Date_file']


    if 0:
        # TEST - Behavior for other cutoff dates.
        cdf = ca.load_merged_summary(*(('2021-05-01',)*2))  # TEST
        fdate = cdf.iloc[-1]['Date_file']



    # Dataframe with DON/DOO/DPL counts for most recent file date and all
    # Date_statistics since the July 2020, ignore last 3 days
    cdf = cdf.loc[cdf['Date_file'] == fdate]
    cdf = cdf.loc[cdf['Date_statistics'] >= '2020-07-01'].iloc[:-3].copy()
    cdf.drop(columns='Date_file', inplace=True)

    cdf['Date_statistics'] += pd.Timedelta(12, 'h') # Set timestamps at noon

    cdf.set_index('Date_statistics', inplace=True)




    # Apply smoothing to total cases (Dtot)
    # Smoothing parameters, various combinations (n, sigma)
    # Larger sigma = more agressive smoothing.
    sm_params = [
        (13, 2.1),
        (11, 2.1), # This works well
        (9, 2.1),
        (7, 2.8)
        ]

    sm_n, sm_sigma = sm_params[1]  # pick one

    cdf['Dsm'] = gauss_smooth(cdf['Dtot'].values, sm_n, sm_sigma)

    # Estimate R value 'JBF method'.
    Tgen = 4  # generation interval (integer days)
    Tdelay = 0  # additional time shift (integer days)

    Rs = cdf.iloc[Tgen:]['Dsm'].values / cdf.iloc[:-Tgen]['Dsm'].values
    cdf['Rt'] = np.nan
    idx_R = cdf.index[:-Tgen]
    if Tdelay != 0:
        idx_R = idx_R - pd.Timedelta(int(Tdelay), 'd')
        if Tdelay > 0:
            idx_R = idx_R[Tdelay:]
            Rs = Rs[Tdelay:]
        else:
            idx_R = idx_R[:Tdelay]
            Rs = Rs[:Tdelay]
    cdf.loc[idx_R, 'Rt'] = Rs

    # Data gets progressively unreliable for less than 14 days ago.
    # Cutoff at 10 days.
    cdf.loc[cdf.index >= fdate - pd.Timedelta(10, 'd'), 'Rt'] = np.nan


    # RIVM R series: index=Datum (12:00)
    R_rivm = nlcs.load_Rt_rivm(False)['R']
    R_jbf = pd.Series(cdf['Rt'].values, index=cdf.index)

    row_select = (cdf.index >= '2020-08-01')

    plt.close('all')
    fig, (axC, axR1, axR2, axDR) = plt.subplots(
        4, 1, figsize=(7, 9),
        tight_layout=True, sharex=True
        )

    xdates = cdf.index[row_select]  # Dates on x axis
    axC.set_title(f'Gauss \'smoothing\', n={sm_n}, σ={sm_sigma}')
    axC.semilogy(xdates, cdf.loc[row_select, 'Dtot'], label='Casus DOO+DPL+DON')
    axC.semilogy(xdates, cdf.loc[row_select, 'Dsm'], label='Casus Dxx, smooth')
    axC.legend()
    axC.set_ylabel('$N$ (Casussen/dag)')
    axC.grid('y', which='minor')
    # R graphs
    for axR in (axR1, axR2):
        axR.plot(R_jbf.loc[xdates], label='R (JBF)', color='r')
        axR.plot(R_rivm.loc[xdates], label='R (RIVM)', color='b',
                 linestyle=(0, (5, 1)))
        axR.set_ylabel('$R_t$')

    if Tdelay == 0:
        axR1.set_title(rf'$R_{{JBF}}(t) = N_{{sm}}(t+{Tgen})/N_{{sm}}(t)$')
    else:
        axR1.set_title(rf'$R_{{JBF}}(t) = N_{{sm}}(t+{Tdelay+Tgen})/N_{{sm}}(t+{Tdelay})$')

    axR1.legend()
    axR2.set_title('Ingezoomd')
    axR2.set_ylim(0.7, 1.2)

    axDR.set_title('Verschil $R_{JBF} - R_{RIVM}$')
    axDR.plot((R_jbf - R_rivm).loc[xdates])
    axDR.set_ylim(-0.1, 0.1)
    axDR.set_ylabel(r'$\Delta R$')

    for ax in fig.get_axes():
        set_xaxis_dateformat(ax)
    ax.set_xlabel('Datum')
    fig.show()

