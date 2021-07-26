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


def smoothen(data, kernel):
    """Convolve data with odd-size kernel, with boundary handling."""

    n, = kernel.shape
    assert n % 2 == 1
    m = (n-1) // 2
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


def gauss_smooth(data, n=11, sigma=1.67, mu=0.0):
    """Apply gaussian smoothing kernel on data array.

    Parameters:

    - n: window size (should be odd)
    - sigma: standard deviation of the window.
    - mu: center

    Return:

    - smoothed data, same size as input.
    """
    assert n%2 == 1
    m = (n-1) // 2
    xs = np.arange(-m, m+1)
    kernel = np.exp(-0.5/sigma**2 * (xs-mu)**2)
    kernel /= kernel.sum()

    smooth = smoothen(data, kernel)
    return smooth


def calc_Rjbf(cdf, Tgen=4, Tdelay=0, sm_preset='g11', update_cdf=True, dt_dpl=0):
    """Calculate R_jbf:

    Parameters:

    - cdf: casus DataFrame with index Date_statistics, columns
      DON, DPL, DOO.
    - Tgen: generation interval (integer days)
    - Tdelay: additional time shift (integer days)
    - sm_preset: smoothing-parameter preset (g11, g9, custom11, etc.).
      See source code.
    - update_cdf: True to add columns 'Dsm' and 'Rt' to cdf.
    - dt_dpl: time offset (int days) of DPL data

    Return:

    - Dsm: smoothed daily cases Series - same index as cdf.
    - Rt: reproduction number as Series - same index.
    - sm_desc: Smoothing description str
    """

    # Apply smoothing to total cases (Dtot)
    # Smoothing parameters, various combinations (n, sigma)
    # Larger sigma = more agressive smoothing.
    sm_presets = {
        # Gaussian kernels n, sigma, mu
        'g13': (13, 2.1, 0),
        'g11': (11, 2.1, 0), # This works well
        'g9': (9, 2.1, 0),
        'g7': (7, 2.8, 0),
        'g11offs': (11, 2.1, -0.3),
        # Custom kernelsf'Gaussian, n={sm_n}, σ={sm_sigma:.3g}'
        'custom11': np.array(
            [0.00210191, 0.03323479, 0.02142772, 0.07180504, 0.14321991,
             0.18248858, 0.19054455, 0.17576429, 0.09342508, 0.05432827,
             0.02243447, 0.00861231, 0.00061307])

        }
    sm_preset = sm_presets[sm_preset]
    fdate = cdf.index[-1]

    Dtot = cdf['DOO'] + cdf['DON']
    dpl = cdf['DPL']
    dpl = dpl.shift(
        dt_dpl,
        fill_value=dpl.iloc[0 if dt_dpl>0 else -1]
        )
    Dtot += dpl

    if isinstance(sm_preset, tuple):
        sm_n, sm_sigma, sm_mu = sm_preset
        Dsm = gauss_smooth(Dtot.values, sm_n, sm_sigma, sm_mu)
        sm_desc = f'Gaussian, n={sm_n}, σ={sm_sigma:.3g}, μ={sm_mu:.3g}'
    else:
        Dsm = smoothen(Dtot.values, sm_preset)
        sm_desc = f'Speciale kernel, n={len(sm_preset)}'
    Dsm = pd.Series(Dsm, index=cdf.index)

    # Estimate R value 'JBF method'.

    Rs = Dsm.iloc[Tgen:].values / Dsm.iloc[:-Tgen].values
    idx_R = cdf.index[:-Tgen]
    if Tdelay != 0:
        idx_R = idx_R - pd.Timedelta(int(Tdelay), 'd')
        if Tdelay > 0:
            idx_R = idx_R[Tdelay:]
            Rs = Rs[Tdelay:]
        else:
            idx_R = idx_R[:Tdelay]
            Rs = Rs[:Tdelay]


    Rt_series = pd.Series(np.nan, index=cdf.index)
    Rt_series.loc[idx_R] = Rs

    # Data gets progressively unreliable for less than 14 days ago.
    # Cutoff at 10 days.
    Rt_series.loc[cdf.index >= fdate - pd.Timedelta(10, 'd')] = np.nan

    if update_cdf:
        cdf['Dsm'] = Dsm
        cdf['Rt'] = Rt_series

    return Dsm, Rt_series, sm_desc

def get_cdf(fdate=None):
    """Return casus DataFrame for 1 day (yyyy-mm-dd, defaulrt most recent).

    Index as datetime at 12:00.
    """

    if fdate:
        cdf = ca.load_merged_summary(fdate, fdate, reprocess=False)
    else:
        # most recent
        yesterday = (pd.to_datetime('now') - pd.Timedelta(1, 'd')).strftime('%Y-%m-%d')
        cdf = ca.load_merged_summary(yesterday, '2099-01-01', reprocess=False)

    fdate = cdf.iloc[-1]['Date_file']


    # Dataframe with DON/DOO/DPL counts for most recent file date and all
    # Date_statistics since the July 2020, ignore last 3 daysTgen=4, Tdelay=0,Tgen=4, Tdelay=0,
    cdf = cdf.loc[cdf['Date_file'] == fdate]
    cdf = cdf.loc[cdf['Date_statistics'] >= '2020-07-01'].iloc[:-3].copy()
    cdf.drop(columns='Date_file', inplace=True)

    cdf['Date_statistics'] += pd.Timedelta(12, 'h') # Set timestamps at noon

    cdf.set_index('Date_statistics', inplace=True)

    return cdf

def infer_smoothing_kernel_rivm(ts_lo='2020-08-08', ts_hi='2021-06-15',
                                nk=13, plot=True):
    """Find best smoothing kernel for RIVM data.

    Parameters:

    - ts_lo: date_statistics low
    - ts_hi: date_statistics high
    - nk: kernel size (odd) to fit.
    - plot: True to generate a plot.

    Return:

    - ker: kernel, array shape (nk,).
    """
    assert nk % 2 == 1
    Tgen = 4
    ts_lo, ts_hi = [pd.to_datetime(x) + pd.Timedelta(Tgen+0.5, 'd') for x in [ts_lo, ts_hi]]
    cdf = get_cdf()
    cdf['Dsm'] = gauss_smooth(cdf['Dtot'])
    cdf = cdf.loc[ts_lo:ts_hi]
    cdf['R_rivm'] = nlcs.load_Rt_rivm(False)['R']

    # Reconstruct Dsm counts by applying R_rivm repeatedly.
    cdf['Dreco'] = cdf['Dsm'].copy()
    for i in range(Tgen, len(cdf)):
        idxa, idxb = cdf.index[[i-Tgen, i]]
        nreco = cdf.at[idxa, 'Dreco'] * cdf.at[idxa, 'R_rivm']
        # This is to dampen out differences from rounding errors
        f = 0.25
        nreco = (1-f)*nreco + f*cdf.at[idxb, 'Dsm']
        cdf.loc[idxb, 'Dreco'] = nreco

    if plot:
        fig, ax = plt.subplots()
        for col in ['Dsm', 'Dreco']:
            ax.plot(cdf[col], label=col)
        ax.legend()
        set_xaxis_dateformat(ax)
        fig.show()

    # Construct smoothing kernel K
    # Solve y = X*k for k.
    n = len(cdf)
    m = (nk - 1)//2
    X = np.zeros((n-2*m, nk))
    y = cdf['Dreco'].values[m:-m]
    x = cdf['Dtot'].values
    for i in range(n - 2*m):
        X[i, :] = x[i:i+nk]

    # Regularize to equal sums on all rows, so that periods with low
    # and high case counts get the same weight.
    rfac = 1/X.sum(axis=1)
    X *= rfac.reshape(-1, 1)
    y *= rfac

    ker, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    ker /= ker.sum()

    if plot:
        # comparison: gaussian kernel
        a = np.zeros(11)
        a[5] = 1.0
        sigma, mu = 2.1, -0.3
        ker_gauss = gauss_smooth(a, n=11, sigma=sigma, mu=mu)

        fig, ax = plt.subplots()
        ax.stem(np.arange(-m, m+1), ker, markerfmt='o', use_line_collection=True,
                basefmt='none', label='Inferred kernel')
        markerline, stemlines, baseline  = ax.stem(
            np.arange(-5, 6)+0.1, ker_gauss, markerfmt='^', linefmt='--',
            use_line_collection=True,
            basefmt='none', label='Gauss'
            )
        stemlines.set_color('gray')
        markerline.set_color('gray')
        ax.axhline(0, color='black')
        ax.set_title(f'Inferred smoothing kernel and Gauss (n=11, σ={sigma}, μ={mu})')
        ax.legend()
        ax.grid()
        fig.show()

    return ker


def calc_plot_Rjbf(fdate=None, Tgen=4, Tdelay=0, sm_preset='g11', dt_dpl=0):
    """Calculate R_jbf and plot.

    Parameters:

    - fdate: casus file date ('yyyy-mm-dd'); default: most recent.
    - Tgen: generation interval (integer days)
    - Tdelay: additional time shift (integer days)
    - sm_preset: smoothing-parameter preset (int), see source code of calc_Rjbf.
    - dt_dpl: time offset (int days) of DPL data

    Return:

    - cdf: DataFrame with date index, columns a.o. Dtot, Dsm, Rt.
    """
    cdf = get_cdf(fdate)
    # This adds 'Dsm', 'Rt' columns to cdf.
    Tgen, Tdelay = 4, 0
    _, _, sm_desc = calc_Rjbf(
        cdf, Tgen, Tdelay, sm_preset=sm_preset, dt_dpl=0
        )

    # RIVM R series: index=Datum (12:00)
    R_rivm = nlcs.load_Rt_rivm(False)['R']
    R_jbf = pd.Series(cdf['Rt'].values, index=cdf.index)

    row_select = (cdf.index >= '2020-08-01')

    fig, (axC, axR1, axR2, axDR) = plt.subplots(
        4, 1, figsize=(7, 9),
        tight_layout=True, sharex=True
        )

    xdates = cdf.index[row_select]  # Dates on x axis
    axC.set_title(f'Smoothing: {sm_desc}')
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

    return cdf


if __name__ == '__main__':

    if 0:
        # Run this manually, interactively to refresh data.
        # (Slow, don't do this automatically.)
        ca.create_merged_summary_csv()

    plt.close('all')
    infer_smoothing_kernel_rivm()
    calc_plot_Rjbf('2021-06-15')
    calc_plot_Rjbf('2021-06-15', sm_preset='custom11')
    calc_plot_Rjbf('2021-06-15', sm_preset='g11offs')
