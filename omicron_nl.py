#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 12 18:20:28 2021

Twitter: @hk_nien
"""


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tools

PLT_COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b',
              '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']



def add_or_stats(df):
    """Add columns or, or_std, or_fit, ga1 to df with ntest, npos, return update.

      Set datetime index from sdate

      or (odds ratio),
      or_std (odds ratio standard error),
      or_fit (exponential fit on or).
    - ga1: growth advantage per day (logistic k parameter).
    """
    df = df.copy()
    if 'sdate' in df.columns:
        df.set_index('sdate', inplace=True)

    s_nneg = df['ntest'] - df['npos']
    df['or'] = df['npos'] / s_nneg
    df['or_std'] = df['or'] * np.sqrt(1/df['npos'] + 1/s_nneg)

    # Remove days with zero. This is not entirely correct, but
    # too much work to do a real binomial fit.
    df = df.loc[df['or_std'].notna()].copy()

    # linear regression of log
    tm_days = (df.index - df.index[0]) / pd.Timedelta('1 d')
    ga1, a0 = np.polyfit(tm_days, np.log(df['or']), w=df['or']/df['or_std'], deg=1)
    df['or_fit'] = np.exp(a0 + ga1*tm_days)
    df['ga1'] = ga1

    return df


def get_data_dk(plot=True, fig_ax=None):
    """Get DK data, optionally plot.

    Return:

    - df: DataFrame with index sdate (sample date), columns including
      ntest, npos.

    Source: https://www.ssi.dk/
    # https://www.ssi.dk/-/media/cdn/files/covid19/omikron/statusrapport/rapport-omikronvarianten-18122021-wj25.pdf?la=da
    https://www.ssi.dk/-/media/cdn/files/covid19/omikron/statusrapport/rapport-omikronvarianten-21122021-14tk.pdf?la=da
    """

    # Sample_date, samples_total, samples_omicron, %omicron
    data_dk = """\
    22-11-2021 4,514 1 0.0% 0%-0.1%
    23-11-2021 4,717 1 0.0% 0%-0.1%
    24-11-2021 4,034 1 0.0% 0%-0.1%
    25-11-2021 4,105 2 0.0% 0%-0.2%
    26-11-2021 4,161 2 0.0% 0%-0.2%
    27-11-2021 3,853 2 0.1% 0%-0.2%
    28-11-2021 3,894 12 0.3% 0.2%-0.5%
    29-11-2021 5,096 11 0.2% 0.1%-0.4%
    30-11-2021 5,424 24 0.4% 0.3%-0.7%
    01-12-2021 4,552 74 1.6% 1.3%-2%
    02-12-2021 4,596 60 1.3% 1%-1.7%
    03-12-2021 5,174 72 1.4% 1.1%-1.8%
    04-12-2021 5,098 101 2.0% 1.6%-2.4%
    05-12-2021 4,808 171 3.6% 3.1%-4.1%
    06-12-2021 7,115 356 5.0% 4.5%-5.5%
    07-12-2021 7,339 569 7.8% 7.2%-8.4%
    08-12-2021 6,692 704 10.5% 9.8%-11.3%
    09-12-2021 6,637 752 11.3% 10.6%-12.1%
    10-12-2021 6,961 887 12.7% 12%-13.6%
    11-12-2021 6,701 1,096 16.4% 15.5%-17.3%
    12-12-2021 7,139 1,560 21.9% 20.9%-22.8%
    13-12-2021 10,580 3,046 28.8% 27.9%-29.7%
    14-12-2021 11,471 4,454 38.8% 37.9%-39.7%
    15-12-2021 11,273 5,106 45.3% 44.4%-46.2%
    """
    data_dk = data_dk.replace(',', '').replace('%', '')
    records = [x.split()[:4] for x in data_dk.splitlines() if '-202' in x]
    df = pd.DataFrame.from_records(records, columns=['sdate', 'ntest', 'npos', 'pos%'])
    df['sdate'] = pd.to_datetime(df['sdate'], format='%d-%m-%Y')
    df.set_index('sdate', inplace=True)
    for c in df.columns:
        df[c] = df[c].astype(float)
    df = add_or_stats(df)

    return df


def get_data_nl():
    """Return DataFrame.

    Source: https://twitter.com/ARGOSamsterdam/status/1473390513646669831


    """
    # old: Source: https://twitter.com/JosetteSchoenma/status/1471536542757883918
    txt_old = """\
        2021-12-06 2.5
        2021-12-13 14
        2021-12-14 25
        """

    txt = """\
        2021-12-12 4
        2021-12-13 11
        2021-12-14 14
        2021-12-15 25
        2021-12-16 18.5
        2021-12-17 21
        2021-12-18 35.5
        2021-12-19 35
        2021-12-20 49
        """
    records = [li.split() for li in txt.splitlines() if '202' in li]
    df = pd.DataFrame.from_records(records, columns=['sdate', 'pos%'])
    df['sdate'] = pd.to_datetime(df['sdate'])
    df.set_index('sdate')
    df['pos%'] = df['pos%'].astype(float)
    ntest = 100
    df['ntest'] = ntest
    df['npos'] = ntest/100 * df['pos%']
    df = add_or_stats(df)
    return df


def plot(dfdict, fig_ax=None):
    """Plot from dict region_name -> dataframe with (or, or_std, or_fit)."""

    if fig_ax is None:
        fig, ax = plt.subplots(tight_layout=True, figsize=(6, 4))
    else:
        fig, ax = fig_ax

    ax.set_yscale('log')
    ax.set_ylabel(r'Odds ratio omicron/other')
    ax.set_xlabel('Sampling date')
    pcycle = plt.rcParams['axes.prop_cycle']()


    for region_name, df in dfdict.items():
        ga1 = df['ga1'].iloc[-1]
        props = next(pcycle)
        label = f'{region_name} (k={ga1:.2f} per day)'
        yerr = 2*df['or_std']

        # big y error bars look ugly on log scale...
        yerr_big_mask = yerr >= df['or']
        yerr[yerr_big_mask] = df['or'][yerr_big_mask]*0.75

        ax.errorbar(df.index, df['or'], yerr=yerr, ls='none', marker='x',
                    label=label, **props)
        ax.plot(df.index, df['or_fit'],
                **props)


    ax.axhline(1, color='k', linestyle='--')
    ax.text(df.index[-1], 1.3, '50% omicron', va='bottom', ha='right')

    # ax.set_ylim(None, np.exp(a0 + ga1*(tm_days[-1] + 4)))

    ax.legend(loc='upper left')
    ax.set_title('Omicron/Delta ratios')
    tools.set_xaxis_dateformat(ax)
    fig.show()

    return df, ga1

def estimate_cases_nd_no(n_cases, gf7, f_o, ga1):
    """Estimate parameters from today's status.

    Parameters:

    - n_cases: number of positive cases on reference date.
    - gf7: growth factor compared to 7 days ago.
    - f_o: estimated fraction of omicron in cases today, 0 < f_o < 1.
    - ga1: growth advantage per day (ln)

    Return:

    - nc_d: number of cases delta
    - nc_o: number of cases omicron
    - k_d: log growth rate of delta
    - k_o: log growth rate of omicron
    """

    nc_o, nc_d = f_o * n_cases, (1 - f_o) * n_cases

    k_d = (1/7) * np.log(gf7 * (nc_d + nc_o*np.exp(-7*ga1)) / n_cases)
    k_o = k_d + ga1

    return nc_d, nc_o, k_d, k_o


if __name__ == '__main__':
    plt.close('all')

    dfdict = {
        'Denmark': get_data_dk(),
        'Netherlands': get_data_nl(),
        }

    plot(dfdict)
    ax = plt.gcf().get_axes()[0]
    ax.set_ylim(None, 10)
    ax.set_xlim(None, pd.to_datetime('2021-12-25'))
    ax.set_title('Omicron/Delta ratios - data source SSI, Argos - plot @hk_nien')

    #%%

    fig, axs = plt.subplots(2, 1, figsize=(7, 6), tight_layout=True, sharex=True)

    # DK data plot
    ax = axs[0]

    # Dutch growth advantage
    ga1_nl = 0.3


    plot(dfdict, fig_ax=(fig, axs[0]))


    ax.set_ylim(1e-4, 110)
    ax.set_xlabel(None)
    ax.set_title('Verhouding Omicron/Delta Denemarken en Nederland')



    # Here are the model parameters
    date_ref = pd.to_datetime('2021-12-18')
    ncases_ref = 15000
    daynos = np.arange(-21, 7)
    nc0_d, nc0_o, k_d, k_o = estimate_cases_nd_no(
        ncases_ref, gf7=0.9, f_o=0.9, ga1=ga1_nl
        )



    dates = (date_ref + pd.Series(daynos * pd.Timedelta('1 d'))).values

    ncs_d = nc0_d * np.exp(daynos * k_d)
    ncs_o = nc0_o * np.exp(daynos * k_o)

    Tgen = 4.0
    R_delta = np.exp(k_d * Tgen)
    R_om = np.exp(k_o * Tgen)

    ax.plot(dates, ncs_o/ncs_d, label='Nederland (model)', ls='--',
            color=PLT_COLORS[len(dfdict)]
            )
    ax.legend()

    #for ax in axs:
    #    ax.set_xlim(*dates[[0, -1]])
    ax = axs[1]
    i0 = 7 # skip old NL data on delta
    ax.plot(dates[i0:], (ncs_d+ncs_o)[i0:], label='Totaal')
    ax.scatter([date_ref], [ncases_ref])
    ax.plot(dates[i0:], ncs_d[i0:], label=f'Delta (R={R_delta:.2f})', ls='-.')
    ax.plot(dates, ncs_o, label=f'Omicron (R={R_om:.2f})', ls=':')
    ax.set_ylim(100, 1e5)
    ax.legend()
    ax.set_yscale('log')
    ax.set_title('Cases per dag Nederland (model)')
    ax.set_xlabel('Datum monstername')
    ax.text(pd.to_datetime('2021-11-25'), 200, 'RIVM:\ngeen Omicron', ha='center', va='bottom')


    tools.set_xaxis_dateformat(ax)



