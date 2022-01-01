#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 12 18:20:28 2021

Twitter: @hk_nien
"""

import re
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import calc_R_from_ggd_tests as ggd_R
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

    # binomial distribution standard deviation
    df['or_std'] = np.sqrt(df['ntest'] * df['npos']) * s_nneg**(-1.5)

    # tweak to get nonzero errors for npos=0
    df['or_std'] = np.sqrt(df['or_std']**2 + (0.5/df['ntest'])**2)

    # Remove days with zero. This is not entirely correct, but
    # too much work to do a real binomial fit.
    df = df.loc[df['or'] > 0]
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

    # Sample_date, ...., %omicron CI-95
    data_dk = """\
    21-11-2021 3,693 0 2 130 1.5% 0.2%-5.5%
    22-11-2021 4,514 0 0 197 0.0% 0%-1.9%
    23-11-2021 4,718 0 0 231 0.0% 0%-1.6%
    24-11-2021 4,034 0 0 3,505 0.0% 0%-0.1%
    25-11-2021 4,106 0 0 3,658 0.0% 0%-0.1%
    26-11-2021 4,164 1 0 3,703 0.0% 0%-0.1%
    27-11-2021 3,853 0 2 3,637 0.1% 0%-0.2%
    28-11-2021 3,892 0 9 3,581 0.3% 0.1%-0.5%
    29-11-2021 5,098 0 11 4,534 0.2% 0.1%-0.4%
    30-11-2021 5,426 2 24 4,887 0.5% 0.3%-0.7%
    01-12-2021 4,554 0 76 4,023 1.9% 1.5%-2.4%
    02-12-2021 4,595 2 57 4,013 1.4% 1.1%-1.8%
    03-12-2021 5,182 6 74 4,626 1.6% 1.3%-2%
    04-12-2021 5,107 1 107 4,730 2.3% 1.9%-2.7%
    05-12-2021 4,809 8 158 4,467 3.5% 3%-4.1%
    06-12-2021 7,117 17 325 6,416 5.1% 4.5%-5.6%
    07-12-2021 7,354 74 502 6,558 7.7% 7%-8.3%
    08-12-2021 6,696 75 632 5,921 10.7% 9.9%-11.5%
    09-12-2021 6,669 95 689 5,904 11.7% 10.9%-12.5%
    10-12-2021 6,961 84 802 6,060 13.2% 12.4%-14.1%
    11-12-2021 6,716 55 1,053 6,097 17.3% 16.3%-18.2%
    12-12-2021 7,168 72 1,484 6,371 23.3% 22.3%-24.4%
    13-12-2021 10,625 280 2,626 8,873 29.6% 28.7%-30.6%
    14-12-2021 11,539 420 3,963 9,981 39.7% 38.7%-40.7%
    15-12-2021 11,235 421 4,658 9,860 47.2% 46.3%-48.2%
    16-12-2021 10,603 471 4,273 8,952 47.7% 46.7%-48.8%
    17-12-2021 11,075 402 4,999 9,311 53.7% 52.7%-54.7%
    18-12-2021 10,465 178 4,889 8,559 57.1% 56.1%-58.2%
    19-12-2021 10,640 113 4,625 7,343 63,0% 61.9%-64.1%
    20-12-2021 13,954 149 1,633 2,548 64.1% 62.2%-66%
    21-12-2021 13,722 101 2,366 3,096 76.4% 74.9%-77.9%
    22-12-2021 12,299 90 1,022 1,292 79.1% 76.8%-81.3%
    23-12-2021 13,298 63 2,587 3,259 79.4% 78%-80.8%
    24-12-2021 7,428 62 505 602 83.9% 80.7%-86.7%
    25-12-2021 8,264 91 677 859 78.8% 75.9%-81.5%
    """
    data_dk = data_dk.replace(',', '').replace('%', '')
    records = [x.split()[:7] for x in data_dk.splitlines() if '-202' in x]
    df = pd.DataFrame.from_records(
        records,
        columns=['sdate', 'ncases', 'np_kda', 'npos', 'ntest', 'pos%', 'pos_ci']
        )
    df['sdate'] = pd.to_datetime(df['sdate'], format='%d-%m-%Y')
    df.set_index('sdate', inplace=True)
    for c in df.columns[:-1]:
        df[c] = df[c].astype(float)
    df = add_or_stats(df)

    return df


def get_data_ams_nl():
    """Return DataFrame.

    Source: https://twitter.com/ARGOSamsterdam


    """
    # old: Source: https://twitter.com/JosetteSchoenma/status/1471536542757883918
    txt_old = """\
        2021-12-06 2.5
        2021-12-13 14
        2021-12-14 25
        """

    txt = """\
        2021-12-12 100 4
        2021-12-13 100 11
        2021-12-14 100 14
        2021-12-15 100 25
        2021-12-16 100 18.5
        2021-12-17 100 21
        2021-12-18 100 35.5
        2021-12-19 100 35
        2021-12-20 95 46
        2021-12-21 106 63
        2021-12-22 89 56
        2021-12-23 79 46
        2021-12-24 93 60
        2021-12-27 94 72
        2021-12-28 92 74
        2021-12-29 87 73
        2021-12-30 87 71
        """
    records = [li.split() for li in txt.splitlines() if '202' in li]
    df = pd.DataFrame.from_records(records, columns=['sdate', 'ntest', 'npos'])
    for c in 'ntest', 'npos':
        df[c] = df[c].astype(float)
    df['sdate'] = pd.to_datetime(df['sdate'])
    df.set_index('sdate')
    df['pos%'] = 100 * (df['npos'] / df['ntest'])
    df = add_or_stats(df)
    return df


def _get_data_synsal(txt, date_shift=0):
    """Parse data from https://www.rivm.nl/coronavirus-covid-19/virus/varianten/omikronvariant"""

    txt = re.sub(r'[ \t]+', ' ', txt).replace('%', '').replace(',', '.')
    txt = re.sub(r'(\d+)-dec', r'2021-12-\1', txt)
    records = [li.split() for li in txt.splitlines() if '202' in li]
    df = pd.DataFrame.from_records(records, columns=['sdate', 'ntest', 'npos', 'pos%'])
    for c in 'ntest', 'npos', 'pos%':
        df[c] = df[c].astype(float)
    df['sdate'] = pd.to_datetime(df['sdate']) + pd.Timedelta(date_shift, 'd')
    df.set_index('sdate')
    df['pos%'] = 100 * (df['npos'] / df['ntest'])
    df = add_or_stats(df)
    return df


def get_data_synlab():
    """Source: https://www.rivm.nl/coronavirus-covid-19/virus/varianten/omikronvariant"""

    txt="""\
    1-dec 	2622 	  	1 	0.0%
    2-dec 	3345 	  	0  	0.0%
    3-dec 	3500 	  	1 	0.0%
    4-dec 	3921 	  	5 	0.1%
    5-dec 	3665 	  	3 	0.1%
    6-dec 	2635 	  	5 	0.2%
    7-dec 	3669 	  	5 	0.1%
    8-dec 	2941 	  	13 	0.4%
    9-dec 	3155 	  	11 	0.3%
    10-dec 	2724 	  	23 	0.8%
    11-dec 	2563 	  	18 	0.7%
    12-dec 	1868 	  	22 	1.2%
    13-dec 	1940 	  	23 	1.2%
    14-dec 	2667 	  	72 	2.7%
    15-dec 	2137 	  	74 	3.5%
    16-dec 	2513 	  	161 	6.4%
    17-dec 	1027 	  	81 	7.9%
    18-dec 	1989 	  	195 	9.8%
    19-dec 	1284 	  	135 	10.5%
    20-dec 	1660 	  	237 	14,3%
    21-dec 	1545 	  	303 	19,6%
    22-dec 	1607 	  	408 	25,4%
    23-dec 	1809 	  	517 	28,6%
    24-dec 	1875 	  	605 	32,3%
    25-dec 	1816 	  	645 	35,5%
    26-dec 	1807 	  	781 	43,2%
    27-dec 	1288 	  	574 	44,6%
    28-dec 	1963 	  	1020 	52,0%
    29-dec 	1893 	  	1112 	58,7%
    30-dec 	1923 	  	1267 	65,9%
    31-dec 	1743 	  	1179 	67,6%
    """
    return _get_data_synsal(txt, -2)


def get_data_saltro():
    """Source: https://www.rivm.nl/coronavirus-covid-19/virus/varianten/omikronvariant"""

    txt="""\
    1-dec 	885 	0 	0.0%
    2-dec 	916 	1 	0.1%
    3-dec 	888 	4 	0.5%
    4-dec 	1121 	3 	0.3%
    5-dec 	514 	2 	0.4%
    6-dec 	798 	2 	0.3%
    7-dec 	852 	2 	0.2%
    8-dec 	811 	1 	0.1%
    9-dec 	1041 	0 	0.0%
    10-dec 	821 	3 	0.4%
    11-dec 	553 	7 	1.3%
    12-dec 	656 	6 	0.9%
    13-dec 	263 	0 	0.0%
    14-dec 	168 	5 	3.0%
    15-dec 	775 	30 	3.9%
    16-dec 	964 	45 	4.7%
    17-dec 	922 	64 	6.9%
    18-dec 	751 	82 	10.9%
    19-dec 	777 	92 	11.8%
    20-dec 	574 	75 	13,1%
    21-dec 	1008 	176 	17,5%
    22-dec 	1171 	297 	25,4%
    23-dec 	1673 	491 	29,3%
    24-dec 	1350 	422 	31,3%
    25-dec 	1152 	455 	39,5%
    26-dec 	796 	343 	43,1%
    27-dec 	1125 	561 	49,9%
    28-dec 	1430 	800 	55,9%
    29-dec 	1318 	792 	60,1%
    30-dec 	1481 	939 	63,4%
    31-dec 	451 	292 	64,7%
    """
    return _get_data_synsal(txt, -2)

def get_data_nl():
    """Return DataFrame.

    Source: RIVM https://www.rivm.nl/coronavirus-covid-19/virus/varianten
    """

    txt = """\
        2021-11-25 1874 4
        2021-12-02 1864 8
        2021-12-09 1753 21
        2021-12-16 1004 88
        """
    records = [li.split() for li in txt.splitlines() if '202' in li]
    df = pd.DataFrame.from_records(records, columns=['sdate', 'ntest', 'npos'])
    for c in 'ntest', 'npos':
        df[c] = df[c].astype(float)
    df['sdate'] = pd.to_datetime(df['sdate'])
    df.set_index('sdate')
    df['pos%'] = 100 * (df['npos'] / df['ntest'])
    df = add_or_stats(df)
    return df


def get_data_all():
    """Return dict with name -> dataframe"""
    dfdict = {
        'Denmark': get_data_dk(),
        'Amsterdam': get_data_ams_nl(),
        'Synlab': get_data_synlab(),
        'Saltro': get_data_saltro(),
        'Nederland': get_data_nl(),
        }
    return dfdict


def plot_omicron_delta_odds(dfdict, fig_ax=None):
    """Plot from dict region_name -> dataframe with (or, or_std, or_fit)."""

    if fig_ax is None:
        fig, ax = plt.subplots(tight_layout=True, figsize=(6, 4))
    else:
        fig, ax = fig_ax

    ax.set_yscale('log')
    ax.set_ylabel(r'Odds ratio omicron/other')
    ax.set_xlabel('Sampling date')
    pcycle = plt.rcParams['axes.prop_cycle']()

    markers = 'xo<>'*3

    for i, (region_name, df) in enumerate(dfdict.items()):
        marker = markers[i]
        x_offs = pd.Timedelta((i % 3 - 1)*0.05, 'd')
        ga1 = df['ga1'].iloc[-1]
        props = next(pcycle)
        label = f'{region_name} (k={ga1:.2f} per day)'
        yerr = 2*df['or_std']

        # big y error bars look ugly on log scale...
        yerr_big_mask = yerr >= df['or']
        yerr[yerr_big_mask] = df['or'][yerr_big_mask]*0.75

        ax.errorbar(df.index+x_offs, df['or'], yerr=yerr, ls='none', marker=marker,
                    label=label, **props)
        ax.plot(df.index, df['or_fit'],
                **props)


    ax.axhline(1, color='k', linestyle='--')
    ax.text(
        ax.get_xlim()[0], 1.3,
        '50% omicron', va='bottom', ha='left'
        )

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



def run_scenario(date_ref, *, odds_ref, kga, cf=1.04, regions=None,
                 dayrange=(-21, 14), kgachange=None):
    """Run scenario with plot.

    Parameters:

    - date_ref: 'yyyy-mm-dd' sample date reference.
    - odds_ref: ratio omikron/delta on that day.
    - kga: growth advantage in 1/d units as float.
    - cf: correction factor on number of positive cases
      (data source is GGD data, which is incomplete).
    - regions: optional list of regions to plot. Default: all.
    - dayrange: range relative to reference date to simulate.
    - kgachange: optional (dayno, kga2) for a change in kga.
      (dayno > 0).
    """

    dfdict = get_data_all()
    if regions is not None:
        dfdict = {k:v for k, v in dfdict.items() if k in regions}

    fig, axs = plt.subplots(2, 1, figsize=(9, 7), tight_layout=True, sharex=True)

    # odds-ratio plot.
    ax = axs[0]

    plot_omicron_delta_odds(dfdict, fig_ax=(fig, axs[0]))

    ax.set_ylim(1e-4, 110)
    ax.set_xlabel(None)
    ax.set_title('Verhouding Omicron/Delta Denemarken en Nederland')

    # Here are the model parameters
    # ref date based on sampling date
    # GGD test date 2021-12-28: 13800, 2021-12-21: 11800
    # (smoothed values)
    # Add 4% for non-GGD data sources
    # Omikron on test date 2021-12-28: ratio 1.7
    df_ggd = ggd_R.load_ggd_pos_tests()  # index at date_tested
    df_ggd.index -= pd.Timedelta('12 h')  # make sure that index is at 0:00:00.
    df_ggd.loc[df_ggd.index[-2:], 'n_pos_7'] = np.nan  # these are extrapolated values

    date_ref = pd.to_datetime(date_ref)
    ncases_ref = df_ggd.loc[date_ref, 'n_pos_7'] * cf
    ncases_prev = df_ggd.loc[date_ref-pd.Timedelta('7 d'), 'n_pos_7']*cf
    f_omicron = odds_ref / (1+odds_ref)

    daynos = np.arange(*dayrange)
    nc0_d, nc0_o, k_d, k_o = estimate_cases_nd_no(
        ncases_ref, gf7=ncases_ref/ncases_prev, f_o=f_omicron, ga1=kga
        )

    dates = (date_ref + pd.Series(daynos * pd.Timedelta('1 d'))).values

    ncs_d = nc0_d * np.exp(daynos * k_d)
    ncs_o = nc0_o * np.exp(daynos * k_o)

    Tgen = 4.0
    R_delta = np.exp(k_d * Tgen)
    R_om = np.exp(k_o * Tgen)

    if kgachange is not None:
        dn_change, kga2 = kgachange
        i0 = np.argmax(daynos == dn_change)
        m = len(daynos) - i0
        k2_o = k_d + kga2
        ncs_o[i0:] = ncs_o[i0]*np.exp(np.arange(m)*k2_o)  # may be off-by-one error
        R2_om = np.exp(k2_o * Tgen)

    ax.plot(
        dates, ncs_o/ncs_d,
        label=f'Nederland (model: k={kga:.2f})',
        ls='--',
        color=PLT_COLORS[len(dfdict)]
        )

    ax.legend()

    #for ax in axs:
    #    ax.set_xlim(*dates[[0, -1]])
    ax = axs[1]
    i0 = 7 # skip old NL data on delta
    ax.plot(dates[i0:], (ncs_d+ncs_o)[i0:], label='Totaal')
    ax.scatter(
        [date_ref-pd.Timedelta('7 d'), date_ref],
        [ncases_prev, ncases_ref],
        label='IJkpunnten'
        )
    ax.plot(dates[i0:], ncs_d[i0:], label=f'Delta (R={R_delta:.2f})', ls='-.')

    if kgachange is None:
        label = f'Omicron (R={R_om:.2f})'
    else:
        label = f'Omicron (R={R_om:.2f} → {R2_om:.2f}'
    ax.plot(dates, ncs_o, label=label, ls=':')


    # add ggd data
    ax.set_xlim(*ax.get_xlim())
    select = (
        (df_ggd.index > dates[0]) & (df_ggd.index <= dates[-1])
        )
    ax.plot(
        df_ggd.index[select], df_ggd.loc[select, 'n_pos_7']*cf,
        label=f'GGD positief x {cf}, 7d-gemiddelde',
        linestyle='--'
        )

    ax.set_ylim(1000, 1e5)
    ax.legend()
    ax.set_yscale('log')
    ax.set_title(
        f'Cases per dag Nederland (model) / Rekenwaarde generatie-interval {Tgen:.0f} d'
        )
    ax.set_xlabel('Datum monstername')
    # ax.text(pd.to_datetime('2021-11-25'), 200, 'RIVM:\ngeen Omicron', ha='center', va='bottom')

    ax.grid(axis='y', which='minor')
    tools.set_xaxis_dateformat(ax)



if __name__ == '__main__':
    plt.close('all')
    dfdict = get_data_all()
    if 0:
        plot_omicron_delta_odds(dfdict)
        ax = plt.gcf().get_axes()[0]
        ax.set_ylim(None, 20)
        ax.set_xlim(None, pd.to_datetime('2022-01-05'))
        ax.set_title('Omicron/Delta ratios - data source SSI, Argos, RIVM - plot @hk_nien')
        ax.legend(loc='lower right')

    plt.close('all')

    run_scenario('2021-12-28', odds_ref=1.7, kga=0.2)
    run_scenario('2021-12-28', odds_ref=1.7, kga=0.23, kgachange=(4, 0.10))


