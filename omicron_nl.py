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


def get_data_dk(plot=True, fig_ax=None):
    """Get DK data, optionally plot.

    Return:

    - df: DataFrame with index sdate (sample date), columns
      ntest, npes, pos%, or (odds ratio), or_std (odds ratio standard error),
      or_fit (exponential fit on or).
    - ga1: growth advantage per day.
    """

    # Sample_date, samples_total, samples_omicron, %omicron
    data_dk = """\
    23-11-2021 4,666 1 0.0
    24-11-2021 3,982 1 0.0
    25-11-2021 4,059 4 0.1
    26-11-2021 4,114 7 0.2
    27-11-2021 3,813 3 0.1
    28-11-2021 3,849 10 0.3
    29-11-2021 5,048 11 0.2
    30-11-2021 5,368 25 0.5
    01-12-2021 4,491 76 1.7
    02-12-2021 4,528 60 1.3
    03-12-2021 5,126 77 1.5
    04-12-2021 5,058 101 2.0
    05-12-2021 4,768 170 3.6
    06-12-2021 7,028 356 5.1
    07-12-2021 7,162 581 8.1
    08-12-2021 1,759 311 17.7
    """
    data_dk = data_dk.replace(',', '')
    records = [x.split() for x in data_dk.splitlines() if '2021' in x]
    df = pd.DataFrame.from_records(records, columns=['sdate', 'ntest', 'npos', 'pos%'])
    df['sdate'] = pd.to_datetime(df['sdate'], format='%d-%m-%Y')
    df.set_index('sdate', inplace=True)
    for c in df.columns:
        df[c] = df[c].astype(float)


    s_nneg = df['ntest'] - df['npos']
    df['or'] = df['npos'] / s_nneg
    df['or_std'] = df['or'] * np.sqrt(1/df['npos'] + 1/s_nneg)

    # linear regression of log
    tm_days = (df.index - df.index[0]) / pd.Timedelta('1 d')
    ga1, a0 = np.polyfit(tm_days, np.log(df['or']), w=1/df['or_std'], deg=1)
    df['or_fit'] = np.exp(a0 + ga1*tm_days)

    if plot:
        if fig_ax is None:
            fig, ax = plt.subplots(tight_layout=True, figsize=(6, 4))
        else:
            fig, ax = fig_ax
        ax.errorbar(df.index, df['or'], yerr=2*df['or_std'], ls='none', marker='x',
                    label='Sequencing result (95% CI)')
        ax.plot(df.index, df['or_fit'], color=PLT_COLORS[0],
                label=f'Fit: ln growth advantage {ga1:.2f} per day')
        ax.set_yscale('log')
        ax.set_ylabel(r'Odds ratio omicron/other')
        ax.set_xlabel('Sampling date')
        ax.legend(loc='upper left')
        ax.set_title('Denmark Omicron occurrences')
        #ax.set_xlim(None, df.index[-1] + 4*pd.Timedelta('1 d'))
        ax.set_ylim(None, np.exp(a0 + ga1*(tm_days[-1] + 4)))
        ax.axhline(1, color='k', linestyle='--')
        ax.text(df.index[-1], 1.3, '50% omicron', va='bottom', ha='right')

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


plt.close('all')


get_data_dk()
ax = plt.gcf().get_axes()[0]
ax.set_ylim(None, 5)
ax.set_xlim(None, pd.to_datetime('2021-12-17'))

fig, axs = plt.subplots(2, 1, figsize=(7, 6), tight_layout=True, sharex=True)

# DK data plot
ax = axs[0]
df, ga1 = get_data_dk(fig_ax=(fig, ax))


# Dutch growth advantage
ga1_nl = 0.3

ax.set_ylim(1e-4, 110)
ax.set_xlabel(None)
ax.set_title('Verhouding Omicron/Delta Denemarken en Nederland')




date_ref = pd.to_datetime('2021-12-12')
ncases_ref = 17000

nc0_d, nc0_o, k_d, k_o = estimate_cases_nd_no(
    ncases_ref, gf7=0.8, f_o=0.3, ga1=ga1_nl
    )


daynos = np.arange(-14, 14)
dates = (date_ref + pd.Series(daynos * pd.Timedelta('1 d'))).values

ncs_d = nc0_d * np.exp(daynos * k_d)
ncs_o = nc0_o * np.exp(daynos * k_o)

Tgen = 4.0
R_delta = np.exp(k_d * Tgen)
R_om = np.exp(k_o * Tgen)

ax.plot(dates, ncs_o/ncs_d, label='Nederland (model)', ls='--')
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



