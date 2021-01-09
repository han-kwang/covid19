#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Simulate case numbers and effective R for mix of old/new strains

Created on Fri Jan  8 18:01:37 2021

@author: @hk_nien
"""

import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.signal
import tools
import nlcovidstats as nlcs

def simulate_cases(date_ra=('2020-12-18', '2021-02-15'),
                   Rs=(0.88, 0.88*1.6), f0=0.01, Tg = 4.0,
                   use_r7=True, report_delay = 7.0, n0=10000):
    """Simulate case/infection numbers for old/B117 mix.

    Parameters:

    - date_ra: (date_start, date_end)
    - Rs: (R_old, R_B117)
    - Tg: generation interval (days)
    - f0: fraction of B117 infections at date_start
    - use_r7: whether to work with rolling 7-day average
    - report_delay: time (d) from infection to positive test result
    - n0: number of infections at date_start

    Return: DataFrame with:

    - index: Date
    - ni_old: infections per day (old variant)
    - ni_b117: infections per day (B1.1.7)
    - npos: positive cases per day
    - f_b117: fraction of B117 positive cases per day
    - Rt: reproduction number (at moment of infection)
    """
    date_ra = pd.to_datetime(date_ra)
    dates = pd.date_range(date_ra[0], date_ra[1], freq='1 d')
    npad = 4 # days of padding at begin and end
    ts = np.arange(-npad, len(dates)+npad)
    Rs = np.array(Rs).reshape(2)
    ks = np.log(Rs) / Tg # growth constants per day
    k_ratio = ks[1] - ks[0]
    print(f'Log odds growth rate: {k_ratio:.3f} /day')

    # number of infections at t=0
    ni0 = np.array([1-f0, f0])*n0
    nis = ni0.reshape(2, 1) * np.exp(ks.reshape(2, 1) * ts)

    if use_r7:
        nis = scipy.signal.convolve(nis, np.full((1, 7), 1/7), mode='same')

    # log growth rate
    n_tot = nis.sum(axis=0)
    dln_dt = ((n_tot[2:] - n_tot[:-2])/(2*n_tot[1:-1]))[npad-1:1-npad]
    R_eff = np.exp(Tg*dln_dt)

    nis = nis[:, npad:-npad]
    n_tot = n_tot[npad:-npad]

    df1 = pd.DataFrame(
        dict(Date=dates, ni_old=nis[0, :], ni_b117=nis[1, :], Rt=R_eff)
        )
    df1 = df1.set_index('Date')
    df2 = pd.DataFrame(dict(Date=dates+pd.Timedelta(report_delay, 'd'),
                            npos=n_tot,
                            f_b117=nis[1, :]/n_tot))
    df2 = df2.set_index('Date')
    df = df1.merge(df2, how='outer', left_index=True, right_index=True)

    return df

def get_Rt_cases(delay=7, Tc=4):
    """Get smoothed Rt from case stats and cases stats (r7 smooth)

    Return:

    - Dataframe with date index, columns Rt, Rlo, Rhi.
    """
    df1, _npop = nlcs.get_region_data('Nederland', -1)
    source_col = 'Delta7r'

    # skip the first 10 days because of zeros
    Rt, delay_str = nlcs.estimate_Rt_series(df1[source_col].iloc[10:], delay=delay, Tc=Tc)
    Rt = Rt.iloc[-200:]
    Rt_smooth = scipy.signal.savgol_filter(Rt.iloc[:-2], 13, 2)[:-1]
    Rt_smooth = pd.Series(Rt_smooth, index=Rt.index[:-3])
    Rt_err = np.full(len(Rt_smooth), 0.05)
    Rt_err[-4:] *= np.linspace(1, 1.4, 4)
    Rt = Rt.iloc[:-3]

    df = pd.DataFrame(dict(Rt=Rt_smooth, Rlo=Rt_smooth-Rt_err, Rhi=Rt_smooth+Rt_err))
    return df, df1

def _get_data_dk():

    # https://covid19.ssi.dk/-/media/cdn/files/opdaterede-data-paa-ny-engelsk-virusvariant-sarscov2-cluster-b117--01012021.pdf?la=da

    dk_data = [
        # Year-week-da, n_pos, f_b117
        ('2020-W49-4', 12663, 0.002),
        ('2020-W50-4', 21710, 0.005),
        ('2020-W51-4', 24302, 0.009),
        ('2020-W52-4', 15143, 0.023)
        ]
    # Convert week numbers to date (Wednesday of the week)
    twelvehours = pd.Timedelta('12 h')
    dk_data = [
        dict(Date=datetime.datetime.strptime(f'{r[0]}', "%Y-W%W-%w") + twelvehours,
             n_pos=r[1], f_b117=r[2])
        for r in dk_data
        ]
    df = pd.DataFrame.from_records(dk_data).set_index('Date')
    return dict(Denemarken=df)

def _get_data_uk():

    #https://twitter.com/hk_nien/status/1344937884898488322
    # data points read from plot (as ln prevalence)
    seedata = {
        '2020-12-21': [
            ['2020-09-25', -4.2*1.25],
            ['2020-10-02', -3.5*1.25],
            ['2020-10-15', -3.2*1.25],
            ['2020-10-20', -2.3*1.25],
            ['2020-10-29', -2.3*1.25],
            ['2020-11-05', -1.5*1.25],
            ['2020-11-12', -0.9*1.25],
            ['2020-11-19', -0.15*1.25],
            ['2020-11-27', 0.8*1.25]
            ],
        '2020-12-31': [
            ['2020-10-31', -2.1],
            ['2020-11-08', -1.35],
            ['2020-11-15', -0.75],
            ['2020-11-22', -0.05],
            ['2020-11-29', 0.05],
            ]
        }

    cdict = {}
    for report_date, records in seedata.items():
        df = pd.DataFrame.from_records(records, columns=['Date', 'ln_odds'])
        df['Date'] = pd.to_datetime(df['Date'])
        odds = np.exp(df['ln_odds'])
        df['f_b117'] = odds / (1 + odds)
        df = df[['Date', 'f_b117']].set_index('Date')
        cdict[f'SE England ({report_date})'] = df

    return cdict

def _get_data_ch():
    """Confirmed B117 cases, but not as a fraction."""

    # https://twitter.com/b117science/status/1347503372719558656

    cumdata = [
        ('2020-12-23', 0),
        ('2020-12-24', 3),
        ('2020-12-29', 5),
        ('2021-01-05', 29),
        ('2021-01-06', 36),
        ('2021-01-07', 46),
        ('2021-01-08', 86),
        ]

    dates = pd.to_datetime([d for d, _ in cumdata])
    ncum = np.array([n for _, n in cumdata])
    deltas = ncum[1:] - ncum[:-1]
    delta_ts = np.array((dates[1:] - dates[:-1]).days) # in days
    dates_mid = dates[:-1] + delta_ts * pd.Timedelta('1 d')
    n_per_day = deltas / delta_ts

    df2 = pd.DataFrame(
        dict(Date=dates_mid, n_b117=n_per_day)).set_index('Date')

#%%

def get_data_countries():
    """Return dict, key=country_name, value=dataframe with Date, n_pos, f_b117."""

    cdict = _get_data_dk()
    cdict = { **cdict, **_get_data_uk() }

    return cdict


def simulate_and_plot(Rs, f0, title_prefix=''):
    """Simulate and plot, given R and initial prevelance."""
    df =  simulate_cases(f0=f0, Rs=Rs, use_r7=True)
    df_R, dfc = get_Rt_cases()
    df_R = df_R.loc[df_R.index >= '2020-12-18']
    dfc = dfc.loc[dfc.index >= '2020-12-25'] # cases

    colors = plt.rcParams['axes.prop_cycle']()
    colors = [next(colors)['color'] for _ in range(10)]

    from matplotlib.ticker import LogFormatter, FormatStrFormatter


    fig, axs = plt.subplots(3, 1, tight_layout=True, sharex=True, figsize=(9, 9))
    ax = axs[0]
    ax.set_ylabel('Aantal per dag')
    ax.semilogy(df['ni_old'], label='Infecties oude variant (simulatie)',
                color=colors[0], linestyle='-.')
    ax.semilogy(df['ni_b117'], label='Infecties B117 variant (simulatie)',
                color=colors[0], linestyle='--')
    ax.semilogy(df['ni_old'] + df['ni_b117'], label='Infecties totaal (simulatie)',
                color=colors[0], linestyle='-', linewidth=2)
    ax.semilogy(df['npos'], label='Positieve tests (simulatie)',
                color=colors[1], linestyle='-.')
    ax.semilogy(dfc['Delta7r']*17.4e6, label='Positieve tests (NL)',
                color=colors[2], linestyle='--', linewidth=3)
    ax.set_ylim(df['npos'].min()/5, 20000)
    ax.text(pd.to_datetime('2020-12-15'), df['npos'].min()/4.5, 'Lockdown', rotation=90,
        horizontalalignment='center')
    ax.grid()
    ax.grid(which='minor', axis='y')
    ax.legend(loc='lower left')

    ax = axs[1]
    ax.set_ylabel('R')
    ax.plot(df['Rt'], label='$R_t$ (simulatie)',
            color=colors[1])
    ax.fill_between(df_R.index, df_R['Rlo'], df_R['Rhi'],
                    color='#0000ff', alpha=0.15, label='$R_t$ (observatie NL)')
    ax.text(pd.to_datetime('2020-12-15'), 0.82, 'Lockdown', rotation=90,
            horizontalalignment='center')
    ax.grid()
    ax.legend()

    ax = axs[2]
    ax.set_ylabel('Aandeel B117 pos. tests (%)')
    ax.semilogy(df['f_b117']*100, label='Nederland (simulatie)',
                color=colors[1])

    for country_name, df in get_data_countries().items():
        ax.plot(df.index, df['f_b117']*100, 'o:', linewidth=2, label=country_name)

    ax.grid(which='both', axis='y')
    ax.legend()

    ax.yaxis.set_major_formatter(FormatStrFormatter('%g'))
    # Monkey-patch to prevent '%e' formatting.
    LogFormatter._num_to_string = lambda _0, x, _1, _2: ('%g' % x)
    ax.yaxis.set_minor_formatter(LogFormatter(minor_thresholds=(2, 1)))
    title = f'{title_prefix}R_oud={Rs[0]:.2f};  R_B117={Rs[1]:.2f}'
    axs[0].set_title(title)

    for i in range(3):
        tools.set_xaxis_dateformat(axs[i], maxticks=15, ticklabels=(i==2))
    fig.show()


if __name__ == '__main__':

    plt.close('all')
    nlcs.init_data()

    simulate_and_plot(Rs=(0.85, 0.85*1.6), f0=0.01, title_prefix='Scenario 1: ')
    simulate_and_plot(Rs=(0.8, 0.8*1.6), f0=0.15, title_prefix='Scenario 2 (horror): ')
    # simulate_and_plot(Rs=(0.77, 0.77*1.6), f0=0.2, title_prefix='Scenario 3 (horror): ')

