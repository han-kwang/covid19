#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Simulate case numbers and effective R for mix of old/new strains

Created on Fri Jan  8 18:01:37 2021

@author: @hk_nien
"""

import datetime
import numpy as np
import scipy.optimize
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


def _ywd2date(ywd):
    """Convert 'yyyy-Www-d' string to date (12:00 on that day)."""

    twelvehours = pd.Timedelta('12 h')

    dt = datetime.datetime.strptime(ywd, "%G-W%V-%w") + twelvehours
    return dt

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

    dk_data = [
        dict(Date=_ywd2date(r[0]),
             n_pos=r[1], f_b117=r[2])
        for r in dk_data
        ]
    df = pd.DataFrame.from_records(dk_data).set_index('Date')
    return {'Denemarken (1 jan)': df}

def _get_data_dk_2():

    # https://www.covid19genomics.dk/statistics
    dk_data = [
        # Year-week-da, n_pos, f_b117
        ('2020-W48-4', -1, 0.002),
        ('2020-W49-4', -1, 0.002),
        ('2020-W50-4', -1, 0.004),
        ('2020-W51-4', -1, 0.008),
        ('2020-W52-4', -1, 0.020),
        ('2020-W53-4', -1, 0.024),
        ('2021-W01-4', -1, 0.036), # preliminary
        ]
    # Convert week numbers to date (Wednesday of the week)

    dk_data = [
        dict(Date=_ywd2date(r[0]),
             n_pos=r[1], f_b117=r[2])
        for r in dk_data
        ]

    df = pd.DataFrame.from_records(dk_data).set_index('Date')
    return {'Denemarken (15 jan)': df}


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

def _get_data_nl_koopmans():

    # https://nos.nl/video/2363435-tussen-1-en-5-procent-nederlandse-coronagevallen-besmet-met-britse-variant.html
    # 2021-01-07: "1%-5% of positive cases" (sampling date not specified, assuming preceding week).

    df = pd.DataFrame(dict(Date=pd.to_datetime(['2021-01-04']), f_b117=[0.03]))
    df = df.set_index('Date')

    cdict = {
        'NL "1%-5%"': df
        }
    return cdict


def _get_data_nl():

    # OMT advies #96
    # https://www.tweedekamer.nl/kamerstukken/brieven_regering/detail?id=2021Z00794&did=2021D02016
    # https://www.rivm.nl/coronavirus-covid-19/omt (?)

    nl_data = [
        # Year-week-da, n_pos, f_b117
        ('2020-W49-4', -1, 0.011),
        ('2020-W50-4', -1, 0.007),
        ('2020-W51-4', -1, 0.011),
        ('2020-W52-4', -1, 0.014),
        ('2020-W53-4', -1, 0.052),
        ('2021-W01-4', -1, 0.119), # preliminary
        ]
    # Convert week numbers to date (Wednesday of the week)

    dk_data = [
        dict(Date=_ywd2date(r[0]),
             n_pos=r[1], f_b117=r[2])
        for r in nl_data
        ]

    df = pd.DataFrame.from_records(dk_data).set_index('Date')
    return {'NL OMT-advies #96': df}



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

    return df2

def get_data_countries():
    """Return dict, key=country_name, value=dataframe with Date, n_pos, f_b117."""

    cdict = { **_get_data_dk_2(), **_get_data_uk(), **_get_data_nl() }

    return cdict

def f2odds(f):
    """Convert fraction B117 to odds ratio B117/other."""

    return f/(1-f)


def simulate_and_plot(Rs, f0, title_prefix='', date_ra=('2020-12-18', '2021-02-15'), n0=1e4,
                      clip_nRo=('2099', '2099', '2099')):
    """Simulate and plot, given R and initial prevelance.

    - clip_nRo: optional max dates for (n_NL, R_nL, f_other)
    """

    df =  simulate_cases(f0=f0, Rs=Rs, date_ra=date_ra, n0=n0, use_r7=True)
    df_R, dfc = get_Rt_cases()
    df_R = df_R.loc[df_R.index >= '2020-12-01']
    dfc = dfc.loc[dfc.index >= '2020-12-25'] # cases

    colors = plt.rcParams['axes.prop_cycle']()
    colors = [next(colors)['color'] for _ in range(10)]

    from matplotlib.ticker import LogFormatter, FormatStrFormatter


    fig, axs = plt.subplots(3, 1, tight_layout=True, sharex=True, figsize=(9, 9))
    ## top panel
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

    select = dfc.index <= clip_nRo[0]
    ax.semilogy(dfc.loc[select, 'Delta7r']*17.4e6, label='Positieve tests (NL)',
                color=colors[2], linestyle='--', linewidth=3)
    # first valid point of positive tests
    firstpos = df.loc[~df['npos'].isna()].iloc[0]
    ax.scatter(firstpos.name, firstpos['npos'], color=colors[1], zorder=10)

    ax.set_ylim(df['npos'].min()/5, 20000)
    ax.yaxis.set_minor_formatter(LogFormatter(minor_thresholds=(2, 1)))

    ax.text(pd.to_datetime('2020-12-15'), df['npos'].min()/4.5, 'Lockdown', rotation=90,
        horizontalalignment='center')
    ax.grid()
    ax.grid(which='minor', axis='y')
    ax.legend(loc='lower left')

    ## R plot
    ax = axs[1]
    ax.set_ylabel('R')
    ax.plot(df['Rt'], label='$R_t$ (simulatie)',
            color=colors[1])
    ax.scatter(df.index[0], df['Rt'][0], color=colors[1], zorder=10)
    dfR1 = df_R.loc[df_R.index <= clip_nRo[1]]
    ax.fill_between(dfR1.index, dfR1['Rlo'], dfR1['Rhi'],
                    color='#0000ff', alpha=0.15, label='$R_t$ (observatie NL)')
    ax.text(pd.to_datetime('2020-12-15'), 0.82, 'Lockdown', rotation=90,
            horizontalalignment='center')
    ax.grid(zorder=0)
    ax.legend()

    ## odds plot
    ax = axs[2]
    ax.set_ylabel('Verhouding B117:overig (pos. tests)')
    ax.semilogy(f2odds(df['f_b117']), label='Nederland (simulatie)',
                color=colors[1])

    markers = iter('os^vD*os^vD*')
    for country_name, df in get_data_countries().items():
        df = df.loc[df.index <= clip_nRo[2]]
        marker = next(markers)
        ax.plot(df.index, f2odds(df['f_b117']), f'{marker}:', linewidth=2, label=country_name)

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


def simulate_and_plot_alt(start_t_R=('2021-01-04', 0.85),
                          req_t_f_npos=('2021-01-15', 0.05, 6e3),
                          ndays=60,
                          title_prefix='', R_ratio=1.6, Tg=4.0, report_delay=7.0,
                          clip_nRo=('2099', '2099', '2099')):
    """Simulation/plotting starting from apparent R and #positive tests.

    - start_t_R: tuple (start_date, R_eff)
    - req_t_f_npos: tuple (date, required_f, required_npos), where 'f'
      is the fraction of B117 as of the positive test results.
    - ndays: number of days from start_t to simulate.
    - R_ratio: ratio R_B117/R_old
    - Tg: generation interval (days)
    """

    day = pd.Timedelta(1, 'd')
    t_start = pd.to_datetime(start_t_R[0])
    R_start = start_t_R[1]
    t_req = pd.to_datetime(req_t_f_npos[0])
    f_req, npos_req = req_t_f_npos[1:]

    # convert times to times in days since t_start.
    tt_req = (t_req - t_start) / day

    odds_req = f_req / (1-f_req) # at specified positive-test date.
    odds_start = odds_req * R_ratio ** (-(tt_req - report_delay)/Tg)
    f_start = odds_start/(odds_start + 1)

    def _get_Rt(R_old):
        """Return effective Rt for this R_old."""

        Ra, Rb = R_old, R_ratio*R_old
        Rt = np.exp(Tg/(1 + odds_start) * (np.log(Ra)/Tg + odds_start*np.log(Rb)/Tg))
        return Rt

    # Solve Rt = Rstart for R_old
    func = lambda R_old: _get_Rt(R_old) - R_start
    R_old = scipy.optimize.newton(func, x0=R_start)

    # time from start to moment that the required number of positives
    # were infected.
    tti_req = tt_req - report_delay

    # Calculate n_total growth factor over dt_pos_spec
    mgen = tti_req / Tg # number of generations
    growth_fac = (R_old**mgen + odds_start*(R_ratio*R_old)**mgen) / (1 + odds_start)
    n0 = npos_req / growth_fac

    simulate_and_plot(
        (R_old, R_ratio*R_old), f_start, title_prefix=title_prefix, n0=n0,
        date_ra=(t_start, t_start+ndays*day), clip_nRo=clip_nRo
        )

def fit_log_odds(xs, ys):
    """Fit ln(y) = a*x + b; assume larger relative errors for small y."""

    # sigmas in ln(y). For Poisson statistics, exponent -0.5.
    sigmas = ys**-0.5
    a, b = np.polyfit(xs, np.log(ys), deg=1, w=1/sigmas)
    return a, b



def plot_countries_odds_ratios():
    cdict = get_data_countries()

    fig, ax = plt.subplots(tight_layout=True, figsize=(7, 4))


    markers = iter('o^vs*o^vs*')
    colors = plt.rcParams['axes.prop_cycle']()
    colors = iter([next(colors)['color'] for _ in range(10)])


    tm0 = pd.to_datetime('2020-12-01')
    one_day = pd.Timedelta(1, 'd')

    for desc, df in cdict.items():

        odds = f2odds(df['f_b117']).values
        tms = df.index
        xs = (tms - tm0) / one_day

        # 1st and last point in each curve deviates from the 'trend by eye'.
        # Therefore, ignore it.
        oslope, odds0 = fit_log_odds(xs[1:-1], odds[1:-1])

        xse = np.array([xs[0] - 3, xs[-1] + 3]) # expanded x range
        odds_fit = np.exp(oslope * xse + odds0)

        p = next(markers)
        col = next(colors)
        label = f'{desc} [{oslope:.3f}]'
        ax.semilogy(tms, odds, f'{p}:', color=col, label=label)
        ax.semilogy([tms[0]-3*one_day, tms[-1]+3*one_day], odds_fit, '-', color=col)

    ax.set_ylabel('Odds ratio B117/other')
    ax.legend(loc='upper left')
    tools.set_xaxis_dateformat(ax)
    ax.set_title('B.1.1.7 presence in positive cases, with $\\log_e$ slopes')
    fig.show()

#%%





# cases for simulate_and_plot_alt:
nl_alt_cases = dict(
    nl_20201228=dict(
        start_t_R=('2020-12-28', 0.94),
        req_t_f_npos=('2021-01-04', 0.05, 7.7e3),
        ndays=45, title_prefix='Extrapolatie vanaf R=0.94 op 28 jan; ',
        clip_nRo=('2021-01-08', '2020-12-28', '2020-12-31')
        ),
    nl_20210104=dict(
        start_t_R=('2021-01-04', 0.85),
        req_t_f_npos=('2021-01-15', 0.09, 5.6e3),
        ndays=45, title_prefix='Extrapolatie vanaf R=0.85 op 2021-01-04; ',
        clip_nRo=('2021-01-16', '2021-01-05', '2021-01-10')
        ),
    nl_20210106=dict(
        start_t_R=('2021-01-06', 0.86),
        req_t_f_npos=('2021-01-17', 0.3, 5.2e3),
        ndays=45, title_prefix='Extrapolatie vanaf R=0.86 op 2021-01-06; ',
        clip_nRo=('2021-01-17', '2021-01-15', '2021-01-15')
        )

    )



if __name__ == '__main__':

    plt.close('all')
    nlcs.init_data()

    plot_countries_odds_ratios()

    simulate_and_plot_alt(**nl_alt_cases['nl_20201228'])
    #simulate_and_plot_alt(**nl_alt_cases['nl_20210104'])
    simulate_and_plot_alt(**nl_alt_cases['nl_20210106'])

    # simulate_and_plot(Rs=(0.85, 0.85*1.6), f0=0.01, title_prefix='Scenario 1: ')

    # # This was wrong; matched the wrong curve
    # # Extrapolating from test cases as of 2021-01-15
    # simulate_and_plot(Rs=(0.82, 0.82*1.6), f0=0.09, title_prefix='Extrapolatie vanaf heden: ',
    #                   date_ra=('2021-01-04', '2021-02-15'), n0=8e3
    #                      )
    #  simulate_and_plot(Rs=(0.8, 0.8*1.6), f0=0.15, title_prefix='Scenario 2 (horror): ')
    # simulate_and_plot(Rs=(0.77, 0.77*1.6), f0=0.2, title_prefix='Scenario 3 (horror): ')

