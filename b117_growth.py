#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Simulate case numbers and effective R for mix of old/new strains

Created on Fri Jan  8 18:01:37 2021

@author: @hk_nien
"""

import numpy as np
import scipy.optimize
import pandas as pd
import matplotlib.pyplot as plt
import scipy.signal
import tools
import nlcovidstats as nlcs
from b117_country_data import get_data_countries

def simulate_cases(date_ra=('2020-12-18', '2021-02-15'),
                   Rs=(0.88, 0.88*1.6), f0=0.01, Tg = 4.0,
                   use_r7=True, report_delay = 7.0, n0=10000,
                   R_changes=None):
    """Simulate case/infection numbers for old/B117 mix.

    Parameters:

    - date_ra: (date_start, date_end)
    - Rs: (R_old, R_B117)
    - Tg: generation interval (days)
    - f0: fraction of B117 infections at date_start
    - use_r7: whether to work with rolling 7-day average
    - report_delay: time (d) from infection to positive test result
    - n0: number of infections at date_start
    - R_changes: list of R-changes as tuples (date, R_scaling, label).
      For example: R_changes=[('2021-01-23', 0.8, 'Avondklok')]
      Scalings are applied successively.

    Return: DataFrame with:

    - index: Date
    - ni_old: infections per day (old variant)
    - ni_b117: infections per day (B1.1.7)
    - npos: positive cases per day
    - f_b117: fraction of B117 positive cases per day
    - Rt: reproduction number (at moment of infection)
    - label: label str to apply to that date (or None).
    """
    date_ra = pd.to_datetime(date_ra)
    dates = pd.date_range(date_ra[0], date_ra[1], freq='1 d')
    npad = 4 # days of padding at begin and end
    ts = np.arange(-npad, len(dates)+npad)
    one_day = pd.Timedelta(1, 'd')
    dates_padded = pd.date_range(date_ra[0]-npad*one_day, date_ra[1]+npad*one_day,
                                 freq='1 d')
    Rs = np.array(Rs).reshape(2)
    ks = np.log(Rs) / Tg # growth constants per day
    k_ratio = ks[1] - ks[0]
    #print(f'Log odds growth rate: {k_ratio:.3f} /day')

    # number of infections at t=0
    ni0 = np.array([1-f0, f0])*n0
    nis = ni0.reshape(2, 1) * np.exp(ks.reshape(2, 1) * ts)

    # Apply changes in R
    if R_changes is None:
        R_changes = []
    for date_start, Rfac, label in R_changes:
        date_start = pd.to_datetime(date_start)
        if date_start <= date_ra[0]:
            raise ValueError(f'R_change {label} at {date_start} is too early.')
        # construct array tx, value 0 for t < start, value 1, 2, ... for t >= start
        txs = np.array((dates_padded - date_start)/one_day) + 0.5
        txs[txs < 0] = 0
        nis *= Rfac ** (txs/Tg)

    if use_r7:
        nis = scipy.signal.convolve(nis, np.full((1, 7), 1/7), mode='same')

    # log growth rate
    n_tot = nis.sum(axis=0)
    dln_dt = ((n_tot[2:] - n_tot[:-2])/(2*n_tot[1:-1]))[npad-1:1-npad]
    R_eff = np.exp(Tg*dln_dt)

    nis = nis[:, npad:-npad]
    n_tot = n_tot[npad:-npad]

    # Merge data
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


def f2odds(f):
    """Convert fraction B117 to odds ratio B117/other."""

    return f/(1-f)

def odds2f(o):
    return o/(1+o)

def _fill_between_df(ax, df, col_lo, col_hi, **kwargs):
    """Fill-between from two dataframe columns."""

    ax.fill_between(df.index, df[col_lo], df[col_hi],
                    **kwargs)



def simulate_and_plot(Rs, f0, title_prefix='', date_ra=('2020-12-18', '2021-02-15'), n0=1e4,
                      clip_nRo=('2099', '2099', '2099'), R_changes=None, use_r7=True,
                      df_lohi=None
                      ):
    """Simulate and plot, given R and initial prevelance.

    - clip_nRo: optional max dates for (n_NL, R_nL, f_other)
    - R_changes: list of R-changes as tuples (date, R_scaling, label).
      For example: R_changes=[('2021-01-23', 0.8, 'Avondklok')]
    - df_lohi: optional DataFrame with columns nlo, nhi, Rlo, Rhi to use
      as confidence intervals.
    """

    df = simulate_cases(f0=f0, Rs=Rs, date_ra=date_ra, n0=n0, use_r7=use_r7,
                         R_changes=R_changes)

    # simulation for 'no interventions'
    df_nointv = simulate_cases(f0=f0, Rs=Rs, date_ra=date_ra, n0=n0, use_r7=use_r7,
                         R_changes=None)

    df_R, dfc = get_Rt_cases()
    df_R = df_R.loc[df_R.index >= '2020-12-01']
    dfc = dfc.loc[dfc.index >= '2020-12-25'] # cases

    colors = plt.rcParams['axes.prop_cycle']()
    colors = [next(colors)['color'] for _ in range(10)]

    from matplotlib.ticker import LogFormatter, FormatStrFormatter


    fig, axs = plt.subplots(3, 1, tight_layout=True, sharex=True, figsize=(9, 10))
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

    if df_lohi is not None:
        _fill_between_df(ax, df_lohi, 'nlo', 'nhi',
                        color=colors[1], alpha=0.15, zorder=-10)

    if R_changes:
        ax.semilogy(df_nointv['npos'], label='P.T. (sim., geen maatregelen)',
                    color=colors[1], linestyle=':')

    select = dfc.index <= clip_nRo[0]
    ax.semilogy(dfc.loc[select, 'Delta7r']*17.4e6, label='Positieve tests (NL)',
                color=colors[2], linestyle='--', linewidth=3)
    # first valid point of positive tests
    firstpos = df.loc[~df['npos'].isna()].iloc[0]
    ax.scatter(firstpos.name, firstpos['npos'], color=colors[1], zorder=10)

    ax.set_ylim(df['npos'].min()/3, 20000)
    ax.yaxis.set_minor_formatter(LogFormatter(minor_thresholds=(2, 1)))

    ax.text(pd.to_datetime('2020-12-15'), df['npos'].min()/2.6, 'Lockdown', rotation=90,
        horizontalalignment='center')
    ax.grid()
    ax.grid(which='minor', axis='y')
    ax.legend(loc='lower left')

    ## R plot
    ax = axs[1]
    ax.set_ylabel('R')
    ax.plot(df['Rt'], label='$R_t$ (simulatie)',
            color=colors[1])


    if df_lohi is not None:
        _fill_between_df(ax, df_lohi, 'Rlo', 'Rhi',
                        color=colors[1], alpha=0.15, zorder=-10)

    if R_changes:
        ax.plot(df_nointv['Rt'], label='$R_t$ (sim., geen maatregelen)',
                color=colors[1], linestyle=':')
    ax.scatter(df.index[0], df['Rt'][0], color=colors[1], zorder=10)
    dfR1 = df_R.loc[df_R.index <= clip_nRo[1]]
    ax.fill_between(dfR1.index, dfR1['Rlo'], dfR1['Rhi'],
                    color='#0000ff', alpha=0.15, label='$R_t$ (observatie NL)')

    date_labels = [('2020-12-15', 0, 'Lockdown')] + (R_changes or [])
    for date, _, label in date_labels:
        ax.text(pd.to_datetime(date), 0.82, label, rotation=90,
            horizontalalignment='center')


    ax.grid(zorder=0)
    ax.legend()

    ## odds plot
    ax = axs[2]
    ax.set_ylabel('Verhouding B117:overig (pos. tests)')
    ax.semilogy(f2odds(df['f_b117']), label='Nederland (simulatie)',
                color=colors[1])

    if df_lohi is not None:
        df_lohi['odds_lo'] = f2odds(df_lohi['flo'])
        df_lohi['odds_hi'] = f2odds(df_lohi['fhi'])
        _fill_between_df (ax, df_lohi, 'odds_lo', 'odds_hi',
                          color=colors[1], alpha=0.15, zorder=-10)


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
    if R_changes:
        title += f'\n(R wijzigt vanaf {R_changes[0][0]})'
    axs[0].set_title(title)

    for i in range(3):
        tools.set_xaxis_dateformat(axs[i], maxticks=15, ticklabels=(i==2))
    fig.show()

def get_sim_args_from_start_cond(
        start_t_R=('2021-01-04', 0.85),
        req_t_f_npos=('2021-01-15', 0.05, 6e3),
        ndays=60,
        title_prefix='', R_ratio=1.6, Tg=4.0, report_delay=7.0,
        clip_nRo=('2099', '2099', '2099'),
        R_changes=None, use_r7=True,
        ):
    """Return dict with kwargs for simulate_cases()."""


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

    kwargs = dict(
        date_ra=(t_start, t_start+ndays*day),
        Rs=(R_old, R_ratio*R_old),
        f0=f_start, Tg=Tg, use_r7=use_r7, report_delay=report_delay, n0=n0,
        R_changes=R_changes
        )

    return kwargs


def monte_carlo_runs(startcond_nom, Rstart_ra, Rratio_ra, f_ra, n=120):
    """Run many simulations for 95% intrevals on npos and R.

    Ranges are supposed to represent ±2 sigma intervals.

    Parameters:

    - startcond_nom: nominal start conditions (dict).
    - Rstart_ra: tuple (R_lo, R_hi), range of effective R.
    - f_ra: tuple (f_lo, f_hi), range of f at start condition.
    - number of runs.

    Return:

    DataFrame with columns nlo, nhi, Rlo, Rhi, flo, fhi.
    """

    # output for each run, list of arrays
    runs = dict(Rt=[], npos=[], f_b117=[])

    def sample_from_ra(ra):
        """Return random number, normal distribution, +/-2 sigma range specified."""

        lo, hi = ra
        mu = (lo + hi)/2
        sigma = (hi - lo)/4
        return np.random.normal(mu, sigma)

    np.random.seed(1)
    for _ in range(n):

        f = sample_from_ra(f_ra)
        stR = startcond_nom['start_t_R']
        rtfn = startcond_nom['req_t_f_npos']

        startcond = {
            **startcond_nom,
            'start_t_R': (stR[0], sample_from_ra(Rstart_ra)),
            'R_ratio': sample_from_ra(Rratio_ra),
            'req_t_f_npos': (rtfn[0], f, rtfn[2])
            }

        sim_kwargs = get_sim_args_from_start_cond(**startcond)
        df = simulate_cases(**sim_kwargs)
        for col in runs.keys():
            runs[col].append(df[col].values)

    index = df.index

    df_lohi = pd.DataFrame(index=index)
    for col, key in [('n', 'npos'), ('R', 'Rt'), ('f', 'f_b117')]:
        quantiles = np.quantile(runs[key], [0.025, 0.975], axis=0)
        df_lohi[f'{col}lo'] = quantiles[0, :]
        df_lohi[f'{col}hi'] = quantiles[1, :]

    return df_lohi


def simulate_and_plot_alt(start_t_R=('2021-01-04', 0.85),
                          req_t_f_npos=('2021-01-15', 0.05, 6e3),
                          ndays=60,
                          title_prefix='', R_ratio=1.6, Tg=4.0, report_delay=7.0,
                          clip_nRo=('2099', '2099', '2099'),
                          R_changes=None, use_r7=True,
                          var_Rstart_Rratio_f=None,
                          ):
    """Simulation/plotting starting from apparent R and #positive tests.

    - start_t_R: tuple (start_date, R_eff)
    - req_t_f_npos: tuple (date, required_f, required_npos), where 'f'
      is the fraction of B117 as of the positive test results.
    - ndays: number of days from start_t to simulate.
    - R_ratio: ratio R_B117/R_old
    - Tg: generation interval (days)
    - R_changes: list of R-changes as tuples (date, R_scaling, label).
      For example: R_changes=[('2021-01-23', 0.8, 'Avondklok')]
    - var_Rstart_Rratio_f: optional tuple (R_start_err, R_ratio_lo, R_ratio_hi, f_lo, f_hi)
      e.g. (0.05, 1.4, 1.6, 0.3, 0.4).
    """

    # Nominal start conditions and resulting simulation parameters.
    startcond_nom = dict(
        start_t_R=start_t_R, req_t_f_npos=req_t_f_npos,
        ndays=ndays, R_ratio=R_ratio, Tg=Tg, report_delay=report_delay,
        clip_nRo=clip_nRo, R_changes=R_changes, use_r7=use_r7
        )

    sim_kwargs_nom = get_sim_args_from_start_cond(**startcond_nom)
    dR = var_Rstart_Rratio_f[0]
    Rstart = start_t_R[1]

    if var_Rstart_Rratio_f:
        df_lohi = monte_carlo_runs(
            startcond_nom,
            Rstart_ra=(Rstart-dR, Rstart+dR),
            Rratio_ra=var_Rstart_Rratio_f[1:3],
            f_ra=var_Rstart_Rratio_f[3:5]
            )
    else:
        df_lohi = None


    simplot_kwargs = 'Rs,f0,title_prefix,date_ra,n0,clip_nRo,R_changes,use_r7'.split(',')
    simplot_kwargs = {k:v for (k,v) in sim_kwargs_nom.items() if k in simplot_kwargs}

    simulate_and_plot(**simplot_kwargs, df_lohi=df_lohi,
                      title_prefix=title_prefix.format(R=Rstart, start_date=start_t_R[0]))

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
        ax.semilogy(tms, odds, f'{p}', color=col, label=label)
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
        clip_nRo=('2021-01-17', '2021-01-06', '2021-01-15')
        ),
    nl_20210108_ak=dict(
        start_t_R=('2021-01-08', 0.91),
        req_t_f_npos=('2021-01-20', 0.38, 5.2e3),
        ndays=45, title_prefix='Extrapolatie vanaf R=0.91 op 2021-01-06; ',
        # Don't clip.
        clip_nRo=('2099', '2099', '2099'),
        use_r7=False,
        # OMT estimates -8% to -13% effect on Rt
        # https://nos.nl/artikel/2365254-het-omt-denkt-dat-een-avondklok-een-flink-effect-heeft-waar-is-dat-op-gebaseerd.html
        R_changes=[('2021-01-23', 0.9, 'Avondklok')],
        ),
    nl_202101ak_latest=dict(
        start_t_R=('2021-01-10', 0.93), R_ratio=1.5,
        req_t_f_npos=('2021-01-21', odds2f(0.30), 5.1e3),
        ndays=45, title_prefix='Extrapolatie vanaf R={R:.2f} op {start_date}; ',
        # Don't clip.
        clip_nRo=('2099', '2099', '2099'),
        use_r7=False,
        # OMT estimates -8% to -13% effect on Rt
        # https://nos.nl/artikel/2365254-het-omt-denkt-dat-een-avondklok-een-flink-effect-heeft-waar-is-dat-op-gebaseerd.html
        R_changes=[('2021-01-23', 0.9, 'Avondklok')],
        var_Rstart_Rratio_f=(0.03, 1.4, 1.6, odds2f(0.23), odds2f(0.37))
        ),
    )



if __name__ == '__main__':

    plt.close('all')
    nlcs.init_data()

    plot_countries_odds_ratios()

    #simulate_and_plot_alt(**nl_alt_cases['nl_20201228'])
    #simulate_and_plot_alt(**nl_alt_cases['nl_20210106'])

    simulate_and_plot_alt(**nl_alt_cases['nl_202101ak_latest'])

    # simulate_and_plot(Rs=(0.85, 0.85*1.6), f0=0.01, title_prefix='Scenario 1: ')

    # # This was wrong; matched the wrong curve
    # # Extrapolating from test cases as of 2021-01-15
    # simulate_and_plot(Rs=(0.82, 0.82*1.6), f0=0.09, title_prefix='Extrapolatie vanaf heden: ',
    #                   date_ra=('2021-01-04', '2021-02-15'), n0=8e3
    #                      )
    #  simulate_and_plot(Rs=(0.8, 0.8*1.6), f0=0.15, title_prefix='Scenario 2 (horror): ')
    # simulate_and_plot(Rs=(0.77, 0.77*1.6), f0=0.2, title_prefix='Scenario 3 (horror): ')

