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


def add_percentage_y2_axis(ax_o, label='Frequency B.1.1.7 (%)'):
    """Add y2 axis with percentages on odds axis. Return new Axis."""

    ax_p = ax_o.twinx()
    olo, ohi = ax_o.get_ylim()
    ax_o.set_ylim(olo, ohi) # make sure they don't change anymore
    ax_p.set_ylabel(label)
    ax_p.set_ylim(olo, ohi)
    ax_p.set_yscale('log')
    ax_p.minorticks_off()

    pvals = np.array([0.001, 0.01, 0.1, 1, 10, 30, 60, 80,
                      90, 95, 98, 99, 99.5, 99.8,
                      99.9, 99.95, 99.98, 99.99
                      ])
    ovals = f2odds(0.01*pvals)

    mask = (ovals >= olo) & (ovals <= ohi)
    ovals = ovals[mask]
    pvals = pvals[mask]

    y2ticks = ovals
    y2labels = [f'{p:.4g}' for p in pvals]
    ax_p.set_yticks(y2ticks, minor=False)
    ax_p.set_yticklabels(y2labels)

    return ax_p



def simulate_and_plot(Rs, f0, title_prefix='', date_ra=('2020-12-18', '2021-02-15'), n0=1e4,
                      clip_nRo=('2099', '2099', '2099'), R_changes=None, use_r7=True,
                      df_lohi=None, country_select=None
                      ):
    """Simulate and plot, given R and initial prevelance.

    - clip_nRo: optional max dates for (n_NL, R_nL, f_other)
    - R_changes: list of R-changes as tuples (date, R_scaling, label).
      For example: R_changes=[('2021-01-23', 0.8, 'Avondklok')]
    - df_lohi: optional DataFrame with columns nlo, nhi, Rlo, Rhi to use
      as confidence intervals.
    - country_select: selection preset (str) for which countries to show.
      See get_data_countries() for details.
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
    ## top panel: number of cases
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

    date_labels = [('2020-12-15', 0, 'Lockdown')] + (R_changes or [])
    for date, _, label in date_labels:
        ax.text(pd.to_datetime(date), df['npos'].min()/2.6, label, rotation=90,
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


    markers = iter('o^vs*Do^vs*D'*2)
    cdict = get_data_countries(select=country_select).items()
    for country_name, df in cdict:
        df = df.loc[df.index <= clip_nRo[2]]
        marker = next(markers)
        label = country_name if len(country_name) < 25 else country_name[:23] + '...'
        ax.plot(df.index, f2odds(df['f_b117']), f'{marker}:', linewidth=2, label=label,
                zorder=100)



    ax.grid(which='both', axis='y')
    ax.legend(fontsize=10, framealpha=0.9)

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

    add_percentage_y2_axis(ax, label='Aandeel B.1.1.7 (%)')

    plt.pause(0.75)

class _StartCond:
    def __init__(self, date, value):
        self.date = pd.to_datetime(date)
        self.val = value
    def __repr__(self):
        ymd=self.date.strftime("%Y-%m-%d")
        return f'({ymd}, {self.val})'

def get_sim_args_from_start_cond(
        conditions,
        ndays=60,
        title_prefix='', R_ratio=1.6, Tg=4.0, report_delay=7.0,
        clip_nRo=('2099', '2099', '2099'),
        R_changes=None, use_r7=True,
        ):
    """Return dict with kwargs for simulate_cases().

    - conditions: dict with R, f, npos keys; values tuples (date, value).
    """


    one_day = pd.Timedelta(1, 'd')
    conds = conditions.copy()
    for key, date_value in conditions.items():
        conds[key] = _StartCond(*date_value)

    # Start time of the curve is the R date condition.
    t_start = conds['R'].date

    def delta_t(t1, t2):
        """From datetime to number of days."""
        return (t1 - t2)/one_day

    # f_start: fraction of UK variant at infection time, start time
    odds_req = f2odds(conds['f'].val)
    num_gen = (delta_t(conds['f'].date, conds['R'].date) - report_delay)/Tg
    odds_start = odds_req * R_ratio ** (-num_gen)
    f_start = odds2f(odds_start)

    # R number for old strain
    R_old = conds['R'].val * R_ratio**(-f_start)

    # calculate infection cases at t_start
    num_gen = (delta_t(conds['npos'].date, conds['R'].date) - report_delay) / Tg # number of generations
    growth_fac = (R_old**num_gen + odds_start*(R_ratio*R_old)**num_gen) / (1 + odds_start)
    n0 = conds['npos'].val / growth_fac

    kwargs = dict(
        date_ra=(t_start, t_start+ndays*one_day),
        Rs=(R_old, R_ratio*R_old),
        f0=f_start, Tg=Tg, use_r7=use_r7, report_delay=report_delay, n0=n0,
        R_changes=R_changes
        )

    return kwargs


def monte_carlo_runs(startcond_nom, dR=0.03, fac_RR=1.077, fac_odds=1.2, n=120):
    """Run many simulations for 95% intrevals on npos and R.

    Ranges are supposed to represent ±2 sigma intervals.

    Parameters:

    - startcond_nom: nominal start conditions (dict).
    - dR: R deviation (±).
    - fac_RR: R-ratio factor (>1); range will be RR/fac .. RR*fac.
    - fac_odds: odds factor; range will be odds/fac .. odds*fac
    - n: number of runs.

    Return:

    DataFrame with columns nlo, nhi, Rlo, Rhi, flo, fhi.
    """

    # output for each run, list of arrays
    runs = dict(Rt=[], npos=[], f_b117=[])

    def sample_fac(fac):
        return np.exp(np.random.normal(scale=0.5*np.log(fac)))
    def sample_norm(twosigma):
        return np.random.normal(scale=0.5*twosigma)

    conds = startcond_nom['conditions']

    np.random.seed(1)
    for _ in range(n):
        f = odds2f(f2odds(conds['f'][1]) * sample_fac(fac_odds))
        R = conds['R'][1] + sample_norm(dR)
        RR = startcond_nom['R_ratio'] * sample_fac(fac_RR)
        new_conds = {
            **conds,
            'R': (conds['R'][0], R),
            'f': (conds['f'][0], f),
            }
        startcond = {
            **startcond_nom,
            'conditions': new_conds,
            'R_ratio': RR,
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


def simulate_and_plot_alt(conditions,
                          ndays=60,
                          title_prefix='', R_ratio=1.6, Tg=4.0, report_delay=7.0,
                          clip_nRo=('2099', '2099', '2099'),
                          R_changes=None, use_r7=True,
                          variations=None, country_select=None,
                          ):
    """Simulation/plotting starting from apparent R and #positive tests.

    Parameters:

    - conditions: dict with keys 'R', 'f', 'npos';
      values are tuples (date_str, value).
    - ndays: number of days from start_t to simulate.
    - R_ratio: ratio R_B117/R_old
    - Tg: generation interval (days)
    - R_changes: list of R-changes as tuples (date, R_scaling, label).
      For example: R_changes=[('2021-01-23', 0.8, 'Avondklok')]
    - variations: optional dict with Monte-Carlo 2-sigma deviations;
      keys: dR, fac_RR, fac_odds.
    - country_select: selection preset (str) for which countries to show.
      See get_data_countries() for details.
    """

    # Nominal start conditions and resulting simulation parameters.
    startcond_nom = dict(
        conditions=conditions,
        ndays=ndays, R_ratio=R_ratio, Tg=Tg, report_delay=report_delay,
        clip_nRo=clip_nRo, R_changes=R_changes, use_r7=use_r7
        )

    sim_kwargs_nom = get_sim_args_from_start_cond(**startcond_nom)
    if variations:
        df_lohi = monte_carlo_runs(
            startcond_nom,
            **variations
            )
    else:
        df_lohi = None


    simplot_kwargs = 'Rs,f0,title_prefix,date_ra,n0,clip_nRo,R_changes,use_r7'.split(',')
    simplot_kwargs = {k:v for (k,v) in sim_kwargs_nom.items() if k in simplot_kwargs}

    title_prefix = title_prefix.format(
        R=conditions['R'][1],
        start_date=conditions['R'][0]
        )

    simulate_and_plot(**simplot_kwargs, df_lohi=df_lohi,
                      title_prefix=title_prefix,
                      clip_nRo=clip_nRo,
                      country_select=country_select
                      )

def fit_log_odds(xs, ys, last_weight=0.33):
    """Fit ln(y) = a*x + b; assume larger relative errors for small y.

    Optionally decrease weight of last point.
    """

    # sigmas in ln(y). For Poisson statistics, exponent -0.5.
    sigmas = ys**-0.5
    sigmas[-1] /= last_weight

    sigmas += sigmas.max()


    a, b = np.polyfit(xs, np.log(ys), deg=1, w=1/sigmas)
    return a, b




def plot_countries_odds_ratios(subtract_eng_bg=True, country_select='all_recent',
                               wiki=False):

    cdict = get_data_countries(country_select, subtract_eng_bg=subtract_eng_bg)

    fig, ax = plt.subplots(tight_layout=True, figsize=(10, 5.5))


    markers = iter('o^vs*Do^vs*D'*4)
    colors = plt.rcParams['axes.prop_cycle']()
    colors = iter([next(colors)['color'] for _ in range(40)])

    tm0 = pd.to_datetime('2020-12-01')
    one_day = pd.Timedelta(1, 'd')

    oddsfit_records = []

    for desc, df in cdict.items():

        odds = f2odds(df['f_b117']).values
        tms = df.index
        xs = np.array((tms - tm0) / one_day)

        # early points and last point in each curve deviates from the 'trend by eye'.
        # Therefore, ignore them.
        ifirst = max(1, len(xs)-6)
        oslope, odds0 = fit_log_odds(xs[ifirst:], odds[ifirst:], last_weight=0.33)

        # show fit result
        odds_latest = np.exp(odds0 + oslope * xs[-1])
        tm_latest = tms[-1].strftime("%Y-%m-%d")
        oddsfit_records.append(dict(
            region=desc,
            date=tm_latest,
            odds=float('%.4g' % odds_latest),
            log_slope=float('%.4g' % oslope)
            ))

        xse = np.array([xs[0] - 3, xs[-1] + 3]) # expanded x range
        odds_fit = np.exp(oslope * xse + odds0)

        p = next(markers)
        col = next(colors)
        label = f'{desc} [{oslope:.3f}]'
        ax.semilogy(tms, odds, f'{p}', color=col, label=label)
        ax.semilogy([tms[0]-3*one_day, tms[-1]+3*one_day], odds_fit, '-', color=col)

    odds_fit_df = pd.DataFrame.from_records(oddsfit_records).set_index('region')
    print(f'Slope fit results:\n{odds_fit_df}')

    # label 'today' in the graph
    tm_now = pd.to_datetime('now')
    tm_now += pd.Timedelta(12-tm_now.hour, 'h') # 12:00 noon

    if not wiki:
        ymax = ax.get_ylim()[1]
        ax.axvline(tm_now, color='#888888')
        ax.text(tm_now, ymax, tm_now.strftime('%d %b  '),
                horizontalalignment='right', verticalalignment='top', rotation=90)

    ax.set_ylabel('Odds ratio B.1.1.7/other')


    tools.set_xaxis_dateformat(ax) # must be before adding a second y axis.
    add_percentage_y2_axis(ax)
    ax.legend(bbox_to_anchor=(1.2, 1), fontsize=9)
    ax.set_title('B.1.1.7 presence in positive cases, with $\\log_e$ slopes')
    fig.canvas.set_window_title('B117 in countries/regions')


    if subtract_eng_bg:
        sgtf_subtracted = ' (backgroud positive rate subtracted for England regions)'
    else:
        sgtf_subtracted = ''
    ax.text(1.10, -0.05 + 0.1 * (len(cdict) < 16),
            'UK data is based on population sampling\n'
            f'and mostly SGTF{sgtf_subtracted}.\n'
            'UK SGTF data shifted by 14 days to estimate symptom onset.\n'
            'Other data is from genomic sequencing (\'seq\').\n'
            'Sources: Walker et al., Imperial College, ons.gov.uk\n'
            'covid19genomics.dk, RIVM NL, Borges et al., sciencetaskforce.ch.'
            , transform=ax.transAxes, fontsize=9)

    if not wiki:
        fig.text(0.99, 0.01, '@hk_nien', fontsize=8,
                  horizontalalignment='right', verticalalignment='bottom')


    fig.show()
    plt.pause(0.5)


# cases for simulate_and_plot_alt:
nl_alt_cases = dict(
    nl_20210115=dict(
        conditions=dict(
            R=('2021-01-04', 0.85),
            f=('2020-12-24', odds2f(0.024)),
            npos=('2021-01-15', 5.6e3),
            ),
        R_ratio=1.6,
        ndays=45, title_prefix='Extrapolatie vanaf R={R:.2f} zoals bekend op 15 jan; ',
        clip_nRo=('2021-01-16', '2021-01-05', '2021-01-10'),
        country_select='DK_SEE_20210101',
        variations=dict(dR=0.03, fac_odds=1.2, fac_RR=1.077),
        ),
    nl_ak_20210121=dict(
        conditions=dict(
            R=('2021-01-10', 0.93),
            f=('2021-01-07', odds2f(0.112)),
            npos=('2021-01-21', 5.1e3), # 7 days after R date
            ),
        R_ratio=1.5,
        ndays=45, title_prefix='Extrapolatie vanaf R={R:.2f} zoals bekend op 21 jan; ',
        # Don't clip.
        clip_nRo=('2021-01-21', '2021-01-10', '2099'),
        use_r7=True,
        # OMT estimates -8% to -13% effect on Rt
        # https://nos.nl/artikel/2365254-het-omt-denkt-dat-een-avondklok-een-flink-effect-heeft-waar-is-dat-op-gebaseerd.html
        R_changes=[('2021-01-23', 0.9, 'Avondklok')],
        variations=dict(dR=0.03, fac_odds=1.2, fac_RR=1.077),
        country_select='NL_DK_SEE_20210119'
        ),
    nl_ak_20210130=dict(
        conditions=dict(
            R=('2021-01-19', 0.88),
            f=('2021-01-08', 0.09),
            npos=('2021-01-25', 4.8e3), # 7 days after R date
            ),
        R_ratio=1.40,
        ndays=45, title_prefix='Extrapolatie vanaf R={R:.2f} op {start_date}; ',
        # Don't clip.
        clip_nRo=('2099', '2099', '2099'),
        use_r7=True,
        # OMT estimates -8% to -13% effect on Rt
        # https://nos.nl/artikel/2365254-het-omt-denkt-dat-een-avondklok-een-flink-effect-heeft-waar-is-dat-op-gebaseerd.html
        R_changes=[('2021-01-23', 0.9, 'Avondklok'),
                   ('2021-02-08', 1.111, 'Basisscholen open'),
                   ],
        # variations=dict(dR=0.04, fac_odds=1.2, fac_RR=1.077),
        variations=dict(dR=0.04, fac_odds=1.1, fac_RR=1.077),
        country_select='picked'
        ),
     nl_ak_latest=dict(
        conditions=dict(
            R=('2021-01-19', 0.88),
            f=('2021-01-15', odds2f(0.2193)),
            npos=('2021-01-25', 4.8e3), # 7 days after R date
            ),
        R_ratio=np.exp(0.105*4),
        ndays=45, title_prefix='Extrapolatie vanaf R={R:.2f} op {start_date}; ',
        # Don't clip.
        clip_nRo=('2099', '2099', '2099'),
        use_r7=True,
        # OMT estimates -8% to -13% effect on Rt
        # https://nos.nl/artikel/2365254-het-omt-denkt-dat-een-avondklok-een-flink-effect-heeft-waar-is-dat-op-gebaseerd.html
        R_changes=[('2021-01-23', 0.9, 'Avondklok'),
                   ('2021-02-08', 1.111, 'Basisscholen open'),
                   ('2021-02-10', 1.111, 'Einde avondklok'),
                   ],
        # variations=dict(dR=0.04, fac_odds=1.2, fac_RR=1.077),
        variations=dict(dR=0.04, fac_odds=1.1, fac_RR=1.077),
        country_select='picked'
        ),
    )



if __name__ == '__main__':

    plt.close('all')
    nlcs.init_data()

    plot_countries_odds_ratios(subtract_eng_bg=True, country_select='all_recent')

    ## effect of no background subtraction
    # plot_countries_odds_ratios(subtract_eng_bg=False)

    1 and simulate_and_plot_alt(**nl_alt_cases['nl_ak_latest'])
    if 0: # set to 1/True to plot old data
        pass
        # simulate_and_plot_alt(**nl_alt_cases['nl_20201228'])
        simulate_and_plot_alt(**nl_alt_cases['nl_ak_20210121'])
        simulate_and_plot_alt(**nl_alt_cases['nl_20210115'])
