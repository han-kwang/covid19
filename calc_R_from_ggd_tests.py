#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Estimating R (Netherlands) based on GGD tests only. Experimental.

The dataset https://data.rivm.nl/covid-19/COVID-19_uitgevoerde_testen.csv
can be used to estimate R. It suffers much less from noise due
to administrative delays than the RIVM 'by municipality' or 'casus' datasets.
Weekday patterns are much more predictable.

Two methods, based on N[day_number] of positive tests, delay D, generation
interval Tg:

- "Week-on-week": R[i-D] = (N[i]/N[i-7]) ** (Tg/7)
- "Derivative of rolling-7":

   - N7[i] = N[i-3:i+4].mean()
   - R[i-D] = (N7[i+1] / N7[i-1]) ** (Tg/2)
     (implemented using derivative of log(N7).)

   Here, the last 3 entries of N7 can be estimated using historical
   N7[i]/N[i] ratios (default: previous 2 weeks). Because of the predictable
   weekday patterns in this dataset, this seems to work better than in the
   municipality dataset. Still not super accurate though; resulting R value
   typically +/- 0.1 off, occasionally much more.

Created on Sat Nov 13 22:26:05 2021

Author: @hk_nien on Twitter.
"""

# from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import nlcovidstats as nlcs
import tools
import ggd_data
from tvt_Rvalue import get_R_from_TvT

# Clone of the github.com/mzelst/covid-19 repo.
test_path = '../mzelst-covid19-nobackup/data-rivm/tests'

def load_ggd_pos_tests(lastday=-1, pattern_weeks=2, Tgen=4.0, region_re=None):
    """Get latest GGD test data as DataFrame.

    Parameters:

    - lastday: data up to which date to use (-1 = most recent, -2=second most
      recent) or 'yyyy-mm-dd' string for specific date.
    - pattern_weeks: how many previous weeks to estimate weekday patterns.
    - Tgen: generation interval (days)
    - region_re: optional regexp for GGD regions.

    Return DataFrame:

    - Index: Date_tested (timestamps at 12:00)
    - n_tested: number tested on that day (over entire country)
    - n_pos: number positive on that day
    - n_pos_7: 7-day average (last 3 days are estimated).
    - gfac_7: growth factor week-on-week for n_pos
    - R_wow: R based on week-on-week (no time offset)
    - R_d7r: R based on derivative of rolling-7 derivative
      (no time offset)

    Days with known bad data will have NaN.
    """
    df = ggd_data.load_ggd_pos_tests(lastday, region_regexp=region_re)
    df['n_pos_7'] = nlcs.get_rolling7_with_est3(df['n_pos'], 3, m=2)

    # Shift dates to mid-day
    df.index = df.index + pd.Timedelta(12, 'h')

    # R estimated based on week-on-week growth
    df['gfac_7'] = df['n_pos'] / df['n_pos'].shift(7)
    df['R_wow'] = df['gfac_7'] ** (Tgen/7)

    # R estimated on rolling average
    log_n = np.log(df['n_pos_7'])
    d_log_n = 0.5*(log_n.shift(-1) - log_n.shift(1))
    df['R_d7r'] = np.exp(d_log_n * Tgen)

    return df


def add_dataset(ax, Rs, delay_days, label, marker, color, err_fan=(7, 0.15),
                markersize=5, zorder=0):
    """Add an R dataset with smoothed line.

    - err_fan (n, dR): specify error fan-out to last n-1 days up to +/- dR.
    """
    delay = pd.Timedelta(delay_days, 'd')
    ax.plot(
        Rs.index - delay, Rs,
        marker=marker, linestyle='none', color=color, label=label,
        markersize=markersize, zorder=zorder
        )

    # create smooth line.
    Rs.interpolate(inplace=True)  # for missing values in the middle

    # remove leading/trailing NaN
    Rs = Rs[Rs.notna()]

    from scipy.signal import savgol_filter
    Rsmooth = savgol_filter(Rs.values, 11, 2)
    ax.plot(Rs.index - delay, Rsmooth, '-', color=color, zorder=zorder)
    if err_fan:
        n, dR = err_fan
        Rsmooth1 = Rsmooth[-n:]
        fan = np.linspace(0, dR, n)
        ax.fill_between(Rs.index[-n:] - delay, Rsmooth1-fan, Rsmooth1+fan,
                        color=color, alpha=0.2, zorder=zorder)


def _plot_steps_and_smooth(ax, dates, daydata, r7data, label, color):
    """Plot steps and rolling-7 data. Dates should be at mid-day."""
    ax.step(
        dates + pd.Timedelta('12 h'),
        daydata,
        where='pre', color=color, alpha=0.6, label=label
        )
    ax.plot(
        dates[:-3], r7data.iloc[:-3],
        color=color, alpha=0.8, linestyle='-'
        )
    ax.plot(
        dates[-4:], r7data.iloc[-4:],
        color=color, alpha=0.8, linestyle=(2, (1.5, 0.7))
        )


def plot_rivm_and_ggd_positives(
        num_days=100, correct_anomalies=None,
        yscale=('log', 300, 25000), trim_end=0,
        rivm_regions=('Landelijk',), ggd_regions=('Landelijk',)
        ):
    """Plot RIVM daily cases and GGD positive tests together in one graph.

    Parameters:

    - num_days: number of days to display.
    - correct_anomalies: ignored. For backward compatibility.
    - yscale: ('log', ymin, ymax) or ('linear', ymin, ymax).
    - trim_end: how many days up to present to remove.
      Data shown is for the interval (today-trim_end-num_days) ...
      (today-trim_end).
    - rivm_regions: list of region strings, see doc of
      ``nlcovidstats.get_region_data`` er 'Landelijk'.
    - ggd_regions: list of region strings, see doc of
      ``load_positive_tests``, or 'Landelijk'.

    """
    corr = ' schatting datastoring' if correct_anomalies else ''
    fig, ax = plt.subplots(figsize=(9, 5), tight_layout=True)
    ax.set_title('Positieve tests per dag')
    fig.canvas.manager.set_window_title(ax.title.get_text())

    ## RIVM data
    colors = ['#0044cc', '#217cd3', '#219ed3', '#1eaab9', '#1eb99d', '#1aaa57']*5
    for color, region in zip(colors, rivm_regions):
        region1 = 'Nederland' if region == 'Landelijk' else region
        df_mun_c, population = nlcs.get_region_data(
            region1, -1, correct_anomalies=True
            )
        df_mun, population = nlcs.get_region_data(
            region1, -1, correct_anomalies=False
            )
        df_mun_c = df_mun_c.iloc[:len(df_mun_c)-trim_end]
        df_mun = df_mun.iloc[:len(df_mun)-trim_end]
        _plot_steps_and_smooth(
            ax,
            df_mun.index+pd.Timedelta('2 h'),  # df_mun timestamps daily at 10:00
            df_mun['Delta']*population,
            df_mun_c['Delta7r']*population,
            label=f'RIVM meldingen ({region}){corr}',
            color=color
            )

    colors = ['#cc001f', '#cc3d00', '#cc6f00', '#af912e', '#a9af2e'] * 5
    for color, region in zip(colors, ggd_regions):
        if region == 'Landelijk':
            region1 = None
            label = 'GGD pos. tests (datum monstername + 2)'
        else:
            region1 = region
            label = f'GGD pos. ({region.replace("HR:", "")})'
        df_ggd = load_ggd_pos_tests(-1, region_re=region1)

        df_ggd = df_ggd.iloc[:len(df_ggd)-trim_end]
        _plot_steps_and_smooth(
            ax,
            df_ggd.index + pd.Timedelta('2 d'),  # index timestamps daily at 12:00
            df_ggd['n_pos'],
            df_ggd['n_pos_7'],
            label=label,
            color=color
            )
    ax.set_yscale(yscale[0])
    ax.set_ylim(*yscale[1:])
    dfa = nlcs.DFS['anomalies'].copy()  # index: date 10:00, columns ..., days_back
    dfa = dfa.loc[dfa['days_back'] < 0]  # discard non-shift disturbances
    # dfa['Date_anomaly'] = dfa.index + pd.to_timedelta(2/24 + dfa['days_back'].values, 'd')

    # mun_idxs = pd.DatetimeIndex([
    #     df_mun.index[df_mun.index.get_loc(tm, method='nearest')]
    #     for tm in dfa['Date_anomaly'].values
    #     ])

    # show corrected values
    mask_anom = df_mun['Delta'] != df_mun_c['Delta']
    ax.scatter(
        df_mun.index[mask_anom]+pd.Timedelta('2 h'),
        df_mun_c.loc[mask_anom, 'Delta']*population,
        s=20, marker='x', alpha=0.8, color='#004488', zorder=10,
        label='RIVM schattingen/onvolledig i.v.m. datastoring/drukte',
        )

    ax.legend()
    date_range = (df_ggd.index[-num_days], df_ggd.index[-1] + pd.Timedelta(4, 'd'))
    ax.set_xlim(date_range[0], date_range[1])

    nlcs._add_event_labels(
        ax,
        date_range[0], date_range[1],
        with_ribbons=False,
        flagmatch='CaseGraph'
        )
    tools.set_xaxis_dateformat(ax)
    ax.grid(axis='y', which='minor')
    if yscale[0] == 'log':
        tools.set_yaxis_log_minor_labels(ax)

    fig.show()


def scatterplot_rivm_ggd_positives(
        t_a='2021-09-23', t_b='2021-11-12'
        ):
    """Plot RIVM daily cases versus GGD positive tests.

    Parameters:

    - num_days: number of days to display.
    - trim_end: how many days up to present to remove.
      Data shown is for the interval (today-trim_end-num_days) ...
      (today-trim_end).
    """
    t_a = pd.Timestamp(t_a) + pd.Timedelta('12 h')
    t_b = pd.Timestamp(t_b) + pd.Timedelta('12 h')
    npos_ggd = load_ggd_pos_tests(-1)['n_pos']
    df_mun_c, population = nlcs.get_region_data(
        'Nederland', -1,
        correct_anomalies=True
        )
    npos_mun = (df_mun_c['Delta'] * population).astype(int)

    # Set all data to 12:00 (noon) and map GGD to publication date.
    npos_ggd.index += pd.Timedelta(2, 'd') # publication date
    npos_mun.index += pd.Timedelta(2, 'h')


    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    kernels = [
        ('as-is', [0, 0, 1, 0, 0]),
        #('0.25 d', [0, 0.25, 0.75, 0, 0]),
        ('0.5 d', [0, 0.5, 0.5, 0, 0]),
        #('0.75 d', [0, 0.75, 0.25, 0, 0]),
        ('1 d', [0, 1, 0, 0, 0]),
        ('1.5 d', [0.5, 0.5, 0, 0, 0]),
        ]

    fig, axs = plt.subplots(2, 1, figsize=(7, 9), tight_layout=True)
    ax = axs[0]
    ax.set_xscale('log')
    ax.set_yscale('log')
    axs[1].set_yscale('log')
    ax.set_xlabel('GGD')
    ax.set_ylabel('RIVM')
    for color, (label, kernel) in zip(colors, kernels):
        # Redistribute GGD bins.
        assert len(kernel) % 2 == 1
        npos_ggd_shifted = pd.Series(
            data=np.convolve(npos_ggd.values, kernel, 'same'),
            index=npos_ggd.index
            )

        ax= axs[0]
        if label == 'as-is':
            ax.set_title(f'Coverage: {t_a.strftime("%Y-%m-%d")} '
                         f'- {t_b.strftime("%Y-%m-%d")}')
            axs[1].plot(
                npos_mun.loc[t_a:t_b], linestyle=':', color='k', label='RIVM',
                zorder=10
                )

        xs = npos_ggd_shifted.loc[t_a:t_b]
        ys = npos_mun.loc[t_a:t_b]
        ax.plot(xs, ys, 'o', label=label, color=color, alpha=0.5)

        # mark Mondays
        #mask = (xs.index.dayofweek == 0)
        #ax.scatter(xs[mask], ys[mask], zorder=10, s=10, color=color)

        ax = axs[1]
        ax.plot(xs, linestyle='-', color=color, label=f'GGD {label}')



    axs[0].grid(which='both')
    tools.set_xaxis_dateformat(axs[1])

    axs[0].legend()
    axs[1].legend()
    fig.show()

def _ggdregion(region, lastday, Tgen):
    """Return DataFrame of GGD R for holiday region HR:Noord/Midden/Zuid or regexp."""
    df = load_ggd_pos_tests(
        lastday=lastday, pattern_weeks=2, Tgen=Tgen, region_re=region
        )
    return df

def plot_R_graph_multiple_methods(
        num_days=100, ylim=(0.6, 1.5),
        methods=('rivm', 'melding', 'ggd_wow', 'ggd_der', 'tvt', 'ggd_regions'),
        Tgen=4.0
        ):
    """Plot national R graph with annotations and multiple calculation methods.

    Tgen: generation interval (days)
    """
    lastday = -1  # -1 for most recent

    nlcs.plot_Rt(
        num_days,
        regions=('Nederland' if 'melding' in methods else 'DUMMY'),
        lastday=lastday, delay=nlcs.DELAY_INF2REP, ylim=ylim, Tc=Tgen
        )
    fig = plt.gcf()
    ax = fig.get_axes()[0]

    df = load_ggd_pos_tests(lastday=lastday, pattern_weeks=2, Tgen=Tgen)
    if 'ggd_wow' in methods:
        add_dataset(ax, df['R_wow'], 6.5, 'GGD week-op-week', 'x', '#008800', err_fan=None)
    if 'ggd_der' in methods:
        add_dataset(ax, df['R_d7r'], 3.0, 'GGD-positief landelijk', '+', 'red', err_fan=None,
                    markersize=7)

    if 'ggd_regions' in methods:
        ggdr_items = [
            ('^', 'Noord', '#da5b00'),
            ('o', 'Midden', '#b34e06'),
            ('v', 'Zuid', '#935224')
            ]
        for marker, region, color in ggdr_items:
            add_dataset(
                ax,
                _ggdregion(f'HR:{region}', lastday, Tgen)['R_d7r'],
                3.0, f'GGD {region}', marker, color, err_fan=None,
                markersize=3, zorder=-10
                )

    if 'tvt' in methods:
        df_tvt = get_R_from_TvT(Tgen=Tgen)
        ax.plot(df_tvt['R_interp'], color='purple', linestyle='--', alpha=0.5)
        ax.errorbar(df_tvt.index, df_tvt['R'], df_tvt['R_err'], alpha=0.5,
                    color='purple')
        ax.scatter(df_tvt.index, df_tvt['R'], marker='*', s=45,
                    label='TvT %positief', color='purple', zorder=10)


    leg = ax.legend(framealpha=0.5, loc='upper left')
    leg.set_zorder(20)


if __name__ == '__main__':

    plt.close('all')
    nlcs.init_data(autoupdate=True)
    #plot_rivm_and_ggd_positives()
    plot_R_graph_multiple_methods()
