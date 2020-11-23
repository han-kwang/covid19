# -*- coding: utf-8 -*-
"""
Created on Sat Aug 15 11:51:39 2020

This is best run inside Spyder, not as standalone script.

Author: @hk_nien on Twitter.
"""
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import urllib
import urllib.request
from pathlib import Path
import time
import locale
from holiday_regions import add_holiday_regions
import scipy.signal


try:
    DATA_PATH = Path(__file__).parent / 'data'
except NameError:
    DATA_PATH = Path('data')


#%%



def load_municipality_data():
    """Return municipality dataframe; index=municipality, column: 'Inwoners'"""

    path = DATA_PATH / 'Regionale_kerncijfers_Nederland_15082020_130832.csv'

    df_mun = pd.read_csv(path, sep=';')
    df_mun.rename(columns={
     #'Perioden',
     #"Regio's",
     'Bevolking/Bevolkingssamenstelling op 1 januari/Totale bevolking (aantal)': 'total',
     'Bevolking/Bevolkingssamenstelling op 1 januari/Burgerlijke staat/Bevolking 15 jaar of ouder/Inwoners 15 jaar of ouder (aantal)': 'n15plus',
     'Bevolking/Bevolkingssamenstelling op 1 januari/Burgerlijke staat/Bevolking 15 jaar of ouder/Gehuwd (in % van  inwoners 15 jaar of ouder)': 'n15gehuwd',
     'Bevolking/Bevolkingssamenstelling op 1 januari/Bevolkingsdichtheid (aantal inwoners per km²)': 'dichtheid',
     'Bouwen en wonen/Woningvoorraad/Voorraad op 1 januari (aantal)': 'woningen',
     'Milieu en bodemgebruik/Bodemgebruik/Oppervlakte/Totale oppervlakte (km²)': 'opp'
     }, inplace=True)

    df_mun = pd.DataFrame({'Municipality': df_mun['Regio\'s'], 'Inwoners': df_mun['total']})
    df_mun.set_index('Municipality', inplace=True)
    df_mun = df_mun.loc[~df_mun.Inwoners.isna()]
    import re
    df_mun.rename(index=lambda x: re.sub(r' \(gemeente\)$', '', x), inplace=True)
    return df_mun


def load_restrictions():
    """Return restrictions DataFrame; index=DateTime, column=Description."""

    df = pd.read_csv(DATA_PATH / 'restrictions.csv')
    df['Date'] = pd.to_datetime(df['Date']) + pd.Timedelta('12:00:00')
    df.set_index('Date', inplace=True)
    return df

def load_Rt_rivm():
    """Return Rt DataFrame, with Date index (12:00), columns R, Rmax, Rmin."""

    # File source:
    # https://raw.githubusercontent.com/J535D165/CoronaWatchNL/master/data-dashboard/data-reproduction/RIVM_NL_reproduction_index.csv
    df_full = pd.read_csv('data/rivm_reproduction_number.csv')
    df_full['Datum'] = pd.to_datetime(df_full['Datum']) + pd.Timedelta(12, 'h')
    df = df_full[df_full['Type'] == ('Reproductie index')][['Datum', 'Waarde']].copy()
    df.set_index('Datum', inplace=True)
    df.rename(columns={'Waarde': 'R'}, inplace=True)
    df['Rmax'] = df_full[df_full['Type'] == ('Maximum')][['Datum', 'Waarde']].set_index('Datum')
    df['Rmin'] = df_full[df_full['Type'] == ('Minimum')][['Datum', 'Waarde']].set_index('Datum')
    return df


def update_cum_cases_csv(force=False):
    """Update 'cumulative data' csv file (if not recently updated)."""

    fpath = DATA_PATH / 'COVID-19_aantallen_gemeente_cumulatief.csv'
    if fpath.is_file():
        local_file_data = fpath.read_bytes()
    else:
        local_file_data = None

    if not force:
        if fpath.is_file():
            # estimated last update
            tm = time.time()
            loc_time = time.localtime(tm)
            day_seconds = loc_time[3]*3600 + loc_time[4]*60 + loc_time[5]
            tm_latest = tm - day_seconds + 14*3600
            if tm_latest > tm:
                tm_latest -= 86400

            tm_file = fpath.stat().st_mtime
            if tm_file > tm_latest + 1800: # after 14:30
                print('Not updating file; seems to bxe recent enough.')
                return
            if tm_file > tm_latest:
                print('file may or may not be the latest version.')
                print('Use update_csv(force=True) to be sure.')
                return


    url = 'https://data.rivm.nl/covid-19/COVID-19_aantallen_gemeente_cumulatief.csv'
    print(f'Getting new file ...')
    with urllib.request.urlopen(url) as response:
        data_bytes = response.read()
        if data_bytes == local_file_data:
            print(f'{fpath}: already latest version.')
        else:
            fpath.write_bytes(data_bytes)
            print(f'Wrote {fpath} .')


def load_cumulative_cases():
    """Return df with cumulative cases by municipality. Retrieve from internet if needed."""

    update_cum_cases_csv()

    df = pd.read_csv('data/COVID-19_aantallen_gemeente_cumulatief.csv', sep=';')
    # Removing 'municipality unknown' records.
    # Not clear; including these, the daily numbers are higher than the official
    # report. With them removed, they are lower.
    # Examples:
    # date: incl NA, w/o NA, nos.nl, allecijfers.nl
    # 2020-11-08: 5807 5651 5703 5664
    # 2020-11-06: 7638 7206 7272 7242
    df = df.loc[~df.Municipality_code.isna()] # Remove NA records.
    df = add_holiday_regions(df)
    df['Date_of_report'] = pd.to_datetime(df['Date_of_report'])
    return df




def get_mun_data(df, mun, n_inw, lastday=-1, printrows=0):
    """Return dataframe for one municipality, added 'Delta', 'Delta7r' columns.

    Parameter:

    - n_inw: minimum population
    - printrows: print this many of the most recent rows

    Special municipalities:

    - 'Nederland': all
    - 'HR:Zuid', 'HR:Noord', 'HR:Midden', 'HR:Midden+Zuid': holiday regions.
    - 'P:xx': province

    Use data up to lastday.
    """

    if mun == 'Nederland':
        df1 = df.groupby('Date_of_report').sum()
    elif mun == 'HR:Midden+Zuid':
        df1 = df.loc[df['HolRegion'].str.match('Midden|Zuid')].groupby('Date_of_report').sum()
    elif mun.startswith('HR:'):
        df1 = df.loc[df['HolRegion'] == mun[3:]].groupby('Date_of_report').sum()
    elif mun.startswith('P:'):
        df1 = df.loc[df['Province'] == mun[2:]].groupby('Date_of_report').sum()
    else:
        df1 = df.loc[df.Municipality_name == mun].copy()
        df1.set_index('Date_of_report', inplace=True)

    if lastday < -1 or lastday > 0:
        df1 = df1.iloc[:lastday+1]

    if len(df1) == 0:
        raise ValueError(f'No data for mun={mun!r}.')

    # nc: number of cases
    nc = df1['Total_reported'].diff()
    if printrows > 0:
        print(nc[-printrows:])


    nc.iat[0] = 0
    nc7 = nc.rolling(7, center=True).mean()
    nc7a = nc7.to_numpy()
    # last 3 elements are NaN, use mean of last 4 raw entries to
    # get an estimated trend and use exponential growth or decay
    # for filling the data.

    nc1 = nc.iloc[-4:].mean() # mean number at t=-1.5 days
    log_slope = (np.log(nc1) - np.log(nc7a[-4]))/1.5
    nc7.iloc[-3:] = nc7a[-4] * np.exp(np.arange(1, 4)*log_slope)

    # 1st 3 elements are NaN
    nc7.iloc[:3] = np.linspace(0, nc7.iloc[3], 3, endpoint=False)

    df1['Delta'] = nc/n_inw
    df1['Delta7r'] = nc7/n_inw

    return df1

def estimate_Rt_series(r7, delay=10, Tc=4.0):
    """Return Rt data, assuming delay infection-reporting.

    - r7: Series with 7-day rolling average of new reported infections.
    - delay: assume delay (days) from date of infection.
    - Tc: assume generation interval.

    Return:

    - Series with name 'Rt' (shorter than r7 by delay+1).
    """

    log_r7 = np.log(r7.to_numpy()) # shape (n,)
    assert len(log_r7.shape) == 1

    log_slope = (log_r7[2:] - log_r7[:-2])/2 # (n-2,)
    Rt = np.exp(Tc*log_slope) # (n-2,)

    # Attach to index with proper offset
    index = r7.index[1:-1] - pd.Timedelta(delay, unit='days')

    return pd.Series(index=index, data=Rt, name='Rt')



def get_t2_Rt(df1, delta_t, i0=-3, use_r7=True):
    """Return most recent doubling time and Rt."""

    # daily new cases
    dnc = df1['Delta7r'] if use_r7 else df1['Delta']
    # exponential fit
    t_gen = 5.0 # generation time (d)
    t_double = delta_t / np.log2(dnc.iloc[i0]/dnc.iloc[i0-delta_t])
    Rt = 2**(t_gen / t_double)
    return t_double, Rt

def add_labels(ax, labels, xpos, mindist_scale=1.0, logscale=True):
    """Add labels, try to have them avoid bumping.


    - labels: list of tuples (y, txt)
    - mindist_scale: set to >1 or <1 to tweak label spacing.
    """
    from scipy.optimize import fmin_cobyla

    ymin, ymax = ax.get_ylim()
    mindist = np.log10(ymax/ymin)*0.025*mindist_scale


    labels = sorted(labels)

    # log positions and sorted
    if logscale:
        Ys = np.log10([l[0] for l in labels])
    else:
        Ys = np.array([l[0] for l in labels])
    n = len(Ys)

    # Distance matrix: D @ y = distances between adjacent y values
    D = np.zeros((n-1, n))
    for i in range(n-1):
        D[i, i] = -1
        D[i, i+1] = 1

    def cons(Y):
        ds = D @ Y
        errs = np.array([ds - mindist, ds])
        #print(f'{np.around(errs, 2)}')
        return errs.reshape(-1)

    # optimization function
    def func(Y):
        return ((Y - Ys)**2).sum()

    new_Ys = fmin_cobyla(func, Ys, cons, catol=mindist*0.05)

    for Y, (_, txt) in zip(new_Ys, labels):
        y = 10**Y if logscale else y
        ax.text(xpos, y, txt, verticalalignment='center')


def plot_daily_trends(df, minpop=2e+5, ndays=100, lastday=-1, mun_regexp=None,
                      use_r7=True):
    """Plot daily-case trends.

    - df: DataFrame with processed per-municipality data.
    - minpop: minimum city population
    - lastday: up to this day.
    - use_r7: whether to use 7-day rolling average.
    """

    fig, ax = plt.subplots(figsize=(12, 6))
    fig.subplots_adjust(top=0.945, bottom=0.085, left=0.09, right=0.83)

    # dict: municitpality -> population
    muns = df_mun.loc[df_mun['Inwoners'] > minpop]['Inwoners'].to_dict()
    muns['Nederland'] = float(df_mun.sum())

    labels = [] # tuples (y, txt)f
    citystats = [] # tuples (Rt, T2, cp100k, cwk, popk, city_name)
    for mun, n_inw in muns.items():

        if mun_regexp and not re.match(mun_regexp, mun):
            continue

        df1 = get_mun_data(df, mun, n_inw, lastday=lastday)
        df1 = df1.iloc[-ndays:]

        fmt = 'o-' if ndays < 70 else '-'
        psize = 5 if ndays < 30 else 3

        dnc_column = 'Delta7r' if use_r7 else 'Delta'
        ax.semilogy(df1[dnc_column]*1e5, fmt, label=mun, markersize=psize)
        delta_t = 7
        i0 = -3 if use_r7 else -1
        t_double, Rt = get_t2_Rt(df1, delta_t, i0=i0, use_r7=use_r7)
        citystats.append((np.around(Rt, 2), np.around(t_double, 2),
                          np.around(df1['Delta'][-1]*1e5, 2),
                          int(df1['Delta7r'][-4] * n_inw * 7 + 0.5),
                          int(n_inw/1e3 + .5), mun))

        if abs(t_double) > 60:
            texp = f'Stabiel'
        elif t_double > 0:
            texp = f'×2: {t_double:.3g} d'
        elif t_double < 0:
            texp = f'×½: {-t_double:.2g} d'

        ax.semilogy(
            df1.index[[i0-delta_t, i0]], df1[dnc_column].iloc[[i0-delta_t, i0]]*1e5,
            'k--', zorder=-10)

        labels.append((df1[dnc_column][-1]*1e5, f' {mun} ({texp})'))


    y_lab = ax.get_ylim()[0]

    for res_t, res_d in df_restrictions['Description'].iteritems():
    #    if res_t >= t_min:
            ax.text(res_t, y_lab, f'  {res_d}', rotation=90, horizontalalignment='center')


    dfc = pd.DataFrame.from_records(
        sorted(citystats), columns=['Rt', 'T2', 'C/100k', 'C/wk', 'Pop/k', 'Gemeente'])
    dfc.set_index('Gemeente', inplace=True)
    print(dfc)


    lab_x = df1.index[-1] + pd.Timedelta('1.2 d')
    add_labels(ax, labels, lab_x)


    ax.grid(which='both')

    if use_r7:
        ax.axvline(df1.index[-4], color='gray')
        # ax.text(df1.index[-4], 0.3, '3 dagen geleden - extrapolatie', rotation=90)
        ax.set_title('7-daags voortschrijdend gemiddelde; laatste 3 dagen zijn een schatting')
    ax.set_ylabel('Nieuwe gevallen per 100k per dag')

    #ax.set_ylim(0.05, None)
    ax.set_xlim(None, df1.index[-1] + pd.Timedelta('1 d'))
    from matplotlib.ticker import LogFormatter, FormatStrFormatter
    ax.yaxis.set_major_formatter(FormatStrFormatter('%g'))
    # Monkey-patch to prevent '%e' formatting.
    LogFormatter._num_to_string = lambda _0, x, _1, _2: ('%g' % x)
    ax.yaxis.set_minor_formatter(LogFormatter(minor_thresholds=(3, 1)))
    #plt.xticks(pd.to_dateTime(['2020-0{i}-01' for i in range(1, 9)]))
    ax.legend() # loc='lower left')
    for tl in ax.get_xticklabels():
        tl.set_ha('left')

    fig.canvas.set_window_title(f'Case trends (ndays={ndays})')
    fig.show()


def plot_Rt(df, ndays=100, lastday=-1, delay=9,
            regions='Nederland',
            Tc=4.0, Rt_rivm=None):
    """Plot daily-case trends (using global DataFrame df).

    - df: DataFrame with processed per-municipality data.
    - lastday: up to this day.
    - delay: assume delay days from infection to positive report.
    - Tc: generation interval time
    - Rt_rivm: optional series with RIVM estimates.
    - regions: comma-separated string (or list of str);
      'Nederland', 'V:xx' (holiday region), 'P:xx' (province), 'M:xx'
      (municipality).
    """

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.subplots_adjust(top=0.90, bottom=0.085, left=0.09, right=0.92)
    plt.xticks(rotation=-20)
    # dict: municitpality -> population

    labels = [] # tuples (y, txt)
    if isinstance(regions, str):
        regions = regions.split(',')

    for region in regions:

        df1 = get_mun_data(df, region, 1e9, lastday=lastday)
        Rt = estimate_Rt_series(df1['Delta7r'].iloc[-ndays-delay:], delay=delay, Tc=Tc)

        fmt = 'o-' if ndays < 70 else '-'
        psize = 5 if ndays < 30 else 3

        label = re.sub('^[A-Z]+:', '', region)
        ax.plot(Rt, fmt, label=label, markersize=psize)

        labels.append((Rt[-1], f' {label}'))

    if len(labels) == 0:
        fig.close()
        raise ValueError(f'No data to plot.')

    if Rt_rivm is not None:
        tm_lo, tm_hi = Rt.index[[0, -1]] # lowest timestamp
        tm_rivm_est = Rt_rivm[Rt_rivm['R'].isna()].index[0] # 1st index with NaN
        # final values
        Rt_rivm_final = Rt_rivm.loc[tm_lo:tm_rivm_est, 'R']
        ax.plot(Rt_rivm_final.iloc[:-1], 'k-', label='RIVM')
        ax.plot(Rt_rivm_final.iloc[-2::-7], 'ko', markersize=4)
        # estimates
        Rt_rivm_est = Rt_rivm.loc[tm_rivm_est-pd.Timedelta(1, 'd'):tm_hi+pd.Timedelta(12, 'h')]
        # print(Rt_rivm_est)
        ax.fill_between(Rt_rivm_est.index, Rt_rivm_est['Rmin'], Rt_rivm_est['Rmax'],
                        color='#bbbbbb', label='RIVM prognose')



    y_lab = ax.get_ylim()[0]

    # add_labels(ax, labels, lab_x)
    ax.grid(which='both')
    ax.axvline(Rt.index[-4], color='gray')
    ax.axhline(1, color='k', linestyle='--')
    ax.text(Rt.index[-4], ax.get_ylim()[1], Rt.index[-4].strftime("%d %b "),
            rotation=90, horizontalalignment='right', verticalalignment='top')

    ax.set_title('Reproductiegetal o.b.v. positieve tests; laatste 3 dagen zijn een extrapolatie\n'
                 f'(Generatie-interval: {Tc:.3g} dg, rapportagevertraging {delay} dg)' )
    ax.set_ylabel('Reproductiegetal $R_t$')

    # get second y axis
    ax2 = ax.twinx()
    T2s = np.array([-2, -4,-7, -10, -14, -21, -60, 9999, 60, 21, 14, 10, 7, 4, 2])
    y2ticks = 2**(Tc/T2s)
    y2labels = [f'{t2 if t2 != 9999 else "∞"}' for t2 in T2s]
    ax2.set_yticks(y2ticks)
    ax2.set_yticklabels(y2labels)
    ax2.set_ylim(*ax.get_ylim())
    ax2.set_ylabel('Halverings-/verdubbelingstijd (dagen)')



    xlim = (Rt.index[0] - pd.Timedelta('12 h'), Rt.index[-1] + pd.Timedelta('3 d'))
    ax.set_xlim(*xlim)
    for res_t, res_d in df_restrictions['Description'].iteritems():
        if res_t >= xlim[0] and res_t <= xlim[1]:
            ax.text(res_t, y_lab, f'  {res_d}', rotation=90, horizontalalignment='center')

    ax.text(0.99, 0.98, '@hk_nien', transform=ax.transAxes,
            verticalAlignment='top', horizontalAlignment='right',
            rotation=90)

    ax.legend(loc='upper center')
    for tl in ax.get_xticklabels():
        tl.set_ha('left')

    fig.canvas.set_window_title(f'Rt ({", ".join(regions)[:30]}, ndays={ndays})')
    fig.show()

def plot_Rt_oscillation():
    """Uses global Rt_rivm variable."""

    fig, axs = plt.subplots(2, 1, tight_layout=True)
    Rr = Rt_rivm['R'][~Rt_rivm['R'].isna()].to_numpy()
    Rr = Rr[-120:]
    Rr_smooth = scipy.signal.savgol_filter(Rr, 15, 2)

    ax = axs[0]
    n = len(Rr)
    ts = np.arange(n)/7
    ax.plot(ts, Rr, label='Rt (RIVM)')
    ax.plot(ts, Rr_smooth, label='Rt smooth', zorder=-1)
    ax.plot(ts, Rr - Rr_smooth, label='Difference')
    ax.set_xlabel('Time (wk)')
    ax.set_ylabel('R')
    ax.set_xlim(ts[0], ts[-1])
    ax.legend()
    ax.grid()
    ax = axs[1]
    # window = 1 - np.linspace(-1, 1, len(Rr))**2


    window = scipy.signal.windows.tukey(n, alpha=(n-14)/n)
    spectrum = np.fft.rfft((Rr-Rr_smooth) * window)
    freqs = 7/len(Rr) * np.arange(len(spectrum))
    ax.plot(freqs, np.abs(spectrum)**2)
    ax.set_xlabel('Frequency (1/wk)')
    ax.set_ylabel('Power')
    ax.grid()

    fig.canvas.set_window_title('Rt oscillation')

    fig.show()


if __name__ == '__main__':

    # Note: need to run this twice before NL locale takes effect.
    locale.setlocale(locale.LC_ALL, 'nl_NL.UTF-8')
    plt.rcParams["date.autoformatter.day"] = "%d %b"

    df_mun = load_municipality_data()
    df = load_cumulative_cases()
    Rt_rivm = load_Rt_rivm()
    df_restrictions = load_restrictions()

    print(f'CSV most recent date: {df["Date_of_report"].iat[-1]}')

    plt.close('all')

    get_mun_data(df, 'Nederland', 5e6, printrows=5)

    plot_Rt_oscillation()
    plot_daily_trends(df, ndays=100, lastday=-1, use_r7=True, minpop=2e5)
    plot_Rt(df, ndays=130, lastday=-1, delay=9, Rt_rivm=Rt_rivm)
    plot_Rt(df, ndays=40, lastday=-1, delay=9, Rt_rivm=Rt_rivm,
            regions='HR:Noord,HR:Midden+Zuid')
