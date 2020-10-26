# -*- coding: utf-8 -*-
"""
Created on Sat Aug 15 11:51:39 2020

This is best run inside Spyder, not as standalone script.

Author: @hk_nien on Twitter.
"""
import re
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import urllib
from pathlib import Path
import time

## This is old stuff. It turns out that the 'casus' file is a pain to use.
# # https://data.rivm.nl/covid-19/COVID-19_casus_landelijk.json

# # Date_file: Datum en tijd waarop de gegevens zijn gepubliceerd door het RIVM

# # Date_statistics: Datum voor statistiek; eerste ziektedag, indien niet bekend, datum lab positief, indien niet bekend, melddatum aan GGD (formaat: jjjj-mm-dd)

# # Date_statistics_type: Soort datum die beschikbaar was voor datum voor de variabele "Datum voor statistiek", waarbij:
# # DOO = Date of disease onset : Eerste ziektedag zoals gemeld door GGD. Let op: het is niet altijd bekend of deze eerste ziektedag ook echt al Covid-19 betrof.
# # DPL = Date of first Positive Labresult : Datum van de (eerste) positieve labuitslag.
# # DON = Date of Notification : Datum waarop de melding bij de GGD is binnengekomen.

# # Agegroup: Leeftijdsgroep bij leven; 0-9, 10-19, ..., 90+; bij overlijden

# df = pd.read_json('COVID-19_casus_landelijk.json')
# for col in ['Date_file', 'Date_statistics']:
#     df[col] = pd.to_datetime(df[col])


# #%%

# dfg_cases = df.groupby('Date_statistics').count()

# plt.close('all')

# plt.plot(dfg_cases.index, dfg_cases['Date_file'])

#%%

df_mun = pd.read_csv('data/Regionale_kerncijfers_Nederland_15082020_130832.csv', sep=';')
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

df_restrictions = pd.DataFrame([
    ('2020-05-11', 'Heropening, max 30'),
    ('2020-06-01', 'Mondkapje OV'),
    ('2020-07-01', "Max 100"),
    ('2020-08-17', 'Scholen open Noord'),
    ('2020-08-18', 'Max 6 gasten thuis'),
    ('2020-08-31', 'Scholen open Midden'),
    ('2020-09-20', 'Max 50/6, Horeca 01u'),
    ('2020-09-29', 'Max 30/3, horeca 22u'),
    ('2020-10-05', 'Mondkapjes VO'),
    ('2020-10-14', 'Niveau Zeer Ernstig'),
    ], columns=['Date', 'Description'])
df_restrictions['Date'] = pd.to_datetime(df_restrictions['Date']) + pd.Timedelta('12:00:00')
df_restrictions.set_index('Date', inplace=True)


Rt_rivm = pd.DataFrame.from_records([
    ('2020-07-31T12:00', 1.22),
    ('2020-08-07T12:00', 0.96),
    ('2020-08-14T12:00', 0.95),
    ('2020-08-21T12:00', 1.15),
    ('2020-08-28T12:00', 1.39),
    ('2020-09-04T12:00', 1.34),
    ('2020-09-11T12:00', 1.28),
    ('2020-09-18T12:00', 1.20),
    ('2020-09-24T12:00', 1.28),
    ('2020-10-02T12:00', 1.22),
    ])
Rt_rivm = pd.Series(data=Rt_rivm[1].to_numpy(), index=pd.to_datetime(Rt_rivm[0]), name='Rt_rivm')


#%%



def get_mun_data(df, mun, n_inw, lastday=-1):
    """Return dataframe for one municipality, added 'Delta', 'Delta7r' columns.

    Use data up to lastday.
    """

    if mun == 'Nederland':
        df1 = df.groupby('Date_of_report').sum()
    else:
        df1 = df.loc[df.Municipality_name == mun].copy()
        df1.set_index('Date_of_report', inplace=True)

    if lastday < -1 or lastday > 0:
        df1 = df1.iloc[:lastday+1]

    # nc: number of cases
    nc = df1['Total_reported'].diff()
    if mun == 'Nederland':
        print(nc[-3:])
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
        if mun == 'Nederland':
            print(df1.iloc[-3:][['Delta']]*n_inw)


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


def plot_Rt(df, minpop=2e+5, ndays=100, lastday=-1, delay=10, mun_regexp='Nederland',
            Tc=4.0, Rt_rivm=None):
    """Plot daily-case trends (using global DataFrame df).

    - df: DataFrame with processed per-municipality data.
    - minpop: minimum city population
    - lastday: up to this day.
    - delay: assume delay days from infection to positive report.
    - Tc: generation interval time
    - Rt_rivm: optional series with RIVM estimates.
    """

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.subplots_adjust(top=0.90, bottom=0.085, left=0.09, right=0.83)

    # dict: municitpality -> population
    muns = df_mun.loc[df_mun['Inwoners'] > minpop]['Inwoners'].to_dict()
    muns['Nederland'] = float(df_mun.sum())

    labels = [] # tuples (y, txt)
    for mun, n_inw in muns.items():

        if mun_regexp and not re.match(mun_regexp, mun):
            continue

        df1 = get_mun_data(df, mun, n_inw, lastday=lastday)
        Rt = estimate_Rt_series(df1['Delta7r'].iloc[-ndays-delay:], delay=delay, Tc=Tc)

        fmt = 'o-' if ndays < 70 else '-'
        psize = 5 if ndays < 30 else 3
        ax.plot(Rt, fmt, label=mun, markersize=psize)

        labels.append((Rt[-1], f' {mun}'))

    if len(labels) == 0:
        fig.close()
        raise ValueError(f'No data to plot.')

    if Rt_rivm is not None:
        ax.plot(Rt_rivm, 'o', markersize=7, color='k', label='RIVM')

    y_lab = ax.get_ylim()[0]
    for res_t, res_d in df_restrictions['Description'].iteritems():
        ax.text(res_t, y_lab, f'  {res_d}', rotation=90, horizontalalignment='center')

    lab_x = Rt.index[-1] + pd.Timedelta('1.2 d')
    add_labels(ax, labels, lab_x)
    ax.grid(which='both')
    ax.axvline(Rt.index[-4], color='gray')
    ax.set_title('Reproductiegetal o.b.v. positieve tests; laatste 3 dagen zijn een extrapolatie\n'
                 f'(Generatie-interval: {Tc:.3g} dg, rapportagevertraging {delay} dg)' )
    ax.set_ylabel('Reproductiegetal $R_t$')

    #ax.set_ylim(0.05, None)
    ax.set_xlim(Rt.index[0] - pd.Timedelta('12 h'), Rt.index[-1] + pd.Timedelta('1 d'))
    ax.legend() # loc='lower left')
    for tl in ax.get_xticklabels():
        tl.set_ha('left')


def update_csv(force=False):
    """Update csv file (if not recently updated)."""

    fpath = Path('data/COVID-19_aantallen_gemeente_cumulatief.csv')
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
                print('Not updating file; seems to be recent enough.')
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

if __name__ == '__main__':


    update_csv()

    plt.close('all')
    df = pd.read_csv('data/COVID-19_aantallen_gemeente_cumulatief.csv', sep=';')
    df = df.loc[~df.Municipality_code.isna()]
    df['Date_of_report'] = pd.to_datetime(df['Date_of_report'])


    plt.close('all')

    print(f'CSV most recent date: {df["Date_of_report"].iat[-1]}')

    plot_daily_trends(df, ndays=40, lastday=-1, use_r7=True, minpop=2e5)
    plot_Rt(df, ndays=60, lastday=-1, minpop=4e5, Rt_rivm=Rt_rivm)
