# -*- coding: utf-8 -*-
"""
Created on Sat Aug 15 11:51:39 2020

This is best run inside Spyder, not as standalone script.

@author: hnienhuy
"""
import re
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


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

#%%

def get_mun_data(df, mun, n_inw, lastday=-1):
    """Return dataframe for one municipality, added 'Delta' and 'Delta7r' columns.

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
    # get an estimated trend

    nc1 = nc.iloc[-4:].mean() # mean number at t=-1.5 days
    slope = (nc1 - nc7a[-4])/1.5
    nc7.iloc[-3:] = nc7a[-4] + np.arange(1, 4)*slope

    # 1st 3 elements are NaN
    nc7.iloc[:3] = np.linspace(0, nc7.iloc[3], 3, endpoint=False)

    df1['Delta'] = nc/n_inw
    df1['Delta7r'] = nc7/n_inw

    return df1


def get_t2_Rt(df1, delta_t, i0=-3):
    """Return most recent doubling time and Rt."""

    nc7 = df1['Delta7r']
    # exponential fit
    t_gen = 5.0 # generation time (d)
    t_double = delta_t / np.log2(nc7.iloc[i0]/nc7.iloc[i0-delta_t])
    Rt = 2**(t_gen / t_double)
    return t_double, Rt

def add_labels(ax, labels, xpos, mindist=0.2):
    """Add labels, try to have them avoid bumping.


    - labels: list of tuples (y, txt)
    - mindist: minimum log10 distance.
    """
    from scipy.optimize import fmin_cobyla

    labels = sorted(labels)

    # log positions and sorted
    logys = np.log10([l[0] for l in labels])
    n = len(logys)

    # Distance matrix: D @ y = distances between adjacent y values
    D = np.zeros((n-1, n))
    for i in range(n-1):
        D[i, i] = -1
        D[i, i+1] = 1

    def cons(ly):
        ds = D @ ly
        errs = np.array([ds - mindist, ds])
        #print(f'{np.around(errs, 2)}')
        return errs.reshape(-1)

    # optimization function
    def func(ly):
        return ((ly - logys)**2).sum()

    new_logys = fmin_cobyla(func, logys, cons, catol=mindist*0.05)

    for lgy, (_, txt) in zip(new_logys, labels):
        ax.text(xpos, 10**lgy,txt, verticalalignment='center')

def plot_daily_trends(minpop=2e+5, ndays=100, lastday=-1, mun_regexp=None, mindist=0.04):
    """Plot daily-case trends.

    - minpop: minimum city population
    - lastday: up to this day.
    - mindist: minimum spacing of curve labels.
      Adjust depending on y range.
    """

    fig, ax = plt.subplots(figsize=(12, 6))
    fig.subplots_adjust(top=0.945, bottom=0.085, left=0.09, right=0.83)

    # dict: municitpality -> population
    muns = df_mun.loc[df_mun['Inwoners'] > minpop]['Inwoners'].to_dict()
    muns['Nederland'] = float(df_mun.sum())

    labels = [] # tuples (y, txt)
    citystats = [] # tuples (Rt, T2, cp100k, cwk, popk, city_name)

    for mun, n_inw in muns.items():

        if mun_regexp and not re.match(mun_regexp, mun):
            continue

        df1 = get_mun_data(df, mun, n_inw, lastday=lastday)
        df1 = df1.iloc[-ndays:]
        use_r7 = True

        if mun == 'Nederland':
            print(df1.iloc[-3:][['Delta']]*n_inw)


        fmt = 'o-' if ndays < 30 else '-'

        if use_r7:
            ax.semilogy(df1['Delta7r']*1e5, fmt, label=mun)
        else:
            ax.semilogy(df1['Delta']*1e5, fmt, label=mun)

        delta_t = 7
        i0 = -3
        t_double, Rt = get_t2_Rt(df1, delta_t, i0=i0)
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
            df1.index[[i0-delta_t, i0]], df1['Delta7r'].iloc[[i0-delta_t, i0]]*1e5,
            'k--', zorder=-10)

        labels.append((df1['Delta7r'][-1]*1e5, f' {mun} ({texp})'))

    dfc = pd.DataFrame.from_records(
        sorted(citystats), columns=['Rt', 'T2', 'C/100k', 'C/wk', 'Pop/k', 'Gemeente'])
    dfc.set_index('Gemeente', inplace=True)
    print(dfc)


    lab_x = df1.index[-1] + pd.Timedelta('1.2 d')
    add_labels(ax, labels, lab_x, mindist=mindist)


    ax.grid()

    if use_r7:
        ax.axvline(df1.index[-4], color='gray')
        # ax.text(df1.index[-4], 0.3, '3 dagen geleden - extrapolatie', rotation=90)
        ax.set_title('7-daags voortschrijdend gemiddelde; laatste 3 dagen zijn een schatting')
    ax.set_ylabel('Nieuwe gevallen per 100k per dag')

    #ax.set_ylim(0.05, None)
    ax.set_xlim(None, df1.index[-1] + pd.Timedelta('1 d'))
    #plt.xticks(pd.to_dateTime(['2020-0{i}-01' for i in range(1, 9)]))
    ax.legend() # loc='lower left')
    for tl in ax.get_xticklabels():
        tl.set_ha('left')

if __name__ == '__main__':


    # To refresh the data:
    # wget -O data/COVID-19_aantallen_gemeente_cumulatief.csv https://data.rivm.nl/covid-19/COVID-19_aantallen_gemeente_cumulatief.csv

    df = pd.read_csv('data/COVID-19_aantallen_gemeente_cumulatief.csv', sep=';')
    df = df.loc[~df.Municipality_code.isna()]
    df['Date_of_report'] = pd.to_datetime(df['Date_of_report'])


    plt.close('all')

    print(f'CSV most recent date: {df["Date_of_report"].iat[-1]}')
    
    plot_daily_trends(ndays=28, lastday=-1)
