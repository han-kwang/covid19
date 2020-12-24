#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 22 19:16:09 2020

@author: hnienhuy
"""
import json
import pandas as pd
import nlcovidstats as nlcs
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

def load_religion_by_municipality():
    """Return dataframe with religion by municipality.

    Index: Gemeente;
    Columns: 'Provincie', 'Maandelijks%', 'Religieus%',
     'Katholiek%', 'Hervormd%', 'Gereformeerd%', 'PKN%', 'Islam%', 'Joods%',
     'Hindoe%', 'Boeddhist%', 'Anders%'
    """
    dfr = pd.read_csv('data/Religie_en_kerkbezoek_naar_gemeente_2010-2014.csv',
                    comment='#', index_col=False)
    dfr = dfr.set_index('Gemeente').drop(columns=['dummy', 'Gemcode'])

    # Remove municipalities with no data
    dfr.drop(index=dfr.index[dfr['Religieus%'] == '.'], inplace=True)

    # Convert to numbers
    for col in dfr.columns:
        if col.endswith('%'):
            dfr[col] = dfr[col].astype(float)

    return dfr

def load_household_size_by_municipality():
    """Return dataframe, index 'Gemeente', column 'HHsize'."""

    dfhh = pd.read_csv('data/huishoudens_samenstelling_gemeentes.csv', comment='#')
    dfhh.sort_values('Gemeente', inplace=True)
    dfhh.set_index('Gemeente', inplace=True)

    # remove rows for nonexistent municipalites
    dfhh.drop(index=dfhh.index[dfhh['nHH'].isna()], inplace=True)

    # rename municipalities
    rename_muns = {
        'Beek (L.)': 'Beek',
        'Hengelo (O.)': 'Hengelo',
        'Laren (NH.)': 'Laren',
        'Middelburg (Z.)': 'Middelburg',
        'Rijswijk (ZH.)': 'Rijswijk',
        'Stein (L.)': 'Stein',
        'Groningen (gemeente)': 'Groningen',
        'Utrecht (gemeente)': 'Utrecht',
        "'s-Gravenhage (gemeente)": "'s-Gravenhage",
        }
    dfhh.rename(index=rename_muns, inplace=True)
    return dfhh


def load_extended_municipality_data():
    """Load municipality data with extra columns for religion, household size.

    Use global variable to cache unless force=True.
    """

    dfr = load_religion_by_municipality()

    # Combine with 2020 population data.
    # Note that religion data is from 2014 and municipalities have changed in the meantime.
    # Remove municipalities that don't match municipalities of 2020.
    dfm = nlcs.DFS['mun'].copy()
    dfm = dfm.join(dfr, how='inner')

    dfhh = load_household_size_by_municipality()

    dfm['numHH'] = dfhh['nHH']
    dfm['HHsize'] = np.around(dfm['Population']/dfm['numHH'], 2)

    return dfm


def plot_trends_by_religious_visits(ndays=80, maxpop=30e3):
    """Plot trends.

    - ndays: number of days up to today.
    - maxpop: maximum population size per municipality.
    """

    dfm = load_extended_municipality_data()

    # Select <30k population
    dfm   = dfm.loc[dfm['Population'] < maxpop]
    print(f'Statistics for {len(dfm)} municipalities.')

    # Find quantiles. DataFrame with index q, columns 'Maandelijks%' etc.
    qs = np.array([0, 0.4, 0.8, 0.9, 1.0])
    dfq = dfm.quantile(q=qs, axis='index')
    regions = ['Nederland']

    for i in range(1, len(qs)):
        qa, qb = qs[i-1], qs[i]
        maa, mab = dfq.at[qa, 'Maandelijks%'], dfq.at[qb, 'Maandelijks%']
        select = (dfm['Maandelijks%'] >= maa-0.01) & (dfm['Maandelijks%'] <= mab+0.01)
        muns = list(dfm.index[select])
        mean = dfm.loc[select]['Population'].mean()
        print(f'Quantile {qa}-{qb}: Maandelijks% {maa:.1f}-{mab:.1f}, Mean pop: {mean/1e3:.1f}k\n --> {", ".join(muns)}\n')
        rdict = dict(label=f'{maa:.0f}%-{mab:.0f}%', muns=muns)
        regions.append('JSON:' + json.dumps(rdict))


    subtitle = f'Gemeentes < {maxpop/1e3:.0f}k inwoners, naar maandelijks kerkbezoek'
    nlcs.plot_daily_trends(ndays, region_list=regions, subtitle=subtitle)


def get_extended_mun_data_with_weekly_cases(ref_date='now', pop_range=(0, 30e3)):
    """Return dataframe with extended municipality data and
    'Weekly_cases' added column.

    Parameters:

    - ref_date: reference date ('yyyy-mm-dd' or 'now'); count 7 days back from
      that date.
    - pop_range: min/max population to include.

    Return:

    - dataframe
    - date_end
    """

    dfm = load_extended_municipality_data()

    pop_mask = (dfm['Population'] >= pop_range[0]) & (dfm['Population'] < pop_range[1])
    dfm = dfm.loc[pop_mask]


    dfc = nlcs.DFS['cases']
    # find the exact report date closest to the requested reference date
    # (just in case the reporting time shifts away from 10:00)
    date_end = pd.to_datetime(pd.to_datetime(ref_date).strftime('%Y-%m-%d 10:01'))
    date_end = dfc['Date_of_report'].iloc[np.argmin(np.abs(dfc['Date_of_report'] - date_end))]

    dfm['Weekly_cases'] = np.nan

    for mun in dfm.index:
        # iterate municipalities

        dfc_1mun = dfc.loc[dfc['Municipality_name'] == mun]
        # ncs: number of cases (cumulative) as time series.
        ncs = dfc_1mun[['Date_of_report', 'Total_reported']].set_index('Date_of_report')
        ncs = ncs[:date_end]
        delta_cases = int(ncs.iloc[-1] - ncs.iloc[-8])
        assert ncs.index[-8] == ncs.index[-1] - pd.Timedelta(7, 'd') # paranoia
        dfm.loc[mun, 'Weekly_cases'] = delta_cases

    return dfm, date_end


def plot_hhsize_relig_cases(ref_date='now', pop_range=(0, 60e3)):
    """Generate a scatter plot household size, religion, covid cases.

    Parameters:

            Parameters:

    - ref_date: reference date ('yyyy-mm-dd' or 'now'); count 7 days back from
      that date.
    - pop_range: min/max population to include.

    Return:

    - dataframe with extended municipality data.
    - date_end
    """

    dfm_wc, date_end = get_extended_mun_data_with_weekly_cases(ref_date, pop_range=pop_range)

    ##  Urk is an outlier? Try without
    #dfm_wc.drop(index='Urk', inplace=True)

    dfm_wc['WeeklyPer100k'] = dfm_wc['Weekly_cases']/dfm_wc['Population']*(1e5)
    mean_per100k = dfm_wc['Weekly_cases'].sum() / dfm_wc['Population'].sum() * (1e5)
    maxdev = np.abs(dfm_wc['WeeklyPer100k'] - mean_per100k).max()
    vmin, vmax = mean_per100k-maxdev, mean_per100k + maxdev

    print(f'\nData for end date {date_end.strftime("%Y-%m-%d")}')

    print(f'Mean per day per 100k: {mean_per100k:.1f}')

    cmap = matplotlib.colors.LinearSegmentedColormap.from_list('bgr',
        [(0, 0, 0.6), (0, 0, 1),
         (0.8, 0.8, 0.8),
         (1, 0, 0), (0.6, 0, 0)]
        )


    fig, ax = plt.subplots(1, 1, figsize=(6, 4), tight_layout=True)
    ax.set_xlabel('Gemiddelde grootte huishouden')
    ax.set_ylabel('% Actief religieus')

    # pop_scale is for scaling the marker size by population
    pop_scale=30/np.sqrt(dfm_wc['Population'].max())

    for zorder, fmin, fmax in [(11, 0, 0.7), (10, 0.7, 1.3), (12, 1.3, 99)]:
        # plot the points close to the average (f=1) at the bottom.
        fs = dfm_wc['WeeklyPer100k'] / mean_per100k
        df = dfm_wc.loc[(fs >= fmin) & (fs < fmax)]
        cm = ax.scatter(
            df['HHsize'], df['Maandelijks%'],
            c=df['WeeklyPer100k'],
            s=np.sqrt(df['Population']) * pop_scale,
            cmap=cmap, vmin=vmin, vmax=vmax,
            # norm=matplotlib.colors.LogNorm(),
            zorder=zorder
            )

    # grid and y ticks
    ax.set_yscale('log')
    yt_vals = [3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    ax.set_yticks(yt_vals)
    yt_labs = [str(y) for y in yt_vals]
    ax.set_yticklabels(['' if y[0] in '689' else y for y in yt_labs])
    ax.grid(zorder=-10)

    # Plot colorbar
    cbar = fig.colorbar(cm)
    cyvals = cbar.get_ticks()
    cyvals = cyvals[cyvals >= 0]
    cbar.set_ticks(cyvals)

    # Annotate the outliers
    annot_mask = [
        dfm_wc['HHsize'] > 2.8,
        dfm_wc['HHsize'] < 1.85,
        dfm_wc['Maandelijks%'] > 65,
        dfm_wc['Maandelijks%'] < 5.5
        ]
    annot_muns = dfm_wc.index[np.any(annot_mask, axis=0)]
    for mun in annot_muns:
        xy = dfm_wc.loc[mun, ['HHsize', 'Maandelijks%']]
        annot_kwargs = {
            'horizontalalignment': 'right' if xy[0] > 3 else 'left',
            'verticalalignment': 'top' if xy[1] > 90 else 'bottom'
            }

        ax.annotate(mun, xy, zorder=99, **annot_kwargs)

    # Finalize plot

    title = f'Covidgevallen per 100k per week t/m {date_end.strftime("%Y-%m-%d")}'
    ax.set_title(f'{title}\nGemeentes met {pop_range[0]/1e3:.0f}k-{pop_range[1]/1e3:.0f}k inwoners.')
    fig.canvas.set_window_title(title)

    df_renamed = dfm_wc[['HHsize', 'Maandelijks%', 'WeeklyPer100k']].copy()
    df_renamed.rename(columns={'HHsize': 'HHgrootte',
                               'Maandelijks%':'Religieus%',
                               'WeeklyPer100k':'CovidGevallen'}, inplace=True)
    corr = df_renamed.corr()
    print(f'Correlation matrix:\n{np.around(corr, 2)}')

    fig.show()

    return dfm_wc, date_end

if __name__ == '__main__':


    nlcs.reset_plots()

    nlcs.init_data(autoupdate=True)

    plot_hhsize_relig_cases(pop_range=(0, 100e3))
    plot_hhsize_relig_cases(ref_date='2020-11-20', pop_range=(0, 100e3))
    # plot_trends_by_religious_visits(ndays=80, maxpop=30e3)


