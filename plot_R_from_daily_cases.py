#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Just plot the estimate of R based on daily cumulative case statistics.

See also nlcovidstats_show.py.
"""

import matplotlib.pyplot as plt
import tools
import nlcovidstats as nlcs
import pandas as pd

city_list = [
    'Nederland', 'Amsterdam', 'Rotterdam', "'s-Gravenhage", 'Utrecht',
    'Eindhoven', 'Groningen', 'Tilburg', 'Enschede'
    ]
province_list = [
    'Zuid-Holland', 'Noord-Holland', 'Noord-Brabant', 'Gelderland',
   'Utrecht', 'Overijssel', 'Limburg', 'Friesland', 'Groningen', 'Drenthe',
   'Flevoland', 'Zeeland'
   ]
province_list = ['Nederland'] + [f'P:{x}' for x in province_list]


#%%g
if __name__ == '__main__':

    nlcs.download_Rt_rivm(force=True)

    nlcs.reset_plots()
    nlcs.init_data(autoupdate=True)
    print('--Corrected data--')
    nlcs.construct_Dfunc(nlcs.DELAY_INF2REP, plot=True)
    nlcs.plot_Rt(ndays=120, lastday=-1, delay=nlcs.DELAY_INF2REP, source='r7', correct_anomalies=True)
    #nlcs.plot_Rt(ndays=200, lastday=-1, delay=nlcs.DELAY_INF2REP, source='r7', correct_anomalies=True)
    #nlcs.plot_Rt(ndays=42, lastday=-1, delay=nlcs.DELAY_INF2REP, source='r7', correct_anomalies=True,
    #             regions=city_list, only_trendlines=True)
    #nlcs.plot_Rt(ndays=42, lastday=-1, delay=nlcs.DELAY_INF2REP, source='r7', correct_anomalies=True,
    #             regions=province_list, only_trendlines=True)
    #print('--Raw data--')
    #nlcs.plot_Rt(ndays=120, lastday=-1, delay=nlcs.DELAY_INF2REP, source='r7', correct_anomalies=False)
    tools.pause_commandline()

    #%% Holiday regions
    # nlcs.plot_Rt(regions=['HR:Midden+Noord', 'HR:Zuid'],
    #              ndays=120, lastday=-1, delay=nlcs.DELAY_INF2REP, source='r7', correct_anomalies=True)

    #%%
    plt.close('all')
    nlcs.plot_daily_trends(45, region_list=city_list)

    nlcs.plot_daily_trends(45, region_list=province_list)
    #%%
    univ_towns = [
         'Amsterdam',
         'Rotterdam',
         'Utrecht',
         'Eindhoven',
         'Groningen',
         'Tilburg',
         'Nijmegen',
         'Enschede',
         'Leiden',
         'Maastricht',
         'Delft',
         'Wageningen'
         ]
    nlcs.plot_daily_trends(
        30,  region_list=['Nederland'] + univ_towns
    )


    #%% anomalies

    print('--Daily cases--')
    df=nlcs.get_region_data('Nederland')[0].iloc[-50:]

    fig, ax = plt.subplots(figsize=(10, 4), tight_layout=True);
    width = pd.Timedelta('10 h')
    ax.bar(df.index-width/2, df['Delta_orig']*17.4e6, width=width,  label='Ruwe data')
    ax.bar(df.index+width/2, df['Delta']*17.4e6, width=width, label='Schatting na correctie')

    mask = df.index.dayofweek == 3
    ax.plot(df.index[mask]-width/2, df.loc[mask, 'Delta_orig']*17.4e6, 'gv', markersize=8, label='Donderdagen')
    ax.legend()
    ax.set_yscale('log')
    ax.set_ylabel('Positieve gevallen per dag')
    tools.set_xaxis_dateformat(ax)
    ax.grid(which='minor', axis='y')
    title = 'Positieve tests per dag'
    ax.set_title(title)
    fig.canvas.set_window_title(title)
    fig.show()

