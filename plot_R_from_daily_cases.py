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
    'Eindhoven', 'Groningen', 'Tilburg', 'Almere',
    ]
province_list = [
    'Zuid-Holland', 'Noord-Holland', 'Noord-Brabant', 'Gelderland',
   'Utrecht', 'Overijssel', 'Limburg', 'Friesland', 'Groningen', 'Drenthe',
   'Flevoland', 'Zeeland'
   ]
province_list = ['Nederland'] + [f'P:{x}' for x in province_list]


#%% R graph for daily Twitter update
if __name__ == '__main__':

    nlcs.download_Rt_rivm(force=True)
    nlcs.check_RIVM_message()
    nlcs.reset_plots()
    nlcs.init_data(autoupdate=True)
    print('--Corrected data--')
    nlcs.construct_Dfunc(nlcs.DELAY_INF2REP, plot=True)
    nlcs.plot_Rt(ndays=120, lastday=-1, delay=nlcs.DELAY_INF2REP, source='r7', correct_anomalies=True,
                 ylim=(0.5, 1.5))

    #%%



    nlcs.plot_Rt(regions=['Nederland', 'Amsterdam'],
                 ndays=120, lastday=-1, delay=nlcs.DELAY_INF2REP, source='r7', correct_anomalies=True,
                 ylim=(0.5, 1.5))


    #%% R by holiday rogion

    import nlcovidstats as nlcs
    nlcs.reset_plots()
    nlcs.init_data(autoupdate=True)

    nlcs.plot_Rt(ndays=55, lastday=-1, delay=nlcs.DELAY_INF2REP, source='r7', correct_anomalies=True,
             regions=['HR:Noord', 'HR:Midden', 'HR:Zuid'], ylim=(0.6, 1.2),
             only_trendlines=False)

    #%%
    nlcs.plot_daily_trends(
        45,  region_list=['HR:Noord', 'HR:Midden', 'HR:Zuid']
    )


    #%% R graph of cities, provinces
    #nlcs.plot_Rt(ndays=200, lastday=-1, delay=nlcs.DELAYhttps://nos.nl/liveblog/2386246_INF2REP, source='r7', correct_anomalies=True)
    nlcs.plot_Rt(ndays=42, lastday=-1, delay=nlcs.DELAY_INF2REP, source='r7', correct_anomalies=True,
                 regions=city_list, only_trendlines=True, ylim=(0.5, 1.5))
    nlcs.plot_Rt(ndays=42, lastday=-1, delay=nlcs.DELAY_INF2REP, source='r7', correct_anomalies=True,
                 regions=province_list, only_trendlines=True, ylim=(0.5, 1.5))
    #print('--Raw data--')
    #nlcs.plot_Rt(ndays=120, lastday=-1, delay=nlcs.DELAY_INF2REP, source='r7', correct_anomalies=False)

    # plt.close('all')
    # nlcs.plot_daily_trends(45, region_list=city_list)

    # nlcs.plot_daily_trends(45, region_list=province_list)
    tools.pause_commandline()
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
    import nlcovidstats as nlcs
    nlcs.init_data()
    nlcs.plot_anomalies(-100, None)
    #nlcs.plot_anomalies(-180, -70)
