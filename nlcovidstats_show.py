#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate plots related to daily cumulative case statistics.

Running nlcovidstats.py directly generates too many plots.
"""


import tools
import nlcovidstats as nlcs


if __name__ == '__main__':
    # Which plots to show.
    # RtD9 is not included here (its purpose is to show bad data, not labeled explicitly).
    show_plots = 'Rtosc,trends,trends-pop,trends-prov,Rt,RtRegion,delay,anomalies,dowcorr'.split(',')

    # Uncomment and change line below to do a selection of plots rather than
    # all of them.
    # show_plots = 'Rt,dowcorr'.split(',')
    nlcs.reset_plots()
    nlcs.init_data(autoupdate=True)

    if 'Rtosc' in show_plots:
        nlcs.plot_Rt_oscillation()

    if 'dowcorr' in show_plots:
        nlcs.get_dow_correction((-50, -1), verbose=True)

    if 'trends' in show_plots:
        nlcs.plot_daily_trends(
            ndays=100, lastday=-1,
            region_list=nlcs.get_municipalities_by_pop(2e5, 9e5))

    if 'trends-pop' in show_plots:
        pop_regions = ['Nederland', 'POP:200-900', 'POP:80-200', 'POP:30-80',
                        'POP:0-30']
        nlcs.plot_daily_trends(
            ndays=90, lastday=-1,
            region_list=pop_regions)
        nlcs.plot_cumulative_trends(ndays=200, regions=pop_regions)

    if 'trens-prov' in show_plots:
        regions = [
            f'P:{p}'
            for p in nlcs.DFS['mun']['Province'].drop_duplicates().sort_values()
            ]
        nlcs.plot_daily_trends(
            ndays=60, lastday=-1,
            region_list=['Nederland'] + regions)

    if 'delay' in show_plots:
        nlcs.construct_Dfunc(nlcs.DELAY_INF2REP, plot=True)

    if 'RtD9' in show_plots:
        nlcs.plot_Rt(ndays=120, lastday=-1, delay=9)

    if 'Rt' in show_plots:
        nlcs.plot_Rt(ndays=120, lastday=-1, delay=nlcs.DELAY_INF2REP)

    if 'RtRegion' in show_plots:
        nlcs.plot_Rt(ndays=40, lastday=-1, delay=nlcs.DELAY_INF2REP,
                     regions='HR:Noord,HR:Midden+Zuid,POP:80-900,POP:0-80')



    if 'anomalies' in show_plots:
        nlcs.plot_anomalies_deltas(ndays=120)


    # pause (for command-line use)
    tools.pause_commandline('Press Enter to end.')
