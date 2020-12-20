#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate plots related to daily cumulative case statistics.

Running nlcovidstats.py directly generates too many plots.
"""


import tools
import nlcovidstats as nlcs


if __name__ == '__main__':
    # Which plots to show.
    show_plots = 'Rtosc,trends,RtD9,Rt,RtRegion,delay,anomalies'.split(',')

    # Uncomment and change line below to do a selection of plots rather than
    # all of them.
    # show_plots = 'RtD9,Rt,delay'.split(',')
    nlcs.reset_plots()
    nlcs.init_data(autoupdate=True)

    if 'Rtosc' in show_plots:
        nlcs.plot_Rt_oscillation()

    if 'trends' in show_plots:
        nlcs.plot_daily_trends(ndays=100, lastday=-1, minpop=2e5)

    if 'delay' in show_plots:
        nlcs.construct_Dfunc(nlcs.DELAY_INF2REP, plot=True)

    if 'RtD9' in show_plots:
        nlcs.plot_Rt(ndays=120, lastday=-1, delay=9)

    if 'Rt' in show_plots:
        nlcs.plot_Rt(ndays=120, lastday=-1, delay=nlcs.DELAY_INF2REP)

    if 'RtRegion' in show_plots:
        nlcs.plot_Rt(ndays=40, lastday=-1, delay=nlcs.DELAY_INF2REP,
                     regions='HR:Noord,HR:Midden+Zuid')

    if 'anomalies' in show_plots:
        nlcs.plot_anomalies_deltas(ndays=120)


    # pause (for command-line use)
    tools.pause_commandline('Press Enter to end.')
