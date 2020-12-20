#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Just plot the estimate of R based on daily cumulative case statistics.

See also nlcovidstats_show.py.
"""


import tools
import nlcovidstats as nlcs

if __name__ == '__main__':
    nlcs.reset_plots()
    nlcs.init_data(autoupdate=True)
    nlcs.plot_Rt(ndays=120, lastday=-1, delay=nlcs.DELAY_INF2REP, source='r7')
    # nlcs.plot_Rt(ndays=120, lastday=-1, delay=nlcs.DELAY_INF2REP, source='r7', correct_anomalies=False)
    tools.pause_commandline()


