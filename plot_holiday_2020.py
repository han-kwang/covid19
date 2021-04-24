#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 16 12:23:07 2021

@author: @hk_nien
"""


import pandas as pd
import matplotlib.pyplot as plt
import tools
import nlcovidstats as nlcs
import scipy.signal

if __name__ == '__main__':
    nlcs.reset_plots()
    nlcs.init_data(autoupdate=True)

    ndays = 130
    lastday = -60

    nlcs.plot_Rt(ndays=ndays, lastday=lastday, regions='HR:Noord,HR:Midden+Zuid')

    Rts = [] # Rt for North, Mid+South


    for region in ['HR:Noord', 'HR:Midden+Zuid']:
        df1, _npop = nlcs.get_region_data(region, lastday=lastday, correct_anomalies=True)
        source_col = 'Delta7r'

        # skip the first 10 days because of zeros
        Rt, delay_str = nlcs.estimate_Rt_series(df1[source_col].iloc[10:],
                                                delay=nlcs.DELAY_INF2REP)
        Rt = Rt.iloc[-ndays:]

        Rts.append(Rt)

    Rtdiff = Rts[0] - Rts[1]
    deltaR_smooth = scipy.signal.savgol_filter(Rtdiff.values, 13, 2)
    deltaR_smooth = pd.Series(deltaR_smooth, index=Rtdiff.index)

    fig, ax = plt.subplots(figsize=(10, 4), tight_layout=True)
    ax.plot(deltaR_smooth, label='Verschil Noord vs. Mid+Zuid', color='r')
    ax.set_ylabel(r'$\Delta R_t$')
    ax.axhline(0, color='k', linestyle='--')
    ax.legend()
    nlcs._add_restriction_labels(ax, Rt.index[0], Rt.index[-1])
    tools.set_xaxis_dateformat(ax)

    fig.show()

