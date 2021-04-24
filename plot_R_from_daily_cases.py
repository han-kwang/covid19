#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Just plot the estimate of R based on daily cumulative case statistics.

See also nlcovidstats_show.py.
"""

import matplotlib.pyplot as plt
import tools
import nlcovidstats as nlcs


#%%
if __name__ == '__main__':

    nlcs.download_Rt_rivm(force=True)

    #%%

    nlcs.reset_plots()
    nlcs.init_data(autoupdate=True)
    print('--Corrected data--')
    nlcs.plot_Rt(ndays=120, lastday=-1, delay=nlcs.DELAY_INF2REP, source='r7', correct_anomalies=True)
    #print('--Raw data--')
    #nlcs.plot_Rt(ndays=120, lastday=-1, delay=nlcs.DELAY_INF2REP, source='r7', correct_anomalies=False)
    tools.pause_commandline()

    #%% Holiday regions
    nlcs.plot_Rt(regions=['HR:Midden+Noord', 'HR:Zuid'],
                 ndays=120, lastday=-1, delay=nlcs.DELAY_INF2REP, source='r7', correct_anomalies=True)


    #%% anomalies

    print('--Daily cases--')
    df=nlcs.get_region_data('Nederland')[0].iloc[-50:]

    fig, ax = plt.subplots(figsize=(10, 4), tight_layout=True);
    ax.plot(df.index, df['Delta_orig']*17.4e6, '^-', label='Ruwe data', markersize=3.5);
    ax.plot(df['Delta']*17.4e6, label='Schatting na correctie');

    mask = df.index.dayofweek == 3
    ax.plot(df.index[mask], df.loc[mask, 'Delta_orig']*17.4e6, 'go', label='Donderdagen')
    ax.legend()
    ax.set_yscale('log')
    ax.set_ylabel('Positieve gevallen per dag')
    tools.set_xaxis_dateformat(ax)
    ax.grid(which='minor', axis='y')
    title = 'Positieve tests per dag'
    ax.set_title(title)
    fig.canvas.set_window_title(title)
    fig.show()

