#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""New daily plot generation - using GGD data.

Created on Tue Nov 16 22:02:09 2021

@hk_nien
"""

import matplotlib.pyplot as plt
import tools
import nlcovidstats as nlcs
import nlcovidstats_data as ncd
import pandas as pd

import plot_aantal_tests_updates as ggd_tests
import calc_R_from_ggd_tests as ggd_R
import ggd_data
import os
import tools


# #%% R graph for daily Twitter update
if __name__ == '__main__':

    plt.close('all')
    tools.wait_for_refresh('15:00:00', '15:15:00')
    nlcs.reset_plots()
    nlcs.init_data(autoupdate=True)
    ggd_data.update_ggd_tests()
    ncd.check_RIVM_message()
    print('---GGD tests---')
    ggd_tests.plot_daily_tests_and_delays('2021-10-01')
    # ggd_tests.plot_daily_tests_and_delays('2021-09-01', src_col='n_pos')
    plt.pause(0.25)
    print('--R calculation--')
    ggd_R.plot_rivm_and_ggd_positives(90, yscale=('log', 2500, 50000))
    plt.pause(0.25)
    ggd_R.plot_R_graph_multiple_methods(num_days=100, ylim=(0.6, 2))
    plt.gcf().get_axes()[0].legend(loc='upper left')
    plt.pause(0.25)
    nlcs.construct_Dfunc(nlcs.DELAY_INF2REP, plot=True)

    #%%
    if 0:
        #%% check recent anomaly correction
        plt.close('all')
        nlcs.init_data(autoupdate=True)
        #ggd_R.plot_rivm_and_ggd_positives(25, yscale=('linear', 7000, 25000))
        #plt.pause(0.4)
        ggd_R.plot_rivm_and_ggd_positives(25, True, yscale=('linear', 5000, 25000))

        #%% Show TvT performance
        ggd_R.plot_R_graph_multiple_methods(
            num_days=240, ylim=(-0.9, 4.2),
            methods=('rivm', 'melding', 'tvt'),
            )
        #%% GGD tests in regions
        fig, ax = plt.subplots(figsize=(7, 4), tight_layout=True)
        for rre in ['Utrecht', 'Midden- en West-Brabant', 'Groningen', 'Drenthe', 'Twente']:
            df = ggd_data.load_ggd_pos_tests(region_regexp=rre)
            ax.step(df.index[-100:], df['n_tested'][-100:], where='mid', label=rre)
        ax.set_yscale('log')
        ax.set_title('Uitgevoerde GGD tests per regio')
        ax.set_xlabel('Datum monstername')
        ax.legend()
        import tools
        tools.set_xaxis_dateformat(ax)
        fig.show()

