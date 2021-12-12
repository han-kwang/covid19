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


# def download_ggd_tests(force=False):




# #%% R graph for daily Twitter update
if __name__ == '__main__':

    plt.close('all')
    nlcs.reset_plots()
    ggd_data.update_ggd_tests()

    nlcs.init_data(autoupdate=True)
    ncd.check_RIVM_message()
    print('---GGD tests---')
    ggd_tests.plot_daily_tests_and_delays('2021-09-15')
    # ggd_tests.plot_daily_tests_and_delays('2021-09-01', src_col='n_pos')
    plt.pause(0.25)
    print('--R calculation--')
    ggd_R.plot_rivm_and_ggd_positives(140, yscale=('log', 1000, 30000))
    plt.pause(0.25)
    ggd_R.plot_R_graph_multiple_methods(num_days=100)
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



