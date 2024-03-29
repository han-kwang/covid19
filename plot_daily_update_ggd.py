#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""New daily plot generation - using GGD data.

Created on Tue Nov 16 22:02:09 2021

@hk_nien
"""
import sys
import matplotlib.pyplot as plt
import tools
import nlcovidstats as nlcs
import nlcovidstats_data as ncd
# import pandas as pd
import numpy as np

import plot_aantal_tests_updates as ggd_tests
import calc_R_from_ggd_tests as ggd_R
import ggd_data

def check_anomalies():
    """Abort if anomalies not up to date."""
    cdf = nlcs.DFS['cases']
    adf = nlcs.DFS['anomalies']

    date = cdf.iloc[-1]['Date_of_report']
    if date not in adf.index or np.any(np.abs(adf.loc[date, 'fraction']) < 0.011):
        print('---')
        tools.print_warn(
            f'Anomalies not up to date for {date}.\n'
            'Update/run make_anomaly_corrections.py and update data/daily_numbers_anomalies.csv'
            )
        sys.exit(1)


# #%% R graph for daily Twitter update
if __name__ == '__main__':

    plt.close('all')
    tools.wait_for_refresh('15:00:00', '15:16:00')
    ncd.check_RIVM_message()
    nlcs.reset_plots()
    nlcs.init_data(autoupdate=True)
    ggd_data.update_ggd_tests()
    # check_anomalies()

    print('---GGD tests---')
    ggd_tests.plot_daily_tests_and_delays(-(13*7+3))
    ggd_tests.plot_daily_tests_and_delays(-(13*7+3), region_re='HR:Zuid')
    # ggd_tests.plot_daily_tests_and_delays('2021-09-01', src_col='n_pos')
    plt.pause(0.25)
    print('--R calculation--')
    ggd_R.plot_rivm_and_ggd_positives(
        90, yscale=('log', 2500, 400000),
        ggd_regions=['Landelijk', 'HR:Noord', 'HR:Midden', 'HR:Zuid']
        )
    plt.pause(0.25)

    ggd_R.plot_R_graph_multiple_methods(
        num_days=70, ylim=(0.3, 3),
        methods=('rivm', 'melding', 'ggd_der', 'tvt', 'ggd_regions')
        )
    fig = plt.gcf()
    fig.figure.set_size_inches(10, 6.5)
    plt.pause(0.25)
    # nlcs.plot_Rt(
    #     60, regions=['Nederland', 'HR:Noord', 'HR:Midden', 'HR:Zuid'],
    #     ylim=(0.5, 2.5)
    #     )

    # nlcs.construct_Dfunc(nlcs.DELAY_INF2REP, plot=True)

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

