#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""This is to find out why my R peaks at a value around 2021-07-01, that is
much higher than RIVM's.

Created on Fri Jul 23 12:52:53 2021

@author: hk_nien
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import tools
import nlcovidstats as nlcs


def get_Rt_rivm(mindate, maxdate):
    """Return Series with R(rivm). Note timestamps are always at time 12:00:00."""

    df_rivm = nlcs.DFS['Rt_rivm'].copy()

    # get 4 days extra from 'prognosis'
    prog = df_rivm.loc[df_rivm['R'].isna()].iloc[:4]
    prog_R = np.around(np.sqrt(prog['Rmin']*prog['Rmax']), 2)

    df_rivm.loc[prog_R.index, 'R'] = prog_R
    R_rivm = df_rivm.loc[~df_rivm['R'].isna(), 'R']

    return R_rivm.loc[(R_rivm.index >= mindate) & (R_rivm.index <= maxdate)]

def get_Rt_mine(mindate, maxdate, slide_delay=True, cdf=None):
    """Return my Rt estimate, sampled at 12:00 daily.

    Optionally provide cdf as test case; DataFrame with time index and
    'Delta7r' column (7-day rolling average daily positive cases).
    """
    from scipy.interpolate import interp1d

    delay = nlcs.DELAY_INF2REP if slide_delay else 4.0

    if cdf is None:
        cdf, _npop = nlcs.get_region_data('Nederland', lastday=-1, correct_anomalies=True)
    Rdf = nlcs.estimate_Rt_df(cdf['Delta7r'].iloc[10:], delay=delay, Tc=4.0)

    r_interp = interp1d(
        Rdf.index.astype(np.int64), Rdf['Rt'], bounds_error=False,
        fill_value=(Rdf['Rt'].iloc[0], Rdf['Rt'].iloc[-1])
        )

    tlims = [pd.to_datetime(t).strftime('%Y-%m-%dT12:00')
             for t in [mindate, maxdate]
             ]
    index = pd.date_range(*tlims, freq='1d')
    R_mine = pd.Series(r_interp(index.astype(int)), index=index)
    return R_mine


def get_Rt_test_case(mindate, maxdate, case='step', slide_delay=True):

    index = pd.date_range('2021-01-01', 'now', freq='1d')
    cdf = pd.DataFrame(index=index + pd.Timedelta(4, 'd'))
    cdf['Delta7r'] = 1000

    if case == 'step':
        cdf.loc[index >= '2021-07-01', 'Delta7r'] = 10000
        # sudden factor 10 increase should result in
        # 1 day R=1e+4 or 2 days R=1e+2, which is the case.
    else:
        raise ValueError(f'case={case!r}')


    return get_Rt_mine(mindate, maxdate, slide_delay=slide_delay, cdf=cdf)





#%%
if __name__ == '__main__':

    nlcs.reset_plots()
    nlcs.init_data(autoupdate=True)
    #%%

    Rt_mine_fixD = get_Rt_mine('2021-06-22', '2021-07-20', slide_delay=False)
    Rt_mine_varD = get_Rt_mine('2021-06-22', '2021-07-20', slide_delay=True)


    Rt_rivm = get_Rt_rivm('2021-06-22', '2021-07-13')

    # Rt_mine = get_Rt_test_case('2021-06-22', '2021-07-09', 'step', slide_delay=False)

    cases_df, n_pop = nlcs.get_region_data('Nederland')
    cases = cases_df['Delta'] * n_pop
    cases7 = cases_df['Delta7r'] * n_pop
    cases_mask = (cases.index >= '2021-06-22') & (cases.index <= '2021-07-23')
    day = pd.Timedelta(1, 'd')
    cases = cases.loc[cases_mask]
    cases7 = cases7.loc[cases_mask]

    plt.close('all')
    fig, (axR, axC) = plt.subplots(2, 1, tight_layout=True, sharex=True,
                                   figsize=(6, 7))
    Tgen = 4.0 # generation interval

    cases_scale = 590

    for Rt, label, delay, marker in [
            (Rt_mine_fixD, 'hk_nien, fixD', 4.5*day, 'o'),
            #(Rt_mine_varD, 'hk_nien, varD', 5.5*day, 'o'),
            (Rt_rivm, 'RIVM', 4*day, '^')
            ]:
        axR.plot(Rt, marker=marker, label=label)
        cases_from_R = Rt.cumprod() ** (1/Tgen)
        axC.plot(
            cases_from_R.index + delay,
            cases_from_R.values * cases_scale,
            marker=marker, label=f'Volgens R[{label}]'
            )

    axC.plot(cases.index, cases.values, marker='*', linestyle='', label='Gerapporteerd')
    axC.plot(cases7, marker='v', label='Gerapporteerd (7d gemid.)')

    axR.set_ylabel('Rt')
    axC.set_ylabel('Aantal positief')
    axC.set_yscale('log')
    axR.legend()
    axC.legend()
    axC.annotate(f'Hier {cases_scale}',
                 (Rt.index[0] + delay, cases_scale),
                 xytext=(Rt.index[0] + delay, cases_scale * 2),
                 arrowprops=dict(arrowstyle='->'),
                 horizontalalignment='center')

    axR.set_title('R-schattingen')
    axC.set_title('R teruggerekend naar aantal positief per dag:\n'
                  r'$n_{pos}(t) = n_{pos}(t-1) \times R^{1/4}$')




    tools.set_xaxis_dateformat(axC)
    tools.set_xaxis_dateformat(axR)

    fig.show()

