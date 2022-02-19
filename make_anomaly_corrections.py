#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Create anomaly correction entries from "RIVM achterstanden".
Entries to be pasted into data/daily_numbers_anomalies.csv.


Created on Sat Feb  5 13:15:48 2022

@hk_nien
"""

import pandas as pd
import numpy as np
import nlcovidstats as nlcs

# Each row: publication date; backlog size according to RIVM.
BACKLOG_TXT = """\
    2022-01-17 0
    2022-01-18 5000
    2022-01-19 15000
    2022-01-20 27000
    2022-01-21 36000
    2022-01-22 46000
    2022-01-23 48000
    2022-01-24 46000
    2022-01-25 60000
    2022-01-26 78000
    2022-01-27 72000
    2022-01-28 76000
    2022-01-29 122000
    2022-01-30 131000
    2022-01-31 105000
    2022-02-01 81000
    2022-02-02 104000
    2022-02-03 113000
    2022-02-04 124000
    2022-02-05 176000
    2022-02-06 186000
    2022-02-07 191000
    2022-02-08 0
    """

def get_count_corrections():
    """Return DataFrame with datetime and count correction."""

    dates = []
    counts = []

    for li in BACKLOG_TXT.splitlines():
        f = li.split()
        if len(f) == 2:
            dates.append(pd.Timestamp(f[0]))
            counts.append(int(f[1]))
    df = pd.DataFrame(dict(backlog=counts), index=dates)
    df.index += pd.Timedelta('10 h')
    df['corr_count'] = df['backlog'].diff()
    df = df.iloc[1:].copy()
    return df


if __name__ == '__main__':
    nlcs.init_data(autoupdate=True)
    dfNL, n_pop = nlcs.get_region_data('Nederland', correct_anomalies=False)

    dfc = get_count_corrections()
    dfc['cases_unc'] = dfNL.loc[dfc.index, 'Delta'] * n_pop
    dfc['cases_cor'] = dfc['cases_unc'] + dfc['corr_count']
    dfc['relative_cor'] = np.around(dfc['corr_count'] / dfc['cases_unc'], 3)
    print(dfc)

    for tstamp, row in dfc.iterrows():
        tstamp = tstamp.strftime('%Y-%m-%d')
        backlog_k = row["backlog"]/1000
        corr_k = row['corr_count']/1000
        comment = f'Achterstand {backlog_k:.3g}k, correctie {corr_k:.3g}k'
        print(f'{tstamp},{row["relative_cor"]},0,"{comment}"')


