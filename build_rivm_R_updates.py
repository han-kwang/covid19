#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Create data/rivm_R_updates.csv from data/rivm_reproductiegetal-yyyy-mm-dd.csv files.

Created on Sat May  8 19:51:05 2021

@author: @hk_nien
"""

from pathlib import Path
import pandas as pd


csv_fnames = sorted(Path('data').glob('rivm_reproductiegetal-????-??-??.csv'))

# from each file, get the last entry with 'Rt_avg' defined.
first = True
records = None
for fn in csv_fnames:
    df = pd.read_csv(fn).set_index('Date')
    df = df.loc[df['population'] == 'testpos'][['Rt_avg']]
    df = df.loc[~df['Rt_avg'].isna()]

    if first:
        # get values for Fridays
        friday_select = (pd.to_datetime(df.index).dayofweek == 4)
        df = df.loc[friday_select]
        records = list(df.to_records())
        first =False
    else:
        records.append((df.index[-1], df['Rt_avg'][-1]))

df_updates = pd.DataFrame.from_records(records, columns=['Date', 'Rt_update']).set_index('Date')
df_updates.to_csv('data/rivm_R_updates.csv')
