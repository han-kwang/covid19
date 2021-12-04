#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 30 17:22:06 2021

@author: @hk_nien
"""
import pandas as pd
import matplotlib.pyplot as plt
import tools


df=pd.read_csv('data-rivm/tests/rivm_daily_2021-11-30.csv.gz')

df['sdate'] = pd.to_datetime(df['Date_of_statistics'])
df.set_index('sdate', inplace=True)
fig, ax = plt.subplots(tight_layout=True, figsize=(15, 5))


for rolling7 in [False, True]:

    for i, sr in enumerate(sorted(df['Security_region_name'].unique())):
        s = df.loc[df['Security_region_name']==sr, 'Tested_with_result']
        if rolling7:
            s = s.rolling(7).mean()


        lsty = ['-', '--', '-.', ':'][i//10]
        ax.plot(s, label=sr, linestyle=lsty)

ax.grid()
ax.legend(ncol=2, loc='upper left', bbox_to_anchor=(1, 1))
ax.set_xlim(s.index[-70], s.index[-1])
ax.set_title('GGD uitgevoerde tests per regio')
tools.set_xaxis_dateformat(ax)
fig.show()