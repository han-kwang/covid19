#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 17 15:31:12 2021

@author: @hk_nien
"""

import pandas as pd
import matplotlib.pyplot as plt
import tools

dft_full = pd.read_csv('data/COVID-19_uitgevoerde_testen.csv', sep=';')
for col in ['Date_of_report', 'Date_of_statistics']:
    dft_full[col] = pd.to_datetime(dft_full[col])

dft_full['Date_of_statistics'] = dft_full['Date_of_statistics'] + pd.Timedelta('10 h')

dft = dft_full.groupby('Date_of_statistics').sum()
dft['perc_positive'] = dft['Tested_positive']/dft['Tested_with_result'] * 100

fig, axs = plt.subplots(2, 1, figsize=(10, 6), tight_layout=True, sharex=True)
ax = axs[0]

ax.plot(dft['Tested_with_result'].iloc[-50:])
title = 'Dagelijks afgenomen tests bij GGD'
ax.set_title(title)
fig.canvas.manager.set_window_title(title)

ax = axs[1]
ax.set_title('Percentage positief')
ax.plot(dft['perc_positive'].iloc[-50:])


tools.set_xaxis_dateformat(axs[0])
tools.set_xaxis_dateformat(axs[1])
fig.show()

#%% oost-west

vrcodes = {
    'all': set(x for x in dft_full['Security_region_code'].unique()
               if isinstance(x, str))
}

vrcodes['low_pollen_20210226'] = {
    'VR01', 'VR02', 'VR03', 'VR25', 'VR10', 'VR11', 'VR12', 'VR13',
    'VR14', 'VR15', 'VR16', 'VR17', 'VR18', 'VR19', 'VR24'
    }

vrcodes['high_pollen_20210226'] = vrcodes['all'] - vrcodes['low_pollen_20210226']

dfts = {}

for key, vrs in vrcodes.items():
    _df = dft_full.loc[dft_full['Security_region_code'].isin(vrs)]
    _df = _df.groupby('Date_of_statistics').sum()
    _df['perc_positive'] = _df['Tested_positive']/_df['Tested_with_result'] * 100
    dfts[key] = _df

fig, axs = plt.subplots(2, 1, figsize=(10, 6), tight_layout=True, sharex=True)
ax = axs[0]
title = 'Dagelijks afgenomen tests bij GGD'
axs[0].set_title(title)
fig.canvas.manager.set_window_title(title)
axs[1].set_title('Percentage positief')

for key in ['low_pollen_20210226', 'high_pollen_20210226']:
    dft = dfts[key]
    axs[0].plot(dft['Tested_with_result'].iloc[-50:], label=key)
    axs[1].plot(dft['perc_positive'].iloc[-50:], label=key)

for ax in axs:
    ax.legend()
    tools.set_xaxis_dateformat(ax)

fig.show()
