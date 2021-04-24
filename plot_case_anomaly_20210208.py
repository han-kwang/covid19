#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 10 15:39:15 2021

@author: @hk_nien
"""

import nlcovidstats as nlcs
import matplotlib.pyplot as plt
import tools
nlcs.init_data(autoupdate=True)

plt.close('all')

#%%
df=nlcs.get_region_data('Nederland')[0].iloc[-50:]
fig, ax = plt.subplots(figsize=(10, 4), tight_layout=True);
ax.plot(df.index, df['Delta_orig']*17.4e6, '^-', label='orig', markersize=3.5);
ax.plot(df['Delta']*17.4e6, label='corrected');
ax.plot(df.index[::7], df['Delta_orig'][::7]*17.4e6, 'bo')
ax.plot(df.index[6::7], df['Delta_orig'][6::7]*17.4e6, 'go')
ax.legend()
ax.set_yscale('log')
ax.set_ylabel('Positieve gevallen per dag')
tools.set_xaxis_dateformat(ax)
ax.grid(which='minor', axis='y')
fig.show()
