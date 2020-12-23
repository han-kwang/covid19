#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 22 19:16:09 2020

@author: hnienhuy
"""
import json
import pandas as pd
import nlcovidstats as nlcs
import numpy as np

nlcs.reset_plots()
nlcs.init_data(autoupdate=True)

# Religion per municipality
# Index: Gemeente; columns: 'Provincie', 'Maandelijks%', 'Religieus%',
# 'Katholiek%', 'Hervormd%', 'Gereformeerd%', 'PKN%', 'Islam%', 'Joods%',
# 'Hindoe%', 'Boeddhist%', 'Anders%']
dfr = pd.read_csv('data/Religie_en_kerkbezoek_naar_gemeente_2010-2014.csv',
                comment='#', index_col=False)
dfr = dfr.set_index('Gemeente').drop(columns=['dummy', 'Gemcode'])

# Remove municipalities with no data
dfr.drop(index=dfr.index[dfr['Religieus%'] == '.'], inplace=True)

# Convert to numbers
for col in dfr.columns:
    if col.endswith('%'):
        dfr[col] = dfr[col].astype(float)

# Combine with 2020 population data.
# Note that religion data is from 2014 and municipalities have changed in the meantime.
# Remove municipalities that don't match municipalities of 2020.
dfm = nlcs.DFS['mun'].copy()
dfm = dfm.join(dfr, how='inner')

# Select <30k population
dfm   = dfm.loc[dfm['Population'] < 3e4]
print(f'Statistics for {len(dfm)} municipalities.')

# Find quantiles. DataFrame with index q, columns 'Maandelijks%' etc.
qs = np.array([0, 0.4, 0.8, 0.9, 1.0])
dfq = dfm.quantile(q=qs, axis='index')
regions = ['Nederland']

for i in range(1, len(qs)):
    qa, qb = qs[i-1], qs[i]
    maa, mab = dfq.at[qa, 'Maandelijks%'], dfq.at[qb, 'Maandelijks%']
    select = (dfm['Maandelijks%'] >= maa-0.01) & (dfm['Maandelijks%'] <= mab+0.01)
    muns = list(dfm.index[select])
    mean = dfm.loc[select]['Population'].mean()
    print(f'Quantile {qa}-{qb}: Maandelijks% {maa:.1f}-{mab:.1f}, Mean pop: {mean/1e3:.1f}k\n --> {", ".join(muns)}\n')
    rdict = dict(label=f'{maa:.0f}%-{mab:.0f}%', muns=muns)
    regions.append('JSON:' + json.dumps(rdict))


nlcs.plot_daily_trends(80, region_list=regions,
                       subtitle='Gemeentes < 30k inwoners, naar maandelijks kerkbezoek')


