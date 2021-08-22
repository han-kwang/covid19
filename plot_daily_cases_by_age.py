#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 20 11:13:16 2021

@author: @hk_nien
"""

from multiprocessing import Pool
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import casus_analysis as ca
import tools
import nlcovidstats as nlcs

pd.options.display.width=120
pd.options.display.max_colwidth=12
pd.options.display.max_columns=10



fdate = pd.to_datetime('2020-09-01')
tm_now = pd.to_datetime('now')
fdates = []
while fdate < tm_now:
    fdates.append(fdate.strftime('%Y-%m-%d'))
    fdate += pd.Timedelta(1, 'd')


def get_cumulatives_one_date(fdate):
    try:
        df = ca.load_casus_data(fdate)
    except FileNotFoundError:
        print('*', end='', flush=True)
        return None

    totals = df[['Agegroup', 'Date_file']].groupby('Agegroup').count()['Date_file']
    print('.', end='', flush=True)
    return totals

with Pool() as pool:
    # list, each entry is a Series, with Agegroup as index.
    cumulatives = pool.map(get_cumulatives_one_date, fdates)
    print()

#%% construct delta dataframe

# df will be a DateFrame with index: Date_file; columns: age groups;
# values: daily changes.
# df_fracs will be dataframe with fractions of the total.

dfx = None
for fdate, cum in zip(fdates, cumulatives):
    if cum is None:
        break
    dfy = pd.DataFrame(index=cum.index, data={fdate: cum.values})
    if dfx is None:
        dfx = dfy
    else:
        dfx[fdate] = dfy[fdate]
df = dfx.T.diff().iloc[1:]
df.index.name = 'Date_file'
df.index = pd.to_datetime(df.index)
del dfx, dfy

age_groups = ['0-9', '10-19', '20-29', '30-39', '40-49', '50-59', '60-69',
              '70-79', '80-89', '90+']

data = df.loc[:, age_groups].values
data *= (100/data.sum(axis=1).reshape(-1, 1))
df_pct = pd.DataFrame(data, index=df.index, columns=age_groups)




#%% plot barchart

def _series2steps(s, step='1 d'):
    """oversample series to make suitable for line plots rather than bar charts.

    Return x values, y values.
    """

    step = pd.Timedelta(step)
    xs = np.vstack([s.index + step*0.15, s.index + step*0.85]).T.ravel()
    ys = np.vstack([s.values, s.values]).T.ravel()
    return xs, ys

nlcs.init_data()

fig, axs = plt.subplots(2, 1, figsize=(10, 6), tight_layout=True, sharex=True)

bar_bottoms_n = None
bar_bottoms_f = None

# For the bottom panel, store y positions here as tuples (y, txt).
flabels = []

for agroup in age_groups:

    sn = df[agroup]
    sf = df_pct[agroup]
    xs, ns = _series2steps(sn)
    _, fs = _series2steps(sf)
    if bar_bottoms_n is None:
        bar_bottoms_n = pd.Series(0, index=xs)
        bar_bottoms_f = pd.Series(0, index=xs)
    axs[0].fill_between(xs, bar_bottoms_n, bar_bottoms_n+ns, label=agroup)
    axs[1].fill_between(xs, bar_bottoms_f, bar_bottoms_f+fs, label=agroup)

    flabels.append((bar_bottoms_f[-1] + fs[-1]*0.5, agroup))

    bar_bottoms_n += ns
    bar_bottoms_f += fs

axs[0].set_ylim(bar_bottoms_n.max()*(-0.01), bar_bottoms_n.max()*1.01)
axs[1].set_ylim(-1, 101)
axs[1].set_xlim(xs[0], xs[-1])

handles, labels = axs[0].get_legend_handles_labels()
axs[0].legend(handles[::-1], labels[::-1], loc='upper left', bbox_to_anchor=(1.01, 1))
axs[0].set_title('Aantal posititieve tests per dag naar leeftijd en rapportagedatum')
axs[1].set_title('Percentages posititieve tests naar leeftijd en rapportagedatum')

nlcs.add_labels(axs[1], flabels, xpos=xs[-1] + (xs[-1]-xs[0])*0.02, logscale=False,
                mindist_scale=2.5)

nlcs._add_event_labels(axs[0], xs[0], xs[-1], with_ribbons=False, textbox=True,
                             bottom=False)

for ax in axs:
    tools.set_xaxis_dateformat(ax)
fig.show()



