#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 13 08:34:25 2021

@author: @hk_nien on Twitter
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tools

def load_summary_df():
    """Get cases summary DataFrame.

    DataFrame will have columns 'sdate', 'fdate' for Date_statistics,
    Date_file. Integer index.
    """
    df = pd.read_csv('data/casus_history_summary.csv', usecols=[1, 2, 3, 4, 5, 6])
    df = df.rename(columns=dict(Date_statistics='sdate', Date_file='fdate'))
    date_columns = ['sdate', 'fdate']
    for c in date_columns:
        df[c] = pd.to_datetime(df[c])

    fdates = pd.DatetimeIndex(sorted(df['fdate'].unique()))

    # Add changes in DPL (by report date)
    # This can probably be done with a smart join operation,
    # but this'll do the job.
    Dcolumns = ['DON', 'DPL', 'DOO', 'Dtot']
    for c in Dcolumns:
        df[f'd{c}'] = 0

    for fdate in fdates[280:]:
        df_prev = df.loc[df['fdate'] == fdate - pd.Timedelta(1, 'd')]
        df_tody = df.loc[df['fdate'] == fdate]

        for c in Dcolumns:
            ic = df_tody.columns.get_loc(c)
            df.loc[df_tody.index[:-1], f'd{c}'] = \
                df_tody.iloc[:-1, ic].values - df_prev.iloc[:, ic].values
            df.at[df_tody.index[-1], f'd{c}'] = df.at[df_tody.index[-1], c]
        print('.', end='', flush=True)
    return df


def plot_casus_summary(df, sd_rel_irange=(-21, 0), fd_irange=(-21, None)):
    """
    - sd_rel_irange range relative to fdate to consider (in days).
      (interpreted as slice; None and/or third value allowed)
    - fd_irange: range of file dates to consider (in days).
    """

    fdates = pd.DatetimeIndex(sorted(df['fdate'].unique()))

    fd_select = fdates[slice(*fd_irange)]

    fig, axs = plt.subplots(2, 1, tight_layout=False, sharex=True)
    fig.subplots_adjust(
        top=0.97,
        bottom=0.122,
        left=0.094,
        right=0.7,
        hspace=0.171,
        wspace=0.2
        )
    axs[-1].set_xlabel('date_stat - date_file (d)')

    lcolors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']
    lstyles = ['-', '--', '-.', ':']

    descs = {'dDPL':'Positive lab result', 'dDOO':'Disease onset'}

    # for each fdate the mean DOO delay (positive value).
    doo_delays = []

    for i_fdate, fdate in enumerate(fd_select):
        row_select = (
            (df['fdate'] == fdate) &
            (df['sdate'] >= fdate + pd.Timedelta(sd_rel_irange[0], 'd')) &
            (df['sdate'] <= fdate + pd.Timedelta(sd_rel_irange[1], 'd'))
            )
        df1 = df.loc[row_select].copy()
        # delta time
        df1['date_diff'] = (df1['sdate'] - df1['fdate']).astype(np.int64) / 86400e9

        lsty = lstyles[i_fdate // 7]
        lcol = lcolors[i_fdate % 7]

        for c, ax in zip(['dDPL', 'dDOO'], axs):
            ax.plot(df1['date_diff'], df1[c]/df1['dDtot'].sum(),
                    color=lcol, linestyle=lsty,
                    label=fdate.strftime('%Y-%m-%d %a'))
            if i_fdate == 0:
                ax.text(0.01, 0.95, f'{c}: {descs[c]}',
                        transform=ax.transAxes,
                        verticalAlignment='top')

        doo_delays.append(
            -np.sum(df1['dDOO'] * df1['date_diff']) / df1['dDOO'].sum()
            )


    for ax in axs:
        ax.grid()

    # axs[0].set_title('Each line is one Date_file')
    from matplotlib.ticker import MaxNLocator
    axs[-1].xaxis.set_major_locator(MaxNLocator('auto', steps=[7]))
    axs[-1].xaxis.set_minor_locator(MaxNLocator('auto', steps=[1]))
    axs[-1].tick_params(which='both')
    axs[-1].set_xlim(None, 0)

    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right', title='Date_file')

    fig.show()


    fig2, ax2 = plt.subplots(tight_layout=True)
    ax2.set_xlabel('File date')
    ax2.set_ylabel('Mean DOO delay')
    ax2.plot(fd_select, doo_delays, 'o-')
    tools.set_xaxis_dateformat(ax2)
    fig2.show()


plt.close('all')

if 0:
    import casus_analysis as ca
    # Warning: slow! Needs to process 30+ GB of data.
    ca.create_merged_summary_csv()
if 0:
    df = load_summary_df()
#%%
for i_dow in range(0, -5, -1):
    plot_casus_summary(df, (-18, 0), (-56+i_dow, None, 7))


plot_casus_summary(df, (-10, 0), (-14, None))
plot_casus_summary(df, (-10, 0), (-28, -14))


