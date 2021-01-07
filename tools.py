#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Small, generic helper functions.

- pause_commandline(): prompt user to continue (when outside Spyder).
- set_xaxis_dateformat(): set layout parameters for date x-axis.
"""
import os
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd


def pause_commandline(msg='Press Enter to continue.'):
    """Prompt user to continue (when outside Spyder).

    Only do this script appears to be run from command line.
    This is to prevent plot windows from closing at program termination.
    """

    if 'SPYDER_ARGS' not in os.environ:
        input(msg)




def set_xaxis_dateformat(ax, xlabel=None, maxticks=10, yminor=False):
    """Set x axis formatting for dates; call after adjusting ranges etc."""

    plt.xticks(rotation=-20)

    md = matplotlib.dates
    date_lo, date_hi = pd.to_datetime(md.num2date(ax.get_xlim()))
    day_span = (date_hi - date_lo).days

    monday_locator = md.WeekdayLocator(0)

    minor_grid = False
    if day_span <= maxticks:
        ax.xaxis.set_major_locator(md.DayLocator())
        ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%Y-%m-%d'))
    elif day_span <= maxticks*7:
        ax.xaxis.set_major_locator(monday_locator)
        ax.xaxis.set_minor_locator(md.DayLocator())
        minor_grid=True
        ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%Y-%m-%d'))
    elif day_span <= maxticks*30.5:
        ax.xaxis.set_major_locator(md.MonthLocator())
        ax.xaxis.set_minor_locator(monday_locator)
        ax.tick_params('x', which='minor', direction='in')
        ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%Y-%m-%d'))
        if day_span <= maxticks*15:
            minor_grid = True
    else:
        ax.xaxis.set_major_locator(md.MonthLocator([1, 4, 7, 10]))
        ax.xaxis.set_minor_locator(md.MonthLocator())
        ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%Y-%m'))
        minor_grid = (day_span < maxticks*90)

    ax.grid(which='major', axis='both', color='#aaaaaa')
    if minor_grid:
        ax.grid(which='minor', axis='x', color='#dddddd')

    if yminor:
        ax.grid(which='minor', axis='y', color='#dddddd')

    # ax.grid(which=('both' if minor_grid else 'major'))
    for tl in ax.get_xticklabels():
        tl.set_ha('left')

    if xlabel:
        ax.set_xlabel(xlabel)

plt.close('all')


def _test_set_xaxis_dateformat():
    date1 = pd.to_datetime('2010-05-03')
    date2s = ['2010-05-20', '2010-07-01', '2010-11-01', '2012-01-01']
    fig, axs = plt.subplots(len(date2s), 1, figsize=(10, 8))

    for date2, ax in zip(date2s, axs):
        date2 = pd.to_datetime(date2)
        tseries = pd.Series([0, 0], index=[date1, date2])
        ax.plot(tseries)
        set_xaxis_dateformat(ax, maxticks=5)
        fig.show()


if __name__ == '__main__':

    _test_set_xaxis_dateformat()
