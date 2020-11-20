"""Script for casus analysis (using functions from casus_analysis.py)."""

from copy import deepcopy
import matplotlib.pyplot as plt
import pandas as pd
import scipy.signal
import numpy as np
import casus_analysis as ca


def get_summary_df(maxageh=1):
    """Get summary dataframe with cache handling and csv update."""

    if ca.download_rivm_casus_files():
        # If there is new data, process it.
        # This is a bit slow. Someday, be smarter in merging.
        ca.create_merged_summary_csv()

    cache_fname = 'casus_summary_cache.pkl'
    df = ca.load_data_cache(cache_fname, maxageh=maxageh)
    if df is None:
        df = ca.load_merged_summary_csv() # (2020-08-01', '2020-10-01')
        df = ca.add_eDOO_to_summary(df)
        ca.save_data_cache(df, cache_fname)
    print('---snippet from summary dataframe df---')
    print(df)

    return df


def demo_dow_prediction(df, doo_corr, dow_corr, drange, dows):
    """Demonstrate DoW effect in plots.

    Parameters:

    - df: summary DataFrame with multi-index and eDOO column.
    - doo_corr: DOOCorrection (doocorr.G_dow_corr will be ignored.)
    - dow_corr: day-of-week correction, (7, m) array.
    - drange: (start_date, end_date) tuple (refers to Date_statistics).
    - dows: list of DoWs (0=Monday).

    This will generate two plots, with and without DoW correction enabled.
    one without. Report dates on selected DoWs will be shown.
    """

    # Select file dates to plot (based on dows parameter).
    # The one with the final date is for the best case-number values.
    drange = [pd.Timestamp(d) for d in drange]
    print('demo_dow_prediction...')
    fdates = df.index.get_level_values('Date_file').unique().sort_values()
    fdates = fdates[(fdates >= drange[0]) & (fdates <= drange[1])]
    dows = set(dows)
    fdates_dow = [d for d in fdates if d.dayofweek in dows]
    if fdates_dow[-1] != fdates[-1]:
        fdates_dow.append(fdates[-1])

    # build list of dataframes for different file dates.
    df_list = [
        df.loc[fd, :] # result has single index: Date_statistics
        for fd in fdates_dow
        ]
    # trim old entries
    df_list = [
        d.loc[(d.index >= drange[0]) & (d.index <= drange[1])]
        for d in df_list
    ]

    doo_corr = deepcopy(doo_corr)
    dow_names = 'ma,di,wo,do,vr,za,zo'.split(',')
    dows_str = ', '.join([dow_names[i] for i in sorted(dows)]) # e.g. 'ma, wo'
    dows_str = f'rapportagedag {dows_str}' if len(dows)==1 else f'rapportagedagen {dows_str}'

    doo_corr.G_dow_corr[:, :] = 1.0
    doo_corr.plot_nDOO(df_list, title=f'Zonder weekdagcorrectie; {dows_str}')

    doo_corr.G_dow_corr[:, :] = dow_corr
    doo_corr.plot_nDOO(df_list, title=f'Met weekdagcorrectie; {dows_str}')

def demo_dow_prediction_202010(df):
    """Demo DoW correction for OCt 2020, from full summary DataFrame."""

    # Get DoW statistics (dow_corr)
    date_range_dow = ('2020-10-01', '2020-11-15')
    G_dow_corr = ca.DOOCorrection.calc_dow_correction(
        df, m=18, date_range=date_range_dow)

    # Get and plot DOO correction
    doo_corr = ca.DOOCorrection.from_doo_df(
        df, m=18,  G_dow_corr=G_dow_corr)
    doo_corr.plot_dow_corr()

    plt.pause(0.5)
    # Show effect of DoW correction for Monday and Sunday
    for dow in [0, 6]:
        demo_dow_prediction(df, doo_corr, G_dow_corr, ('2020-07-01', '2020-11-01'),
                            dows=[dow])
        plt.pause(0.5)


def demo_doo_correction_by_month(df):

    dranges = [
        ('2020-07-01', '2020-08-18'),
        ('2020-08-01', '2020-09-18'),
        ('2020-09-01', '2020-10-18'),
        ('2020-10-01', '2020-11-18')
        ]

    gds = [
           ca.DOOCorrection.from_doo_df(df, m=18, date_range=dr)
           for dr in dranges
        ]
    print('done.')
    gds[0].plot(gds[1:])


def analyze_case_wk_effects(df, date_range=('2020-09-01', '2099-01-01'), skip=14):
    """Figure out per-week distribution of DOO.

    Parameters:

    - df: full summary dataframe. (will use the most recent file date.)
    - date_range: range of DOO dates to consider.
    - skip: skip this many of the most recent DOO days (because DOO statistics
      are not stable yet).
    """

    # Get data for one file date.
    fdates = df.index.get_level_values('Date_file').unique().sort_values()
    df1 = df.loc[fdates[-1]] # most recent file; new dataframe has index Date_statistics

    # Get data for range of DOO dates
    sdates = df1.index # Date_statistics
    sdmask_1 = (sdates >= date_range[0])
    sdmask_2 = (sdates <= pd.to_datetime(date_range[1]) - pd.Timedelta(skip, 'd'))
    sdmask_3 = (sdates <= fdates[-1] - pd.Timedelta(skip, 'd'))
    sdates = sdates[sdmask_1 & sdmask_2 & sdmask_3]
    ndoos = df1.loc[sdates, 'eDOO']

    # convert to log, get smooth component and deviation.
    # Savitzky-Golay filter width 15 (2 weeks) will not follow weekly oscillation.
    log_ys = np.log(ndoos)
    log_ys_sm = scipy.signal.savgol_filter(log_ys, 15, 2)
    log_ys_sm = pd.Series(log_ys_sm, index=sdates)
    log_ys_dev = log_ys - log_ys_sm

    # Statistics on per-weekday basis.
    dows = sdates.dayofweek
    dowdev = np.array([
        np.mean(log_ys_dev[dows==dow])
        for dow in range(7)
        ])
    log_ys_dev_resid = log_ys_dev - dowdev[dows]
        
    fig, axs = plt.subplots(1, 3, tight_layout=True, figsize=(10, 3))
    ax = axs[0]
    ax.plot(log_ys, label='Ruwe data')
    ax.plot(ndoos.index, log_ys_sm, label='Trend', zorder=-1)
    # ax.plot(ndoos.index, log_ys_sm + dowdev[dows], label='Gecorrigeerde trend', zorder=-1)
    ax.set_ylabel('ln(nDOO)')
    ax.grid()
    ax.legend()
    ax.tick_params(axis='x', labelrotation=-20)
    ax.legend()
    # plt.xticks(rotation=-20)
    for tl in ax.get_xticklabels():
        tl.set_ha('left')

    ax = axs[1]
    ax.plot(log_ys_dev, label='t.o.v. trend')
    ax.plot(log_ys_dev_resid, label='t.o.v. gecorrigeerde trend')
    ax.set_ylabel('Verschil')
    ax.grid()
    ax.tick_params(axis='x', labelrotation=-20)
    ax.legend()
    # plt.xticks(rotation=-20)
    for tl in ax.get_xticklabels():
        tl.set_ha('left')

    # 3rd panel: per-weekday mean deviation
    ax = axs[2]
    ax.bar(np.arange(7), (np.exp(dowdev)-1)*100)
    ax.set_ylabel('Afwijking (%)')
    ax.set_xticks(np.arange(7))
    ax.set_xticklabels(['ma', 'di', 'wo', 'do', 'vr', 'za', 'zo'])

    fig.show()

if __name__ == '__main__':

    # Note, because of multiprocessing for data loading,
    # avoid adding interactive/slow code outside the main block.

    plt.close('all')
    plt.rcParams["date.autoformatter.day"] = "%d %b"
    df = get_summary_df()

    # demo_dow_prediction_202010(df)
    # demo_doo_correction_by_month(df)
    analyze_case_wk_effects(df)


#    ca.analyze_dow_effect(df)

#    print('gdata...')
#
#    dfns = [
#        df.loc[fdate].reset_index(0)
#        for fdate in ['2020-11-16', '2020-11-01', '2020-11-08', '2020-10-15',
#                      '2020-10-01']
#        ]
#    gds[3].plot_nDOO(dfns)


