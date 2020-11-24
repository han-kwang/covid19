"""Script for casus analysis (using functions from casus_analysis.py)."""

from copy import deepcopy
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import numpy as np
import casus_analysis as ca
import urllib.request


def get_summary_df(maxageh=1, force_today=False):
    """Get summary dataframe with cache handling and csv update."""

    if ca.download_rivm_casus_files(force_today=force_today):
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


def demo_dow_prediction(df, doo_corr, f_dow_corr, date_range, dows):
    """Demonstrate DoW effect in plots.

    Parameters:

    - df: summary DataFrame with multi-index and eDOO column.
    - doo_corr: DOOCorrection (doocorr.G_dow_corr will be ignored.)
    - f_dow_corr: file day-of-week correction, (7, m) array.
    - date_range: (start_date, end_date) tuple (refers to Date_statistics).
    - dows: list of DoWs (0=Monday).

    This will generate two plots, with and without DoW correction enabled.
    one without. Report dates on selected DoWs will be shown.
    """

    # Select file dates to plot (based on dows parameter).
    # The one with the final date is for the best case-number values.
    date_range = [pd.Timestamp(d) for d in date_range]
    print('demo_dow_prediction...')
    fdates = df.index.get_level_values('Date_file').unique().sort_values()
    fdates = fdates[(fdates >= date_range[0]) & (fdates <= date_range[1])]
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
        d.loc[(d.index >= date_range[0]) & (d.index <= date_range[1])]
        for d in df_list
    ]

    doo_corr = deepcopy(doo_corr)
    dow_names = 'ma,di,wo,do,vr,za,zo'.split(',')
    dows_str = ', '.join([dow_names[i] for i in sorted(dows)]) # e.g. 'ma, wo'
    dows_str = f'rapportagedag {dows_str}' if len(dows)==1 else f'rapportagedagen {dows_str}'

    doo_corr.f_dow_corr[:, :] = 1.0
    doo_corr.plot_nDOO(df_list, title=f'Zonder weekdagcorrectie; {dows_str}')

    doo_corr.f_dow_corr[:, :] = f_dow_corr
    doo_corr.plot_nDOO(df_list, title=f'Met weekdagcorrectie; {dows_str}')


def demo_dow_prediction_202010(df):
    """Demo DoW correction for OCt 2020, from full summary DataFrame."""

    # Get DoW statistics (dow_corr)
    date_range_dow = ('2020-10-01', '2020-11-15')
    f_dow_corr = ca.DOOCorrection.calc_fdow_correction(
        df, m=18, date_range=date_range_dow)

    # Get and plot DOO correction
    doo_corr = ca.DOOCorrection.from_doo_df(
        df, m=18,  f_dow_corr=f_dow_corr)
    doo_corr.plot_fdow_corr()

    plt.pause(0.5)
    # Show effect of DoW correction for Monday and Sunday
    for dow in [0, 6]:
        demo_dow_prediction(df, doo_corr, f_dow_corr=f_dow_corr,
                            date_range=('2020-07-01', '2020-11-01'),
                            dows=[dow])
        plt.pause(0.5)

def demo_fdow_corr(df, date_range, show=True, fname=None):

    f_dow_corr = ca.DOOCorrection.calc_fdow_correction(
        df, m=18, date_range=date_range)

    doo_corr = ca.DOOCorrection.from_doo_df(
        df, m=18,  f_dow_corr=f_dow_corr)

    doo_corr.plot_fdow_corr(show=show, fname=fname)


def demo_prediction(df, m=12, fdate_range=('2020-10-22', '2020-11-21'),
                    dow_extra=45, with_dc=True, show=True, fbname=None):
    """Demo prediction for Nov 2020, from full summary DataFrame.

    Parameters:

    - m: number of days to wait until DOO reports are "final".
    - fdate_range: date range for calibrating DOOCorrection.
      (last date minus m days is the actual coverage).
    - dow_extra: number of extra days for calculating DoW correction,
      which needs a longer time interval.
    - with_dc: whether to enable DOO correction. Set to False
      to get uncorrected data.
    - show: whether to plot on-screen.
    - fbname: optional file basename (no suffix) for plot output.

    Show how well predicitions were in November.
    Correction factors based on data available in Sept/October.

    Parameters:
    """

    fdra_lo = pd.to_datetime(fdate_range[0])
    fdra_hi = pd.to_datetime(fdate_range[1])
    fdra_lodow = fdra_lo - pd.Timedelta(dow_extra, 'd')

    # include data from mid-Aug for DoW effocts
    f_dow_corr = ca.DOOCorrection.calc_fdow_correction(
           df, date_range=(fdra_lodow, fdra_hi), m=m)
    s_dow_ce = ca.DOOCorrection.calc_sdow_correction(
        df, date_range=(fdra_lodow, fdra_hi), show=False)

    # Use data from mid-Sept for generic correction
    if with_dc:
        doo_corr = ca.DOOCorrection.from_doo_df(
            df, date_range=(fdra_lo, fdra_hi), m=m,
            f_dow_corr=f_dow_corr, s_dow_ce=s_dow_ce
            )
    else:
        doo_corr = ca.DOOCorrection(
            G=np.array([0]*3 + [1.0]*(m-3)),
            eG=np.zeros(m),
            date_range=(fdra_lo, fdra_lo),
            f_dow_corr=np.ones((7, m)),
            s_dow_ce=(np.ones(7), 0.0)
            )

    # Go back from today in steps of 4 days.
    fdates = df.index.get_level_values(0).unique().sort_values()
    fdates = fdates[fdates >= fdra_lodow]
    fdates = fdates[-1::-4]

    dfs_edoo = [
        df.loc[fdate].loc[fdra_lodow-pd.Timedelta(30, 'd'):]
        for fdate in fdates
    ]

    fdra_hi_m = fdra_hi - pd.Timedelta(m, 'd')
    if with_dc:
        title = ('Correctiefactor recente cijfers o.b.v.'
                 f' {ca._ymd(fdra_lo)} .. {ca._ymd(fdra_hi_m)}.')
    else:
        title = 'Ruwe data eerste ziektedag.'
    doo_corr.plot_nDOO(dfs_edoo, title=title, show=show,
                       fname=(fbname and f'{fbname}_ndoo.pdf'))

    if with_dc:
        doo_corr.plot(show=show, fname=f'{fbname}-doo_corr.pdf')


def demo_doo_correction_by_month(df, show=True, fname=None):

    dranges = [
        ('2020-07-01', '2020-08-18'),
        ('2020-08-01', '2020-09-18'),
        ('2020-09-01', '2020-10-18'),
        ('2020-10-01', '2020-11-18')
        ]

    dcs = [
           ca.DOOCorrection.from_doo_df(df, m=18, date_range=dr)
           for dr in dranges
        ]
    dcs[0].plot(dcs[1:], show=show, fname=fname)

def demo_sdow(df, fdate_range, m=14, show=True, fbname=None):
    """Show statistics DoW effect."""

    fd0, fd1 = [pd.to_datetime(d) for d in fdate_range]
    day = pd.Timedelta('1 d')
    f_dow_corr = ca.DOOCorrection.calc_fdow_correction(
           df, date_range=fdate_range, m=m)
    s_dow_ce = ca.DOOCorrection.calc_sdow_correction(
        df, date_range=fdate_range, show=show,
        fname=(fbname and f'{fbname}-sdow.png'))

    doo_corr = ca.DOOCorrection.from_doo_df(
        df, date_range=fdate_range, m=m,
        f_dow_corr=f_dow_corr, s_dow_ce=s_dow_ce
        )

    df_slice = df.loc[(fd1, slice(fd0-m*day, fd1)), :].reset_index(0)

    doo_corr.plot_nDOO(df_slice, kind='nDOO', show=show,
                       fname=(fbname and f'{fbname}_nDOO.pdf'))
    doo_corr.plot_nDOO(df_slice, kind='ncDOO', show=show,
                       fname=(fbname and f'{fbname}_ncDOO.pdf'))

def demo_corr_recom(df, n=120, m=18, show=True, fbname=None):
    """Show recommended correction values based on latest data.

    Parameters:

    - df: full (multi-index) dataset with eDOO data.
    - n: number of days to go back (file dates)
    - m: number of days for DOO correction to stabilize.
    - fbname: file basename (no suffix) for output.
    """

    fdates = df.index.get_level_values('Date_file').unique().sort_values()
    fdate_range = (fdates[-n], fdates[-1])
    f_dow_corr = ca.DOOCorrection.calc_fdow_correction(
           df, date_range=fdate_range, m=m)
    s_dow_ce = ca.DOOCorrection.calc_sdow_correction(
        df, date_range=fdate_range, show=False)

    doo_corr_bare = ca.DOOCorrection.from_doo_df(
        df, date_range=fdate_range, m=m,
        )

    fig, ax  = plt.subplots(1, 1, tight_layout=True, figsize=(6, 4))
    ax2 = ax.twinx()

    ax.errorbar(np.arange(m), doo_corr_bare.iG, doo_corr_bare.eiG,
                fmt='bo-', capsize=10, label='Correctiefactor 1/G')
    ax2.plot(np.arange(m), 100*doo_corr_bare.eiG/doo_corr_bare.iG,
             'r^-',  label='Relatieve fout (%)')
    # ax.legend()
    ax.set_xlabel('Dagen geleden')
    ax.set_ylabel('Correctiefactor 1/G', color='blue')
    ax2.set_ylabel('Relatieve fout (%)', color='red')
    ax.legend(loc='upper right')
    ax2.legend(loc='center right')

    ax.set_ylim(1, 7)
    ax.set_xlim(0, m-0.9)
    ax2.set_ylim(0, 30)

    locator = matplotlib.ticker.MaxNLocator(steps=[1, 2, 5])
    ax.xaxis.set_major_locator(locator)

    ax.grid()
    ax.set_title(f'O.b.v. DOO {ca._ymd(fdates[-n])} .. {ca._ymd(fdates[-m])}')

    ca._show_save_fig(fig, show, (fbname and f'{fbname}_creco.pdf'))

def demo_fixed_j(df, fdate_range, ages=(5, 9, 14), stats_age=30, m=18, sdow=True,
                 show=True, fname=None):
    """Demonstrate case trends using fixed DOO ages.

    Parameters:

    - df: Dataframe with eDOO column and multi-index.
    - fdate_range: (fdate_lo, fdate_hi) tuple.
    - ages: sequence of case ages to consider.
    - stats_age: number of recent fdates not to consider in getting
      correction factors.
    - m: the 'm' parameter as usual.
    - sdow: whether to apply s_dow_corr correction.
      (counterproductive for small age values)
    - show: whether to show plot on-screen.
    - fname: optional filename for writing plot image.
    """

    # Slice df dataframe to requested
    fdates = df.index.get_level_values('Date_file').unique().sort_values()
    fdates = fdates[(fdates >= fdate_range[0]) & (fdates <= fdate_range[1])]
    fdate_range = (fdates[0], fdates[-1])
    df = df.loc[fdates]

    # get statistics
    doo_corr = ca.DOOCorrection.from_doo_df(
        df, m=m,
        date_range=(fdate_range[0], fdates[-stats_age]),
        f_dow_corr='auto', s_dow_ce='auto')

    # list of nDOO series, one for each age
    ndoo_by_age = []
    for age in ages:
        sdates = fdates - pd.Timedelta(age, 'd')
        idx = list(zip(fdates, sdates))
        eDOOs = df.loc[idx].reset_index('Date_file')['eDOO'] # Series

        # Correct for G and DoW effects
        nDOOs = eDOOs * doo_corr.iG[age]
        if sdow:
            nDOOs *= doo_corr.s_dow_corr[eDOOs.index.dayofweek]
        nDOOs /= doo_corr.f_dow_corr[fdates.dayofweek, age]

        ndoo_by_age.append(nDOOs)

    # Plot 'm all
    fig, ax = plt.subplots(1, 1, figsize=(9, 5), tight_layout=True)
    for age, ndoo in zip(ages, ndoo_by_age):
        ax.semilogy(ndoo, label=f'j={age}')

    ti_sdate = ', sdate' if sdow else ''
    ax.set_title(f'Methode "vaste j"; Weekdagcorrectie (fdate{ti_sdate}) o.b.v. DOO '
                 f'{ca._ymd(doo_corr.date_range[0])} .. {ca._ymd(doo_corr.date_range[1])}')
    ax.set_ylabel('Aantal per dag')
    ax.set_xlabel('Datum eerste ziektedag')
    ax.tick_params(axis='x', labelrotation=-20)
    for tl in ax.get_xticklabels():
        tl.set_ha('left')

    ax.legend()
    ax.grid()
    ax.grid(which='minor', axis='y')

    ca._show_save_fig(fig, show, fname)


def plots_for_report(plots='dcm,fdc,nnc,nc10,sdow,crecom,fixj1,fixj2',
                     show=False):
    """Generate plots to include in report.

    - show: whether to show plot on-screen. (always save)
    plots: comma-separated string, which things to plot:

        - dcm: correction factor for different months.
        - fdc: correction factor for file DoW.
        - nnc: new cases, no correction (with multiple file dates).
        - nc10: new cases, correction based on October
        - nc10b: new cases, correction based from late October
        - sdow: statistics DoW effect.
        - crecom: correction data recommendations.
        - fixj1, fixj2: fixed-j strategy (with/without sdow correction).
    """
    df = get_summary_df()
    now = df.iloc[-1].name[0] # DateTime of most recent entry
    day = pd.Timedelta(1, 'd')

    plot_cases = dict(
        # Tuples: function name, args, kwargs
        dcm=(
            demo_doo_correction_by_month,
            (df,),
            dict(show=show, fname='output/casus_dcmonth.pdf')),
        fdc=(
            demo_fdow_corr,
            (df,),
            dict(date_range=(now-90*day, now),
                 show=show, fname='output/casus_fdow_corr.png')),
        nnc=(
            demo_prediction,
            (df,),
            dict(with_dc=False, fdate_range=(now-60*day, now),
                 show=show, fbname='output/casus_nnc')),
        nc10=(
            demo_prediction,
            (df,),
            dict(fdate_range=('2020-10-01', '2020-11-12'),
                 m=12, show=show, fbname='output/casus_nc10')),
        nc10b=(
            demo_prediction,
            (df,),
            dict(fdate_range=('2020-10-22', '2020-11-21'),
                 m=12, show=show, fbname='output/casus_nc10b')),
        sdow=(
            demo_sdow,
            (df,),
            dict(fdate_range=(now-90*day, now), m=18, show=show,
                 fbname='output/casus_sdow')),
        crecom=(
            demo_corr_recom,
            (df,),
            dict(n=120, m=18, show=show, fbname='output/casus_crecom')),
        fixj1=(
            demo_fixed_j,
            (df,),
            dict(fdate_range=(now-90*day, now),
                 ages=(5, 7, 9, 11, 17), stats_age=30, m=18, show=show,
                 fname='output/casus_fixj1.pdf', sdow=False)),
        fixj2=(
            demo_fixed_j,
            (df,),
            dict(fdate_range=(now-90*day, now),
                 ages=(5, 7, 9, 11, 17), stats_age=30, m=18, show=True,
                 fname='output/casus_fixj2.pdf', sdow=False)),
        )


    if not plots:
        print('Nothing to plot.')
        return

    for plot in plots.split(','):
        func, args, kwargs = plot_cases[plot]
        print(f'Doing {plot} ...')
        func(*args, **kwargs)




if __name__ == '__main__':
    pass

    # # Note, because of multiprocessing for data loading,
    # # avoid adding interactive/slow code outside the main block.

    plt.close('all')

    # Default is a larger font for the title, overrule this.
    plt.rcParams['axes.titlesize'] = plt.rcParams['xtick.labelsize']


    ## This will create all plots (both on-screen and stored as files).
    ## Warning: major screen clutter.
    # plots_for_report(show=True)

    ## This will create all plots as files.
    # plots_for_report(show=False)

    ## This will create and show one plot. See doc of plots_for_report.
    plots_for_report('fixj2', show=True)
