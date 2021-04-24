#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  7 20:43:14 2021
"""



import pandas as pd
import tools
import nlcovidstats as nlcs

if __name__ == '__main__':
    nlcs.reset_plots()
    nlcs.init_data(autoupdate=True)

    df = nlcs.load_cumulative_cases(autoupdate=True)
    df = df.drop(columns=['Municipality_code', 'Province'])

    jan6 = pd.to_datetime('2021-01-06T10')
    jan7 = pd.to_datetime('2021-01-07T10')

    mask6 = df['Date_of_report'] == jan6
    mask7 =  df['Date_of_report'] == jan7

    muns_2020 = set(df.loc[mask6, 'Municipality_name'].unique())
    muns_2021 = set(df.loc[mask7, 'Municipality_name'].unique())

    print(f'Disappeared: {muns_2020.difference(muns_2021)}')
    print(f'Appeared: {muns_2021.difference(muns_2020)}')

    regions = [
        'Haaren', # verdwijnt
        'Boxtel', 'Vught', 'Tilburg',
        'Eemsdelta', # nieuw
        'Appingedam', 'Delfzijl', 'Loppersum',
        'Hengelo', 'Hengelo (O.)',
        ]
    for r in regions:
        mask = (df['Municipality_name'] == r) & (df['Date_of_report'] > '2021-01-06')
        df1 = df.loc[mask]
        print(df1.iloc[-2:])
