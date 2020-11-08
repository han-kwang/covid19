#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Holiday zones in the Netherlands; module for importing.

Function:

    - add_holiday_regions()

Created on Sat Nov  7 16:08:51 2020  @hk_nien
"""
from pathlib import Path
import pandas as pd

def add_holiday_regions(df_cov):
    """Add a column 'HolRegion' with holiday region names (Noord, Midden, Zuid).

    Parameter:

    - df_cov: Dataframe with columns 'Municipality_name' and 'Province'.

    Return new dataframe including 'HolRegion' column.
    """

    # Build dataframe with columns 'Municipality_name', 'Province', 'HolRegion'.
    dfh = df_cov[['Municipality_name', 'Province']].drop_duplicates()
    dfh = dfh.loc[~dfh['Municipality_name'].isna()]
    dfh['HolRegion'] = None

    # Definitions holiday regions
    # https://www.rijksoverheid.nl/onderwerpen/schoolvakanties/regios-schoolvakantie

    rules = [
        # Region name, (P:|M:) province/municipality name
        ['Noord',
             'P:Drenthe', 'P:Flevoland', 'P:Friesland', 'P:Groningen',
             'P:Overijssel', 'P:Noord-Holland'],
        ['Midden', 'M:Zeewolde', 'P:Utrecht', 'P:Zuid-Holland'],
        ['Zuid', 'P:Limburg', 'P:Noord-Brabant', 'P:Zeeland'],
        ['Noord', 'M:Hattem', 'M:Eemnes'],
        ['Zuid', 'P:Gelderland'],
        ['Midden', 'M:Aalten', 'M:Apeldoorn', 'M:Barneveld', 'M:Berkelland',
        'M:Bronckhorst', 'M:Brummen', 'M:Buren', 'M:Culemborg', 'M:Doetinchem',
        'M:Ede', 'M:Elburg', 'M:Epe', 'M:Ermelo', 'M:Harderwijk', 'M:Heerde',
        'M:Lochem', 'M: Montferland', 'M:Neder-Betuwe', 'M:Nijkerk', 'M:Nunspeet',
        'M:Oldebroek', 'M:Oost-Gelre', 'M:Oude IJsselstreek', 'M:Putten',
        'M:Scherpenzeel', 'M:Tiel', 'M:Voorst', 'M:Wageningen', 'M:West Betuwe',
        'M:Winterswijk en Zutphen', 'M:Werkendam', 'M:Woudrichem'],
        ]

    for rule in rules:
        hrname = rule[0]
        for pmname in rule[1:]:
            # print(f'Applying {pmname} to {hrname}.') # DEBUG
            if pmname.startswith('P:'):
                dfh.loc[dfh['Province'] == pmname[2:], 'HolRegion'] = hrname
            elif pmname.startswith('M:'):
                dfh.loc[dfh['Municipality_name'] == pmname[2:] , 'HolRegion'] = hrname
            else:
                raise ValueError(f'pmname {pmname!r}: bad pattern.')

    dfh.set_index('Municipality_name', inplace=True)
    dfh.drop(columns='Province', inplace=True)

    new_df = df_cov.join(dfh, on='Municipality_name')
    return new_df


if __name__ == '__main__':
    # test
    df = pd.read_csv('data/COVID-19_aantallen_gemeente_cumulatief.csv', sep=';')
    add_holiday_regions(df)
