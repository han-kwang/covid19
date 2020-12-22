#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Holiday zones, municipality classes in the Netherlands; module for importing.

Function:

- build_municipality_csv()
- get_municipality_data()
- select_cases_region()

Created on Sat Nov  7 16:08:51 2020  @hk_nien
"""
from pathlib import Path
import re
import pandas as pd

DATA_PATH = Path(__file__).parent / 'data'
DF_MUN = None

def build_municipality_csv(df_cases):
    """Write data/municipalities.csv.

    The csv will have columns:

    - Municipality_name
    - Population
    - Province
    - HolRegion

    This function only needs to be called only rarely (output will be committed).

    Parameters:

    - df_cases: dataframe with columns 'Municipality_name' and 'Province'.
    """

    df_mun =  _load_municipality_data_cbs(df_cases)
    df_mun.rename(columns={'Inwoners': 'Population'}, inplace=True)

    ### Get provinces from cases dataframe.
    # dataframe: index=Municipality_name, column 'Province'
    mun_provs = df_cases.groupby('Municipality_name').first()[['Province']]
    df_mun['Province'] = mun_provs['Province']

    ## Holiday regions
    _add_holiday_regions(df_mun)

    fpath = DATA_PATH / 'municipalities.csv'
    df_mun.to_csv(fpath, float_format='%.7g', header=True)
    print(f'Wrote {fpath}')


def get_municipality_data():
    """Return dataframe with municipality data:

    Index: Municipality (name)
    Columns: Population, Province, HolRegion,

    This just loads the csv file created by build_municipality_csv(),
    or use a previously cached version.
    """

    global DF_MUN
    if DF_MUN is None:
        df = pd.read_csv(DATA_PATH / 'municipalities.csv')
        df.set_index('Municipality', inplace=True)
        DF_MUN = df

    return DF_MUN.copy()

def _load_municipality_data_cbs(df_cases):
    """Return municipality dataframe from cases dataframe.

    Cases dataframe must have 'Municipality_name' column.
    This takes data from the CBS table 'Regionale_kerncijfers*.csv'.

    Return dataframe with:

    - index: municipality
    - 'Inwoners' column
    - 'Province' column
    """

    ## Load municipality populations
    path = DATA_PATH / 'Regionale_kerncijfers_Nederland_15082020_130832.csv'

    df_mun = pd.read_csv(path, sep=';')
    df_mun.rename(columns={
     #'Perioden',
     #"Regio's",
     'Bevolking/Bevolkingssamenstelling op 1 januari/Totale bevolking (aantal)': 'total',
     'Bevolking/Bevolkingssamenstelling op 1 januari/Burgerlijke staat/Bevolking 15 jaar of ouder/Inwoners 15 jaar of ouder (aantal)': 'n15plus',
     'Bevolking/Bevolkingssamenstelling op 1 januari/Burgerlijke staat/Bevolking 15 jaar of ouder/Gehuwd (in % van  inwoners 15 jaar of ouder)': 'n15gehuwd',
     'Bevolking/Bevolkingssamenstelling op 1 januari/Bevolkingsdichtheid (aantal inwoners per km²)': 'dichtheid',
     'Bouwen en wonen/Woningvoorraad/Voorraad op 1 januari (aantal)': 'woningen',
     'Milieu en bodemgebruik/Bodemgebruik/Oppervlakte/Totale oppervlakte (km²)': 'opp'
     }, inplace=True)

    df_mun = pd.DataFrame({'Municipality': df_mun['Regio\'s'], 'Inwoners': df_mun['total']})
    df_mun.set_index('Municipality', inplace=True)
    df_mun = df_mun.loc[~df_mun.Inwoners.isna()]
    import re
    df_mun.rename(index=lambda x: re.sub(r' \(gemeente\)$', '', x), inplace=True)

    rename_muns = {
        'Beek (L.)': 'Beek',
        'Hengelo (O.)': 'Hengelo',
        'Laren (NH.)': 'Laren',
        'Middelburg (Z.)': 'Middelburg',
        'Rijswijk (ZH.)': 'Rijswijk',
        'Stein (L.)': 'Stein',
        }

    df_mun.rename(index=rename_muns, inplace=True)
    # df_mun.drop(index=['Valkenburg (ZH.)'], inplace=True)

    return df_mun

def _add_holiday_regions(df_mun):
    """Add a column 'HolRegion' with holiday region names (Noord, Midden, Zuid).

    Parameter:

    - df_mun: Dataframe with index 'Municipality_name' and at least column 'Province'.

    Update df_mun in-place with new 'HolRegion' column.
    """

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

    df_mun['HolRegion'] = None

    for rule in rules:
        hrname = rule[0]
        for pmname in rule[1:]:
            if pmname.startswith('P:'):
                df_mun.loc[df_mun['Province'] == pmname[2:], 'HolRegion'] = hrname
            elif pmname.startswith('M:'):
                df_mun.loc[df_mun.index == pmname[2:] , 'HolRegion'] = hrname
            else:
                raise ValueError(f'pmname {pmname!r}: bad pattern.')



def select_cases_region(dfc, region):
    """Select daily cases by region.

    Parameters:

    - dfc: cases dataframe, with columns
      'Date_of_report', 'Municipality', and various numerical columns.
    - region: one of:
      - the name of a municipality
      - 'Nederland': all
      - 'HR:Zuid', 'HR:Noord', 'HR:Midden', 'HR:Midden+Zuid': holiday regions.
      - 'POP:xx-yy': municipalities with population xx <= pop/1000 < yy'
      - 'P:xx': province

    Return:

    - Dataframe with Date_of_report as index and
      numerical columns summed as appropriate.
    - npop: population.
    """

    df_mun = get_municipality_data()

    # First, mselect is Dataframe of selected municipalities.
    if region == 'Nederland':
        mselect = df_mun
    elif region == 'HR:Midden+Zuid':
        mselect = df_mun.loc[df_mun['HolRegion'].str.match('Midden|Zuid')]
    elif region.startswith('HR:'):
        mselect = df_mun.loc[df_mun['HolRegion'] == region[3:]]
    elif region.startswith('P:'):
        mselect = df_mun.loc[df_mun['Province'] == region[2:]]
    elif region.startswith('POP:'):
        ma = re.match(r'POP:(\d+)-(\d+)$', region)
        if not ma:
            raise ValueError(f'region={region!r} does not match \'MS:NUM-NUM\'.')
        pop_lo, pop_hi = float(ma.group(1)), float(ma.group(2))
        mask = (df_mun['Population'] >= pop_lo*1e3) & (df_mun['Population'] < pop_hi*1e3)
        mselect = df_mun.loc[mask]
    else:
        mselect = df_mun.loc[[region]]

    npop = mselect['Population'].sum()

    # Select the corresponding rows in dfc.
    dfc_sel = dfc.join(mselect[[]], on='Municipality_name', how='inner')

    if len(dfc_sel) == 0:
        raise ValueError(f'No data for region={region!r}.')

    # combine
    dfc_sel = dfc_sel.groupby('Date_of_report').sum()

    return dfc_sel, npop



if __name__ == '__main__':
    # recreate municipalities.csv
    df = pd.read_csv('data/COVID-19_aantallen_gemeente_cumulatief.csv', sep=';')
    build_municipality_csv(df)
