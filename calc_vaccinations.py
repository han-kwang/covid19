#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  3 18:03:40 2021

@author: @hk_nien
"""


import numpy as np
import scipy.interpolate
import pandas as pd
import matplotlib.pyplot as plt
import tools

DELAY_2ND_VACC = 35 # 5 weeks waiting (in days)

# Source: https://coronadashboard.rijksoverheid.nl/landelijk/vaccinaties
# date, cumulative vaccinations
vac_recs = [
    ('2021-01-03', 0),
    ('2021-01-17', 78e3),
    ('2021-01-31', 344e3),
    ('2021-02-14', 791e3),  # orig 784e3
    ('2021-02-28', 1419e3), # orig 1331e3),
    ('2021-03-14', 2079e3), # orig 1922e3
    ('2021-03-28', 2636e3), # original 2411e3
    ('2021-04-04', 3130e3), # original estimate 2850e3
    # Official estimates changed retroactively around here;
    ('2021-04-11', 3.78e6),
    ('2021-04-18', 4.4e6),
    ('2021-05-02', 5.6e6),
    ('2021-05-09', 6.5e6),
    ('2021-05-23', 8.5e6),
    ('2021-05-30', 9.4e6),
    ('2021-06-13', 12.4e6),
    ('2021-06-27', 15.3e6),
    ('2021-08-01', 20.9e6),
    # estimate RIVM
    ('2021-08-31', 24.3e6),
    ]


# Fully vaccinated from coronadadashboard
vac_recs_full = [
    ('2021-01-24', 0),
    ('2021-02-21', 163e3),
    ('2021-03-28', 667e3),
    ('2021-04-18', 993e3),
    ('2021-05-09', 1.57e6),
    ('2021-05-23', 2.44e6),
    ('2021-06-13', 4.48e6),
    ('2021-06-27', 5.91e6),
    ('2021-07-11', 7.10e6),
    ('2021-07-18', 7.98e6),
    ('2021-07-25', 8.64e6),
    ('2021-08-01', 9.31e6),
    ('2021-08-08', 10.0e6),
    ('2021-08-15', 10.70e6),
    ('2021-08-22', 10.93e6),
    ('2021-08-29', 11.13e6),
    ('2021-09-05', 11.30e6),
    ('2021-09-12', 11.38e6),
    ('2021-09-19', 11.42e6),
    ]

vdf = pd.DataFrame.from_records(vac_recs, columns=['Date', 'ncum'])
fvdf = pd.DataFrame.from_records(vac_recs_full, columns=['Date', 'ncum'])

for df in [vdf, fvdf]:
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)

    # Extrapolate 3 weeks
    dt = (df.index[-1] - df.index[-2]) / pd.Timedelta('1 d')
    ncum_a, ncum_b = df.iloc[-2:]['ncum']
    dt_x = 21 # 3 weeks
    ncum_x = ncum_b + (ncum_b-ncum_a) / dt * dt_x
    tm_x = df.index[-1] + pd.Timedelta(dt_x, 'd')
    df_x = pd.DataFrame(dict(ncum=ncum_x), index=[tm_x])
    df = df.append(df_x)
    # print(df)

    # interpolate to 1-day resolution
    f_ncum = scipy.interpolate.interp1d(
        df.index.to_julian_date(), df['ncum'], bounds_error=False, fill_value='extrapolate')

    end_date = max(pd.to_datetime('now'), df.index[-1])
    new_df = pd.DataFrame(
        index=pd.date_range(df.index[0], end_date, freq=pd.Timedelta('1 d'))
        )
    new_df['ncum'] = f_ncum(new_df.index.to_julian_date()).astype(int)
    if df is vdf:
        vdf = new_df
    else:
        fvdf = new_df


#%%

# calculate 1st and second.
ndays = len(vdf)
# deltas per day
dn1st = np.zeros(ndays, dtype=int)
dn2nd = dn1st.copy()

delta_ntots = vdf['ncum'].diff()
DATE_RECOVERED_1SHOT_ONLY = pd.to_datetime('2021-04-15')

for i, (delta_n, date) in enumerate(zip(delta_ntots, delta_ntots.index)):
    if i == 0:
        continue
    if i < DELAY_2ND_VACC:
        dn1st[i] = delta_n
    else:
        # starting 2021-04-01: only one shot for those tested positive since
        # Oct 2020. Applies to date of booking; assuming actual vaccination
        # is 2 weeks later.
        f = 1 if date < DATE_RECOVERED_1SHOT_ONLY else 0.9
        dn2nd[i] = dn1st[i-DELAY_2ND_VACC] * f
        dn1st[i] = delta_n - dn2nd[i]

vdf['n1st'] = np.cumsum(dn1st)
vdf['n2nd'] = np.cumsum(dn2nd)

# when past multiple of 5%?
# build inverse interpolation function
t0 = vdf.index[0]
f_f1st = scipy.interpolate.interp1d(
    vdf['n1st'].values/17.4e6, np.arange(len(vdf)), kind='linear',
    bounds_error=True
    )
for f in np.arange(0.05, 1, 0.05):
    try:
        t = t0 + f_f1st(f) * pd.Timedelta(1, 'd')
        print(f'{t.strftime("%Y-%m-%d")},,{f*100:g}% 1e prik')
    except ValueError:
        break

print(vdf)
#%% label texts for full vaccination

t0 = fvdf.index[0]
f_f1st = scipy.interpolate.interp1d(
    fvdf['ncum'].values/17.4e6, np.arange(len(fvdf)), kind='linear',
    bounds_error=True
    )
for f in np.arange(0.05, 1, 0.05):
    try:
        t = t0 + f_f1st(f) * pd.Timedelta(1, 'd')
        print(f'{t.strftime("%Y-%m-%d")},,{f*100:g}% gevaccineerd')
    except ValueError:
        break




#%%
plt.close('all')
fig, ax = plt.subplots(figsize=(9, 4), tight_layout=True)
npop = 17.4e6
ax.plot(vdf['n1st']/npop * 100, label='eerste prik')
ax.plot(vdf['ncum']/npop * 100, label='prikken totaal', linestyle='--')
ax.legend()
ax.set_title('Vaccinaties COVID-19 Nederland')
ax.set_ylabel('Percentage van bevolking')
ax.axvline(pd.to_datetime('now'), color='k')
ax.text(pd.to_datetime('now') + pd.Timedelta('1 d'), 5, 'Vandaag', rotation=90)
ymin, ymax = ax.get_ylim()

ax.set_ylim(ymin, ymax)

tools.set_xaxis_dateformat(ax)
ax2 = ax.twinx()
ax2.set_ylim(ymin*npop/1e8, ymax*npop/1e8)
ax2.set_ylabel('Aantal prikken (miljoen)')

fig.show()
