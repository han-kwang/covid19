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
    ('2021-05-15', 7.5e6), # estimate as of 27 Apr.
    ]

vdf = pd.DataFrame.from_records(vac_recs, columns=['Date', 'ncum'])
vdf['Date'] = pd.to_datetime(vdf['Date'])
vdf = vdf.set_index('Date')


# interpolate to 1-day resolution
f_ncum = scipy.interpolate.interp1d(
    vdf.index.to_julian_date(), vdf['ncum'], bounds_error=False, fill_value='extrapolate')

end_date = max(pd.to_datetime('now'), vdf.index[-1])


vdf = pd.DataFrame(
    index=pd.date_range(vdf.index[0], end_date, freq=pd.Timedelta('1 d'))
    )
vdf['ncum'] = f_ncum(vdf.index.to_julian_date()).astype(int)

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
plt.close('all')
fig, ax = plt.subplots(figsize=(9, 4), tight_layout=True)
ax.plot(vdf['n1st']/17.4e6 * 100)
ax.set_ylabel('Percentage eerste prik')
tools.set_xaxis_dateformat(ax)
fig.show()
