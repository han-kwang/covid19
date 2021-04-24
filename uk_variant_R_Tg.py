#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Effect of generation interval and reproduction number on
UK-strain predictions.

Twitter: @hk_nien

Created on Tue Dec 29 13:03:00 2020
"""

import numpy as np
import pandas as pd


def get_tbl_row(T2, log_diff=0.108, Tgs=(4, 6.5)):
    """Return dict for this doubling time in the old strain.

    - T2: doubling time (days)
    - log_diff: log slope (per day) infectiousness difference for new strain.
    - Tgs: generation intervals to consider.
    """

    T2 = float(T2)
    Tgs = np.array(Tgs)

    # R estimates for dominant, old strain
    Rs = 2**(Tgs/T2)

    # R estimates for new strain
    Rxs = Rs * np.exp(log_diff*Tgs)

    Rs = np.around(Rs, 2)
    Rxs = np.around(Rxs, 2)


    return dict(T2=T2,
                R1=Rs[0],
                Rx1=Rxs[0],
                R2=Rs[1],
                Rx2=Rxs[1])


T2s = [np.inf, -21, -15, -12.5, -10, -7, -6.4, -5]

rows = [get_tbl_row(T2) for T2 in T2s]
df = pd.DataFrame(rows)
print(df)
print(df.to_latex())
