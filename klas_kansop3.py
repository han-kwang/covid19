#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 19 20:25:25 2022

@author: @hk_nien
"""
import numpy as np
from scipy.stats import binom
import matplotlib.pyplot as plt

k = 3

plt.close('all')
fig, ax = plt.subplots(tight_layout=True, figsize=(6, 4))
ax.set_title(f'Kans op ≥ {k} positief in klas van n')
ax.grid()
ax.grid(axis='both', which='minor')
ax.set_xlabel('Prevalentie infecties (%)')
ax.set_ylabel('Kans (%)')
ax.set_ylim(0.1, 120)

for n in [15, 20, 25, 30][::-1]:

    ps = np.linspace(0.005, 0.5)
    pks = binom.cdf(k-1, n, ps)
    ax.loglog(ps*100, 100 - pks*100, label=f'n = {n}')

ax.legend()
fig.show()