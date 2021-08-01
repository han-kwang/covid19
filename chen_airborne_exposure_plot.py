#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 26 18:37:18 2020

@author: @hk_nien
"""

# Chen 2020, https://doi.org/10.1101/2020.03.16.20037291
# Fig 7b, 'exposure from small droplets emitted from talking'

import numpy as np
import matplotlib.pyplot as plt

# fig 7B
xs=np.array([0.5, 1, 1.5, 2])
ys=10**np.array([-3.9, -5.2, -6.0, -6.5])

# fig 7D (coughing)
xs2 = np.array([1.1, 1.6, 2.0])
ys2 = 10**np.array([-3.15, -3.85, -4.5])

fig, ax = plt.subplots(tight_layout=True, figsize=(6, 4))
ax.set_xlabel('Distance (m)')
ax.set_ylabel('Exposure ($\\mu$L)')

ax.loglog(xs, ys, 'o', color='r', label='fig 7B, "talking 2 min, airborne"')
ax.loglog(xs, 10**-5.2/xs**4.3, color='r', label='Power law $1/r^{4.3}$')
ax.loglog(xs2, ys2, 's', color='b', label='fig 7D, "coughing once, airborne"')
ax.loglog(xs2, 10**-3.85/(xs2/1.55)**5, color='b', label='Power law $1/r^{5}$')
ax.legend()
ax.set_title('Short-range airborne droplet exposure\n'
             '(Data: Chen https://doi.org/10.1101/2020.03.16.20037291)')
fig.text(0.98, 0.02, '@hk_nien', horizontalalignment='right')
fig.show()
