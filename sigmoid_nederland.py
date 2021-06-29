#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 24 19:32:42 2021

@author: hnienhuy
"""
import numpy as np

sus = 0.6
inf = 8e-4
R0 = 5
pop = 17.4e6

print('Gen          inf      sus      R0*sus')


import matplotlib.pyplot as plt

suss = [sus]
infs = [inf]


for i in range(1, 40):

    inf_next = sus * (1 - np.exp(-inf*R0))
    sus -= inf_next
    inf = inf_next
    suss.append(sus)
    infs.append(inf)
    print(f'{i:<5} {inf:9.3g} {sus:9.3g} {R0*sus:9.3g}')
    if inf < 1e-6:
        break

suss = np.array(suss)
infs = np.array(infs)
gens = np.arange(len(suss))

plt.close('all')

fig, axs = plt.subplots(2, 1, tight_layout=True, sharex=True)
ax = axs[0]
ax.plot(gens, 100*(1-suss), 'o-', label='% Groepsimmuniteit')
ax.set_ylim(None, 100)

ax = axs[1]
ax.set_yscale('log')
ax.plot(gens, infs*pop, 'o-', label='Infecties per generatie')
ax.set_xlabel('Generatie # (4 dagen)')
axs[0].set_title(f'Covid-19 in Nederland met R0={R0}')

for ax in axs:
    ax.grid()
    ax.legend()
fig.show()