#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""What happens when you have three variants?

a, b are old variants with different repro numbers.
c is a the B.1.1.7 variant.

Created on Sat Jan 23 18:56:36 2021

@author: @hk_nien
"""

import numpy as np
import matplotlib.pyplot as plt
def simulate_abc(Rs, n0s, Tg=4):
    """Simulate with initial populations n0s (3,)."""


    vnames = 'abcde'[:len(Rs)]
    RsT = np.array(Rs).reshape(-1, 1)
    n0sT = np.array(n0s).reshape(-1, 1)
    ts = np.arange(90)

    ns = n0sT * np.exp(np.log(RsT)/Tg * ts)
    dndts = np.log(RsT)/Tg * ns

    ntots = ns.sum(axis=0)
    dntots = dndts.sum(axis=0)


    # apparent R
    Rapp = np.exp(Tg * dntots/ntots)

    # odds ratio
    odds = ns[-1, :]/ns[:-1, :].sum(axis=0)
    log_slopes = np.diff(np.log(odds))[[0, -1]]

    odds_line0 = odds[0] * np.exp(log_slopes[0]*ts)
    odds_line1 = odds[-1] * np.exp(log_slopes[1]*(ts-ts[-1]))


    fig, axs = plt.subplots(3, 1, tight_layout=True, sharex=True, figsize=(7, 7))
    lstyles = ['--', '-.', ':', '-'] * 2

    for R, n1s, vn, lsty in zip(Rs, ns, vnames, lstyles):
        axs[0].semilogy(ts, n1s, linestyle=lsty, label=f'Variant {vn} (R={R})')
        axs[1].plot(ts, np.full_like(ts, R, dtype=float), linestyle=lsty,  label=f'{vn}: R={R}')

    ax = axs[0]
    ax.semilogy(ts, ntots, linestyle=lstyles[3], label='Total')
    ax.legend(loc='upper right')

    ax.set_ylabel('daily infections')
    ax.grid()

    ax = axs[1]
    ax.plot(ts, Rapp, linestyle=lstyles[3], label='Effective R')
    ax.grid()
    ax.set_ylabel('R')
    ax.legend()

    ax = axs[2]
    ax.set_ylabel('Odds ratio')

    odds_title = f'Ratio {vnames[-1]} / ({"+".join(vnames[:-1])})'
    ax.semilogy(ts, odds, 'k', label=odds_title)
    ax.semilogy(ts, odds_line0, 'k--', label=f'log$_e$ slope {log_slopes[0]:.3f}')
    ax.semilogy(ts, odds_line1, 'k-.', label=f'log$_e$ slope {log_slopes[1]:.3f}')
    ax.legend()
    ax.grid()

    axs[0].set_title('Scenario: three competing variants')
    axs[-1].set_xlabel('Day number')

    fig.show()

plt.close('all')
simulate_abc([0.8, 1, 1.25], [9000, 500, 500])
