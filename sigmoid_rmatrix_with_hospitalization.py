#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generational epidemic simulation with partial populations and hospital capacity.

2021-10-31

@author: @hk_nien
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.ticker

def textmatrix(varname, a, fmt='3.1f'):
    """Convert matrix to multiline text representation, like

    varname = [ 1  2 ]
              [ 3  4 ]
    """
    prefix = f'{varname} = '
    prespc = ' '*len(prefix)
    lines = []
    fmt = f'{{:{fmt}}}'
    for i, row in enumerate(a):
        xs = [fmt.format(x) for x in row]
        line = '  '.join(xs)
        line = (prespc if i else prefix) + f'[ {line} ]'
        lines.append(line)
    return '\n'.join(lines)


def run_sim(R0, pop, inf, sus, subpop_names=None,
            ihr=None, hosp_dur=5, hosp_delay=2,
            hicap=(1e4, 1e3), Tgen=4
            ):
    """Run SIR simulation with next-gen matrix; print and plot the results.

    Parameters:

    - R0: next-generation matrix (n, n) array.
    - pop: populations per subpopulation, (n,) array.
    - inf: initial number of infections per subpopulation, (n,) array.
    - sus: initial number of susceptible individuals per subpop, (n,) array.
    - subpop_names: optional list of subpopulation names, length n.
    - ihr: optional infection hospitalization rate,
      array (n, 2) with entries for regular hospitalization and IC admission.
      (regular hospitalization does not include IC admissions).
    - hosp_dur: hospitalization duration (in units of generations, not days!)
    - hosp_delay: delay from infection to hospitalization (in units of generations)
    - hicap: hospital and IC capacity as 2-tuple.
    - Tgen: generation interval in days.

    Next-generation matrix 'R0' as generalization of scalar R0. Accounts for
    herd-immunity buildup.

    At generation i: inf[i+1] = R @ inf[i]

    where `inf` is the 'infectious' vector and R the effective next-gen matrix
    given the remaining susceptible (`sus`) and recovered (`rec`) numbers.
    """
    R0, pop, inf, sus = np.array(R0), np.array(pop), np.array(inf), np.array(sus)
    (n,) = pop.shape
    assert inf.shape == sus.shape == pop.shape
    assert R0.shape == (n, n)
    if subpop_names is None:
        subpop_names = [f'{i}' for i in range(1, n+1)]
    assert len(subpop_names) == n

    # append numbers at each generation to these lists.
    suss = [sus]
    infs = [inf]

    def _strarr(a, fmt='{:9.3g}', sep=' '):
        ss = [fmt.format(x) for x in a]
        return sep.join(ss)

    for i in range(50):
        # incoming infectious contacts per group, (n,) vector.
        con = R0 @ inf
        # probability of at least one infectious incoming contact.
        prob = 1 - np.exp(-con/pop)
        # next generation infections
        inf = sus * prob
        sus = sus - inf
        suss.append(sus)
        infs.append(inf)
        # print(f'Inf: {_strarr(inf)}   Sus: {_strarr(sus)}')
        if np.sum(inf) < np.sum(infs[0]):
            break

    ngen = len(suss)  # number of generations

    suss = np.array(suss)
    infs = np.array(infs)

    # Hospital book-keeping
    nics = np.zeros((ngen, n))
    nhoss = nics.copy()

    def add_occup(occup, delta, cap):
        # add to occupancy [i, :]; handle rejects.
        occup[i, :] = occup[i-1, :] + delta

    if ihr is not None:
        ihr, iir = np.array(ihr).T  # both arrays of length (n)
        for i in range(1, ngen):
            j = i - hosp_delay
            if j >= 0:
                add_occup(nhoss, infs[j, :]*ihr, hicap[0])
                add_occup(nics, infs[j, :]*iir, hicap[1])
            j = i - hosp_delay - hosp_dur
            if j >= 0:
                nhoss[i, :] -= infs[j, :]*ihr
                nics[i, :] -= infs[j, :]*iir

    R0_tot = np.around(np.linalg.eigvals(R0).max(), 2)  # system R0 value
    R0_grp = R0.sum(axis=0)  # outgoing R0 per group
    R0_grp_str = _strarr(R0_grp, '{:g}', ', ')
    daynums = np.arange(ngen) * Tgen

    fig, axs = plt.subplots(
        2 + (ihr is not None), 1,
        figsize=(7, 5 + 2*(ihr is not None)),
        tight_layout=True, sharex=True
        )
    if ihr is None:
        (axrec, axinf) = axs
        axhos = None
    else:
        (axrec, axinf, axhos) = axs
        axhos.set_title('Bezetting ziekenhuis en IC')

    axs[-1].set_xlabel('Dagnummer')
    axrec.set_title(
        f'Systeem $R_0$={R0_tot:.2f}, per deelpopulatie {R0_grp_str}\n\n'
        '% groepsimmuniteit per deelpopulatie')
    axinf.set_title('Nieuwe ziektegevallen per deelpopulatie')

    frac_recs = 1 - (suss + infs)/pop  # (ngen, n) array for m generations
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    for igrp, marker, name in zip(range(n), 'o<>^vsx+*', subpop_names):
        color = colors[igrp]
        axrec.plot(daynums, 100*frac_recs[:, igrp], f'{marker}-', label=name,
                   color=color)
        axinf.semilogy(daynums, infs[:, igrp], f'{marker}-', label=name,
                       color=color)
    if axhos:
        axhos.semilogy(
            daynums, nhoss[:, igrp], 's--',
            label=f'Verpleegafdeling', color='k', markersize=3
            )
        axhos.semilogy(
            daynums, nics[:, igrp], f's-',
            label=f'IC', color='k', markersize=3
            )
        axhos.axhline(hicap[0], linestyle='--', color='red')
        axhos.axhline(hicap[1], linestyle='-', color='red')

    fmt = '3.1f' if R0.min() >= 0.1 else '5.3f'
    R0_str = textmatrix('R0', R0, fmt=fmt)
    print(R0_str)
    print(f'Final group immunity: {_strarr(frac_recs[-1, :]*100, "{:.0f}")} %')
    print()
    axrec.text(
        0.02, 0.4, R0_str,
        transform=axrec.transAxes,
        fontfamily='monospace',
        bbox=dict(facecolor='white', edgecolor='none'),
        )
    for ax in axs:
        ax.legend()
        ax.grid()
    locator = matplotlib.ticker.MaxNLocator(steps=[1, 2, 5, 10])
    axs[-1].xaxis.set_major_locator(locator)
    #for ax in axs[1:]:
    #    ax.grid(axis='y', which='minor')
    fig.show()


if __name__ == '__main__':

    # "Let it rip among the invulnerable."
    R0_rip = [
        [2.3, 0.1],
        [0.2, 0.5]
        ]

    # Moderate measures to "protect the vulnerable"
    R0_moderate = [
        [1.3, 0.1],
        [0.2, 0.5]
        ]

    # Unrealistic measures.
    R0_moderate = [
        [1.3, 0.01],
        [0.01, 0.5]
        ]

    pop = [12e6, 5e6]
    inf = [700, 300]
    sus = np.array(pop) - inf
    plt.close('all')

    for R0_i in [R0_rip, R0_moderate]:
        run_sim(
            R0_i, pop, inf, sus,
            subpop_names=['Niet-kwetsbaren', 'Kwetsbaren'],
            ihr=[[0, 0], [0.046, 0.024]]
            )

    run_sim(
        [[2.5]], [17e6], [1e3], [17e6],
        subpop_names=['Hele bevolking'],
        ihr=[[0.013, 0.007]]
        )


