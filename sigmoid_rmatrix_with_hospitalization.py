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

def textmatrix(varname, a, fmt='4.3g'):
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


def _strarr(a, fmt='9.3g', sep=' '):
    """Return string with array elements."""
    fmt = f'{{:{fmt}}}'
    ss = [fmt.format(x) for x in a]
    return sep.join(ss)


def _run_sim(
        R0s, pop, inf, sus, ihr, hosp_dur=5,
        hosp_delay=2, hicap=(1e4, 1e3), Tgen=4, maxtm=200
        ):
    """Run simulations, return data (no plots).

    Parameters as in run_plot_sim().

    Return:

    - R0s: dict of t->R0 with sorted keys and R0 sanity-checked arrays.
    - suss: array (ngen, n); number of susceptible individuals.
    - infs: array (ngen, n): number of infected individuals.
    - nhoss: array (ngen,): number of hospitalized individuals.
    - nics: array (ngen,): number of individuals in intensive care.
    - nrejs_ho, nrejs_ic: array (ngen,): number of rejected patients from
      hospital, IC.
    """
    pop, inf = np.array(pop), np.array(inf)
    assert inf.shape == pop.shape
    if sus is None:
        sus = pop - inf
    (n,) = pop.shape

    # sanitize R0s into sorted dict of arrays.
    if not isinstance(R0s, dict):
        R0s = {0.0: R0s}
    R0s_ = {}
    if 0 not in R0s:
        raise ValueError('R0s has no entry for t=0.')
    for t in sorted(R0s):
        if t/Tgen != int(t/Tgen):
            raise ValueError(f'Time {t} in R0s is not a multiple of Tgen={Tgen}.')
        R0 = np.array(R0s[t])
        assert R0.shape == (n, n)
        R0s_[t] = R0
    R0s = R0s_
    del R0s_

    # append numbers at each generation to these lists.
    suss = [sus]
    infs = [inf]

    for i in range(int((maxtm)/Tgen+1)):
        t = i*Tgen
        if t in R0s:
            R0 = np.array(R0s[t])
            assert R0.shape == (n, n)
        # con: incoming infectious contacts per group, (n,) vector.
        con = R0 @ inf
        # probability of at least one infectious incoming contact, per
        # susceptible individual; (n,) array.
        prob = 1 - np.exp(-con/pop)
        # next generation infections
        inf = sus * prob
        sus = sus - inf
        suss.append(sus)
        infs.append(inf)
        # print(f'Inf: {_strarr(inf)}   Sus: {_strarr(sus)}')
        if np.sum(inf) < np.sum(infs[0]):
            # End if there are fewer infections than what we started with.
            break

    ngen = len(suss)  # number of generations
    suss = np.array(suss)
    infs = np.array(infs)

    # Hospital book-keeping. These arrays [:, 0] for hospital occupancy
    # in case of unlimited capacity; [:, 1] for actual occupancy.
    nics = np.zeros((ngen, 2))
    nhoss = nics.copy()
    # Number of rejects
    nrejs_ho = np.zeros(ngen)
    nrejs_ic = np.zeros(ngen)
    if not ihr:
        ihr = [(0.0, 0.0)] * n

    def add_occup(i, occup, infs, rejects, rate, cap):
        # add infs[i-hosp_delay] to occupancy[i], subtract infs[i-hosp_dur].
        # handle rejects.
        j = i - hosp_delay - hosp_dur
        occup[i, :] = occup[i-1, :]
        if j >= 0:
            n_released = (infs[j, :]*rate).sum()
            occup[i, 0] -= n_released
            occup[i, 1] -= n_released - rejects[j]
        j = i - hosp_delay
        if j >= 0:
            n_add = (infs[j, :]*rate).sum()
            occup[i, :] += n_add
            if occup[i, 1] > cap:
                rejects[i] = occup[i, 1] - cap
                occup[i, 1] = cap
    # ihr: infection hosp/ic rate; shape (n, 2) with axis 1 for hospital, IC
    ihr = np.array(ihr)
    for i in range(1, ngen):
        add_occup(i, nhoss, infs, nrejs_ho, ihr[:, 0], hicap[0])
        add_occup(i, nics, infs, nrejs_ic, ihr[:, 1], hicap[1])

    nhoss = nhoss[:, 0]
    nics = nics[:, 0]
    return R0s, suss, infs, nhoss, nics, nrejs_ho, nrejs_ic


def run_plot_sim(
        R0s, pop, inf, sus=None, subpop_names=None, ihr=None, hosp_dur=5,
        hosp_delay=2, hicap=(1e4, 1e3), Tgen=4, maxtm=200,
        title=None
        ):
    """Run SIR simulation with next-gen matrix; print and plot the results.

    Parameters:

    - R0s: next-generation matrix (n, n) array OR list of tuples (t, R0_i)
      with R0_i the R0 matrix starting at time t.
      (If Tgen is not an integer, this may be tricky.)
    - pop: populations per subpopulation, (n,) array.
    - inf: initial number of infections per subpopulation, (n,) array.
    - sus: initial number of susceptible individuals per subpop, (n,) array.
      (Default: difference pop - inf).
    - subpop_names: optional list of subpopulation names, length n.
    - ihr: optional infection hospitalization rate,
      array (n, 2) with entries for regular hospitalization and IC admission.
      (regular hospitalization does not include IC admissions).
    - hosp_dur: hospitalization duration (in units of generations, not days!)
    - hosp_delay: delay from infection to hospitalization (in units of generations)
    - hicap: hospital and IC capacity as 2-tuple.
    - Tgen: generation interval in days.
    - maxtm: maximum time to simulate. Simulation will also stop when
      infections are lower than starting value.
    - title: optional title (str).

    Next-generation matrix 'R0' as generalization of scalar R0. Accounts for
    herd-immunity buildup.

    At generation i: inf[i+1] = R @ inf[i]

    where `inf` is the 'infectious' vector and R the effective next-gen matrix
    given the remaining susceptible (`sus`) and recovered (`rec`) numbers.
    """

    # number of individuals in given state, by generation.
    # arrays size (ngen, n) (ngen, n), (ngen,), (ngen,)
    R0s, suss, infs, nhoss, nics, nrej_ho, nrej_ic = _run_sim(
        R0s, pop, inf, sus=sus, ihr=ihr, hosp_dur=hosp_dur,
        hosp_delay=hosp_delay, hicap=hicap, Tgen=Tgen, maxtm=maxtm)
    ngen, n = suss.shape

    if subpop_names is None:
        subpop_names = [f'{i}' for i in range(1, n+1)]
    assert len(subpop_names) == n

    # System R0 value (largest eigenvalue) is the R0 that you'd see
    # after a while if there is no immunity buildup.
    R0 = R0s[0]
    R0_sys = np.around(np.linalg.eigvals(R0).max(), 2)
    R0_grp = R0.sum(axis=0)  # outgoing R0 per group
    R0_grp_str = _strarr(R0_grp, 'g', ', ')

    #### Generate plots. ####
    # Setup plots
    fig, axs = plt.subplots(
        3, 1, figsize=(7, 7),
        tight_layout=True, sharex=True
        )
    (axrec, axinf, axhos) = axs
    axhos.set_title(
        f'Bezetting ziekenhuis en IC en geweigerde patiënten per {Tgen} dagen'
        )

    # Plot the data
    daynums = np.arange(ngen) * Tgen
    if title:
        print(title)
        fig.canvas.set_window_title(f'{title} - SIR-simulation')
        title = f'{title}\n'
    else:
        title = ''
    if len(R0s) == 1:
        title += f'Systeem $R_0$={R0_sys:.2f}, per deelpopulatie {R0_grp_str}\n'
    axs[-1].set_xlabel('Dagnummer')
    axrec.set_title(f'{title}\n% immuniteit')
    axrec.set_ylim(0, 100)
    axinf.set_title(f'Nieuwe ziektegevallen per deelpopulatie, per {Tgen} dagen')

    frac_recs = 1 - (suss + infs)/pop  # (ngen, n) array for m generations
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    marker_scale = 1 if ngen < 50 else 50/ngen if ngen < 100 else 0
    for igrp, marker, name in zip(range(n), 'o<>^vsx+*', subpop_names):
        color = colors[igrp]
        axrec.plot(daynums, 100*frac_recs[:, igrp], f'{marker}-', label=name,
                   color=color, markersize=marker_scale*5)
        axinf.semilogy(daynums, infs[:, igrp], f'{marker}-', label=name,
                       color=color, markersize=marker_scale*5)

    for data, sty, label, color in [
            (nhoss, 's--', 'Verpleegafd. bezetting', 'black'),
            (nics, 's-', 'IC bezetting', 'black'),
            (nrej_ho, 'x--', 'Verpleegafd. geweigerd', 'gray'),
            (nrej_ic, 'x-', 'IC geweigerd', 'gray')
            ]:
        data = np.where(data, data, np.nan)  # prevent log(0) in the graph
        axhos.semilogy(
            daynums, data, sty, label=label, color=color,
            markersize=3*marker_scale*(1.5 if 'x' in sty else 1.0),
            )


    axhos.axhline(hicap[0], linestyle='--', color='red')
    axhos.axhline(hicap[1], linestyle='-', color='red')

    # Print and add other metadata. Ticks and legends.
    fmt = '3.3g' if R0.min() >= 0.1 else '5.3f'
    R0_str = textmatrix('R0', R0, fmt=fmt)
    print(R0_str)
    print(f'Final group immunity: {_strarr(frac_recs[-1, :]*100, ".0f")} %')
    nrej_k = nrej_ho.sum()/1000, nrej_ic.sum()/1000
    print(f'Total rejected {nrej_k[0]:.0f}k (hosp), {nrej_k[1]:.0f}k (IC),'
          f' {sum(nrej_k):.0f}k (total)')

    for i, t in enumerate(sorted(R0s)):
        R0 = R0s[t]
        if len(R0s) > 1:
            for ax in axs:
                ax.axvline(t, color='k', linestyle='--')
        if len(R0s) <= 5:
            # don't bother with matrices if there are many
            axrec.text(
                t, 40 + (i%2)*30,
                textmatrix('R0', R0, fmt=fmt),
                fontfamily='monospace',
                bbox=dict(facecolor='white', edgecolor='none', alpha=0.8),
                )
    for ax in axs:
        ax.legend()
        ax.grid()
    locator = matplotlib.ticker.MaxNLocator(steps=[1, 2, 5, 10])
    axs[-1].xaxis.set_major_locator(locator)
    #for ax in axs[1:]:
    #    ax.grid(axis='y', which='minor')
    fig.show()
    plt.pause(0.25)  # allow drawing of plot

if __name__ == '__main__':

    cases = [
        dict(
            title='Rondrazen',
            R0s=[[2.5]]
            ),
        dict(
            title='Rondrazen door onkwetsbaren',
            R0s=[
                [2.3, 0.1],
                [0.2, 0.5]
                ],
            ),
        dict(
            title='Bescherm de kwetsbaren',
            R0s=[
                [1.3, 0.1],
                [0.2, 0.5]
                ],
            maxtm=365,
            ),
        dict(
            title='Extreme isolatie kwetsbaren',
            R0s=[
                [1.3, 0.01],
                [0.01, 0.5]
                ],
            maxtm=730,
            ),
        dict(
            title='Titreren op zorgcapaciteit',
            R0s={
                0: [[2.5, 0.1], [0.2, 0.5]],
                16: [[1.0, 0.1], [0.1, 0.5]],
                152: [[1.15, 0.1], [0.1, 0.5]],
                372: [[1.4, 0.1], [0.1, 0.5]],
                552: [[1.7, 0.1], [0.1, 0.5]],
                },
            maxtm=730,
            ),
        dict(
            title='Titreren op zorgcapaciteit - extreme isolatie, dan open',
            R0s={
                0: [[2.5, 0.1], [0.02, 0.5]],
                24: [[1.0, 0.1], [0.02, 0.5]],
                80: [[1.4, 0.1], [0.02, 0.5]],
                160: [[1.9, 0.1], [0.02, 0.5]],
                240: [[1.7, 0.8], [0.8, 1.7]],
                },
            maxtm=730,
            ),
        dict(
            title='Titreren op zorgcapaciteit tot groepsimmuniteit',
            R0s={
                0: [[2.5, 0.1], [0.02, 0.5]],
                24: [[1.0, 0.1], [0.02, 0.5]],
                80: [[1.4, 0.1], [0.02, 0.5]],
                160: [[1.9, 0.1], [0.02, 0.5]],
                228: [[2.5, 0.1], [0.1, 0.5]],
                300: [[2.5, 0.1], [0.1, 1.0]],
                400: [[2.5, 0.2], [0.2, 1.1]],
                640: [[2.5, 0.2], [0.3, 1.2]],
                840: [[2.5, 0.2], [0.4, 1.3]],
                },
            maxtm=3*365,
            ),
        ]

    plt.close('all')

    for case in cases[:]:
        R0s = case['R0s']
        R0 = R0s[0.0] if isinstance(R0s, dict) else R0s
        if len(R0) == 2:
            pop = [12e6, 5e6]
            inf = [700, 300]
            subpop_names=['Niet-kwetsbaren', 'Kwetsbaren']
            ihr=[[0, 0], [0.046, 0.024]]
        else:
            pop = [17e6]
            inf = [1000]
            subpop_names=['Hele bevolking']
            ihr = [[0.013, 0.007]]

        run_plot_sim(
            **case, subpop_names=subpop_names, pop=pop, inf=inf,
            ihr=ihr
            )

