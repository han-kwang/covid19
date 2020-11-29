"""Droplet evaporation and falling speed"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker
# Constants
Ru = 8.314 # gas constant J/mol-K
mu = 18e-6 # viscosity of air Pa.s
rho = 1e3 # density of water kg/m3
DHv = 41e3 # enthalpy of vaporization, J/mol
grav = 9.8 # gravity, m2/s
nv0 = 0.94 # vapor molar density at 293 K
alpha = 0.05 # temperature derivative of nv (mol/(m3 K))
k = 0.026 # thermal conductivity of air (W/m-K)
D = 2.4e-5 # diffusivity water in air (m2/s)
mfp = 50e-9 # mean free path air


def v_fall(r):
    """Return falling velocity (m/s) for droplet radius (m).

    Returns NaN if out of valid range.
    """

    Kn = mfp/r
    C = 1 + Kn*(1.26 + 0.40 * np.exp(-2.2/Kn))
    vfall = np.array(2*rho*grav*r**2 / (9*mu*C))
    vfall[r > 50e-6] = np.nan
    return vfall

def t_evap(r, phi):
    """Return estimated evaporation time at T=293 K.

    - r: radius (m)
    - phi: relative humidity
    """
    return 3.6e9 * r**2 / (1 - phi)


def pvap_water(TC):
    """Return vapor pressure of water (Pa) for temperature in C"""

    # Antoine equation
    a, b, c = 8.07131, 1730.63, 233.426
    p_mmHg = 10**(a - (b/(c+TC)))
    p_Pa = p_mmHg * 133.3

    return p_Pa

def plot_rh_curves():
    """Plot RH curves outdoors/indoors.

    This is approximate (do not account for ice, thermal expansion)
    """

    fig, ax = plt.subplots(tight_layout=True)
    ax.set_xlabel('Temperature ($^{\circ}$C)')
    ax.set_ylabel('Relative humidity (%)')


    Ts = np.linspace(-10, 20, num=250)
    rhs = np.linspace(0, 100, 21)
    psat = pvap_water(Ts)

    for i, rh20 in enumerate(rhs):
        rh = rh20 * psat[-1]/psat
        rh[rh > 100] = np.nan
        lstyle = ['-', '--'][i % 2]
        ax.plot(Ts, rh, color='k', linestyle=lstyle)

    ax.set_xlim(-10, 20)
    ax.set_ylim(0, 100)
    ax.grid()
    fig.show()

    fname = 'output/rh_temperature.pdf'
    fig.savefig(fname)
    print(f'Wrote {fname}')



def plot_times():
    """Create plot."""

    r = np.exp(np.linspace(np.log(3e-6), np.log(100e-6), num=200))
    fig, ax = plt.subplots(1, 1, figsize=(6, 4), tight_layout=True)

    # Falling time
    height = 1.8
    tfall = height/v_fall(r)

    ax.set_xlabel('Droplet radius ($\\mu$m)')
    ax.loglog(r*1e6, tfall, label=f'Falling time from {height:.1f} m')
    ax.set_ylabel('Time (s)')

    lstyles = ['-', '--', ':', '-.'] * 2

    rhs = np.array([0.90, 0.80, 0.60, 0.40, 0.20])
    for rh, lstyle in zip(rhs, lstyles[1:]):
        tau = t_evap(r, rh)
        ax.loglog(r*1e6, tau, linestyle=lstyle,
                  label=f'Evaporation time (RH={rh*100:.0f}%)')

    ax.legend()
    ax.grid()

    xticks = np.array([3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
    ax.set_xticks(xticks)
    xtlabels = [f'{x:g}' for x in xticks]
    # eliminate tick labels for 7 and 9.
    xtlabels = [('' if x[0] in '79' else x) for x in xtlabels]

    ax.set_xticklabels(xtlabels)

    fig.show()
    fname = 'output/droplets_simple.pdf'
    fig.savefig(fname)
    print(f'Wrote {fname}.')

if __name__ == '__main__':
    plt.close('all')
    plot_times()
    plot_rh_curves()

