#!/usr/bin/env python3

from copy import deepcopy
import numpy as np
from scipy.sparse import csr_matrix, issparse
import matplotlib.pyplot as plt
import pandas as pd
import itertools

class EpidemyState:
    """Epidemiological state vector.

    Attributes:

    - bins: bin array, consisting of

      - 1: susceptible
      - nb_lat: infected and latent
      - nb_sym: symptomatic, not hospitalized (infectious on 1st bin)
      - nb_hos: hospitalized
      - 1: recovered
      - 1: dead
      - 1: new latents
      - 1: new hospitalizations
      - 1: new deaths

    - i_sus, sl_lat, sl_sym, sl_hos, sl_sus, i_rec, i_ded,
      i_nwh, i_nwd:
      slices and indices for bin ranges.
    - labels: labels for the bins elements
    """

    def __init__(self, nb_lat, nb_sym, nb_hos, start=(1e7, 100, 1.1)):
        """Initialize.

        start: initial state (nsus, n_infected, growth_factor).
        n_infected will be distributed over 'latent' and 'symptomatic'
        bins.
        """

        s = self

        s.nb_lat = int(nb_lat)
        s.nb_sym = int(nb_sym)
        s.nb_hos = int(nb_hos)
        self.nb = 6 + s.nb_lat + s.nb_sym + s.nb_hos

        class Slicer:
            def __init__(self, i):
                self.i = i
            def next(self, n):
                self.i += n
                sl = slice(self.i-n, self.i)
                return self.i-n, sl

        s.i_sus = 0
        sl = Slicer(1)
        s.i0_lat, s.sl_lat = sl.next(s.nb_lat)
        s.i0_sym, s.sl_sym = sl.next(s.nb_sym)
        s.i0_hos, s.sl_hos = sl.next(s.nb_hos)
        s.i_ded, s.i_rec, s.i_nwl, s.i_nwh, s.i_nwd = np.arange(5)+sl.i

        self.labels = self._get_labels()
        # initialize values
        s.bins = np.zeros(self.nb)
        self.reset(*start)

    def reset(self, npop, ninf, gfac):
        """Reset with population of latent and symptomatic.

        - npop: total population
        - ninf: number of latent+asymptomatic
        - gfac: growth factor (bin to bin).
        """
        s = self
        s.bins[:] = 0
        s.bins[s.sl_lat] = gfac**-np.arange(s.nb_lat)
        s.bins[s.sl_sym] = gfac**-np.arange(s.nb_lat, s.nb_lat+s.nb_sym)

        s.bins *= (ninf / s.bins.sum())
        s.bins[s.i_sus] = npop - sum(self.get_nums())

    def apply_matrix(self, mat):
        """Update self.bins := m @ self.bins"""

        self.bins[:] = mat @ self.bins

    def get_nums(self):
        """Return (nsus, nlat, nsym, nhos, nrec, nded)."""

        nsus = self.bins[self.i_sus]
        nlat = self.bins[self.sl_lat].sum()
        nsym = self.bins[self.sl_sym].sum()
        nhos = self.bins[self.sl_hos].sum()
        nrec = self.bins[self.i_rec]
        nded = self.bins[self.i_ded]
        return (nsus, nlat, nsym, nhos, nrec, nded)

    def get_nums_series(self):
        """Return totals and deltas as Series."""

        (nsus, nlat, nsym, nhos, nrec, nded) = self.get_nums()
        s = pd.Series(dict(
            nsus=nsus, nlat=nlat, nsym=nsym, nhos=nhos, nrec=nrec, nded=nded,
            newL=self.bins[self.i_nwl],
            newH=self.bins[self.i_nwh],
            newD=self.bins[self.i_nwd],
            ))
        return s

    def _get_labels(self):
        labels = [None] * self.nb

        def setlabels(prefix, sl):
            for i in range(sl.start, sl.stop):
                labels[i] = f'{prefix}{i-sl.start}'

        labels[self.i_sus] = 'Sus'
        setlabels('La', self.sl_lat)
        setlabels('Sy', self.sl_sym)
        setlabels('Ho', self.sl_hos)
        labels[self.i_rec] = 'Rec'
        labels[self.i_ded] = 'Ded'
        labels[self.i_nwl] = 'NewL'
        labels[self.i_nwh] = 'NewH'
        labels[self.i_nwd] = 'NewD'
        return labels

    def get_R_from_R0(self, R0):
        """Get effective R from R0, accounting for herd immunity."""
        (nsus, nlat, nsym, nhos, nrec, nded) = self.get_nums()

        # susceptible fraction
        f_sus = nsus / (nsus + nlat + nsym + nhos + nrec)

        return f_sus * R0

    def as_dfrow(self):
        df = pd.DataFrame(self.bins.reshape(1, -1), columns=self.labels)
        return df

    def copy(self):
        return deepcopy(self)

    def __repr__(self):

        (nsus, nlat, nsym, nhos, nrec, nded) = self.get_nums()
        return (f'<{self.__class__.__name__}: sus={nsus:.3g}, '
                f'lat={nlat:.3g}, sym={nsym:.3g}, hos={nhos:.3g}, '
                f'rec={nrec:.3g}, ded={nded:.3g}>')


class EpidemyModel:
    """Simple epidemiological model.

    Configuration attributes:

    - tstep: time step (1.0)
    - t_unit: unit name ('d')
    - R: reproduction number (1.3)
    - ifr: infection fatality rate (0.2e-2)
    - ihr: infection hospitalization rate (0.7e-2)
    - T_lat: latent time (4.0 d)
    - T_i2d: time from infection to death (23 d)
    - T_i2h: time from infection to hospital (11 d)
    - dispersion: relative variation (stdev) in duration of phases.

    Other attributes:

    - estate: EpidemyState instance
    """

    def __init__(self, **kwargs):
        """Init the attributes by keyword."""

        self.tstep = 1.0
        self.t_unit = 'd'
        self.R = 1.3
        self.ifr = 0.2e-2
        self.ihr = 0.7e-2
        self.T_lat = 4.5
        self.T_i2d = 23.0
        self.T_i2h = 11.0
        self.dispersion = 0.2

        for k, v in kwargs.items():
            if hasattr(self, k):
                setattr(self, k, type(getattr(self, k))(v))
            else:
                raise KeyError(k)

        assert self.T_lat < self.T_i2h+self.tstep < self.T_i2d+2*self.tstep

        self.T_sym = self.T_i2h - self.T_lat
        self.T_hos = self.T_i2d - self.T_i2h

        n_lat = int(np.ceil(self.T_lat/self.tstep))
        n_sym = int(np.ceil(self.T_sym/self.tstep))
        n_hos = int(np.ceil(self.T_hos/self.tstep))

        self.estate = EpidemyState(n_lat, n_sym, n_hos)
        self.update_matrix()

    def iterate(self, estate=None):
        """Apply one iteration to self.estate or other EpidemyState."""

        if estate is None:
            self.estate.apply_matrix(self.mat)
        else:
            estate.apply_matrix(self.mat)


    def change_R(self, new_R):
        """Update R value."""

        self.R = float(new_R)

        es = self.estate
        self.mat[es.i_sus, es.i0_sym] = -self.R
        self.mat[es.i0_lat, es.i0_sym] = self.R
        self.mat[es.i_nwl, es.i0_sym] = self.R

    @staticmethod
    def _set_transfers_with_dispersion(matf, pf, n, sigma, inplace=True):
        """Add dispersion on a range of transfers.

        - matf: matrix as dataframe.
        - pf: prefix for index/column labels
        - n: source range [0:n] to be transfered to [1:n+1]
        - sigma: requested resulting dispersion (in units of time step).
        - inplace: whether to update matf in-place.

        Return: updated matf
        """

        if not inplace:
            matf = matf.copy()

        orig_shape = matf.shape

        # transfers with dispersion only from range [i0+1:i1-1]
        if n >= 1:
            matf.at[f'{pf}1', f'{pf}0'] = 1
            matf.at[f'{pf}{n}', f'{pf}{n-1}'] = 1

        m = n - 2 # actual transfers with dispersion
        if m <= 0:
            if sigma != 0 :
                raise ValueError(f'Cannot add sigma effect for n={n}.')
            return

        sigma1 = sigma/np.sqrt(m) # dispersion per time step
        if sigma1 > 0.8:
            raise ValueError(
                f'n={n} allows sigma < {0.8*m**0.5:.2f}, not {sigma:.2f}.')

        a = sigma1**2/2
        kernel = np.array([a, 1-2*a, a])
        for i in range(1, n-1):
            matf.loc[f'{pf}{i}':f'{pf}{i+2}', f'{pf}{i}'] = kernel

        assert matf.shape == orig_shape
        return matf

    def update_matrix(self):
        """Build and update sparse transfer matrix for estate.bins.

        self.mat will be updated. Also returned.
        """

        es = self.estate
        (n,) = es.bins.shape
        self.mat = np.zeros((n, n), dtype=float)
        # as dataframe for easier indexing
        matf = self.mat_as_df()

        ## Transfer from sym to lat
        matf.at['Sus', 'Sus'] = 1

        # Note: R feedback is done in the end.
        ## latent transfers
        # number of latent steps, int and fractional part
        f, nlat = np.modf(self.T_lat/self.tstep)
        nlat = int(nlat)
        assert nlat >= 1
        #for i in range(1, nlat):
        #    matf.at[f'La{i}', f'La{i-1}'] = 1
        sigma = self.T_lat / self.tstep * self.dispersion
        self._set_transfers_with_dispersion(matf, 'La', nlat-1, sigma)

        ## transfer from latent to symptomatic
        matf.at['Sy0', f'La{nlat-1}'] = 1-f
        if f > 0:
            matf.at[f'La{nlat}', f'La{nlat-1}'] = f
            matf.at['Sy0', f'La{nlat}'] = 1

        ## Symptomatic transfers
        f, nsym = np.modf(self.T_sym/self.tstep)
        nsym = int(nsym)
        assert nsym >= 1
        sigma = self.T_sym / self.tstep * self.dispersion
        self._set_transfers_with_dispersion(matf, 'Sy', nsym-1, sigma)
        #for i in range(1, nsym):
        #    matf.at[f'Sy{i}', f'Sy{i-1}'] = 1

        ## transfer from symptomatic to hospital and recovered
        if nsym > 0:
            matf.at['Ho0', f'Sy{nsym-1}'] = (1-f)*self.ihr
            matf.at['NewH', f'Sy{nsym-1}'] = (1-f)*self.ihr
            matf.at['Rec', f'Sy{nsym-1}'] = 1-self.ihr
        if f > 0:
            matf.at[f'Sy{nsym}', f'Sy{nsym-1}'] = f*self.ihr
            matf.at['Ho0', f'Sy{nsym}'] = 1
            matf.at['NewH', f'Sy{nsym}'] = 1

        ## Hospital transfers
        f, nhos = np.modf(self.T_hos/self.tstep)
        nhos = int(nhos)
        assert nhos >= 1
        sigma = self.T_hos / self.tstep * self.dispersion
        self._set_transfers_with_dispersion(matf, 'Ho', nhos-1, sigma)
        #for i in range(1, nhos):
        #    matf.at[f'Ho{i}', f'Ho{i-1}'] = 1

        ## transfer from hospital to dead/recovered
        hdr = self.ifr/self.ihr # hospital death rate
        if nhos > 0:
            matf.at['Ded', f'Ho{nhos-1}'] = (1-f)*hdr
            matf.at['NewD', f'Ho{nhos-1}'] = (1-f)*hdr
            matf.at['Rec', f'Ho{nhos-1}'] = 1 - hdr
        if f > 0:
            matf.at[f'Ho{nhos}', f'Ho{nhos-1}'] = f*hdr
            matf.at['Ded', f'Ho{nhos}'] = 1
            matf.at['NewD', f'Ho{nhos}'] = 1

        ## Ded/Rec entries must be preserved.
        matf.at['Ded', 'Ded'] = 1
        matf.at['Rec', 'Rec'] = 1

        # need to put placehollder values for R before converting to sparse
        matf.at['Sus', 'Sy0'] = -999
        matf.at['La0', 'Sy0'] = -999
        matf.at['NewL', 'Sy0'] = -999

        self.mat = csr_matrix(matf.to_numpy())
        self.change_R(self.R)

        return self.mat

    def mat_as_df(self):
        """Return matrix as DataFrame with column/row labels."""

        labels = self.estate.labels

        mat = self.mat
        if issparse(mat):
            mat = mat.todense()

        assert mat.shape == (len(labels), len(labels))
        df = pd.DataFrame(mat, index=labels, columns=labels)
        return df


def test_EpidemyModel():
    """Test case"""

    em = EpidemyModel(
        R=2, T_lat=2.2, T_i2h=3.5, T_i2d=5.4,
        ihr=0.1, ifr=0.01, dispersion=0.0)

    # print(list(em.estate.labels))
    expected_labels = [
        'Sus', 'La0', 'La1', 'La2', 'Sy0', 'Sy1',
        'Ho0', 'Ho1', 'Ded', 'Rec', 'NewL', 'NewH', 'NewD'
        ]
    assert list(em.estate.labels) == expected_labels
    matf = em.mat_as_df()

    assert list(matf.index) == expected_labels
    assert list(matf.columns) == expected_labels
    expected_matrix = np.array([
       [ 1,  0,  0,  0,  -2,  0,  0,   0,  0,  0,  0,  0,  0 ],
       [ 0,  0,  0,  0,  2,   0,  0,   0,  0,  0,  0,  0,  0 ],
       [ 0,  1,  0,  0,  0,   0,  0,   0,  0,  0,  0,  0,  0 ],
       [ 0,  0,  0.2,0,  0,   0,  0,   0,  0,  0,  0,  0,  0 ],
       [ 0,  0,  0.8,1,  0,   0,  0,   0,  0,  0,  0,  0,  0 ],
       [ 0,  0,  0,  0,  0.03,0,  0,   0,  0,  0,  0,  0,  0 ],
       [ 0,  0,  0,  0,  0.07,1,  0,   0,  0,  0,  0,  0,  0 ],
       [ 0,  0,  0,  0,  0,   0,  0.09,0,  0,  0,  0,  0,  0 ],
       [ 0,  0,  0,  0,  0,   0,  0.01,1,  1,  0,  0,  0,  0 ],
       [ 0,  0,  0,  0,  0.9, 0,  0.9 ,0,  0,  1,  0,  0,  0 ],
       [ 0,  0,  0,  0,  2,   0,  0,   0,  0,  0,  0,  0,  0 ],
       [ 0,  0,  0,  0,  0.07,1,  0,   0,  0,  0,  0,  0,  0 ],
       [ 0,  0,  0,  0,  0,   0,  0.01,1,  0,  0,  0,  0,  0 ]])

    assert np.allclose(matf.to_numpy(), expected_matrix)

    # check preservation of number of people (ignore 'newX' rows/columns)
    submat = matf.loc['Sus':'Rec', 'Sus':'Rec']
    assert np.allclose(submat.sum(axis=0), 1)


def test_EpModel_disp(interactive=False):

    n = 4
    labels = [f'Foo{i}' for i in range(n)]
    matf = pd.DataFrame(np.zeros((n, n)), index=labels, columns=labels)
    EpidemyModel._set_transfers_with_dispersion(
        matf, 'Foo', 3, 0.0)
    if interactive:
        print('First matrix:\n{repr(matf.to_numpy())}')
    expected_mat = np.array([
        [0., 0., 0., 0.],
        [1., 0., 0., 0.],
        [0., 1., 0., 0.],
        [0., 0., 1., 0.]])

    assert np.allclose(matf.to_numpy(), expected_mat)

    EpidemyModel._set_transfers_with_dispersion(matf, 'Foo', 3, 0.2)
    if interactive:
        print('Second matrix:\n{repr(matf.to_numpy())}')
    expected_mat = np.array([
        [0., 0.,   0., 0.],
        [1., 0.02, 0., 0.],
        [0., 0.96, 0., 0.],
        [0., 0.02, 1., 0.]])
    assert np.allclose(matf.to_numpy(), expected_mat)

    n = 6
    labels = [f'Foo{i}' for i in range(n)]
    matf = pd.DataFrame(np.zeros((n, n)), index=labels, columns=labels)
    EpidemyModel._set_transfers_with_dispersion(matf, 'Foo', 5, 1)
    if interactive:
        print('Third matrix:\n{repr(matf.to_numpy())}')

def set_pandas_format():
    """Allow wide tables."""

    pd.options.display.float_format = '{:.3g}'.format
    pd.options.display.max_columns = 100
    pd.options.display.width = 200

def estimate_T2(emodel, states, verbose=True):
    """Return doubling time from states list (based on 1st and last entry)."""

    gfac = states[-1]['newL']/states[0]['newL']
    T2 = np.log(2)/np.log(gfac)*emodel.tstep/(len(states)-1)

    if verbose:
        print(f'Doubling time T2={T2:.2f}. Rt={emodel.R:.2f}')

    return T2



def run_simulation(
        T_lat=3.5, T_mild=7.5, T_hos=11, ihr=0.7e-2, ifr=0.2e-2,
        RTa=(1.3, 30), RTb=(0.9, 60), init=(30e3, 25),
        dispersion=0.15,
        fname=None
        ):
    """Simulation run.

    Parameters:

    - T_lat: latent time (d)
    - T_mild: duration of mild-symptoms state (d)
    - T_hos: duration of hospital stay (d)
    - ihr: infection hospitalization rate
    - ifr: infection fatality rate
    - RTa: R-number and duration stage a
    - RTb: R-number and duration stage b
    - init: initialize with (N, T), aim for N latent/mild cases at start;
      allow duration T for simulation to settle.
    - dispersion: relative variation of durations T_lat/mild/hos.
    - fname: optional filename to save data.
    """

    em = EpidemyModel(
        T_lat=T_lat, T_i2h=T_lat+T_mild, T_i2d=T_lat+T_mild+T_hos,
        ihr=ihr, ifr=ifr, R=RTa[0], dispersion=dispersion,
        tstep=0.5
        )

    estate = em.estate.copy()

    # initialize and run to stabilize
    gfac = em.R**(em.tstep/em.T_lat)
    estate.reset(16e6, init[0]*gfac**(-init[1]/em.tstep), gfac)
    for i in range(int(init[1]/em.tstep + 0.5)):
        em.iterate(estate)
        em.change_R(estate.get_R_from_R0(RTa[0]))

    # Here the real simulation starts.
    states = [estate.get_nums_series()]
    Rvals = [em.R]
    for i in range(int(RTa[1]/em.tstep+0.5)):
        em.iterate(estate)
        em.change_R(estate.get_R_from_R0(RTa[0]))
        states.append(estate.get_nums_series())
        Rvals.append(em.R)

    # Verify exponential growth factor
    T2a = estimate_T2(em, states[-2:])

    em.change_R(RTb[0])
    for i in range(int(RTb[1]/em.tstep+0.5)):
        em.iterate(estate)
        em.change_R(estate.get_R_from_R0(RTb[0]))
        states.append(estate.get_nums_series())
        Rvals.append(em.R)
    # Verify exponential growth factor
    T2b = estimate_T2(em, states[-2:])

    sim_df = pd.DataFrame(states)
    sim_df.set_index(np.arange(len(sim_df))*0.5, inplace=True)

    # Check conservation of people:
    for i in [0, -1]:
        npop = sim_df.iloc[i][['nsus', 'nlat', 'nsym', 'nhos', 'nded', 'nrec']].sum()
        print(f'npop={npop:.5g}')

    ### plotting
    cyc_lines = itertools.cycle(["-", "--", "-.", ":"])

    fig, (ax0, ax1) = plt.subplots(
        2, 1, figsize=(8, 7), tight_layout=True,
        gridspec_kw={'height_ratios': [3, 1]}, sharex=True
        )

    ax1.set_xlabel('Time (days)')
    ax0.set_ylabel('Daily numbers')
    ax0.set_xlim(sim_df.index[0], sim_df.index[-1])

    nums_infected = sim_df['nlat'] + sim_df['nsym'] + sim_df['nhos']
    ax0.semilogy(nums_infected, linestyle=next(cyc_lines), label='Active cases')
    ax0.semilogy(sim_df['newL'], linestyle=next(cyc_lines), label='New infections')
    ax0.semilogy(sim_df['nded'], linestyle=next(cyc_lines), label='Cumulative deaths')
    ax0.semilogy(sim_df['newD'], linestyle=next(cyc_lines), label='New deaths')
    ax0.semilogy(sim_df['newH'], linestyle=next(cyc_lines), label='New hospitalizations')

    nded_final = sim_df['nded'].iat[-1]
    ax0.text(sim_df.index[-1], nded_final, f'{nded_final/1000:.3g}k',
             horizontalalignment='right', verticalalignment='top'
            )

    ax0.axvline(RTa[1])

    ax0.grid()
    ax0.legend()

    # bottom panel
    ax1.plot(sim_df.index, Rvals, label='$R_t$')

    f_sus = sim_df['nsus']/(sim_df[['nsus', 'nlat', 'nsym', 'nhos', 'nded', 'nrec']].sum(axis=1))
    ax1.plot(1-f_sus, linestyle='--', label='Herd immunity level')

    ax1.grid()
    ax1.legend()



    info = (
        f'SEIR model, T_latent={em.T_lat:.3g}, T_mild={em.T_sym:.3g}, '
        f'T_hospital={em.T_hos:.3g}, '
        f'IHR={em.ihr*100:.3g}%, IFR={em.ifr*100:.3g}%\n'
        f'$R_1$={RTa[0]}, {RTb[0]} up to/from t={RTa[1]}. '
        f'Doubling time: {T2a:.3g}, {T2b:.3g} before/after'
        )
    ax0.set_title(info)

    fig.text(0.99, 0.02, '@hk_nien (DRAFT)', horizontalalignment='right')
    fig.show()

    if fname:
        fig.savefig(fname)

if __name__ == '__main__':
    set_pandas_format()
    plt.close('all')
    test_EpidemyModel()
    test_EpModel_disp()
    run_simulation(RTa=(1.3, 30), RTb=(0.9, 120), fname='seir0.png')
    run_simulation(RTa=(1.3, 37), RTb=(0.9, 113), fname='seir1.png')
    run_simulation(RTa=(1.3, 150), RTb=(0.9, 0), fname='seir2.png')
