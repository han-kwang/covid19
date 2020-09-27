"""Test cases for code validation."""

import numpy as np
import pandas as pd

from seirmodel import EpidemyModel


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


if __name__ == '__main__':

    test_EpModel_disp()
    test_EpidemyModel()
