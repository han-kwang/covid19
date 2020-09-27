#!/usr/bin/env python3

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from seirmodel import run_simulation
from test_seirmodel import test_EpidemyModel, test_EpModel_disp

def set_pandas_format():
    """Allow wide tables."""

    pd.options.display.float_format = '{:.3g}'.format
    pd.options.display.max_columns = 100
    pd.options.display.width = 200

if __name__ == '__main__':
    set_pandas_format()
    plt.close('all')
    test_EpidemyModel()
    test_EpModel_disp()
    run_simulation(RTa=(1.3, 30), RTb=(0.9, 120), fname='seir0.png')
    run_simulation(RTa=(1.3, 37), RTb=(0.9, 113), fname='seir1.png')
    run_simulation(RTa=(1.3, 150), RTb=(0.9, 0), fname='seir2.png')
