#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Get DoW corrections for daily case numbers.

Created on Tue Dec 29 20:00:21 2020

@hk_nien
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import nlcovidstats as nlcs



if __name__ == '__main__':
    nlcs.init_data()
    plt.close('all')
    get_dow_correction((-100, -1), True)




