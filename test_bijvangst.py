#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 22 14:40:23 2021

@author: @hk_nien
"""

import pandas as pd


def tbl_row(c, *, a=0.2, n=3e6, v=0.01):

    t = n*(v + c*(1-a) - v*c*(1-a))
    p = n*(c*(1-v)*(1-a) + v*c)

    fpos = p/t
    print(f'{v:.3g}\t& {c:.3g}\t& {a:.3g}\t& {t/1e3:.1f}\t& {p/1e3:.2f}\t& {fpos*100:.1f} \\\\')



print('v\t& c\t& a\t& t\t& p\t& fpos')
tbl_row(2e-3, a=0.2, v=0)
tbl_row(2e-3, a=0.2)
tbl_row(2e-3, a=0.5, v=0)
tbl_row(2e-3, a=0.5)


