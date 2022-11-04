# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 18:40:01 2020

@author: pc
"""

import numpy as np
import support_based as spb
import support_read_data as srd
import random
from scipy import interpolate

class dataset:
    def __init__(self, bat_size, sam_n, da_ty):
        # read data
        self.tr_sin, self.tr_mas, self.te_sin, self.te_mas = srd.div_dataset(da_ty)

        # set parameters
        self.da_ty = da_ty
        self.bat_size = bat_size
        self.sam_n = sam_n
        self.sam_len = sam_n*370 - (sam_n*370)%8
        self.aug_rate = np.array([0.40863,0.65809,0.44890,0.51419,0.28135,0.51943,0.62041,0.93148,0.87023,0.90560,0.94062,0.87699])

    # get training batch data
    def get_tr_bat_sigma(self):
        # sample ECG components
        bat_sig, bat_mas = [], []
        for i in range(self.bat_size):
            sig, mas = [], []
            for j in range(self.sam_n):
                ind = int(random.random()*len(self.tr_mas))
                sig.append(self.tr_sin[ind])
                mas.append(self.tr_mas[ind])
            sig = np.concatenate(sig, 0)
            mas = np.concatenate(mas, 0)

            # data augmentation
            sig = sig * self.aug_rate * (1 + np.abs(np.random.normal(0))/4)

            # B-spline interpolation
            sig = np.stack([spb.Bspline(sig[:, x], self.sam_len) for x in range(sig.shape[1])], -1)
            mas = spb.Bspline(mas, self.sam_len)
            mas += 0.3
            mas = mas.astype(np.int16)

            bat_sig.append(sig)
            bat_mas.append(mas)

        return np.stack(bat_sig, 0), np.stack(bat_mas, 0)

    # get test batch data
    def get_te_bat_sigma(self):
        bat_sig = np.concatenate(self.te_sin, 0)
        bat_mas = np.concatenate(self.te_mas, 0)
        num = bat_sig.shape[0] - bat_sig.shape[0]%8
        return bat_sig[np.newaxis, :num, :], bat_mas[np.newaxis, :num]
