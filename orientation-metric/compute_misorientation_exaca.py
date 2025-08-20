#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  6 14:15:43 2025

@author: dx1
"""

import sys
sys.path.append('../state-space')
import numpy                     as np
import orientation_dev           as ordev
import master_wass_metric_binary as mwmb
from scipy import optimize
import matplotlib.pyplot         as plt
import window_sampling_algorithm as wsa
import orientation_metric_tests  as omt
import csv 
import re

rng = np.random.default_rng()

# all_quats = np.zeros((10000,4))
# with open('GrainOrientationQuaternions.csv', newline='') as csvfile:
#     reader = csv.DictReader(csvfile)
#     c = 0
#     for row in reader:
#         all_quats[c,0] = float(row['10000'])
#         for i in range(3):
#             all_quats[c,i+1] = float(row[None][i])
#         c+=1
# np.save('GrainOrientationQuaternions.npy', all_quats)

A = np.load('GrainOrientationQuaternions.npy')[:30]
import time
s = time.time()
dict_quats = ordev.misorientation_dict(A, A)
print(time.time() - s)