#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 29 15:21:53 2025

@author: dx1
"""

import sys
sys.path.append('../state-space')
import numpy as np
import orientation_dev as ordev
import master_wass_metric_binary as mwmb
from scipy import optimize
import matplotlib.pyplot as plt
import window_sampling_algorithm as wsa
import orientation_metric_tests as omt
rng = np.random.default_rng()

a = omt.create_quaternion(rng.integers(0,100000000, size =(100,100)))
uniqa = np.unique(a.reshape(-1,4), axis=0)
n = np.shape(uniqa)
dq = np.zeros(n[0])
for i in range(n[0]):
    dq[i] = ordev.misorientation_angle(uniqa[0],uniqa[i])
    
dqsort = np.copy(dq)
dqsort = np.sort(dqsort)
uniqasort = np.copy(uniqa)

for i in range(n[0]):
    uniqasort[i] = uniqa[np.where(dq == dqsort[i])[0][0]]
    
plt.scatter(np.arange(n[0])/n[0], dqsort, s = .1)
plt.ylabel('Rotation angle')
plt.xlabel('Rotation value')
plt.show()

