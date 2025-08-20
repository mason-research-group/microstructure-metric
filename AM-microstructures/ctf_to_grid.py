#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 12 14:43:56 2025

@author: dx1
""" 

import re
import numpy as np

def read_ctf(file):
    data = []
    with open(file) as f:
        for x in f:
            data.append(re.findall(r"[-+]?(?:\d*\.*\d+)", x))
    return data

def assign_to_grid(data):
    data_np = np.zeros((len(data)-8, 11))
    for i in range(np.shape(data_np)[0]):
        for j in range(11):
            data_np[i,j] = data[i+8][j]
            
    dg = np.zeros((569,854,3))
    for i in range(np.shape(data_np)[0]-1):
        a = data_np[i,1]
        b = data_np[i,2]
        dg[int(a/4), int(b/2)] = data_np[i,5:8]
    return dg

if __name__ == '__main__':
    dL = '/Users/dx1/Downloads/Spatial Variation Baseline EBSD/'
    file = dL + 'P61/SS316 Zeiss specimen Sample 61 Area 2 Site 1 Map Data 4.ctf'
    #file = dL + 'P22/site 7 - ebsd map data.ctf'
    data = read_ctf(file)
    data_grid = assign_to_grid(data)
    np.save('r1_p61_euler_grid.npy', data_grid)