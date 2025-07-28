#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 21 10:25:12 2025

@author: dx1
"""

import sys
#sys.path.append('/Users/dx1/Research/new_metric_code/')
#sys.path.append('/Users/dx1/Research/microstructure_data/')

import numpy as np
import matplotlib.pyplot as plt


def double_lattice_gbs(layer, plotting = False, verbose = False):
    m,n = np.shape(layer)
    layer_double = np.zeros((m*2, n*2))
    
    for i in range(m):
        for j in range(n):
            layer_double[i*2:(i+1)*2, j*2:(j+1)*2] = layer[i,j]
    
    gb_dbl = np.zeros(np.shape(layer_double))
    
    for i in range(m*2-1):
        for j in range(n*2-1):
            a = layer_double[i:i+2, j:j+2]
            if (a[0] != a[1]).all():
                gb_dbl[i, j] = 1
            if (a[:,0] != a[:,1]).all():
                gb_dbl[i, j] = 1

    
    gbs = np.zeros((m,n))
    for i in range(m):
        for j in range(n):
            gbs[i,j] = np.max(gb_dbl[i*2:(i+1)*2, j*2:(j+1)*2])
    
    if plotting:
        plt.figure() 
        plt.imshow(gb_dbl)
        plt.figure() 
        plt.imshow(gbs)

    if verbose:
        print('GB per px area:' , np.average(gb_dbl))
        print('GB per px area:' , np.average(gbs))

    return gbs, gb_dbl

if __name__ == '__main__' :
    fp = '/Users/dx1/Research/microstructure_data/'    
    slices = np.load(fp + 'r1_P22_gids.npy')
    layer = slices[20]
    plt.imshow(layer)
    gbs = double_lattice_gbs(layer, True)
