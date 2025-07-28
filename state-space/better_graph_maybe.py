#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 17 16:23:26 2025

@author: dx1
"""

import sys
#sys.path.append('../AM')
import double_lattice_gbs as dlgb
import imageio
import numpy as np


def make_custom_iwB(a,b):
    ''' window - window block '''
    n = np.shape(a)[0]
    a_b = a - b
    indA = np.where(a_b == 1)
    aij = np.shape(indA)[1]
    
    indB = np.where(a_b == -1)
    bij = np.shape(indB)[1]
    
    dAB = np.zeros((aij,bij))
    for i in range(aij):
        for j in range(bij):
            dAB[i,j] = abs(indA[0][i] - indB[0][j]) + abs(indA[1][i] - indB[1][j])
    
    ''' dummy matrix '''
    D = np.ones((n,n))
    o = int(np.ceil(n/2))
    
    for i in range(1,o):
        D[i:n-i,i:n-i] += 1
        
    for i in range(1,o):
        D[n-i-1] = D[i]
    
    D = D.flatten()
    Da = np.copy(D)[n*indA[0] + indA[1]]
    Db = np.copy(D)[n*indB[0] + indB[1]]
    nB = np.shape(dAB)
    for i in range(nB[0]):
        for j in range(nB[1]):
            if dAB[i,j] > Da[i] + Db[j]:
                dAB[i,j] = 2*o
    
    ''' combined matrix, full bipartite graph '''
    bipartite = np.zeros((aij+bij,aij+bij))
    bipartite[:aij, :bij] = dAB
    bipartite[:aij, bij:] += np.reshape(Da, (aij,1))
    bipartite[aij:, :bij] += Db
    
    return bipartite