#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 25 16:35:43 2025

@author: dx1
"""
import sys
sys.path.append('../state-space')
import numpy as np
import orientation_dev as ordev
import master_wass_metric_binary as mwmb
from scipy import optimize
import matplotlib.pyplot as plt

rng = np.random.default_rng()

def create_quaternion(sl):
    m = np.shape(np.unique(sl))[0]
    qs = rng.random(size=(m,4))
    for i in range(m):
        qs[i] = qs[i] / np.linalg.norm(qs[i]) 
    return qs

def assign_quats(window, quats):
    n = np.shape(window)
    window_quats = np.zeros((n[0],n[0],4))
    c = 0    
    for FID in np.unique(window):
        window_quats[np.where(window == FID)] = quats[rng.integers(3)]
        c+=1
    
    return window_quats
        
fp = '/Users/dx1/Research/microstructure_data/TestMicrostructures/micro_'
slicesA = np.load(fp+'A.npy')
slicesB = np.load(fp+'B.npy')

slA_0 = np.copy(slicesA[0]) 
slA_100 = np.copy(slicesA[100])

new_quats = False
if new_quats:
    quats0 = create_quaternion(slA_0)
    quats100 = create_quaternion(slA_100)
    
    windowA = assign_quats(slA_0, quats0)
    windowB = assign_quats(slA_100, quats100)
    np.save('./test_windowA.npy', windowA)
    np.save('./test_windowB.npy', windowB)

windowA = np.load('./test_windowA.npy')
windowB = np.load('./test_windowB.npy')

striped = False
if striped:
    uniqa = np.unique(windowA.reshape(-1,4), axis=0)
    windowA[:,:] = uniqa[0]
    windowA[0::10] = uniqa[1]

l = 20
tests = 10
w1 = windowA[:l,:l]
avgdists = np.zeros(10)
for j in range(10):
    w1 = ordev.gen_textured_window(l, l)
    w2 = np.zeros((tests,l,l,4))
    
    iwB = mwmb.intrawindow_block(l)
    dists = np.zeros(tests)
    
    #print(ordev.misorientation_dict(windowA, windowA))
    
    for i in range(tests):
        w2[i] = windowA[int((i*2./l)*l):int((i*2./l+1)*l),:l]
    
    for i in range(tests):
        misdict = ordev.misorientation_dict(w1, w2[i])
        G = ordev.texture_graph(w1, w2[i], plotting = False)
        
        Graph = np.copy(iwB)
        Graph[:-1, :-1] += G*(j+1)
        Graph = ordev.full_bipartite(Graph)
        
        r,c = optimize.linear_sum_assignment(Graph)
        dists[i] = Graph[r,c].sum()/l**2
        plt.imshow(w2[i,:,:,0]-w1[:,:,0], vmax=1)
        plt.show()
    avgdists[j] = np.average(dists)
# dists[i] = dist
plt.scatter(np.arange(10), avgdists)
plt.show()