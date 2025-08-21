#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  3 08:20:24 2025

@author: dx1
"""

import numpy as np
import sys
sys.path.append('/Users/dx1/Research/new_metric_code/')
sys.path.append('/Users/dx1/Research/microstructure_data/')

import matplotlib.pyplot as plt
import double_lattice_gbs as dlgb
#import ctf_to_grid as ctf2g

rng = np.random.default_rng()

fp = '/Users/dx1/Research/microstructure_data/gerrys_set/'
fp1 = '/Users/dx1/Research/microstructure-metric/AM-microstructures/'

#files = [fp+'p22_ebsd.npy', fp1+'euler_p22.npy', fp1+'r2_p22_euler_grid.npy', 
#         fp1+'r2_p22_euler_grid.npy']

#data_grid = np.load(files[2])

#data_grid = np.reshape(data_grid, (768,1024,8))
#plt.figure()
#plt.imshow(data_grid[:,:,5:])
#plt.show()


def replace_nulls(data):
    data_grid = data
    m0,m1,m2 = np.shape(data_grid)
    
    ind = np.where(data_grid[:,:,:3] == [0,0,0])
    n = np.shape(ind)[1]
    neighborlist = np.array(([[-1,1], [0,1], [1,1], 
                           [-1,0], [1,0],
                           [-1,-1], [-1,0], [-1,1]]))
    
    #dgcp = np.copy(data_grid)
    for i in range(n):
        neighbors = np.zeros((8,m2))
        for k in range(8):
            a = (ind[0][i] + neighborlist[k,0])%m0
            b = (ind[1][i] + neighborlist[k,1])%m1
            neighbors[k] = data_grid[a, b]
        val = neighbors[rng.integers(0,8)]
        data_grid[ind[0][i], ind[1][i]] = val
       # data_grid[ind[0][i], ind[1][i], :3] = [1,1,1]

            
    plt.figure()
    plt.imshow(data_grid[:,:,0])
    plt.show()
    return data_grid

def to_gray(data):
    dgcp = np.zeros((768,1024))
    dgcp = 0.299 * data_grid[:,:,5] + 0.587 * data_grid[:,:,6] + 0.114 * data_grid[:,:,7]
    plt.figure()
    plt.imshow(dgcp)
    plt.show()
 
    return dgcp
    

def mono_grains(data, thresh):
    data_cp = np.copy(data)
    for i in range(1,767):
        for j in range(1,1023):
            if abs(data[i,j] - data[i+1,j]) < thresh*data[i,j]:
                data_cp[i+1,j] = data_cp[i,j]
                
            if abs(data[i,j] - data[i,j+1]) < thresh*data[i,j]:
                data_cp[i,j+1] = data_cp[i,j]
                
            if abs(data[i,j] - data[i+1,j+1]) < thresh*data[i,j]:
                data_cp[i+1,j+1] = data_cp[i,j]   
                
            if abs(data[i,j] - data[i-1,j]) < thresh*data[i,j]:
                data_cp[i-1,j] = data_cp[i,j]
                
            if abs(data[i,j] - data[i,j-1]) < thresh*data[i,j]:
                data_cp[i,j-1] = data_cp[i,j]
                
            if abs(data[i,j] - data[i+1,j-1]) < thresh*data[i,j]:
                data_cp[i+1,j-1] = data_cp[i,j]   
            
            if abs(data[i,j] - data[i-1,j-1]) < thresh*data[i,j]:
                data_cp[i-1,j-1] = data_cp[i,j]
            
            if abs(data[i,j] - data[i-1,j+1]) < thresh*data[i,j]:
                data_cp[i-1,j+1] = data_cp[i,j]
    return data_cp


def double_lattice_gbs(layer):
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
                
    return gb_dbl

def down_sample(data, factor, quats = False):
    l = np.shape(data)
    m,n = int(l[0]/factor),int(l[1]/factor)
    ds_f = np.zeros((m,n))
    s = 1
    if quats:
        ds_f = np.zeros((m,n,4))
        s = 4
    for i in range(m):
        for j in range(n):
            ds_f[i,j] = rng.choice(np.reshape(data[i*factor:i*factor+factor, j*factor:j*factor+factor], (1,factor**2,s))[0])
    
    return ds_f
    
           
def euler_to_quat(angles):
    p,O,y = angles
    cos = np.cos
    sin = np.sin
    cp = cos(p/2)
    cO = cos(O/2)
    cy = cos(y/2)
    sp = sin(p/2)
    sO = sin(O/2)
    sy = sin(y/2)
    q0 = cp*cO*cy + sp*sO*sy
    q1 = sp*cO*cy - cp*sO*sy
    q2 = cp*sO*cy + sp*cO*sy
    q3 = cp*cO*sy - sp*sO*cy
    q = np.array([q0,q1,q2,q3])
    return q    

def upsample(data, f, gids = False, quats = False):
    m,n = np.shape(data)[:2]
    o = 1
    if gids: o = 3
    if quats: o = 4
    up_data = np.zeros((m*f,n*f,o))
    
    for i in range(m):
        for j in range(n):
            up_data[i*f:(i+1)*f, j*f:(j+1)*f] = data[i,j]
            
    return up_data


r1_p63_sims = []
for i in range(10):
    r1_p63_sims.append(down_sample(np.load('../orientation-metric/r1_p63_sim_qgrid_' + str(i*50) + '.npy'),5,True))

r1_p63_sim = np.zeros((10,80,80,4))
for i in range(10):
    r1_p63_sim[i] = r1_p63_sims[i]
np.save('../orientation-metric/r1_p63_10slices_ds.npy',r1_p63_sim)
    
#while np.shape(np.where(data_grid[:,:,:3] == [0,0,0]))[1]:
#    data_grid = replace_nulls(data_grid)
        
# q_grid = np.load(fp1 + 'r1_p61_quaternion.npy')
# plt.imshow(q_grid[:100,:100,:3])
# plt.imshow(abs(q_grid[:,:,:3]))
# plt.title('Original resolution ebsd from r1 p22')
# plt.show()

downsample = False
if downsample:
    for j in range(5,6):
        dsx = down_sample(q_grid, j)
        plt.imshow(abs(dsx[:,:,:3]))
        plt.title('downsampled r1 p22 by factor of: '+str(j))
        plt.show()
        np.save(fp + '../../microstructure-metric/orientation-metric/ds'+str(j)+'x.npy', dsx)
       
        scale = int(100/j)
        ws = np.zeros((10,35,35,4))
        for i in range(10):
            a = ind[0,i]
            b = ind[1,i]
            ws[i] = windowB[a:a+35, b:b+35]
        
        for i in range(10):
            plt.figure()
            plt.title('Window '+ str(i) +' from ds'+str(j)+'x')
            plt.imshow(abs(ws[i,:,:,:3]))
            #plt.title('Window from ds'+str(j)+'x')
    
            #plt.imshow(abs(dsx[:scale,:scale,:3]))
            
        plt.show()
    
save_data = False
if save_data:
    eulers = np.copy(data_grid[:,:,:3])
    n = np.shape(eulers)
    q_grid = np.zeros((n[0],n[1],4))
    
    for i in range(n[0]):
        for j in range(n[1]):
            q_grid[i,j] = euler_to_quat(eulers[i,j])
    
    plt.imshow(abs(q_grid[:,:,:3]))
    plt.show()
    #np.save(fp1+'r1_p61_quaternion.npy', q_grid)


if False:
    np.save(fp+'p22_ebsd_cleaned.npy',data_grid)
    
    data_gray = to_gray(data_grid)
    gbs = double_lattice_gbs((data_gray[:100,:100]*10).astype(int))
    plt.figure()
    plt.imshow(gbs[:200,:200])
    plt.show()
    
    ds_2x = down_sample(data_gray, 2)
    plt.figure()
    plt.imshow(ds_2x)
    plt.show()
    
    gbs = double_lattice_gbs((ds_2x[:50,:50]*10).astype(int))
    plt.figure()
    plt.imshow(gbs[:100,:100])
    plt.show()
    
    data_mono = mono_grains(data_gray, 0.01)*10
    
    plt.figure()
    plt.imshow(data_mono)
    plt.show()
    
    gbs1 = dlgb.double_lattice_gbs(data_mono[:100,:100].astype(int))[1]
    plt.figure()
    plt.imshow(gbs1[:200,:200])
    plt.show()
    

