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
import window_sampling_algorithm as wsa
import re

rng = np.random.default_rng()

def create_quaternion(sl):
    ### creates a set of randomly generated unit quaternions equal to ###
    ### the number of unique grain ids within a slice/window          ###
    m = np.shape(np.unique(sl))[0]
    qs = rng.normal(size = (m,4))
    for i in range(m):
        qs[i] = qs[i] / np.linalg.norm(qs[i]) 
    return qs

def assign_quats(window, quats, strength):
    ### assigns quaternions to a window containing grain ids (FIDS)     ###
    ### from a list of quaternions. The input texture strength controls ###
    ### the number of possible orientations that will be assigned       ###
    m,n = np.shape(window)
    nq = np.shape(quats)[0]
    s = np.max((int(nq * (1-strength)), 1))
    window_quats = np.zeros((m,n,4))
    c = 0    
    for FID in np.unique(window):
        window_quats[np.where(window == FID)] = quats[rng.integers(s)]
        c+=1
    
    return window_quats
        
def compare(w1, w2, iwB, l, beta = 5):
    ### computes the orientation distance between two windows, w1 and w2 ###
    ### iwB is the transportation graph. l is window size, beta is the   ###
    ### tuning parameter that determines the strength of reorientation   ###
    ### this relies on the orientation library function to create the    ###
    ### graph for orientation and solves the matching problem through    ###
    ### use of the scipy assignment problem implementation               ###    
    G = ordev.texture_graph(w1, w2, plotting = False)
    Graph = np.copy(iwB)
    Graph[:-1, :-1] += G*beta
    Graph = ordev.full_bipartite(Graph)
    r,c = optimize.linear_sum_assignment(Graph)
    d = Graph[r,c].sum()/l**2
    return d, Graph,r,c
    
def sweep(W1, W2, beta):
    ### takes two sets of windows and creates a pairwise distance matrix ###
    ### calling the function 'compare'. degenerate comparisons can be    ###
    ### bypassed if W1 and W2 are the same by the 'start' switch         ###
    wn = np.shape(W1)[0]
    dmat = np.zeros((wn,wn))
    start=0
    if (W1 == W2).all():
        start = 1
    for i in range(wn):
        w1 = W1[i]
        for j in range((i+1)*start,wn):
            w2 = W2[j]
            dmat[i,j] = compare(w1,w2,iwB,l,beta=beta)
    return dmat
          

def euler_to_quat(angles):
    ### converts a set of euler angles to a unit quaternion, assuming ZYX ###
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
 
def eu2qu(eu):
    ### Bunge-Euler angles to quaternion ###
    ### taken from https://github.com/marcdegraef/3Drotations/blob/master/src/python/rotations.py ### 
    ### this should be the standard implementation since Bunge convention is       ###
    ### most commonly used. Other implementations can be found on degraefs github. ###
    P = -1
    ee = 0.5*eu
    cPhi = np.cos(ee[...,1:2])
    sPhi = np.sin(ee[...,1:2])
    qu = np.block([    cPhi*np.cos(ee[...,0:1]+ee[...,2:3]),
                   -P*sPhi*np.cos(ee[...,0:1]-ee[...,2:3]),
                   -P*sPhi*np.sin(ee[...,0:1]-ee[...,2:3]),
                   -P*cPhi*np.sin(ee[...,0:1]+ee[...,2:3])])
    qu[qu[...,0]<0.0]*=-1
    return qu


def exa_to_quat(data, savepath = '', save = False, gid = False):
    ### creates a quaternion representation of ExaCA-format data this ###
    ### works either in the case that the input window has  grain ids ###
    ### (set gids = True) or has euler angles in bunge convention     ### 
    ### the data can be saved to savepath if the window at hand will  ###
    ### be used again in the future. This requires the file           ###
    ### GrainOrientationQuaternions.npy, which contains the quat rep  ###
    ref = np.load('./GrainOrientationQuaternions.npy')
    m,n = np.shape(data)[:2]
    q_grid = np.zeros((m,n,4))
    if gid:
        data = data%10000
        uniqa = np.unique(data)
        for i in range(np.shape(uniqa)[0]):
            q_grid[np.where(data == uniqa[i])] = ref[int(uniqa[i])]
    else: 
        uniqa = np.unique(data.reshape(-1,3), axis=0)
        for i in range(np.shape(uniqa)[0]):
            q_grid[np.where(data == uniqa[i])[:2]] = eu2qu(uniqa[i])
            
    if save:
        np.save(savepath, q_grid)
        
    return q_grid


def upsample(data, f, euls = False, quats = False):
    ### assign the values in the array 'data' to a window which is a factor ###
    ### f times larger than the array. assumes FID/gid representation.      ### 
    ### you can specify euler angle or quaternion representation though.    ###
    m,n = np.shape(data)[:2]
    o = 1
    if euls: o = 3
    if quats: o = 4
    up_data = np.zeros((m*f,n*f,o))
    
    for i in range(m):
        for j in range(n):
            up_data[i*f:(i+1)*f, j*f:(j+1)*f] = data[i,j]
            
    return up_data


def down_sample(data, f, quats = False):
    ### down samples from a given array by a factor of 'f'                 ###
    ### does so by randomly choosing a value within each block of data of  ###
    ### size f and assigns this to the new grid, 'ds_f'                    ###
    l = np.shape(data)
    m,n = int(l[0]/f),int(l[1]/f)
    ds_f = np.zeros((m,n))
    s = 1
    if quats:
        ds_f = np.zeros((m,n,4))
        s = 4
    for i in range(m):
        for j in range(n):
            ds_f[i,j] = rng.choice(np.reshape(data[i*f:i*f+f, j*f:j*f+f], (1,f**2,s))[0])
    
    return ds_f

if __name__ == '__main__': 
    ####################################################################
    ### a bunch of tests created during the development of this code ###
    ####################################################################

    q22 = np.load('qgrid_sample22.npy')
    q61 = np.load('qgrid_sample61.npy')
    q63 = np.load('qgrid_sample63.npy')
    
    q22x8 = upsample(q22, 8, quats=True)
    samples = [q22x8, q61, q63]
    
    
    for i in range(len(samples)):
        if i == 0:
            f = [2,3,3]
            for j in f:
                samples[i] = down_sample(samples[i], j, quats=True)
            np.save('dsp22.npy', samples[i])
        if i == 1:
            f = [2,2]
            for j in f:
                samples[i] = down_sample(samples[i], j, quats=True)
            np.save('dsp61.npy', samples[i])
        if i == 2:
            f = [2,3]
            for j in f:
                samples[i] = down_sample(samples[i], j, quats=True)
            np.save('dsp63.npy', samples[i])
    
        
                
                
    convert_eu2qu = False
    if convert_eu2qu:
        # datafile  ='/Users/dx1/Downloads/Sample61.txt'
        # count = 0
        # c = np.zeros((786432,3))
        # with open(datafile) as f:
        #     for i in f:
        #         c[count] = (re.findall(r"[-+]?(?:\d*\.*\d+)", i)[:3])
        #         count+=1
        # c = c.reshape((768,1024,3))
    
        c = np.load('./site7eulers.npy')
        qgrid_sample22 = exa_to_quat(c)
        np.save('qgrid_sample22.npy', qgrid_sample22)
    
    
    running_comparisons = False
    if running_comparisons:
        fp = '/Users/dx1/Research/microstructure_data/TestMicrostructures/micro_'
        fp1 = '/Users/dx1/Research/microstructure_data/gerrys_set/'
        slicesA = np.load(fp+'A.npy')
        slicesB = np.load(fp+'B.npy')
        
        slA_0 = np.copy(slicesA[0]) 
        slA_10 = np.copy(slicesA[10]) 
        slA_20 = np.copy(slicesA[20])
        slA_50 = np.copy(slicesA[50])
        
        l = 30
        median = 1.393
        alpha = 1
        beta  = 7.5 
        
        iwB = alpha * mwmb.intrawindow_block(l) 
        # iwB[l**2:,:l**2] += beta * median # np.pi #
        # iwB[:l**2,l**2:] += beta * median # np.pi #
    
    new_quats = False
    if new_quats:
        quats0 = create_quaternion(slA_0)
        #quats10 = create_quaternion(slA_10)
        quats50 = create_quaternion(slA_50)
        
        windowA = assign_quats(slA_0, quats0, strength = 0.99)
        windowB = assign_quats(slA_50, quats50, strength = 0.99)
        #np.save('./test_windowA.npy', windowA)
        #np.save('./test_windowB.npy', windowB)
        np.save('./A_sl0_gaussian.npy',windowA)
        np.save('./A_sl50_gaussian.npy',windowB)
    
    
    #windowA = np.load('./A_sl0_gaussian.npy')
    #windowB = np.load('./A_sl50_gaussian.npy')
    #q_grid_ebsd = np.load('./r2_p22_quaternion.npy')
    # simr2p22 = np.load(fp1+'r2_P22.npy')
    # r2_p22_sims = []
    # for i in range(10):
    #     r2_p22_sims.append(exa_to_quat(simr2p22[i*50],'./r2_p22_sim_qgrid_' + str(i*50) + '.npy',save=True))
    
    # q_grid_sim = np.load('./r1_p63_10slices.npy')
    
    striped = False
    if striped:
        uniqa = np.unique(windowA.reshape(-1,4), axis=0)
        windowA[:,:] = uniqa[0]
        windowA[0::10] = uniqa[1]
    
    avg_dist_weight_test = False
    if avg_dist_weight_test:
        tests = 10
        windowA = q_grid
        w1 = windowA[:l,:l]
        avgdists = np.zeros(10)
        a_s = np.zeros(400)
    
        for j in range(10):
            #w1 = ordev.gen_textured_window(l, l)
            w2 = np.zeros((tests,l,l,4))
            
            dists = np.zeros(tests)
            
            alpha = (j+1)/tests
            beta = l/np.pi
    
            iwB = mwmb.intrawindow_block(l) 
    
            for i in range(tests):
                w2[i] = windowA[i*2:int(np.around((i*2/l+1)*l)),:l]
            
            for i in range(tests):
                misdict = ordev.misorientation_dict(w1, w2[i])
                G = ordev.texture_graph(w1, w2[i], plotting = False)
                
                Graph = np.copy(iwB)
                Graph[:-1, :-1] += G*5
                Graph = ordev.full_bipartite(Graph)
                
                r,c = optimize.linear_sum_assignment(Graph)
                dists[i] = Graph[r,c].sum()/l**2
                # plt.imshow(w2[i,:,:,0]-w1[:,:,0], vmax=1)
                # plt.show()
                
                for k in range(400):
                    a_s[k]+=np.shape(np.where(Graph[k,:400] < Graph[k,k]))[1]
                
            plt.hist(a_s/100)
            plt.title('alpha = ' + str(alpha) + ': number with cost < dii')
            plt.show()
                
            avgdists[j] = np.average(dists)
        # dists[i] = dist
        plt.scatter(np.arange(10)/10, avgdists)
        plt.show()
        
    texture_test = False
    if texture_test:
    
        sample_windows  = False
        assign_A0t1     = False
        run_comparison  = False
        solve           = False
        bw              = False
        overlay         = False
    
        wn = 10
        if sample_windows:
            ind = wsa.find_valid_wnds(200, l, wn)[0]
            A0t4 = np.zeros((wn,l,l,4))
            for i in range(10):
                A0t4[i] = windowA[ind[0,i]:ind[0,i]+l, ind[1,i]:ind[1,i]+l]
            for i in range(10):
                plt.imshow(A0t4[i,:,:,0])
                plt.show()
            np.save('./A0t4_str05.npy', A0t4)
            
        full_matrix = True
        sweep_windows = True
        
        if sweep_windows:
    
            resample = False
            if resample:
                ws = np.zeros((wn, l, l, 4))
                wsB = np.zeros((wn, l, l, 4))
                ind = wsa.find_valid_wnds(768, l, wn)[0]
                for i in range(wn):
                    #ws[i] = np.copy(windowA[2*l:3*l, int(l*i*0.1/.5):int(np.around((1+(i*0.1)/.5)*l))])
                    #ws[0] = assign_quats(rng.normal(size=(l,l)), create_quaternion(rng.normal(size=(l**2,4))), strength = 0)
                    a = ind[0,i]
                    b = ind[1,i]
                    ws[i] = q_grid[a:a+l, b:b+l]
                    plt.figure()
                    plt.imshow(abs(ws[i,:,:,:3]), vmax = 1)
                    plt.show()
                    
                ind = wsa.find_valid_wnds(200, l, wn)[0]
                for i in range(wn):
                    #wsB[0] = assign_quats(rng.normal(size=(l,l)), create_quaternion(rng.normal(size=(l**2,4))), strength = 0)
                    a = ind[0,i]
                    b = ind[1,i]
                    wsB[i] = windowB[a:a+l, b:b+l]
                    plt.figure()
                    plt.imshow(abs(wsB[i,:,:,:3]), vmax = 1)
                    plt.show()
                
            Gs = np.zeros((1,wn))
            dis_line = np.zeros((1,wn))
            
            end = 1
            if full_matrix: 
                end = wn
                Gs = np.zeros((wn,wn))
                dis_line = np.zeros((wn,wn))
            
            plotting = False
            for j in range(end):
                for i in range(j+1, wn):
                    #dis_line[i],G = compare(ws[0],ws[i],iwB,l,beta=(0.9*l/np.pi))
                    dis_line[j,i],G = compare(ws[j],wsB[i],iwB,l,beta=(l/np.pi))
                    Gs[j,i] = np.average(G[:l**2, :l**2] - iwB[:l**2, :l**2])
                    
                    # dis_line[i,j] = dis_line[j,i]
                    # Gs[i,j] = Gs[j,i]
    
                    #plt.figure()
                    #plt.imshow(abs(ws[i,:,:,:3]), vmax = 1)
                    #plt.show()
                    if plotting:
                        plt.imshow(np.reshape(G[int(l**2/2 + l/2),:l**2], (l,l)), 'RdBu')
                        plt.colorbar()
                        plt.show()
                        
                #print('average G for: 0' +str(i), np.average(Gs[j]))
                plt.scatter(np.arange(wn),dis_line[j,:])  
                #plt.ylim(np.sort(dis_line)[j,1]-1,np.max(dis_line)+1)
                plt.show()
            
            dis_line += dis_line.T
            if full_matrix: 
                plt.imshow(dis_line)
                plt.colorbar()
                plt.show()
            
            for i in range(wn):
                dis_line[i,i] = 1000
            r,c = optimize.linear_sum_assignment(dis_line)
            print(dis_line[r,c].sum()/wn)
    
        A0t0 = np.load('./A0t0_texture_test.npy')
        A0t3 = np.load('./A0t3_str50.npy')
        A0t4 = np.load('./A0t4_str05.npy')
        A1t0 = np.load('./A1t0_texture_test.npy')
        B0t0 = np.load('./B0t0_texture_test.npy')
    
        if assign_A0t1:
            A0t1 = np.copy(A0t0)
        
            qnew = create_quaternion(np.arange(3))
            c = 0 
            
            uniqa = np.unique(A0t0.reshape(-1,4), axis=0)
            qn = np.shape(qnew)[0]
            for i in range(wn): 
                for j in range(qn):
                    a,b = np.where(A0t0[i,:,:,0] == uniqa[j,0])
                    A0t1[i,a,b,:] = qnew[j]
            np.save('./A0t1_texture_test.npy',A0t1)
        A0t1 = np.load('./A0t1_texture_test.npy')
        samples = [A0t0, A0t1, A1t0, B0t0, A0t3, A0t4]
        
        if run_comparison:
    
            #samples = [A0t0, A0t1, A1t0, A0t3]
            ns = len(samples)
            
            all_ds = np.zeros((wn*ns, wn*ns))
            for i in range(ns):
                for j in range(i,ns):
                    all_ds[i*wn:(i+1)*wn, j*wn:(j+1)*wn] = sweep(samples[i], samples[j], beta = beta)
            
            all_ds+=all_ds.T
            plt.imshow(all_ds)
            plt.show()
         
        if solve:
            maximize = 1
            for i in range(wn*ns):
                all_ds[i,i] = (1-maximize)*100
            Wd = np.zeros((ns, ns))
            for i in range(ns):
                for j in range(i,ns):
                    r,c = optimize.linear_sum_assignment(all_ds[i*10:(i+1)*10, j*10:(j+1)*10], maximize = maximize)
                    Wd[i,j] = all_ds[i*10:(i+1)*10, j*10:(j+1)*10][r,c].sum()/wn
                    Wd[j,i] = Wd[i,j]
                    print(str(i) + str(j), c)
            
            print(np.around(Wd,2))
    
        if bw:
            bigwindows = np.zeros((ns, l, l*10))
            
            for i in range(ns):
                for j in range(wn):
                    bigwindows[i, :l, j*l:(j + 1)*l] = samples[i][j,:,:,0]
                plt.figure()
                plt.imshow(bigwindows[i], vmax=1)
            plt.show()
    
        r = np.arange(10)
        c = [2, 8, 4, 1, 5, 7, 9, 3, 0, 6]
    
        if overlay:
            for i in range(wn):
                w1 = samples[2][r[i]]
                w2 = samples[3][c[i]]
                
                plt.figure()
                plt.imshow(10*w1[:,:,0] - 10*w2[:,:,0], cmap = 'RdBu_r')
                plt.colorbar()
        
                plt.figure()
                image = np.zeros((20,20))
                dists = compare(w1,w2,iwB,l)[1]
                for j in range(400):
                    image[int(np.floor(j/20)), j%20] = dists[j,j]
                plt.imshow( image )
                plt.colorbar()
            plt.show()
            

