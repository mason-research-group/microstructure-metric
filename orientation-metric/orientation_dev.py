import sys
sys.path.append('../state-space/')

import numpy as np
import matplotlib.pyplot as plt
import itertools

rng = np.random.default_rng()

#import master_wass_metric_binary as mwmb

def quatmult(q1, q2):
    ### quaternion multiplication ### 
    ind1 = np.array([0, 1, 2, 3])
    ind2 = np.array([[0, 1, 2, 3],
                     [1, 0, 3, 2],
                     [2, 3, 0, 1],
                     [3, 2, 1, 0]])

    sign = np.array([[1,-1,-1,-1], [1,1,-1,1], [1,1,1,-1], [1,-1,1,1]])
    prod = np.zeros(4)
    c = 0
    for row in ind2:
        prod[c] = np.dot(q1[ind1], q2[row]*sign[c])
        c+=1
    return prod


def misorientation_angle(q1, q2):
    ### calculate orientation distance between two quaternions, assuming that 
    ### the ordering of the quaternion parameters is (real, imaginary)
    wm = abs(q1[0]*q2[0] + q1[1]*q2[1] + q1[2]*q2[2] + q1[3]*q2[3])
    if np.isclose(wm,1):
        return 0
    else:
        return 2*np.arccos(wm)
    

def misorientation(q1, q2):
    ### calculation of rotation quaternion for q1 and q2 ###
    q2inv = np.copy(q2)
    q2inv[1:] *= -1
    r = quatmult(q1, q2)
    return r


def create_FID(G, unique_q, n, plotting = False):
    ### assigning a FID for each unique orientation ###
    FID = 0
    where_q = np.zeros((n,n), dtype = int)
    for quats in unique_q:
        where_q[np.where(G == quats)[:2]] = FID
        FID += 1
        
    if plotting:
        plt.imshow(where_q)
        plt.show()
        
    return where_q


# def memoize(func):
#     cache = dict()

#     def memoized_func(*args):
#         if args in cache:
#             return cache[args]
#         result = func(*args)
#         cache[args] = result
#         return result

#     return memoized_func


# def window_misorientations(A,B,i,j,k,l):
#     quatA = A[i,j]
#     quatB = B[k,l]
#     angle = misorientation_angle(quatA, quatB)
#     return angle, i, j, k, l 


# memoized_window_misorientations = memoize(window_misorientations)  


def misorientation_dict(A,B):
    n = np.shape(A)[0]

    ### we only want to compute disorientation for unique pairs ###
    uniqueA = np.unique(A.reshape(-1,4), axis=0)
    uniqueB = np.unique(B.reshape(-1,4), axis=0)
    fidA = create_FID(A, uniqueA, n)
    fidB = create_FID(B, uniqueB, n)
    misorientations = {}
    i = 0
    
    for quatA in uniqueA:
        j = 0
        for quatB in uniqueB:
            key = str(i)+str(j)
            if key not in misorientations:
                w = misorientation_angle(quatA, quatB)
                misorientations.update({str(i)+str(j) : w})
            j+=1
        i+=1
    return misorientations

def disorientations(q1,q2):
    ### find the minimum angle difference between degenerate quaternions
    # {a0, a1, a2, a3}
    table = list(itertools.product([False, True], repeat=4))
    r = quatmult(q1,q2)
    quatset = np.array(1152,4)
    return
    

def gen_textured_window(l,m):
    ### assign texture randomly for an lxl window using a number of quaternions
    ### equal to m
    A = np.ones((l,l,4))
    qs = rng.random(size=(m,4))
    for i in range(m):
        qs[i] = qs[i] / np.linalg.norm(qs[i]) 
        
    for i in range(l):
        for j in range(l):
            A[i,j] = qs[rng.integers(m)]

    return A

def full_bipartite(Graph):
    '''
    Constructs bipartite graph for a pair of windows.
    In other words, we are phrasing the matching of windows as an assignment
    problem and creating the graph which lets us compute the solution.

    Parameters
    ----------
    Graph : nparray
        output of "intrawindow_block". the unreduced bipartite graph
        
    Returns
    -------
    Graph : nparray
        the reduced bipartite graph, fully dense transport matrix
        which contains nodes for each window (A and B) as well as their 
        associated "boundaries" or "dummy points" or "reservoirs"
    '''
    
    
    # add rest of dummy points #
    s = np.shape(Graph) 
    
    dummyA = np.zeros((s[0],s[0]))
    dummyA += Graph[:,s[1]-1]
    Graph = np.append(Graph, dummyA.T, axis=1)

    dummyB = np.zeros((s[1],s[0]+s[1]))    
    dummyB += Graph[s[0]-1, :]
    Graph = np.append(Graph, dummyB, axis=0)[:-2,:-2]
    
    return Graph

def texture_graph(A,B,plotting):
    ### the graph for orientation difference for two windows, A and B ###
    n = np.shape(A)[0]
    G = np.zeros((n**2, n**2))

    ### we only want to compute disorientation for unique pairs ###
    uniqueA = np.unique(A.reshape(-1,4), axis=0)
    uniqueB = np.unique(B.reshape(-1,4), axis=0)
    nA = np.shape(uniqueA)[0]
    nB = np.shape(uniqueB)[0]
    
    ### creates a feature ID window that can be used to reference the look up table
    fidA = create_FID(A, uniqueA, n, plotting = plotting)
    fidB = create_FID(B, uniqueB, n, plotting = plotting)
    
    ### this computes the lookup table
    misorientations = misorientation_dict(A, B)
    
    ### assign disorientation from the lookup table to the full graph ###
    '''
    slice notation instead of looping 
    Aij to all of B[:,:]?    
    
    simple loop structures can yield 
    - cache alignment
    - loop unrolling
    - embarressingly parallel computations
    
    try except might be too slow
    '''
    
    for i in range(n):
        for j in range(n):
            a = i*n + j
            for k in range(n):
                for l in range(n):
                    b = k*n + l
                    ### these lines try to find the right indexing for the lookup table
                    ### of misorientations
                    try: G[a,b] = misorientations[str(fidA[i,j]) + str(fidB[k,l])]
                    except: G[a,b] = misorientations[str(fidB[k,l]) + str(fidA[i,j])]

    return G
            

if __name__ == '__main__':
    import time
    l = 50
    rng = np.random.default_rng()

    single_comparison = False
    if single_comparison:
        l = 25
        m = 4
        A = gen_textured_window(l, m)
        B = gen_textured_window(l, m)
        s = time.time()
        G = texture_graph(A,B)
        print(time.time()-s)
        
        
    sweep_comparison = False
    if sweep_comparison:
        ### a performance test suite, checking performance as a function of 
        ### window size and strength of texture
        ### ls contains the window sizes (multiples of 5 from 5 to 50)
        ### ms indicates the number of unique orientations in each window
        runs = 10
        ms = 25
        times = np.zeros((ms-1,runs))
        ls = np.arange(1,runs+1)*5
        avgtimes = np.zeros(runs)
        for m in range(1,ms):
            for t in range(runs):
                l = ls[t]
                A = gen_textured_window(l, m)
                B = gen_textured_window(l, m)
                s = time.time()
                G = texture_graph(A,B)
                times[m-1,t] = time.time()-s 
            avgtimes+=times[m-1]
        avgtimes/=m
    
        f, ax = plt.subplots(1)
        ax.plot(ls,avgtimes)
        ax.set_xlabel('window size (length)')
        ax.set_ylabel('Average time to compute distance graph (seconds)')
        stds = np.zeros(runs)
        for i in range(runs): stds[i] = np.std(times[:,i])
        ax.errorbar(ls,avgtimes,yerr=stds)
        plt.show()
    
    
