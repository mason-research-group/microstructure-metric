import sys
sys.path.append('../state-space')
import numpy as np
import matplotlib.pyplot as plt
# import master_wass_metric_binary as mwmb

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


def misorientation(q1, q2):
    ### calculation of rotation quaternion for q1 and q2 ###
    q2inv = np.copy(q2)
    q2inv[1:] *= -1
    r = quatmult(q1, q2inv)
    return r


def create_FID(G, unique_q, n):
    ### assigning a FID for each unique orientation ###
    FID = 0
    where_q = np.zeros((n,n), dtype = int)
    for quats in unique_q:
        where_q[np.where(G == quats)[:2]] = FID
        FID += 1
    plt.imshow(where_q)
    plt.show()
    return where_q


def texture_graph(A,B):
    ### the graph for orientation difference for two windows, A and B ###
    n = np.shape(A)[0]
    G = np.zeros((n**2, n**2))

    ### we only want to compute disorientation for unique pairs ###
    uniqueA = np.unique(A.reshape(-1,4), axis=0)
    uniqueB = np.unique(B.reshape(-1,4), axis=0)
    nA = np.shape(uniqueA)[0]
    nB = np.shape(uniqueA)[0]
    fidA = create_FID(A, uniqueA, n)
    fidB = create_FID(B, uniqueB, n)
    plt.imshow(fidA + 2*fidB)
    plt.show()
    misorientations = np.zeros((nA,nB))
    i = 0
    for quatA in uniqueA:
        j = 0
        for quatB in uniqueB:
            misorientations[i,j] = misorientation(quatA, quatB)[0]
        j+=1
        i+=1
    ### assign disorientation to the full graph ###
    for i in range(n):
        for j in range(n):
            a = i*n + j
            for k in range(n):
                for l in range(n):
                    b = k*n + l
                    G[a,b] = 2*np.arccos(misorientations[fidA[i,j], fidB[k,l]])

    return G



if __name__ == '__main__':
    l = 5
    rng = np.random.default_rng()

    ### create 2 random orientations ###
    q1 = rng.random(size=4)
    q1 = q1 / np.linalg.norm(q1)
    q2 = rng.random(size=4)
    q2 = q2 / np.linalg.norm(q2)

    ### create two windows with randomly assigned q1 and q2 ###
    A = np.ones((l,l,4))
    A[:,:]*=q1
    B = np.ones((l,l,4))
    B[:,:]*=q1
    ijA = rng.integers(0, l, size=(rng.integers(0,l**2), 2))
    for ij in ijA:
        A[ij[0], ij[1], :] = q2
    ijB = rng.integers(0, l, size=(rng.integers(0,l**2), 2))
    for ij in ijB:
        B[ij[0], ij[1], :] = q2

    G = texture_graph(A,B)

print(360/np.pi*np.arccos(misorientation(q1, q2)[0]))
#q1a = Quaternion(q1)
#q2a = Quaternion(q2)
#print(Quaternion.absolute_distance(q1a,q2a)*360/np.pi)