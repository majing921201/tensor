import numpy as np
import tensor_rpca
import math as mt
from numpy import linalg as LA
import matplotlib.pyplot as plt
import time

# start = time.clock()
T = np.load("yelp_LA_TIME36.npy")
A, E = tensor_rpca.trpca(np.load("yelp_LA_TIME36.npy"), 0.08)

N1, N2, N3 = A.shape
for i in range(N1):
    for j in range(N3):
        A[i, ::, j] = A[i, ::, j] / np.sum(A[i, ::, j])
#np.save("trip_lam0.075_base.npy",A)
#np.save("trip_lam0.075_noise.npy",T - A)
for i in range(1, 36):
    print LA.norm((A[i] - A[i - 1]), 'fro')
print "******************************************************"
for i in range(36):
    print LA.norm((T - A)[i], 'fro')
print "******************************************************"
'''
for i in range(50):
    print i
    for j in range(36):
        print np.sqrt(np.sum(np.square((T-A)[j,::,i])))
    print "******************************************************"

'''
# end = time.clock()
# print('Running time: %s Seconds' % (end - start))
'''
N1,N2,N3=A.shape
for i in range(N1):
        for j in range(N3):
            A[i, ::, j] = A[i, ::, j] / np.sum(A[i, ::, j])
for k in range(N1):
    ind = range(N3)
    bar_x = []
    singlebar = np.zeros((N3))
    for j in range(N2):
        singlebar = singlebar + A[k, j, ::]
        bar_x.append(singlebar)
    p5 = plt.bar(ind, bar_x[4], 0.7, color='k')
    p4 = plt.bar(ind, bar_x[3], 0.7, color='g')
    p3 = plt.bar(ind, bar_x[2], 0.7, color='b')
    p2 = plt.bar(ind, bar_x[1], 0.7, color='r')
    p1 = plt.bar(ind, bar_x[0], 0.7, color='y')
    plt.show()
'''

N1, N2, N3 = T.shape
print N1, N2, N3
for i in range(N1):
    for j in range(N3):
        T[i, ::, j] = T[i, ::, j] / np.sum(T[i, ::, j])

for k in range(N1):
    ind = range(N3)
    bar_x = []
    singlebar = np.zeros((N3))
    for j in range(N2):
        singlebar = singlebar + T[k, j, ::]
        bar_x.append(singlebar)
    p5 = plt.bar(ind, bar_x[4], 0.7, color='k')
    p4 = plt.bar(ind, bar_x[3], 0.7, color='g')
    p3 = plt.bar(ind, bar_x[2], 0.7, color='b')
    p2 = plt.bar(ind, bar_x[1], 0.7, color='r')
    p1 = plt.bar(ind, bar_x[0], 0.7, color='y')
    plt.xlim(0, 36)
    plt.ylim(0, 1)
    plt.show()
