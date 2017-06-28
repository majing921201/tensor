import matplotlib.pyplot as plt
import numpy as np

T = np.load("trip_TIME36.npy")

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
    plt.xlim(0, 50)
    plt.ylim(0, 1)
    plt.show()