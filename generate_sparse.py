import numpy as np
import tensor
import random as rd
if __name__ == '__main__':
    N1 = 100
    N2 = 100
    N3 = 100
    r = np.zeros((N1,N2,N3))
    for i in range(N1/10):
        temp = (np.random.random(size=(100,100))>0.98)
        for i in range(10):
            r[i]=temp
    z= np.random.normal(0, 1, (N1, N2, N3))
    q = z*r
    np.save("sparse_100.npy", q)

