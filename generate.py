import numpy as np
import tensor
import random as rd

if __name__ == '__main__':

    n, r = 100, 10
    p = np.random.normal(0, 0.1, (n, n, r))
    q = np.random.normal(0, 0.1, (n, r, n))
    L = tensor.tProduct(p, q)

    np.save("lowrank_100.npy", L)

