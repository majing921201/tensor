import numpy as np
import tensor_rpca
from numpy import linalg as LA
import copy as cp
import alm
import math as mt
import tensor


def com_error(vec):
    error = 0
    for i in vec:
        error += i * i
    return error


if __name__ == '__main__':
    A = np.load("amazon_electronics.npy")
    new_tensor = []
    for i in range(50):
        new_tensor.append(A[i])
    new_tensor = np.array(new_tensor)
    change_tensor = cp.deepcopy(new_tensor)
    for i in range(10):
        change_tensor[i * 5, ::, i * 2] = [0.0, 0.0, 0.0, 0.5, 0.5]
    for i in range(10):
        change_tensor[i * 5 + 1, ::, i * 3] = [0.5, 0.0, 0.0, 0.0, 0.5]
    for i in range(10):
        change_tensor[i * 5 + 2, ::, i * 4] = [0.0, 0.5, 0.0, 0.0, 0.5]
    #T,E = tensor_rpca.trpca(change_tensor,0.13)
    #T = alm.trpca(change_tensor,0.08)
    T = tensor.compression(change_tensor, 1)
    for i in range(50):
        for j in range(50):
            T[i, ::, j] = T[i, ::, j] / np.sum(T[i, ::, j])
    for i in range(1, 50):
        print LA.norm((T[i] - T[i - 1]), 'fro') / LA.norm(T[i - 1], 'fro')

    for i in range(10):
        # print new_tensor[i*5,::,i*2]
        # print T[i*5,::,i*2]
        # print mt.sqrt(com_error(new_tensor[i*5,::,i*2]-T[i*5,::,i*2].real)/com_error(new_tensor[i*5,::,i*2]))
        # print np.sum(abs(new_tensor[i*5,::,i*2]-T[i*5,::,i*2].real))
        print mt.sqrt(com_error(new_tensor[i * 5, ::, i * 2] - T[i * 5, ::, i * 2].real))
        print mt.sqrt(com_error(new_tensor[i * 5 + 1, ::, i * 3] - T[i * 5 + 1, ::, i * 3].real))
        print mt.sqrt(com_error(new_tensor[i * 5 + 2, ::, i * 4] - T[i * 5 + 2, ::, i * 4].real))