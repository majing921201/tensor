# -*- coding:utf8 -*-
import numpy as np
import math as mt
import matplotlib.pyplot as plt
import normf
import copy as cp


# 最外层axis = 0

def loadData(filename):
    dataSet = []
    fr = open(filename)
    for line in fr:
        newLine = map(float, line.strip().split())
        dataSet.append(newLine)
    return dataSet


def trpca(tenX, lam):
    N1, N2, N3 = tenX.shape
    L_new = np.zeros((N1, N2, N3))
    E_new = np.zeros((N1, N2, N3))
    rho = 1.1
    eps = 1e-3

    Y_new = tenX
    U0, S0, V0 = tsvd(Y_new)
    normTwo = np.amax(S0)
    normInf = np.amax(Y_new)
    dualnorm = max((normTwo, normInf))
    mu = 1.25 / normTwo

    muBar = mu * 1e7
    Y_new = Y_new / dualnorm
    L_gap = 10
    S_gap = 10
    iterate = 0
    while L_gap > eps or E_gap > eps:
        iterate += 1
        L_old = cp.deepcopy(L_new)
        E_old = cp.deepcopy(E_new)
        Y_old = cp.deepcopy(Y_new)
        tempE = tenX - L_old + (1 / mu) * Y_old
        for i in range(N2):
            for j in range(N3):
                normtempE = 0
                for k in range(N1):
                    normtempE += tempE[k][i][j] * tempE[k][i][j]
                normtempE = mt.sqrt(normtempE)
                E_new[::, i, j] = max(0, (1 - lam / (mu * normtempE))) * tempE[::, i, j]
        E_L1 = np.sum(E_new * E_new)

        tempL = tenX - E_new + (1 / mu) * Y_old
        # L = shrinkage(tempL, 1 / mu)
        U, S, V = tsvd(tempL)
        for j in range(min(N2, N3), min(N2, N3)):
            if S[::, j, j] > 1 / mu:
                S[::, j, j] = S[::, j, j] - 1 / mu
            elif S[::, j, j] < 1 / mu:
                S[::, j, j] = S[j] + 1 / mu
            else:
                S[::, j, j] = 0

        L_nuc = np.sum(S)
        L_new = tProduct(tProduct(U, S), V)

        '''
        for i in range(N1):
            for j in range(min(N2, N3)):
                if S[i][j][j] > 1 / mu:
                    S[i][j][j] -= 1 / mu
                elif S[i][j][j] < -1 / mu:
                    S[i][i][j] += 1 / mu
                else:
                    S[i][j][j] = 0
        L = tProduct(tProduct(U, S), V)
        '''
        # print L

        mu = min(rho * mu, muBar)
        Y_new = Y_old + mu * (tenX - L_new - E_new)
        L_gap = normf.getnormf(L_new - L_old)
        E_gap = normf.getnormf((E_new - E_old))
        print("iter:%f  L_nuc:%f   E_L1:%f   L_gap:%f   E_gap:%f" % (iterate, L_nuc, E_L1, L_gap, E_gap))

    return L_new, E_new

def tsvd(TenM):
    t, m, n = TenM.shape
    TenD = np.fft.fft(TenM, axis=0)
    TenU_bar = []
    TenS_bar = []
    TenV_bar = []
    for i in range(t):
        MatS = np.zeros((m, n))
        U, S, V = np.linalg.svd(TenD[i], True)
        for j in range(min(m, n)):
            MatS[j][j] = S[j]
        TenU_bar.append(U)
        TenS_bar.append(MatS)
        TenV_bar.append(V)
    TenU = np.fft.ifft(TenU_bar, axis=0)
    TenS = np.fft.ifft(TenS_bar, axis=0)
    TenV = np.fft.ifft(TenV_bar, axis=0)

    TenU = getReal(TenU)  # 取实数部分
    TenS = getReal(TenS)
    TenV = getReal(TenV)

    return TenU, TenS, TenV


def shrinkage(TenM, tau):
    t, m, n = TenM.shape
    TenD = np.fft.fft(TenM, axis=0)
    TenU_bar = []
    TenS_bar = []
    TenV_bar = []
    for i in range(t):
        MatS = np.zeros((m, n))
        U, S, V = np.linalg.svd(TenD[i], True)
        for j in range(min(m, n)):
            if S[j] > tau:
                MatS[j][j] = S[j] - tau
            elif S[j] < -tau:
                MatS[j][j] = S[j] + tau
            else:
                MatS[j][j] = 0

        TenU_bar.append(U)
        TenS_bar.append(MatS)
        TenV_bar.append(V)
    TenU = np.fft.ifft(TenU_bar, axis=0)
    TenS = np.fft.ifft(TenS_bar, axis=0)
    TenV = np.fft.ifft(TenV_bar, axis=0)

    TenU = getReal(TenU)  # 取实数部分
    TenS = getReal(TenS)
    TenV = getReal(TenV)
    print TenS_bar

    return tProduct(tProduct(TenU, TenS), TenV)


def tProduct(A, B):
    n1, n2, n3 = A.shape  # n1是外层的维度,此处指的是时间序列的长度
    m1, m2, m3 = B.shape
    C = []
    for i in range(n1):
        MatSum = np.zeros((n2, m3))
        for k in range(i + 1):
            MatSum = np.add(MatSum, np.dot(A[i - k], B[k]))
        for j in range(i + 1, n1):
            MatSum = np.add(MatSum, np.dot(A[i + n1 - j], B[j]))
        C.append(MatSum)
    return np.array(C)


def getReal(tensor):
    n1, n2, n3 = tensor.shape
    for i in range(n1):
        for j in range(n2):
            for k in range(n3):
                tensor[i][j][k] = tensor[i][j][k].real
    return tensor


if __name__ == '__main__':
    cube = []
    for elements in range(1, 31):
        ele = str(elements)
        matrix = loadData('/Users/jingma/Documents/anomaly detection/data_yelp/' + ele + '.txt')
        cube.append(matrix)
    cube = np.array(cube)

    CubeTime = np.transpose(cube, axes=(1, 2, 0))
    L, E = trpca(CubeTime, 0.1)

    N1, N2, N3 = CubeTime.shape
    for i in range(N1):
        for j in range(N3):
            L[i, ::, j] = L[i, ::, j] / np.sum(L[i, ::, j])
    print L

    for k in range(N3):
        ind = range(N1)  # 画图 时间序列的tensor流
        bar_x = []
        singlebar = np.zeros((N1))
        for j in range(N2):
            singlebar = singlebar + L[::, j, k]
            bar_x.append(singlebar)
        p5 = plt.bar(ind, bar_x[4], 0.7, color='k')
        p4 = plt.bar(ind, bar_x[3], 0.7, color='g')
        p3 = plt.bar(ind, bar_x[2], 0.7, color='b')
        p2 = plt.bar(ind, bar_x[1], 0.7, color='r')
        p1 = plt.bar(ind, bar_x[0], 0.7, color='y')
        plt.show()

    # print np.sum(CubeTime, axis=1)
    # V = np.transpose(V, axes=(0, 2, 1))
    VTrans = []
    UTrans = []
    '''
    VTrans.append(np.transpose(V[0]))  # 转置
    for i in range(35, 0, -1):
        VTrans.append(np.transpose(V[i]))
    VTrans = np.array(VTrans)

    UTrans.append(np.transpose(U[0]))  # 转置
    for i in range(35, 0, -1):
        UTrans.append(np.transpose(U[i]))
    UTrans = np.array(UTrans)

    tensor = tProduct(tProduct(U, S), V)
    print tensor
    '''

'''
        tempL = tenX - E - (1 / mu) * Y
        U, S, V, Sbar = tsvd(tempL)
        # print Sbar
        tenT = np.zeros((N1, min(N2, N3), min(N2, N3)))  # 软阈值处理后的对角张量
        for i in range(min(N2, N3)):
            vectbar = []
            for j in range(N1):
                vectbar.append(max(0, 1 - 1 / (mu * Sbar[j][i][i])))
            vect = np.fft.ifft(vectbar)
            for j in range(N1):
                tenT[j][i][i] = vect[j].real
        softS = tProduct(tenT, S)
        L = tProduct(tProduct(U, softS), V)
        # print softS
        print L
        tempE = tenX - L - (1 / mu) * Y
        for i in range(N2):
            for j in range(N3):
                normtempE = 0
                for k in range(N1):
                    normtempE += tempE[k][i][j] * tempE[k][i][j]
                normtempE = mt.sqrt(normtempE)
                tempE[::, i, j] = max(0, (1 - lam / (mu * normtempE))) * tempE[::, i, j]

        Y = Y + mu * (L + E - tenX)
        '''

'''
        tenT = np.zeros((N1, min(N2, N3), min(N2, N3)))  # 软阈值处理后的对角张量
        for i in range(min(N2, N3)):
            vectbar = []
            for j in range(N1):
                vectbar.append(max(0, 1 - 1 / (mu * Sbar[j][i][i])))
            vect = np.fft.ifft(vectbar)

            for j in range(N1):
                tenT[j][i][i] = vect[j].real

        softS = tProduct(tenT, S)
        '''

'''
        for i in range(N1):
            for j in range(min(N2, N3)):
                if S[i][j][j] > 1 / mu:
                    S[i][j][j] -= 1 / mu
                elif S[i][j][j] < -1 / mu:
                    S[i][i][j] += 1 / mu
                else:
                    S[i][j][j] = 0
        L = tProduct(tProduct(U, S), V)
        '''
