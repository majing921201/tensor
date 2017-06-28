# -*- coding:utf8 -*-
import numpy as np
import matplotlib.pyplot as plt


# 最外层axis = 0

def loadData(filename):
    dataSet = []
    fr = open(filename)
    for line in fr:
        newLine = map(float, line.strip().split())
        dataSet.append(newLine)
    return dataSet


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
    TenU_bar=np.array(TenU_bar)
    TenS_bar=np.array(TenS_bar)
    TenV_bar=np.array(TenV_bar)
    return TenU_bar, TenS_bar, TenV_bar


def tProduct(A, B):
    n1, n2, n3 = A.shape  # n1是外层的维度,此处指的是时间序列的长度
    m1, m2, m3 = B.shape
    C = []
    for i in range(n1):
        MatSum = np.zeros((n2, m3))
        for k in range(i + 1):
            MatSum = MatSum + np.dot(A[i - k], B[k])
        for j in range(i + 1, n1):
            MatSum = MatSum + np.dot(A[i + n1 - j], B[j])
        C.append(MatSum)
    return np.array(C)


def getReal(tensor):
    n1, n2, n3 = tensor.shape
    for i in range(n1):
        for j in range(n2):
            for k in range(n3):
                tensor[i][j][k] = tensor[i][j][k].real
    return tensor


def compression(tenX, k):
    # com_tensor = np.zeros((tenX.shape))

    u, s, v = tsvd(tenX)
    N1, N2, N3 = s.shape
    for i in range(k, min(N2, N3)):
        s[::, i, i] = 0
        u[::, ::, i] = 0
        v[::, i, ::] = 0
    s_new = np.fft.ifft(s, axis=0)
    u_new = np.fft.ifft(u, axis=0)
    v_new = np.fft.ifft(v, axis=0)
    new_tensor = tProduct(tProduct(u_new, s_new), v_new)

    '''
    N1,N2,N3 = s.shape

    for i in range(k):
        u_new = np.zeros((N1,N2,1))
        u_new[::, ::, 0] = u[::, ::, i]
        s_new = np.zeros((N1,1,1))
        s_new[::, 0, 0] = s[::, i, i]
        print s_new
        v_new = np.zeros((N1,1,N3))
        v_new[::, 0, ::] = v[::, i, ::]
        com_tensor += tProduct(tProduct(u_new, s_new), v_new)
    '''
    #print new_tensor - tenX
    return new_tensor


if __name__ == '__main__':
    cube = []
    for elements in range(1, 31):
        ele = str(elements)
        matrix = loadData('/Users/jingma/Documents/anomaly detection/data_yelp/' + ele + '.txt')
        cube.append(matrix)
    cube = np.array(cube)

    CubeTime = np.transpose(cube, axes=(1, 2, 0))
    U, S, V = tsvd(CubeTime)
    print CubeTime
    # V = np.transpose(V, axes=(0, 2, 1))
    '''
    VTrans = []
    UTrans = []

    VTrans.append(np.transpose(V[0]))  # 转置
    for i in range(35, 0, -1):
        VTrans.append(np.transpose(V[i]))
    VTrans = np.array(VTrans)

    UTrans.append(np.transpose(U[0]))  # 转置
    for i in range(35, 0, -1):
        UTrans.append(np.transpose(U[i]))
    UTrans = np.array(UTrans)
    '''
    tensor = tProduct(tProduct(U, S), V)
    N1, N2, N3 = tensor.shape
    for k in range(N3):
        ind = range(N1)  # 画图 时间序列的tensor流
        bar_x = []
        singlebar = np.zeros((N1))
        for j in range(N2):
            singlebar = singlebar + tensor[::, j, k]
            bar_x.append(singlebar)
        p5 = plt.bar(ind, bar_x[4], 0.7, color='k')
        p4 = plt.bar(ind, bar_x[3], 0.7, color='g')
        p3 = plt.bar(ind, bar_x[2], 0.7, color='b')
        p2 = plt.bar(ind, bar_x[1], 0.7, color='r')
        p1 = plt.bar(ind, bar_x[0], 0.7, color='y')
        plt.show()
    # print tProduct(U, S)
    print '***********'
    print tensor
