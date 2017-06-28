import numpy as np
import numpy.linalg as LA
import copy as cp


def loadDate(filename):
    dataSet = []
    fr = open(filename)
    for line in fr:
        newLine = map(float, line.strip().split())
        dataSet.append(newLine)
    dataSet = np.mat(dataSet).T
    return dataSet


def inexactALM(D, lam):
    tol = 1e-7
    (m, n) = D.shape
    Y = D
    normTwo = LA.norm(Y, 2)
    normInf = np.amax(Y) / lam
    dualNorm = max(normTwo, normInf)
    Y = Y / dualNorm
    A = np.mat(np.zeros((m, n)))
    E = np.mat(np.zeros((m, n)))
    mu = 2.5/ normTwo
    muBar = mu * 1e7
    rho = 4.0
    dNorm = LA.norm(D, 'fro')
    iterate = 0
    converged = 0
    while converged == 0:
        iterate += 1
        tempT = D - A + (1 / mu) * Y
        E = np.maximum(tempT - lam / mu, 0)
        E = E + np.minimum(tempT + lam / mu, 0)
        U, S, V = np.linalg.svd(D - E + (1 / mu) * Y, False)

        svp = 0
        for ele in S:
            if ele > 1 / mu:
                svp += 1
        print svp

        A = U[:, 0:svp] * np.mat(np.diag(S[0:svp] - 1 / mu)) * V[0:svp, :]
        Z = D - A - E
        Y = Y + mu * Z
        mu = min(mu * rho, muBar)
        stopCriterion = LA.norm(Z, 'fro') / dNorm
        if stopCriterion < tol:
            converged = 1
        if converged == 0 and iterate >= 1000:
            converged = 1
        print iterate
    return A, E

def trpca(tenX, lam):
    N1, N2, N3 = tenX.shape
    matrix_unfold_1 = np.zeros((N1, N2 * N3))
    matrix_unfold_2 = np.zeros((N2, N1 * N3))
    matrix_unfold_3 = np.zeros((N3, N1 * N2))
    for i in range(N1):
        for j in range(N2):
            for k in range(N3):
                matrix_unfold_1[i][j * N3 + k] = tenX[i][j][k]
    for i in range(N1):
        for j in range(N2):
            for k in range(N3):
                matrix_unfold_2[j][i * N3 + k] = tenX[i][j][k]

    for i in range(N1):
        for j in range(N2):
            for k in range(N3):
                matrix_unfold_3[k][i * N2 + j] = tenX[i][j][k]
    A1, E1 = inexactALM(matrix_unfold_1, lam)
    A2, E2 = inexactALM(matrix_unfold_2, lam)
    A3, E3 = inexactALM(matrix_unfold_3, lam)

    ten_fold_1 = np.zeros((N1, N2, N3))
    ten_fold_2 = np.zeros((N1, N2, N3))
    ten_fold_3 = np.zeros((N1, N2, N3))
    for i in range(N1):
        for j in range(N2):
            for k in range(N3):
                ten_fold_1[i][j][k] = np.array(A1)[i][j * N3 + k]
    for i in range(N1):
        for j in range(N2):
            for k in range(N3):
                ten_fold_2[i][j][k] = np.array(A2)[j][i * N3 + k]
    for i in range(N1):
        for j in range(N2):
            for k in range(N3):
                ten_fold_3[i][j][k] = np.array(A3)[k][i * N2 + j]
    A = (ten_fold_1 + ten_fold_2 + ten_fold_3) / 3
    # A = (N1 / (N1 + N2 + N3)) * ten_fold_1 + (N2 / (N1 + N2 + N3)) * ten_fold_2 + (N3 / (N1 + N2 + N3)) * ten_fold_3
    return A


if __name__ == '__main__':
    ratingSet = loadDate('1.txt')
    inexactALM(ratingSet, 0.18)
