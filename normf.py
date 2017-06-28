import math as mt


def getnormf(tensor):
    n1, n2, n3 = tensor.shape
    value_norm = 0
    for i in range(n1):
        for j in range(n2):
            for k in range(n3):
                value_norm += tensor[i][j][k] * tensor[i][j][k]
    value_norm = mt.sqrt(value_norm)
    return value_norm
