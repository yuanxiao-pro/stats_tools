import numpy as np
import matplotlib
matplotlib.use('Agg')  # 切换到无 GUI 后端
import matplotlib.pyplot as plt

def DE(data, m=3, epsilon=10):
    """
    计算数据集的多样性熵
    """
    N = len(data)
    M = N-(m-1)
    X = np.zeros((m, M)) # m行 M列
    for j in range(M):
        for i in range(m):
            print(i-1 ,j, data[i + j])
            X[i][j] = data[i + j]
    dist = np.zeros((M-1, M-1))
    for i in range(M-1):
        dist[i] = sum(X[:,i] * X[:,i+1]) / (np.sqrt(sum(X[:,i] * X[:,i])) * np.sqrt(sum(X[:,i+1] * X[:,i+1])))
    dist = np.round(dist, 4)
    print(dist)
    return data


if __name__ == "__main__":
    data = np.array([1, 2, 13, 7, 9, 5, 4])
    epsilon = 10
    m = 3
    DE(data, m, epsilon)

