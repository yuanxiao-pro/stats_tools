import numpy as np
import matplotlib
matplotlib.use('Agg')  # 切换到无 GUI 后端
import matplotlib.pyplot as plt
from histcount import histcount
import seaborn as sns
from matrix_visualization import visualize_matrix_3d

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
    print("X.shape", X.shape)
    print("X", X)

    visualize_matrix_3d(X)
    dist = np.zeros((M-1, M-1))
    for i in range(M-1):
        dist[i] = sum(X[:,i] * X[:,i+1]) / (np.sqrt(sum(X[:,i] * X[:,i])) * np.sqrt(sum(X[:,i+1] * X[:,i+1])))
    dist = np.round(dist, 4)
    print("dist.shape", dist.shape)
    visualize_matrix_3d(dist, title="dist")

    counts, _ = histcount(dist, bins=epsilon, range=(-1, 1)) # 当counts数组结尾的数越大，说明输入的数据越可预测
    # print(counts)
    P = counts / sum(counts)
    de = 0
    for i in range(len(P)):
        ep = np.log(epsilon)
        tmp = np.where(P[i] == 0, 0, -1 * P[i] * np.log(P[i]) / ep)
        # print("tmp", tmp)
        de += tmp
    print(de)
    return de


if __name__ == "__main__":
    data = np.array([1, 2, 13, 7, 9, 5, 4])
    # 数据越可预测，熵越小
    data = np.array([1, 2, 13, 7, 9, 15, 14, 0, 1, 13, 1, 0, 99, 1, 0, 13])
    # data = np.array([1, -1, 1, -1,1, -1,1, -1,1, -1,1, -1,1, -1]) 
    epsilon = 10
    m = 3
    DE(data, m, epsilon)

