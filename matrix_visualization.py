import numpy as np
import matplotlib
matplotlib.use('Agg')  # 切换到无 GUI 后端
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

def visualize_matrix_2d(X, title="Matrix Visualization", cmap='viridis'):
    """
    使用热力图可视化矩阵（2D）
    
    Args:
        X: 要可视化的矩阵
        title: 图表标题
        cmap: 颜色映射方案
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(X, cmap=cmap, annot=True, fmt='.2f', 
                xticklabels=True, yticklabels=True)
    plt.title(title)
    plt.xlabel('Column Index')
    plt.ylabel('Row Index')
    plt.tight_layout()
    plt.show()
    plt.savefig(title + '.png')

def visualize_matrix_3d(X, title="3D Matrix Visualization"):
    """
    使用3D表面图可视化矩阵
    
    Args:
        X: 要可视化的矩阵
        title: 图表标题
    """
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # 创建网格
    x, y = np.meshgrid(np.arange(X.shape[1]), np.arange(X.shape[0]))
    
    # 绘制3D散点图
    surf = ax.scatter(x, y, X, c='red', marker='o')
    # 绘制连接线
    for i in range(X.shape[0]):
        for j in range(X.shape[1]-1):
            ax.plot([j, j+1], [i, i], [X[i,j], X[i,j+1]], 'b-', alpha=0.5)
    
    # for j in range(X.shape[1]):
        # for i in range(X.shape[0]-1):
            # ax.plot([j, j], [i, i+1], [X[i,j], X[i+1,j]], 'g-', alpha=0.5)
    
    # 添加颜色条
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
    
    # 设置标签
    ax.set_xlabel('Column Index')
    ax.set_ylabel('Row Index')
    ax.set_zlabel('Value')
    ax.set_title(title)
    
    plt.tight_layout()
    plt.show()
    plt.savefig(title + '.png')

def create_sliding_window_matrix(data: np.ndarray, m: int) -> np.ndarray:
    """
    创建滑动窗口矩阵
    
    Args:
        data: 输入的时间序列数据
        m: 窗口大小
        
    Returns:
        滑动窗口矩阵 X
    """
    # 计算矩阵的列数
    M = len(data) - m + 1
    
    # 创建结果矩阵
    X = np.zeros((m, M))
    
    # 填充矩阵
    for j in range(M):
        for i in range(m):
            X[i, j] = data[i + j]
    
    return X

def example_usage():
    """
    示例使用
    """
    # 创建示例数据
    data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    m = 3  # 窗口大小
    
    # 创建滑动窗口矩阵
    X = create_sliding_window_matrix(data, m)
    
    # 2D可视化
    visualize_matrix_2d(X, title="2D Sliding Window Matrix (m=3)")
    
    # 3D可视化
    visualize_matrix_3d(X, title="3D Sliding Window Matrix (m=3)")
    
    # 打印矩阵值
    print("矩阵值:")
    print(X)

if __name__ == "__main__":
    example_usage() 