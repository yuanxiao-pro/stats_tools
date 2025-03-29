import numpy as np

def create_sliding_window_matrix(data: np.ndarray, m: int) -> np.ndarray:
    """
    创建滑动窗口矩阵
    
    Args:
        data: 输入的时间序列数据
        m: 窗口大小
        
    Returns:
        滑动窗口矩阵 X，其中 X[i,j] = data[(i-1)+j]
    """
    # 计算矩阵的列数
    M = len(data) - m + 1
    
    # 创建结果矩阵
    X = np.zeros((m, M))
    
    # 填充矩阵
    for j in range(M):
        for i in range(m):
            X[i, j] = data[(i-1) + j]
    
    return X

def example_usage():
    """
    示例使用
    """
    # 创建示例数据
    data = np.array([1, 2, 13, 7, 9, 5, 4])
    m = 3  # 窗口大小
    
    # 创建滑动窗口矩阵
    X = create_sliding_window_matrix(data, m)
    
    print("原始数据:", data)
    print("滑动窗口矩阵:")
    print(X)
    
    # 验证矩阵的构建是否正确
    print("\n验证矩阵构建:")
    for j in range(X.shape[1]):
        print(f"第{j+1}列:", X[:, j])

if __name__ == "__main__":
    example_usage() 