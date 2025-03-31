import numpy as np
import matplotlib
matplotlib.use('Agg')  # 切换到无 GUI 后端
import matplotlib.pyplot as plt
from typing import List, Tuple

def coarse_graining(data: np.ndarray, scale: int) -> np.ndarray:
    """
    时间序列的粗粒化处理
    
    Args:
        data: 输入时间序列
        scale: 尺度因子
        
    Returns:
        粗粒化后的时间序列
    """
    N = len(data)
    # 计算粗粒化后的序列长度
    coarse_length = N - scale + 1
    
    # 初始化粗粒化序列
    coarse_data = np.zeros(coarse_length)
    
    # 使用滑动窗口计算平均值
    for j in range(coarse_length):
        coarse_data[j] = np.mean(data[j:j+scale])
    
    return coarse_data

def calculate_distance_matrix(data: np.ndarray) -> np.ndarray:
    """
    计算距离矩阵
    
    Args:
        data: 输入时间序列
        
    Returns:
        距离矩阵
    """
    M = len(data)
    dist = np.zeros((M-1, M-1))
    
    for i in range(M-1):
        for j in range(M-1):
            dist[i, j] = abs(data[i+1] - data[j+1])
    
    return dist

def calculate_DE(data: np.ndarray, m: int = 2, epsilon: float = 0.2) -> float:
    """
    计算多样性熵
    
    Args:
        data: 输入时间序列
        m: 嵌入维度
        epsilon: 阈值参数
        
    Returns:
        DE值
    """
    # 计算距离矩阵
    dist = calculate_distance_matrix(data)
    
    # 计算相似度矩阵
    similarity = np.exp(-dist / epsilon)
    
    # 计算相对密度
    density = np.sum(similarity, axis=1) / (len(data) - 1)
    
    # 计算DE
    de = -np.sum(density * np.log(density)) / (len(data) - 1)
    
    return de

def calculate_MDE(data: np.ndarray, scale_range: int = 20, m: int = 2, epsilon: float = 0.2) -> Tuple[List[float], List[int]]:
    """
    计算多尺度多样性熵
    
    Args:
        data: 输入时间序列
        scale_range: 最大尺度范围
        m: 嵌入维度
        epsilon: 阈值参数
        
    Returns:
        mde_values: 不同尺度下的DE值列表
        scales: 尺度列表
    """
    mde_values = []
    scales = list(range(1, scale_range + 1))
    
    for scale in scales:
        # 1. 粗粒化处理
        coarse_data = coarse_graining(data, scale)
        
        # 2. 计算该尺度下的DE值
        de_value = calculate_DE(coarse_data, m, epsilon)
        mde_values.append(de_value)
    
    return mde_values, scales

def plot_MDE(scales: List[int], mde_values: List[float], title: str = "Multiscale Diversity Entropy"):
    """
    绘制MDE曲线
    
    Args:
        scales: 尺度列表
        mde_values: DE值列表
        title: 图表标题
    """
    plt.figure(figsize=(10, 6))
    plt.plot(scales, mde_values, 'b-o', linewidth=2, markersize=8)
    plt.xlabel('Scale Factor')
    plt.ylabel('Diversity Entropy')
    plt.title(title)
    plt.grid(True)
    plt.show()
    plt.savefig('MDE.png')

def example_usage():
    """
    示例使用
    """
    # 生成示例数据
    np.random.seed(42)
    n_samples = 1000
    t = np.linspace(0, 10, n_samples)
    # 生成一个包含多个频率成分的信号
    data = np.sin(2*np.pi*0.5*t) + 0.5*np.sin(2*np.pi*1*t) + np.random.normal(0, 0.1, n_samples)
    
    # 计算MDE
    mde_values, scales = calculate_MDE(data, scale_range=20, m=2, epsilon=0.2)
    
    # 绘制结果
    plot_MDE(scales, mde_values)
    
    # 打印结果
    print("Scale factors:", scales)
    print("MDE values:", mde_values)

if __name__ == "__main__":
    example_usage() 