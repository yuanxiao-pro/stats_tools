import numpy as np
from typing import Union, Tuple

def histcount(data: np.ndarray, bins: Union[int, np.ndarray] = 10, 
              range: Tuple[float, float] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    计算直方图的计数和边界
    
    Args:
        data: 输入数据数组
        bins: 直方图的箱数或边界数组
        range: 直方图的范围，格式为 (min, max)
        
    Returns:
        counts: 每个箱的计数
        edges: 箱的边界
    """
    # 如果没有指定范围，使用数据的最大最小值
    if range is None:
        range = (np.min(data), np.max(data))
    
    # 计算直方图
    counts, edges = np.histogram(data, bins=bins, range=range)
    
    return counts, edges

def example_usage():
    """
    示例使用
    """
    # 创建示例数据
    np.random.seed(42)
    data = np.random.normal(0, 1, 1000)  # 1000个正态分布随机数
    
    # 使用默认参数
    counts, edges = histcount(data)
    print("默认箱数(10)的结果:")
    print("计数:", counts)
    print("边界:", edges)
    
    # 使用自定义箱数
    counts, edges = histcount(data, bins=20)
    print("\n自定义箱数(20)的结果:")
    print("计数:", counts)
    print("边界:", edges)
    
    # 使用自定义范围
    counts, edges = histcount(data, bins=10, range=(-2, 2))
    print("\n自定义范围(-2到2)的结果:")
    print("计数:", counts)
    print("边界:", edges)
    
    # 使用自定义边界
    custom_edges = np.array([-3, -2, -1, 0, 1, 2, 3])
    counts, edges = histcount(data, bins=custom_edges)
    print("\n使用自定义边界的结果:")
    print("计数:", counts)
    print("边界:", edges)

if __name__ == "__main__":
    example_usage() 