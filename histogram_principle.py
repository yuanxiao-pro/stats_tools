import numpy as np
import matplotlib.pyplot as plt

def calculate_histogram_manual(data: np.ndarray, bins: int = 10) -> tuple:
    """
    手动计算直方图
    
    Args:
        data: 输入数据
        bins: 箱数
        
    Returns:
        counts: 每个箱的计数
        edges: 箱的边界
    """
    # 1. 确定数据范围
    min_value = np.min(data)
    max_value = np.max(data)
    
    # 2. 计算箱宽和边界
    bin_width = (max_value - min_value) / bins
    edges = np.linspace(min_value, max_value, bins + 1)
    
    # 3. 初始化计数数组
    counts = np.zeros(bins)
    
    # 4. 对每个数据点进行计数
    for value in data:
        for i in range(bins):
            if edges[i] <= value < edges[i + 1]:
                counts[i] += 1
                break
    
    return counts, edges

def plot_histogram_comparison(data: np.ndarray, bins: int = 10):
    """
    比较手动计算和numpy计算的直方图
    
    Args:
        data: 输入数据
        bins: 箱数
    """
    # 手动计算直方图
    counts_manual, edges_manual = calculate_histogram_manual(data, bins)
    
    # numpy计算直方图
    counts_numpy, edges_numpy = np.histogram(data, bins=bins)
    
    # 创建子图
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
    
    # 绘制手动计算的直方图
    ax1.bar(edges_manual[:-1], counts_manual, width=np.diff(edges_manual)[0],
            alpha=0.7, label='Manual Calculation')
    ax1.set_title('Manual Histogram Calculation')
    ax1.set_xlabel('Value')
    ax1.set_ylabel('Count')
    ax1.legend()
    
    # 绘制numpy计算的直方图
    ax2.bar(edges_numpy[:-1], counts_numpy, width=np.diff(edges_numpy)[0],
            alpha=0.7, label='Numpy Calculation')
    ax2.set_title('Numpy Histogram Calculation')
    ax2.set_xlabel('Value')
    ax2.set_ylabel('Count')
    ax2.legend()
    
    plt.tight_layout()
    plt.show()
    
    # 打印计算结果比较
    print("手动计算的结果:")
    print("Counts:", counts_manual)
    print("Edges:", edges_manual)
    print("\nNumpy计算的结果:")
    print("Counts:", counts_numpy)
    print("Edges:", edges_numpy)

def example_usage():
    """
    示例使用
    """
    # 创建示例数据
    np.random.seed(42)
    data = np.random.normal(0, 1, 1000)  # 1000个正态分布随机数
    
    # 比较直方图计算结果
    plot_histogram_comparison(data, bins=20)

if __name__ == "__main__":
    example_usage() 