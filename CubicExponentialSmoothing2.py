import numpy as np
import matplotlib
matplotlib.use('Agg')  # 切换到无 GUI 后端
import matplotlib.pyplot as plt
from scipy.optimize import minimize
# 使用三次指数平滑模型预测RUL，基于FPT确定退化开始时间
# A novel exponential model for tool remaining useful life prediction的Cubic First Order Exponential Smoothing
class CubicExponentialSmoothing:
    def __init__(self, alpha=0.38, i=3):
        """
        三次指数平滑模型
        :param alpha: 平滑系数 (0 < alpha < 1)
        """
        self.alpha = alpha
        self.i = i
        self.S1 = None  # 一阶平滑值
        self.S2 = None  # 二阶平滑值
        self.S3 = None  # 三阶平滑值 (最终HI)
    
    def fit(self, y, i=3):
        """
        拟合时间序列数据
        :param y: 原始健康指标序列 (含噪声)
        
        """
        n = len(y)
        self.S1 = np.zeros(n)
        self.S2 = np.zeros(n)
        self.S3 = np.zeros(n)
        
        # 初始化
        self.S1[0] = self.S2[0] = self.S3[0] = y[0]
        
        # 递归计算
        for t in range(1, n):
            self.S1[t] = self.alpha * y[t] + (1 - self.alpha) * self.S1[t-1]
            if(self.i >= 2):
                self.S2[t] = self.alpha * self.S1[t] + (1 - self.alpha) * self.S2[t-1]
            if(self.i >= 3):
                self.S3[t] = self.alpha * self.S2[t] + (1 - self.alpha) * self.S3[t-1] # HI
        if(self.i == 1):
            return self.S1
        if(self.i == 2):
            return self.S2
        return self.S3
    
    def predict(self, steps=1):
        """预测未来HI值 (简单扩展最后趋势)"""
        last_S1, last_S2, last_S3 = self.S1[-1], self.S2[-1], self.S3[-1]
        predictions = []
        for _ in range(steps):
            last_S1 = self.alpha * last_S3 + (1 - self.alpha) * last_S1
            last_S2 = self.alpha * last_S1 + (1 - self.alpha) * last_S2
            last_S3 = self.alpha * last_S2 + (1 - self.alpha) * last_S3
            predictions.append(last_S3)
        return predictions

def optimize_alpha(y_true, y_noisy, alpha_range=(0.1, 0.9)):
    """优化平滑系数alpha,找到和真实趋势最接近的alpha"""
    def loss(alpha):
        model = CubicExponentialSmoothing(alpha=alpha)
        y_smooth = model.fit(y_noisy)
        return np.mean((y_true - y_smooth) ** 2)  # MSE
    
    result = minimize(loss, x0=0.5, bounds=[alpha_range])
    return result.x[0]

def second_derivative(y, h=1):
    """
    计算一维数组的二阶导数
    :param y: 输入数组
    :param h: 采样间隔（默认为1）
    :return: 二阶导数数组（长度比输入少2）
    识别退化趋势的加速或减速阶段，发现退化过程中的突变点，评估退化曲线的弯曲程度
    """
    # 核心计算
    d2y = np.zeros(len(y) - 2)
    for i in range(1, len(y)-1):
        d2y[i-1] = (y[i+1] - 2*y[i] + y[i-1]) / h**2
    
    # 使用NumPy向量化优化（更高效）
    d2y_vectorized = (y[2:] - 2*y[1:-1] + y[:-2]) / h**2
    
    return d2y_vectorized  # 返回向量化结果

def cal3sigma(y):
    """计算3sigma控制限"""
    mean = np.mean(y)
    std = np.std(y)
    return mean + 3*std, mean - 3*std

# ================= 示例使用 =================
if __name__ == "__main__":
    # 生成模拟数据 (真实退化趋势 + 噪声)
    np.random.seed(42)
    t = np.linspace(0, 10, 100)
    sdt = np.linspace(0, 10, 98)

    y_true = 0.5 * t ** 1.5  # 真实退化趋势 (非线性)
    y_noisy = y_true + np.random.normal(0, 1, len(t))  # 添加高斯噪声
    # print("y_true", y_true)
    # print("y_noisy", y_noisy)
    
    # 使用默认alpha=0.38平滑,此时过去时刻的权重更大
    ces_1 = CubicExponentialSmoothing(alpha=0.38, i=1)
    hi_smoothed_1 = ces_1.fit(y_noisy)

    ces_2 = CubicExponentialSmoothing(alpha=0.38, i=2)
    hi_smoothed_2 = ces_2.fit(y_noisy)

    ces_3 = CubicExponentialSmoothing(alpha=0.38, i=3)
    hi_smoothed_3 = ces_3.fit(y_noisy)

    # print("hi_smoothed", hi_smoothed)
    sdhi_smoothed = second_derivative(hi_smoothed_3)
    print("sdhi_smoothed", sdhi_smoothed)
    up, down = cal3sigma(sdhi_smoothed)
    print("up", up)
    print("down", down)


    # 参数优化
    # optimal_alpha = optimize_alpha(y_true, y_noisy)
    # print(f"Optimal alpha: {optimal_alpha:.3f}")
    
    # 使用最优alpha重新计算
    # ces_opt = CubicExponentialSmoothing(alpha=optimal_alpha)
    # hi_opt = ces_opt.fit(y_noisy)
    # print("hi_opt", hi_opt)
    
    # 预测未来5个点
    # future_steps = 5
    # future_hi = ces_opt.predict(steps=future_steps)
    
    # 可视化
    plt.figure(figsize=(12, 6))
    plt.plot(t, y_true, 'g-', label='True HI Trend', linewidth=2)
    plt.scatter(t, y_noisy, c='blue', alpha=0.3, label='Noisy Observations', s=20)
    plt.plot(t, hi_smoothed_1, 'r-', label=f'1st Order (α=0.38)', linewidth=1)
    plt.plot(t, hi_smoothed_2, 'b-', label=f'2nd Order (α=0.38)', linewidth=1)
    plt.plot(t, hi_smoothed_3, 'k-', label=f'3rd Order (α=0.38)', linewidth=1)
    plt.plot(sdt, sdhi_smoothed, 'g--', label=f'SDHI (α=0.38)', linewidth=1)

    plt.axvline(x=t[-1], color='gray', linestyle=':')
    plt.xlabel('Time')
    plt.ylabel('Health Index (HI)')
    plt.title('Cubic Exponential Smoothing for RUL Prediction')
    plt.legend()
    plt.grid(True)
    plt.show()
    plt.savefig('CubicExponentialSmoothing2.png')  # 保存到挂载的目录
    
    plt.figure(figsize=(12, 6))
    plt.plot(sdt, sdhi_smoothed, 'g--', label=f'SDHI (α=0.38)', linewidth=1)
    plt.axhline(y=up, color='r', linestyle='--', label='3σ上限')
    plt.axhline(y=down, color='r', linestyle='--', label='3σ下限')

    # 计算异常点
    # 找到超过3σ的点
    outliers = np.where(np.abs(sdhi_smoothed - sdhi_smoothed.mean()) > 3 * sdhi_smoothed.std())[0]
    print("异常点索引:", outliers)
    # 在图中标记异常点
    plt.scatter(sdt[outliers], sdhi_smoothed[outliers], c='red', marker='x', s=100, label='异常点')
    plt.show()
    plt.savefig('3sigma_outliers.png')  # 保存到挂载的目录  
