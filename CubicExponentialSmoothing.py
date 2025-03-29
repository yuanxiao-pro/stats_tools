import numpy as np
import matplotlib
matplotlib.use('Agg')  # 切换到无 GUI 后端
import matplotlib.pyplot as plt
from scipy.optimize import minimize

class CubicExponentialSmoothing:
    def __init__(self, signal, alpha=0.3, beta=0.1, gamma=0.05):
        """
        三次指数平滑模型
        :param signal: 输入信号（非稳态时间序列）
        :param alpha: 水平平滑系数
        :param beta: 趋势平滑系数
        :param gamma: 曲率平滑系数
        """
        self.signal = np.array(signal)
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.S = np.zeros_like(signal)  # 水平分量
        self.T = np.zeros_like(signal)  # 线性趋势
        self.C = np.zeros_like(signal)  # 曲率分量
        self.anomaly_points = []        # 突变点位置

    def initialize(self):
        """使用前5%数据初始化模型"""
        init_len = max(5, len(self.signal) // 20)
        self.S[:init_len] = self.signal[:init_len]
        self.T[:init_len] = (self.signal[1:init_len] - self.signal[:init_len-1]).mean()
        self.C[:init_len] = 0

    def fit(self, adaptive=True, threshold=3.0):
        """
        拟合非稳态信号
        :param adaptive: 是否启用自适应参数
        :param threshold: 突变点检测阈值（标准差倍数）
        """
        self.initialize()
        
        for t in range(1, len(self.signal)):
            # 自适应调整参数
            if adaptive and t > 10:
                window = self.signal[max(0,t-5):t]
                local_var = np.var(window)
                self.alpha = np.clip(1 - 1/(1+local_var), 0.1, 0.9)
            
            # 核心递归公式
            prev_S, prev_T, prev_C = self.S[t-1], self.T[t-1], self.C[t-1]
            self.S[t] = self.alpha * self.signal[t] + (1-self.alpha) * (prev_S + prev_T + 0.5*prev_C)
            self.T[t] = self.beta * (self.S[t] - prev_S) + (1-self.beta) * prev_T
            self.C[t] = self.gamma * (self.T[t] - prev_T) + (1-self.gamma) * prev_C
            
            # 突变点检测
            if t > 20:
                C_mean = np.mean(self.C[:t])
                C_std = np.std(self.C[:t])
                if abs(self.C[t] - C_mean) > threshold * C_std:
                    self.anomaly_points.append(t)
                    if adaptive:  # 遇到突变时重置参数
                        self.alpha = min(0.8, self.alpha + 0.1)

    def predict(self, steps=1):
        """预测未来值"""
        last_S, last_T, last_C = self.S[-1], self.T[-1], self.C[-1]
        return [last_S + h*last_T + 0.5*h**2*last_C for h in range(1, steps+1)]

    def plot_components(self):
        """可视化分解结果"""
        plt.figure(figsize=(12, 8))
        
        plt.subplot(411)
        plt.title("Original Signal vs Smoothed")
        plt.plot(self.signal, label='Original')
        plt.plot(self.S, 'r--', label='Level')
        plt.legend()
        
        plt.subplot(412)
        plt.title("Trend Component")
        plt.plot(self.T, 'g-')
        
        plt.subplot(413)
        plt.title("Curvature Component") # 曲率
        plt.plot(self.C, 'b-')
        for point in self.anomaly_points:
            plt.axvline(point, color='k', linestyle='--', alpha=0.3)
        
        plt.subplot(414)
        plt.title("Prediction vs Actual")
        pred = self.predict(steps=20)
        extended_range = range(len(self.signal), len(self.signal)+20)
        plt.plot(self.signal[-50:], 'b-')
        plt.plot(extended_range, pred, 'r--')
        
        plt.tight_layout()
        plt.show()
        plt.savefig('CubicExponentialSmoothing.png')  # 保存到挂载的目录


# ================= 示例使用 =================
if __name__ == "__main__":
    # 生成模拟非稳态信号（含趋势突变和曲率变化）
    t = np.linspace(0, 10, 500)
    signal = 0.5*t**2 + 2 * np.sin(3*t) + np.random.normal(0, 1, len(t))
    # 随着t增大，立方项使信号曲率快速增加（加速恶化）
    signal[250:] = signal[250:] + 0.8 * (t[250:] - 5) ** 3  # 添加曲率突变，在信号的后半段（从第250个点开始）人为添加一个三次方突变，模拟设备突然进入加速故障阶段。
    
    # 创建并训练模型
    model = CubicExponentialSmoothing(signal, alpha=0.3, beta=0.1, gamma=0.05)
    model.fit(adaptive=True, threshold=3.0)
    
    # 可视化结果
    model.plot_components()
    
    # 输出突变点位置
    print(f"Detected anomaly points at: {model.anomaly_points}")