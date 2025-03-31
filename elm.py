import numpy as np
from typing import Tuple
import matplotlib
matplotlib.use('Agg')  # 切换到无 GUI 后端
import matplotlib.pyplot as plt
class ELM:
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        """
        初始化ELM
        
        Args:
            input_size: 输入层神经元数量
            hidden_size: 隐藏层神经元数量
            output_size: 输出层神经元数量
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # 随机初始化输入权重和偏置
        self.input_weights = np.random.uniform(-1, 1, (self.hidden_size, self.input_size))
        self.biases = np.random.uniform(-1, 1, (self.hidden_size, 1))
        
    def sigmoid(self, x: np.ndarray) -> np.ndarray:
        """
        Sigmoid激活函数
        """
        return 1 / (1 + np.exp(-x))
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        训练ELM
        
        Args:
            X: 输入数据，形状为 (n_samples, input_size)
            y: 目标值，形状为 (n_samples, output_size)
        """
        # 计算隐藏层输出
        H = self.sigmoid(np.dot(X, self.input_weights.T) + self.biases.T)
        
        # 计算输出权重（使用伪逆）
        self.output_weights = np.dot(np.linalg.pinv(H), y)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        预测
        
        Args:
            X: 输入数据
            
        Returns:
            预测结果
        """
        # 计算隐藏层输出
        H = self.sigmoid(np.dot(X, self.input_weights.T) + self.biases.T)
        
        # 计算输出
        return np.dot(H, self.output_weights)

def example_usage():
    """
    示例使用
    """
    # 生成示例数据
    np.random.seed(42)
    n_samples = 1000
    X = np.random.randn(n_samples, 2)  # 2维输入
    y = np.sin(X[:, 0]) + np.cos(X[:, 1])  # 目标函数
    
    # 创建和训练ELM
    elm = ELM(input_size=2, hidden_size=10, output_size=1)
    elm.fit(X, y.reshape(-1, 1))
    
    # 预测
    y_pred = elm.predict(X)
    
    # 计算误差
    mse = np.mean((y - y_pred.flatten()) ** 2)
    print(f"均方误差: {mse:.6f}")
    
    
    
    plt.figure(figsize=(10, 6))
    plt.scatter(X[:, 0], y, c='blue', alpha=0.5, label='target')
    plt.scatter(X[:, 0], y_pred, c='red', alpha=0.5, label='prediction')
    plt.xlabel('X1')
    plt.ylabel('Y')
    plt.title('ELM')
    plt.legend()
    plt.show()
    plt.savefig('elm.png')

if __name__ == "__main__":
    example_usage() 