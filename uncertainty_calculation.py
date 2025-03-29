import numpy as np
from typing import Tuple, List, Union
"""
实际上EU可以从采样N次后验分布后的预测结果中计算得到，但是异方差的AU必须结合损失函数计算
"""
class UncertaintyCalculator:
    def __init__(self):
        """
        初始化不确定性计算器
        """
        pass

    def epistemic_uncertainty(self, predictions: List[np.ndarray]) -> np.ndarray:
        """
        计算认知不确定性（模型不确定性）
        
        Args:
            predictions: 不同模型或不同参数下的预测结果列表
            
        Returns:
            认知不确定性值（每个时间点的方差）
        """
        # 将预测结果转换为numpy数组
        pred_array = np.array(predictions)
        
        # 计算预测的方差
        epistemic = np.var(pred_array, axis=0)
        
        return epistemic

    def aleatoric_uncertainty(self, predictions: np.ndarray, 
                            targets: np.ndarray) -> np.ndarray:
        """
        计算任意不确定性（数据不确定性）
        
        Args:
            predictions: 模型预测值
            targets: 真实值
            
        Returns:
            任意不确定性值（每个时间点的预测误差方差）
        """
        # 计算预测误差
        errors = predictions - targets
        
        # 使用滑动窗口计算每个时间点的任意不确定性
        window_size = 5  # 可以根据需要调整窗口大小
        aleatoric = np.zeros_like(errors)
        for i in range(len(errors)):
            start_idx = max(0, i - window_size // 2)
            end_idx = min(len(errors), i + window_size // 2 + 1)
            aleatoric[i] = np.var(errors[start_idx:end_idx])
        
        return aleatoric

    def total_uncertainty(self, epistemic: np.ndarray, 
                         aleatoric: np.ndarray) -> np.ndarray:
        """
        计算总不确定性
        
        Args:
            epistemic: 认知不确定性（每个时间点的值）
            aleatoric: 任意不确定性（每个时间点的值）
            
        Returns:
            总不确定性值（每个时间点的值）
        """
        return epistemic + aleatoric

def example_usage():
    """
    示例使用
    """
    # 创建一些示例数据
    # 假设我们有3个不同模型的预测结果
    predictions = [
        np.array([1.0, 2.0, 3.0, 4.0, 5.0]),  # 模型1的预测
        np.array([1.2, 2.1, 3.2, 4.1, 5.2]),  # 模型2的预测
        np.array([0.9, 1.9, 2.9, 3.9, 4.9])   # 模型3的预测
    ]
    
    # 真实值
    targets = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    
    # 创建计算器实例
    calculator = UncertaintyCalculator()
    
    # 计算认知不确定性
    epistemic = calculator.epistemic_uncertainty(predictions)
    print("认知不确定性:", epistemic)
    
    # 计算任意不确定性
    aleatoric = calculator.aleatoric_uncertainty(predictions[1], targets)
    print("任意不确定性:", aleatoric)
    
    # 计算总不确定性
    total = calculator.total_uncertainty(epistemic, aleatoric)
    print("总不确定性:", total)

if __name__ == "__main__":
    example_usage() 