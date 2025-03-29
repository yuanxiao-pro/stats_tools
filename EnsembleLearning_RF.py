from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
import numpy as np

def diversity_entropy(probs, base=2):
    """
    计算多样性熵
    :param probs: 类别概率列表（需和为1）
    :param base: 对数底数（默认2，单位为bit）
    """
    probs = np.array(probs)
    probs = probs[probs > 0]  # 避免log(0)
    return -np.sum(probs * np.log(probs) / np.log(base))

# 示例：3个类别的分布
probs = [0.2, 0.3, 0.5]
print(f"Diversity Entropy: {diversity_entropy(probs):.4f} bits")

# 生成数据
X, y = make_classification(n_samples=1000, n_features=20, n_classes=2)

# 训练随机森林
rf = RandomForestClassifier(n_estimators=10)
rf.fit(X, y)

# 计算样本预测类别分布的熵
def ensemble_diversity(model, X):
    predictions = np.array([tree.predict(X) for tree in model.estimators_]) # 获取每个树的预测结果
    entropy_list = []
    for sample_idx in range(X.shape[0]):
        unique, counts = np.unique(predictions[:, sample_idx], return_counts=True) # 获取每个样本的预测类别分布
        probs = counts / counts.sum() # 计算每个类别的概率
        entropy_list.append(diversity_entropy(probs)) # 计算多样性熵
    return np.mean(entropy_list) # 返回多样性熵的平均值 

print(f"Ensemble Diversity: {ensemble_diversity(rf, X):.4f}")