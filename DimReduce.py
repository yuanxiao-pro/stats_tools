import numpy as np
import matplotlib
matplotlib.use('Agg')  # 切换到无 GUI 后端
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs, make_moons
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap.umap_ as umap

# 设置随机种子保证可复现性
np.random.seed(42)

# 生成高维数据（3类，20维）
X, y = make_blobs(n_samples=300, n_features=20, centers=3, cluster_std=2.0)

# 添加非线性结构（双月形数据）
X_moons, y_moons = make_moons(n_samples=300, noise=0.05)

# 数据标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_moons_scaled = scaler.fit_transform(X_moons)

def apply_pca(X, n_components=2):
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)
    print(f"PCA解释方差比: {pca.explained_variance_ratio_}")
    return X_pca

X_pca = apply_pca(X_scaled)
X_moons_pca = apply_pca(X_moons_scaled)

def apply_tsne(X, n_components=2, perplexity=30):
    """
    使用t-SNE进行降维, 小数据集可视化
    :param X: 输入数据
    :param n_components: 降维后的维度
    :param perplexity: t-SNE参数 参数敏感
    :return: 降维后的数据
    仅适合可视化, 降维结果不能直接用于下游任务
    """
    tsne = TSNE(n_components=n_components, perplexity=perplexity, random_state=42)
    X_tsne = tsne.fit_transform(X)
    return X_tsne

X_tsne = apply_tsne(X_scaled)
X_moons_tsne = apply_tsne(X_moons_scaled, perplexity=30)  # 简单结构可用更小perplexity

def apply_umap(X, n_components=2, n_neighbors=15, min_dist=0.1):
    """
    使用UMAP进行降维, 适合大规模数据可视化/特征提取
    :param X: 输入数据
    :param n_components: 降维后的维度
    :param n_neighbors: UMAP参数 邻居数,n_neighbors 控制局部/全局结构保留
    :param min_dist: UMAP参数 最小距离,min_dist 控制样本点分布
    :return: 降维后的数据
    适合可视化, 降维结果可直接用于下游任务
    """
    reducer = umap.UMAP(n_components=n_components, 
                       n_neighbors=n_neighbors, 
                       min_dist=min_dist, 
                       random_state=42)
    X_umap = reducer.fit_transform(X)
    return X_umap

X_umap = apply_umap(X_scaled)
X_moons_umap = apply_umap(X_moons_scaled, n_neighbors=10, min_dist=0.05)

def plot_results(X_orig, X_transformed, y, title, method_name):
    plt.figure(figsize=(12, 5))
    
    # 原始数据前两维（若>2维）
    plt.subplot(121)
    if X_orig.shape[1] > 2:
        plt.scatter(X_orig[:, 0], X_orig[:, 1], c=y, cmap='viridis', alpha=0.6)
        plt.title(f"Original Data (First 2D)")
    else:
        plt.scatter(X_orig[:, 0], X_orig[:, 1], c=y, cmap='viridis')
        plt.title("Original Data")
    
    # 降维结果
    plt.subplot(122)
    plt.scatter(X_transformed[:, 0], X_transformed[:, 1], c=y, cmap='viridis')
    plt.title(f"{method_name}: {title}")
    plt.colorbar()
    plt.show()
    plt.savefig(f"{method_name}_{title}.png")

# 对Blobs数据结果
plot_results(X_scaled, X_pca, y, "PCA on Blobs", "PCA")
plot_results(X_scaled, X_tsne, y, "t-SNE on Blobs", "t-SNE")
plot_results(X_scaled, X_umap, y, "UMAP on Blobs", "UMAP")

# 对Moons数据结果
plot_results(X_moons_scaled, X_moons_pca, y_moons, "PCA on Moons", "PCA")
plot_results(X_moons_scaled, X_moons_tsne, y_moons, "t-SNE on Moons", "t-SNE")
plot_results(X_moons_scaled, X_moons_umap, y_moons, "UMAP on Moons", "UMAP")