import numpy as np
from scipy.stats import spearmanr
import pandas as pd

# 假设你有多个对logit和OOD评分数据，每个都是长度为6的向量
logits = np.random.rand(10, 6)  # 示例：10对数据，每对logit向量长度为6
ood_scores = np.random.rand(10, 6)  # 示例：10对数据，每对OOD评分向量长度为6

num_pairs = logits.shape[0]  # 数据对的数量
vector_length = logits.shape[1]  # 每个向量的长度

# 创建一个相关性矩阵来存储每对logit和OOD评分向量的Spearman相关性
correlation_matrix = np.zeros(num_pairs)
p_value_matrix = np.zeros(num_pairs)

# 计算每对logit和OOD评分向量之间的Spearman相关性
for i in range(num_pairs):
    correlation, p_value = spearmanr(logits[i], ood_scores[i])
    correlation_matrix[i] = correlation
    p_value_matrix[i] = p_value

# 输出结果
print("Spearman correlation for each pair:")
print(pd.DataFrame({
    'Pair': [f'Pair_{i+1}' for i in range(num_pairs)],
    'Spearman correlation': correlation_matrix,
    'P-value': p_value_matrix
}))