import numpy as np
from collections import defaultdict

# 使用 lambda 函数创建默认值为空的 numpy 数组
class_features = defaultdict(lambda: np.array([]))

# 示例数据：特征列表和对应的标签列表
feature_list = [np.array([1.0, 2.0]), np.array([2.0, 3.0]), np.array([1.5, 2.5]), np.array([3.0, 4.0])]
label_list = [0, 0, 1, 1]

# 将每个特征添加到对应类别的数组中
for feature, label in zip(feature_list, label_list):
    if class_features[label].size == 0:
        class_features[label] = feature
    else:
        class_features[label] = np.vstack((class_features[label], feature))


# 计算每个类别的特征平均值
class_mean_features = {}
for label, features in class_features.items():
    class_mean_features[label] = np.mean(features, axis=0)

# 打印每个类别的特征平均值
for label, mean_feature in class_mean_features.items():
    print(f"Class {label}: {mean_feature}")
