import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 示例数据
np.random.seed(0)
cifar_scores_tinyimagenet = np.random.normal(0.4, 0.01, 5000)
ood_scores_tinyimagenet = np.random.normal(0.43, 0.01, 5000)
cifar_scores_gaussian = np.random.normal(0.4, 0.01, 5000)
ood_scores_gaussian = np.random.normal(0.8, 0.01, 5000)

# 绘制直方图
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# TinyImageNet 图
sns.histplot(cifar_scores_tinyimagenet,kde=False, color='skyblue', ax=axes[0], label='CIFAR')
sns.histplot(ood_scores_tinyimagenet, bins=50, kde=False, color='sandybrown', ax=axes[0], label='OOD')
axes[0].axvline(x=0.42, color='green', linestyle='--')
axes[0].set_title('TinyImageNet')
axes[0].set_xlabel('OOD score')
axes[0].set_ylabel('#Sample')
axes[0].legend()#

# Gaussian Noise 图
sns.histplot(cifar_scores_gaussian, bins=50, kde=False, color='skyblue', ax=axes[1], label='CIFAR')
sns.histplot(ood_scores_gaussian, bins=50, kde=False, color='sandybrown', ax=axes[1], label='OOD')
axes[1].axvline(x=0.42, color='green', linestyle='--')
axes[1].set_title('Gaussian Noise')
axes[1].set_xlabel('OOD score')
axes[1].set_ylabel('#Sample')
axes[1].legend()

plt.tight_layout()
plt.show()
