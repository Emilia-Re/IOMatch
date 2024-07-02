import os.path

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def id_ood_histogram(id_unk_scores= np.random.normal(0.4, 0.01, 5000),ood_unk_scores= np.random.normal(0.43, 0.01, 5000)):
    np.random.seed(0)
    # 绘制直方图
    fig, axes = plt.subplots(1, 1, figsize=(10, 6))
    # TinyImageNet 图
    sns.histplot(id_unk_scores, kde=False, color='skyblue', ax=axes, label='ID')
    sns.histplot(ood_unk_scores, bins=50, kde=False, color='sandybrown', ax=axes, label='OOD')
    axes.axvline(x=0.42, color='green', linestyle='--')
    axes.set_title('TinyImageNet')
    axes.set_xlabel('OOD score')
    axes.set_ylabel(' Num Samples')
    axes.legend()  #
    plt.tight_layout()
    plt.savefig(os.path.join("dir",'test_img'))
    plt.show()


# 示例数据
if __name__=='__main__':
    a=np.float32(1.2)
    print(a)
    print(f'a:{a:.2f}')