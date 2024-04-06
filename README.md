# IOMatch for Open-Set Semi-Supervised Learning

## Introduction

This is the official repository for our **ICCV 2023** paper:

> **IOMatch: Simplifying Open-Set Semi-Supervised Learning with Joint Inliers and Outliers Utilization**</br>
> Zekun Li, Lei Qi, Yinghuan Shi*, Yang Gao</br>

[[`Paper`](https://arxiv.org/abs/2308.13168)] [[`Poster`]](./pubs/Poster.pdf) [[`Slides`]](./pubs/Slides.pdf) [[`Models and Logs`](https://drive.google.com/drive/folders/1pLU6tqxMls55CBRvCgZmDBfHLXm7jGMv?usp=sharing)] [[`BibTeX`](#citation)]

## Preparation

### Required Packages

We suggest first creating a conda environment:

```sh
conda create --name iomatch python=3.8
```

then use pip to install required packages:

```sh
pip install -r requirements.txt
```

### Datasets

Please put the datasets in the ``./data`` folder (or create soft links) as follows:
```
IOMatch
├── config
    └── ...
├── data
    ├── cifar10
        └── cifar-10-batches-py
    └── cifar100
        └── cifar-100-python
    └── imagenet30
        └── filelist
        └── one_class_test
        └── one_class_train
    └── ood_data
├── semilearn
    └── ...
└── ...  
```

The data of ImageNet-30 can be downloaded in [one_class_train](https://drive.google.com/file/d/1B5c39Fc3haOPzlehzmpTLz6xLtGyKEy4/view) and [one_class_test](https://drive.google.com/file/d/13xzVuQMEhSnBRZr-YaaO08coLU2dxAUq/view).

The out-of-dataset testing data for extended open-set evaluation can be downloaded in [this link](https://drive.google.com/drive/folders/1IjDLYfpfsMVuzf_NmqQPoHDH0KAd94gn?usp=sharing).

## Warning
跑困难组实验前一定要把缓存的labeld_idx.npy缓存文件删除了。（先跑完困难组在跑简单组同理，也要把缓存的文件删了）
 该文件是每次实验中选取的标记数据。
## Usage

We implement [IOMatch](./semilearn/algorithms/iomatch/iomatch.py) using the codebase of [USB](https://github.com/microsoft/Semi-supervised-learning).



### Training

Here is an example to train IOMatch on CIFAR-100 with the seen/unseen split of "50/50" and 25 labels per seen class (*i.e.*, the task <u>CIFAR-50-1250</u> with 1250 labeled samples in total). 

```sh
# seed = 1
CUDA_VISIBLE_DEVICES=0 python train.py --c config/openset_cv/iomatch/iomatch_cifar100_1250_1.yaml
```

Training IOMatch on other datasets with different OSSL settings can be specified by a config file:
```sh
# CIFAR10, seen/unseen split of 6/4, 25 labels per seen class (CIFAR-6-150), seed = 1  
CUDA_VISIBLE_DEVICES=0 python train.py --c config/openset_cv/iomatch/iomatch_cifar10_150_1.yaml

# CIFAR100, seen/unseen split of 50/50, 4 labels per seen class (CIFAR-50-200), seed = 1  
CUDA_VISIBLE_DEVICES=0 python train.py --c config/openset_cv/iomatch/iomatch_cifar100_200_1.yaml

# CIFAR100, seen/unseen split of 80/20, 4 labels per seen class (CIFAR-80-320), seed = 1    
CUDA_VISIBLE_DEVICES=0 python train.py --c config/openset_cv/iomatch/iomatch_cifar100_320_1.yaml

# ImageNet30, seen/unseen split of 20/10, 1% labeled data (ImageNet-20-p1), seed = 1  
CUDA_VISIBLE_DEVICES=0 python train.py --c config/openset_cv/iomatch/iomatch_in30_p1_1.yaml
```
### Training with semi-supervised learning 
```shell
CUDA_VISIBLE_DEVICES=0 python train.py --c config/openset_cv/fixmatch/fixmatch_cifar100_2000_2.yaml

CUDA_VISIBLE_DEVICES=0 python train.py --c config/openset_cv/iomatch/iomatch_cifar100_2000_1.yaml
```


### Evaluation

After training, the best checkpoints will be saved in ``./saved_models``. The closed-set performance has been reported in the training logs. For the open-set evaluation, please see [``evaluate.ipynb``](./evaluate.ipynb).

# jhy-experiment result
## cifar10开放集半监督学习
easy group:6个动物类当已知类，其他类为未知类

困难组  
已知类：飞机、鹿、猫、汽车、青蛙、船<br>
        未知类：  鸟、马、狗、卡车
	    不同组中每个类的标记样本分别设置为50，200，1000，原始cifar10测试集作为测试集

[//]: # ()
[//]: # (|          | easy | hard |)

[//]: # (|---------------|-------|-------|)

[//]: # (| cifar-10-300  | 0.939666666666666 | 0.9755 |)

[//]: # (| cifar10_1200  | 0.943333333333333 | 0.975333333333333 |)

[//]: # ( | cifar10_6000  |0.951|0.979|)

 **Closed Accuracy on Closed Test Data**:指的是将原始cifar10中test set中的inlier取出，只在inlier上测试其分类精度

**Open Accuracy on Full Test Data**:指的是在原始cifar10的test set中，将ood数据当作K+1类，在整个原始cifar10 test set上进行K+1分类任务的精确度

**cifar10-300-0-easy**:中的“-0-”是指随机种子,300指标记数据一共有300个

|        group        | Closed Accuracy on Closed Test Data  |  Open Accuracy on Full Test Data  |         AUROC         |
|:-------------------:|:------------------------------------:|:---------------------------------:|:---------------------:|
| cifar10-300-0-easy  |                93.95                 |               79.11               |  38.26603333333334%   |
| cifar10-300-0-hard  |                97.55                 |               83.16               |  49.641620833333334%  |
| cifar10-1200-0-easy |                94.33                 |               80.31               |      43.877975%       |
| cifar10-1200-0-hard |                97.53                 |               83.43               |  51.282487499999995%  |
| cifar10-6000-0-easy |                95.10                 |               81.50               |  55.774950000000004%  |
| cifar10-6000-0-hard |                97.90                 |               83.91               |  54.14243333333333%   |

## cifar10半监督学习（未标记数据中不含OOD）

cifar10数据集上的半监督学习，未标记数据中不含有噪声数据,相当于开放集半监督学习的上界

通过之前的实验可知，cifar10上的6分类任务，区分6个动物类属于相对较困难任务

| group         | acc               |
|---------------|-------------------|
| cifar10-24-0  | 40.63333333333333 |
| cifar10-150-0 | 93.33333333333333 |
| cifar10-300-0 | 94.48333333333333 |

## cifar10全监督学习
cifar10数据集上的全监督学习，利用99%的原始cifar10数据作为标记数据，其他数据作为未标记数据，作为半监督学习的上界

| group         | acc               |
|---------------|-------------------|
| cifar10-29700 | 96.53333333333334 |

# cifar100实验

## cifar100开放集半监督学习 


在开放环境中每类50个标记数据

cifar100-2500-0代表在cifar100数据集上，一共利用2500个标记数据，随机种子为0。

在开放环境中每类50个标记数据

| group           | acc   |
|-----------------|-------|
| cifar100-2500-0(50分类) | 71.96 |


在开放环境中每类25个标记数据

| group                 | acc |
|-----------------------|-----|
| cifar100-1250-0(50分类) | 69.16  |




## cifar100半监督学习（未标记数据不含OOD）
每个类别使用50个标记数据
cifar100-2500-0指在cifar100数据集上实验，共使用2500个标记数据（50类分类）,种子为0

| group                 | acc                 |
|-----------------------|---------------------|
| cifar100-2500-0(50分类) | 75.36               |
| cifar100-2750-0（55分类） | 76.16363636363637   |
| cifar100-4000-0（80分类） | 69.925              |

## cifar100全监督学习
fixmatch方法，每个类别使用99%的数据（每类495个）作为标记数据，其余数据（每类5个）作为未标记数据，未标记数据中不含噪声数据

| group                   | acc               |
|-------------------------|-------------------|
| cifar100-24750-0 (50分类) | 84.02             |
| cifar100-27225-0 (55分类) | 83.96363636363636 |
| cifar100-39600-0 (80分类) | 80.5              |

# ImageNet30实验
原始ImageNet30数据集中有30个类，训练集中每个类都有1300个标记数据。

对类别名称按照字母顺序排序，前20个类当作已知类，后10个类当作未知类。

backbone不同与cifar数据集，使用的是**resnet18**



in-1%表示利用imagenet30的1%的数据作为标记数据，其余数据作为未标记数据，在开放集半监督学习中未标记数据中含OOD，且OOD数据为
来自那其他10个类。

### ImageNet30开放集半监督学习

| group | acc   |auroc|
|-------|-------|---|
| in-1% | 70.05 |  69.986   |
| in-5% | 79.8  |   75.14445 |

### ImageNet30半监督学习（未标记数据中不含OOD）

| group | acc  |
|-------|------|
| in-1% | 65.3 |     
| in-5% | 81.7 |     

### ImageNet30全监督学习
使用99%的数据作为标记数据

|group| acc   |
|---|-------|
|in-99%| 94.85 |

## Example Results


### Close-Set Classification Accuracy

#### CIFAR-10, seen/unseen split of 6/4, 4 labels per seen class (CIFAR-6-24)

| CIFAR-6-24 | Seed=0 | Seed=1 | Seed=2 | Mean | Std. |
|:-------------:|:--------:|:--------:|:--------:|:---------:|:---------:|
| FixMatch    | 90.70  | 75.15  | 78.90  | 81.58   | 6.63 |
| OpenMatch   | 42.05  | 48.18  | 40.67  | 43.63   | 3.26 |
| IOMatch     | 89.28  | 87.40  | 92.35  | 89.68   | 2.04 |

#### CIFAR-10, seen/unseen split of 6/4, 25 labels per seen class (CIFAR-6-150)

| CIFAR-6-150 | Seed=0 | Seed=1 | Seed=2 | Mean | Std. |
|:-------------:|:--------:|:--------:|:--------:|:---------:|:---------:|
| FixMatch    | 93.67  | 91.83  | 93.32  | 92.94   | 0.80 |
| OpenMatch   | 65.00  | 64.90  | 68.90  | 66.27   | 1.86 |
| IOMatch     | 94.05  | 93.88  | 93.67  | 93.87   | 0.16 |

#### CIFAR-100, seen/unseen split of 20/80, 4 labels per seen class (CIFAR-20-80)

| CIFAR-20-80 | Seed=0 | Seed=1 | Seed=2 | Mean | Std. |
|:-------------:|:--------:|:--------:|:--------:|:---------:|:---------:|
| FixMatch    | 45.80  | 46.00  | 47.00  | 46.27   | 0.64 |
| OpenMatch | 34.45 | 38.35 | 39.55 | 37.45 | 2.67 |
| IOMatch | 52.85 | 52.20 | 56.15 | 53.73 | 2.12 |

#### CIFAR-100, seen/unseen split of 20/80, 25 labels per seen class (CIFAR-20-500)

| CIFAR-20-500 | Seed=0 | Seed=1 | Seed=2 | Mean | Std. |
|:-------------:|:--------:|:--------:|:--------:|:---------:|:---------:|
| FixMatch | 66.00 | 66.05 | 67.30 | 66.45 | 0.74 |
| OpenMatch | 60.85 | 62.90 | 64.35 | 62.70 | 1.76 |
| IOMatch | 67.00 | 66.35 | 68.50 | 67.28 | 1.10 |

#### CIFAR-100, seen/unseen split of 50/50, 4 labels per seen class (CIFAR-50-200)

| CIFAR-50-200 | Seed=0 | Seed=1 | Seed=2 | Mean | Std. |
|:-------------:|:--------:|:--------:|:--------:|:---------:|:---------:|
| FixMatch | 48.80 | 43.94 | 54.04 | 48.93 | 5.05 |
| OpenMatch | 33.36 | 34.12 | 33.74 | 33.74 | 0.38 |
| IOMatch | 54.10 | 56.14 | 58.68 | 56.31 | 2.29 |

#### CIFAR-100, seen/unseen split of 50/50, 25 labels per seen class (CIFAR-50-1250)

| CIFAR-50-1250 | Seed=0 | Seed=1 | Seed=2 | Mean | Std. |
|:-------------:|:--------:|:--------:|:--------:|:---------:|:---------:|
| FixMatch | 67.82 | 68.92 | 69.58 | 68.77 | 0.89 |
| OpenMatch | 66.44 | 66.04 | 67.10 | 66.53 | 0.54 |
| IOMatch | 69.16 | 69.84 | 70.32 | 69.77 | 0.58 |

#### CIFAR-100, seen/unseen split of 80/20, 4 labels per seen class (CIFAR-80-320)

| CIFAR-80-320 | Seed=0 | Seed=1 | Seed=2 | Mean | Std. |
|:-------------:|:--------:|:--------:|:--------:|:---------:|:---------:|
| FixMatch | 44.45 | 42.36 | 42.36 | 43.06 | 1.21 |
| OpenMatch | 29.23 | 29.18 | 27.21 | 28.54 | 1.15 |
| IOMatch | 51.86 | 49.89 | 50.73 | 50.83 | 0.99 |

#### CIFAR-100, seen/unseen split of 80/20, 25 labels per seen class (CIFAR-80-2000)

| CIFAR-80-2000 | Seed=0 | Seed=1 | Seed=2 | Mean | Std. |
|:-------------:|:--------:|:--------:|:--------:|:---------:|:---------:|
| FixMatch | 65.02 | 64.06 | 64.25 | 64.44 | 0.51 |
| OpenMatch | 62.11 | 61.09 | 60.50 | 61.23 | 0.81 |
| IOMatch | 65.31 | 64.28 | 64.65 | 64.75 | 0.52 |


## Acknowledgments

We sincerely thank the authors of [USB (NeurIPS'22)](https://github.com/microsoft/Semi-supervised-learning) for creating such an awesome SSL benchmark.

We sincerely thank the authors of the following projects for sharing the code of their great works:

- [UASD (AAAI'20)](https://github.com/yanbeic/ssl-class-mismatch)
- [DS3L (ICML'20)](https://github.com/guolz-ml/DS3L)
- [MTC (ECCV'20)](https://github.com/YU1ut/Multi-Task-Curriculum-Framework-for-Open-Set-SSL)
- [T2T (ICCV'21)](https://github.com/huangjk97/T2T)
- [OpenMatch (NeurIPS'21)](https://github.com/VisionLearningGroup/OP_Match)

## License

This project is licensed under the terms of the MIT License.
See the [LICENSE](LICENSE) file for details.

## Citation

```bibtex
@inproceedings{iomatch,
  title={IOMatch: Simplifying Open-Set Semi-Supervised Learning with Joint Inliers and Outliers Utilization},
  author={Li, Zekun and Qi, Lei and Shi, Yinghuan and Gao, Yang},
  booktitle={ICCV},
  year={2023}
}
```
