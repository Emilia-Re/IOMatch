import os
import sys
from tqdm import tqdm
import pprint
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import umap

from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score

import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F

sys.path.append('..')
from semilearn.core.utils import get_net_builder, get_dataset, over_write_args_from_file
from semilearn.algorithms.iomatch.iomatch import IOMatchNet

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--c', type=str, default='')


def load_model_at(step='best'):
    args.step = step
    if step == 'best':
        args.load_path = '/'.join(args.load_path.split('/')[1:-1]) + "/model_best.pth"
    else:
        args.load_path = '/'.join(args.load_path.split('/')[:-1]) + "/model_at_{args.step}_step.pth"
    print(args.load_path)
    checkpoint_path = os.path.join(args.load_path)
    checkpoint = torch.load(checkpoint_path)
    load_model = checkpoint['ema_model']
    load_state_dict = {}
    for key, item in load_model.items():
        if key.startswith('module'):
            new_key = '.'.join(key.split('.')[1:])
            load_state_dict[new_key] = item
        else:
            load_state_dict[key] = item
    save_dir = '/'.join(checkpoint_path.split('/')[:-1])
    if step == 'best':
        args.save_dir = os.path.join(save_dir, f"model_best")
    else:
        args.save_dir = os.path.join(save_dir, f"step_{args.step}")
    os.makedirs(args.save_dir, exist_ok=True)
    _net_builder = get_net_builder(args.net, args.net_from_name)
    net = _net_builder(num_classes=args.num_classes)
    # if args.cifar100_pure:
    #     net=
    net = IOMatchNet(net, args.num_classes) #TODO：尝试改成FixMatchnet
    keys = net.load_state_dict(load_state_dict)
    print(f'Model at step {args.step} loaded!')
    if torch.cuda.is_available():
        net.cuda()
    net.eval()

    return net

def evaluate_io(args, net, dataset_dict, extended_test=False):
    """
    evaluation function for open-set SSL setting
    """
    # 原始cifar10的测试集作为测试集
    full_loader = DataLoader(dataset_dict['test']['full'], batch_size=256, drop_last=False, shuffle=False, num_workers=4)
    if extended_test:  #原始cifar10的测试集加上其他ood数据，其他ood数据有svhn，高斯噪声、均值噪声、（还有一种忘了），每种ood数据量都是一样的
        extended_loader = DataLoader(dataset_dict['test']['extended'], batch_size=1024, drop_last=False, shuffle=False, num_workers=4)

    total_num = 0.0
    y_true_list = []#真实标签
    p_list = []
    pred_p_list = []
    pred_hat_q_list = []
    pred_q_list = []
    pred_hat_p_list = []
    ood_scores=[]
    with torch.no_grad():
        for data in tqdm(full_loader):
            x = data['x_lb']
            y = data['y_lb']

            if isinstance(x, dict):
                x = {k: v.cuda() for k, v in x.items()}
            else:
                x = x.cuda()
            y = y.cuda()
            y_true_list.extend(y.cpu().tolist())

            num_batch = y.shape[0]
            total_num += num_batch

            outputs = net(x)
            logits = outputs['logits']
            logits_mb = outputs['logits_mb']
            logits_open = outputs['logits_open']

            # predictions p of closed-set classifier
            p = F.softmax(logits, 1)
            pred_p = p.data.max(1)[1]
            pred_p_list.extend(pred_p.cpu().tolist())

            # predictions hat_q from (closed-set + multi-binary) classifiers   #这是本文iomatch提出的方法
            #hat_q就是文中的q tilder，是K+1的分布，是logits
            r = F.softmax(logits_mb.view(logits_mb.size(0), 2, -1), 1)
            tmp_range = torch.arange(0, logits_mb.size(0)).long().cuda()
            hat_q = torch.zeros((num_batch, args.num_classes + 1)).cuda()
            o_neg = r[tmp_range, 0, :]
            o_pos = r[tmp_range, 1, :]
            hat_q[:, :args.num_classes] = p * o_pos #核心方法，对应论文中fig 2
            hat_q[:, args.num_classes] = torch.sum(p * o_neg, 1)  #OOD 打分，即K+1类的预测值
            ood_scores.extend(hat_q[:, args.num_classes].cpu().tolist())
            pred_hat_q = hat_q.data.max(1)[1]
            pred_hat_q_list.extend(pred_hat_q.cpu().tolist())#预测的为标签

            # predictions q of open-set classifier   #这是论文fig2 中最下边的那个开放集分类头，对应fig4 的psi
            q = F.softmax(logits_open, 1)
            pred_q = q.data.max(1)[1]
            pred_q_list.extend(pred_q.cpu().tolist())

            # prediction hat_p of open-set classifier
            hat_p = q[:, :args.num_classes] / q[:, :args.num_classes].sum(1).unsqueeze(1)#归一化,对于经过psi的K+1分类器，对闭集logit做一个归一化
            pred_hat_p = hat_p.data.max(1)[1]
            pred_hat_p_list.extend(pred_hat_p.cpu().tolist())

        y_true = np.array(y_true_list)
        closed_mask = y_true < args.num_classes
        open_mask = y_true >= args.num_classes
        y_true[open_mask] = args.num_classes#将所有的ood数据标记为args.num_classes，比如6分类任务，inlier标记为1-5，ood标记为6


        pred_p = np.array(pred_p_list)
        pred_hat_p = np.array(pred_hat_p_list)
        pred_q = np.array(pred_q_list)
        pred_hat_q = np.array(pred_hat_q_list)

        # closed accuracy of p / hat_p on closed test data
        c_acc_c_p = accuracy_score(y_true[closed_mask], pred_p[closed_mask])#在test set中，把inlier拿出来，测试模型对inlier分类的精确度
        c_acc_c_hp = accuracy_score(y_true[closed_mask], pred_hat_p[closed_mask])#K+1分类的open set classifer当作闭集分类器使用，同样测试test set中先把inlier选出来后，测试其精确度
        c_cfmat_c_p = confusion_matrix(y_true[closed_mask], pred_p[closed_mask], normalize='true')
        c_cfmat_c_hp = confusion_matrix(y_true[closed_mask], pred_hat_p[closed_mask], normalize='true')
        np.set_printoptions(precision=3, suppress=True)

        # open accuracy of q / hat_q on full test data
        o_acc_f_q = balanced_accuracy_score(y_true, pred_q)#psi分类头得到的，这里的y_true是真实标签不变，ood全标记为args.num_classes,即K+1类的分类任务
        o_acc_f_hq = balanced_accuracy_score(y_true, pred_hat_q)#iomatch方法得到的精确度，对于开机集样本，如果被分到闭集，则认为是分错了
        o_cfmat_f_q = confusion_matrix(y_true, pred_q, normalize='true')
        o_cfmat_f_hq = confusion_matrix(y_true, pred_hat_q, normalize='true')

        o_acc_e_q = o_acc_e_hq = 0
        o_cfmat_e_q = None
        o_cfmat_e_hq = None

        # auroc
        # auroc = roc_auc_score(y_true, y_score)#
        ood_scores = np.array(ood_scores)  # 这里为了方便，ood记为1，id记为0
        gt_id_or_ood = [0 if val in range(args.num_classes) else 1 for val in y_true_list]
        gt_id_or_ood = np.array(gt_id_or_ood)
        auroc = roc_auc_score(gt_id_or_ood, ood_scores)


        if extended_test:
            unk_scores = [] # K+1类的概率
            unk_scores_q = []
            for data in tqdm(extended_loader):
                x = data['x_lb']
                y = data['y_lb']

                if isinstance(x, dict):
                    x = {k: v.cuda() for k, v in x.items()}
                else:
                    x = x.cuda()
                y = y.cuda()
                y_true_list.extend(y.cpu().tolist())

                num_batch = y.shape[0]
                total_num += num_batch

                outputs = net(x)
                logits = outputs['logits']
                logits_mb = outputs['logits_mb']
                logits_open = outputs['logits_open']

                # predictions p of closed-set classifier
                p = F.softmax(logits, 1)
                pred_p = p.data.max(1)[1]
                pred_p_list.extend(pred_p.cpu().tolist())

                # predictions hat_q of (closed-set + multi-binary) classifiers
                r = F.softmax(logits_mb.view(logits_mb.size(0), 2, -1), 1)
                tmp_range = torch.arange(0, logits_mb.size(0)).long().cuda()
                hat_q = torch.zeros((num_batch, args.num_classes + 1)).cuda()
                o_neg = r[tmp_range, 0, :]
                o_pos = r[tmp_range, 1, :]
                unk_score = torch.sum(p * o_neg, 1)
                unk_scores.extend(unk_score.cpu().tolist())
                hat_q[:, :args.num_classes] = p * o_pos
                hat_q[:, args.num_classes] = torch.sum(p * o_neg, 1)
                pred_hat_q = hat_q.data.max(1)[1]
                pred_hat_q_list.extend(pred_hat_q.cpu().tolist())

                # predictions q of open-set classifier
                q = F.softmax(logits_open, 1)
                pred_q = q.data.max(1)[1]
                pred_q_list.extend(pred_q.cpu().tolist())

                # prediction hat_p of open-set classifier
                hat_p = q[:, :args.num_classes] / q[:, :args.num_classes].sum(1).unsqueeze(1)#（不太确定）和上边的方式相比，只是归一化方式不同，上边是softmax归一化，下边是直接除以他们的和进行归一
                pred_hat_p = hat_p.data.max(1)[1]
                pred_hat_p_list.extend(pred_hat_p.cpu().tolist())

            y_true = np.array(y_true_list)
            open_mask = y_true >= args.num_classes
            y_true[open_mask] = args.num_classes

            pred_q = np.array(pred_q_list)
            pred_hat_q = np.array(pred_hat_q_list)

            # open accuracy of q / hat_q on extended test data
            o_acc_e_q = balanced_accuracy_score(y_true, pred_q)
            o_acc_e_hq = balanced_accuracy_score(y_true, pred_hat_q)
            o_cfmat_e_q = confusion_matrix(y_true, pred_q, normalize='true')
            o_cfmat_e_hq = confusion_matrix(y_true, pred_hat_q, normalize='true')


        eval_dict = {'c_acc_c_p': c_acc_c_p, 'c_acc_c_hp': c_acc_c_hp,
                     'o_acc_f_q': o_acc_f_q, 'o_acc_f_hq': o_acc_f_hq,
                     'o_acc_e_q': o_acc_e_q, 'o_acc_e_hq': o_acc_e_hq,
                     'c_cfmat_c_p': c_cfmat_c_p, 'c_cfmat_c_hp': c_cfmat_c_hp,
                     'o_cfmat_f_q': o_cfmat_f_q, 'o_cfmat_f_hq': o_cfmat_f_hq,
                     'o_cfmat_e_q': o_cfmat_e_q, 'o_cfmat_e_hq': o_cfmat_e_hq,
                    }

        print(f"#############################################################\n"
              f" Closed Accuracy on Closed Test Data (p / hp): {c_acc_c_p * 100:.2f} / {c_acc_c_hp * 100:.2f}\n"
              f" Open Accuracy on Full Test Data (q / hq):     {o_acc_f_q * 100:.2f} / {o_acc_f_hq * 100:.2f}\n"
              f" Open Accuracy on Extended Test Data (q / hq): {o_acc_e_q * 100:.2f} / {o_acc_e_hq * 100:.2f}\n"
              f' AUROC:{auroc*100}%\n '
              f"#############################################################\n"
            )

        return eval_dict

#待测的实验设置
config='config/openset_cv/jhy_experiment/iomatch_cifar100_4000_0.yaml'
args = parser.parse_args(args=['--c', config])
over_write_args_from_file(args, args.c)
args.data_dir = 'data'
dataset_dict = get_dataset(args, args.algorithm, args.dataset, args.num_labels, args.num_classes, args.data_dir, eval_open=True)
best_net = load_model_at('best')
eval_dict = evaluate_io(args, best_net, dataset_dict)
#默认设置     cifar100_2000   seen/unseen split of 80/20, 25 labels per seen class


#下边这个暂时用不到
# args = parser.parse_args(args=['--c', 'config/openset_cv/iomatch/iomatch_cifar100_1250_1.yaml'])
# over_write_args_from_file(args, args.c)
# args.data_dir = 'data'
# dataset_dict = get_dataset(args, args.algorithm, args.dataset, args.num_labels, args.num_classes, args.data_dir, eval_open=True)
# best_net = load_model_at('best')
# eval_dict = evaluate_io(args, best_net, dataset_dict)


# Confusion matrix of closed-set classification
fig = plt.figure()
f, ax = plt.subplots(figsize=(12,10))
cf_mat = eval_dict['c_cfmat_c_p']
ax = sns.heatmap(cf_mat, cmap='YlGn', linewidth=0.5)
plt.show()



# Confusion matrix of open-set classification
fig = plt.figure()
f, ax = plt.subplots(figsize=(12,10))
cf_mat = eval_dict['o_cfmat_f_q']
ax = sns.heatmap(cf_mat, cmap='YlGn', linewidth=0.5)
plt.show()