import os
import sys
from collections import defaultdict

from scipy.stats import spearmanr
from sklearn.metrics.pairwise import cosine_similarity
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
from semilearn.algorithms.openmatch.openmatch import OpenMatchNet
from semilearn.algorithms.iomatch.iomatch import IOMatchNet
import argparse
def testf():
    print("test fucntion")
# parser = argparse.ArgumentParser()
# parser.add_argument('--c', type=str, default='')


import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def id_ood_histogram(args,id_unk_scores,ood_unk_scores,title:str=None):
    # 绘制直方图
    fig, axes = plt.subplots(1, 1, figsize=(10, 6))
    # TinyImageNet 图
    sns.histplot(id_unk_scores,bins=200,kde=False, color='skyblue', ax=axes, label='ID')
    sns.histplot(ood_unk_scores,bins=200, kde=False, color='sandybrown', ax=axes, label='OOD')
    # axes.axvline(x=0.42, color='green', linestyle='--')
    axes.set_title(title)
    axes.set_xlabel('OOD score')
    axes.set_ylabel(' Num Samples')
    axes.legend()  #
    plt.tight_layout()
    if not os.path.exists(os.path.join(args.img_save_dir,'visualize')):
        os.makedirs(os.path.join(args.img_save_dir,'visualize'))
    plt.savefig(os.path.join(args.img_save_dir,"visualize",'histogram_'+title))
    plt.show()


def load_model_at(args,step='best'):
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
    if args.algorithm == 'openmatch':
        net = OpenMatchNet(net, args.num_classes)
    elif args.algorithm == 'iomatch':
        net = IOMatchNet(net, args.num_classes)
    else:
        raise NotImplementedError
    keys = net.load_state_dict(load_state_dict)
    print(f'Model at step {args.step} loaded!')
    if torch.cuda.is_available():
        net.cuda()
    net.eval()

    return net


def evaluate_open(args,net, dataset_dict, num_classes, extended_test=True,):
    seed=0
    torch.manual_seed(seed)
    np.random.seed(seed)
    full_loader = DataLoader(dataset_dict['test']['full'], batch_size=256, drop_last=False, shuffle=False,
                             num_workers=4)
    if extended_test:
        extended_loader = DataLoader(dataset_dict['test']['extended'], batch_size=1024, drop_last=False, shuffle=False,
                                     num_workers=4)#TODO :shuffle时False，为什么相同位置的数据会不一样.

    total_num = 0.0
    y_true_list = []
    y_pred_closed_list = []
    y_pred_ova_list = []
    unk_scores_list=[]
    results = {}
    logit_list=[]
    feat_list=[]
    with torch.no_grad():
        for data in tqdm(full_loader):
            x = data['x_lb']
            y = data['y_lb']

            if isinstance(x, dict):
                x = {k: v.cuda() for k, v in x.items()}
            else:
                x = x.cuda()
            y = y.cuda()

            num_batch = y.shape[0]
            total_num += num_batch

            out = net(x)
            logits, logits_open,feat = out['logits'], out['logits_open'],out['feat']
            pred_closed = logits.data.max(1)[1]

            probs = F.softmax(logits, 1)
            probs_open = F.softmax(logits_open.view(logits_open.size(0), 2, -1), 1)
            tmp_range = torch.arange(0, logits_open.size(0)).long().cuda()
            unk_score = probs_open[tmp_range, 0, pred_closed]
            pred_open = pred_closed.clone()
            pred_open[unk_score > 0.5] = num_classes    #unk_score是预测为ood的概率,num_classes代表ood类别的标签
            feat_list.append(feat.cpu().tolist())
            logit_list.append(logits.cpu().tolist())
            unk_scores_list.extend(unk_score.cpu().tolist())
            y_true_list.extend(y.cpu().tolist())
            y_pred_closed_list.extend(pred_closed.cpu().tolist())
            y_pred_ova_list.extend(pred_open.cpu().tolist())
    feat_list=np.vstack(feat_list)
    logit_list=np.vstack(logit_list)
    results['unk_scores_list']=np.array(unk_scores_list)
    y_true = np.array(y_true_list)
    results['original_gt']=y_true_list

    closed_mask = y_true < num_classes
    open_mask = y_true >= num_classes
    y_true[open_mask] = num_classes
    results['k_plus_1_gt']=y_true
    results['id_mask']=closed_mask
    results['ood_mask']=open_mask

    y_pred_closed = np.array(y_pred_closed_list)
    y_pred_ova = np.array(y_pred_ova_list)

    #Calculate mean embedding of every id class,
    # and calculate the cosine similarity matrix

    class_features=defaultdict(lambda :np.array([]))
    for label,f in zip( y_true,feat_list):
        if class_features[label].size==0:
            class_features[label]=f
        else:
            class_features[label]=np.vstack((class_features[label],f))
    class_mean_features = {}
    for label, features in class_features.items():
        class_mean_features[label] = np.mean(features, axis=0)
    # for label, mean_feature in class_mean_features.items():
    #     print(f"Class {label}: {mean_feature}")
    class_labels = sorted(list(class_mean_features.keys()))
    mean_embeddings = np.array([class_mean_features[label] for label in class_labels])
    cosine_similarity_matrix = cosine_similarity(mean_embeddings)
    results['cos_sim_mat']=cosine_similarity_matrix

    # visualize cosine similarity matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cosine_similarity_matrix, annot=True, cmap='coolwarm', xticklabels=class_labels,
                yticklabels=class_labels)
    plt.title("Cosine Similarity Matrix Heatmap")
    plt.xlabel("Class Label")
    plt.ylabel("Class Label")
    plt.savefig(os.path.join(args.img_save_dir, 'visualize', "Cosine_Similarity_Matrix_Heatmap"))
    plt.show()

    # Closed Accuracy on Closed Test Data
    y_true_closed = y_true[closed_mask]
    y_pred_closed = y_pred_closed[closed_mask]
    closed_acc = accuracy_score(y_true_closed, y_pred_closed)
    closed_cfmat = confusion_matrix(y_true_closed, y_pred_closed, normalize=None)
    results['c_acc_c_p'] = closed_acc  # Closed Accuracy on Closed Test Data
    results['c_cfmat_c_p'] = closed_cfmat

    # Open Accuracy on Full Test Data
    #Open confusion matrix
    open_acc = balanced_accuracy_score(y_true, y_pred_ova)
    open_cfmat = confusion_matrix(y_true, y_pred_ova, normalize=None)
    results['o_acc_f_hq'] = open_acc     # Open Accuracy on Full Test Data
    results['o_cfmat_f_hq'] = open_cfmat


    #AUROC on original  CIFAR test dataset
    #1:ood
    #0:id
    ood_gt=np.zeros(len(y_true_list))
    ood_gt[closed_mask]=0
    ood_gt[open_mask]=1
    ood_pred=unk_scores_list
    cifar_test_auroc=roc_auc_score(y_true=ood_gt,y_score=unk_scores_list)
    results['cifar_test_auroc']=cifar_test_auroc

    #AUROC on original CIFAR dataset and extended dataset

    #extended_test按照顺序为['svhn', 'lsun', 'gaussian', 'uniform']，每个数据集中数据量为10000
    ood_names = ['svhn', 'lsun', 'gaussian', 'uniform']
    unk_scores_list_extd=[]
    probs_list_extd=[]
    logits_list_extd=[]
    if extended_test:
        with torch.no_grad():
            for data in tqdm(extended_loader):
                x = data['x_lb']
                y = data['y_lb']

                if isinstance(x, dict):
                    x = {k: v.cuda() for k, v in x.items()}
                else:
                    x = x.cuda()
                y = y.cuda()

                num_batch = y.shape[0]
                total_num += num_batch

                out = net(x)
                logits, logits_open = out['logits'], out['logits_open']
                pred_closed = logits.data.max(1)[1]

                probs = F.softmax(logits, 1)
                probs_open = F.softmax(logits_open.view(logits_open.size(0), 2, -1), 1)
                tmp_range = torch.arange(0, logits_open.size(0)).long().cuda()
                unk_score = probs_open[tmp_range, 0, pred_closed]
                pred_open = pred_closed.clone()
                pred_open[unk_score > 0.5] = num_classes
                logits_list_extd.append(logits.cpu().tolist())
                probs_list_extd.extend(probs.cpu().tolist())
                unk_scores_list_extd.extend(unk_score.cpu().tolist())
                y_true_list.extend(y.cpu().tolist())
                y_pred_closed_list.extend(pred_closed.cpu().tolist())
                y_pred_ova_list.extend(pred_open.cpu().tolist())
        logits_list_extd=np.vstack(logits_list_extd)
        y_true = np.array(y_true_list)
        unk_scores_list_extd=np.array(unk_scores_list_extd)
        open_mask = y_true >= num_classes
        y_true[open_mask] = num_classes
        y_pred_ova = np.array(y_pred_ova_list)

        #bulid visulzation and calculate AUROC on extended dataset
        auroc_other={}
        for i,ood_name in enumerate(ood_names):
            id_ood_histogram(args=args, id_unk_scores=results['unk_scores_list'][results['id_mask']],
                         ood_unk_scores=unk_scores_list_extd[10000*(i):10000*(i+1)],
                         title=args.algorithm+' CIFAR10 VS '+ood_name)
            #AUROC against other dataset
            pred=np.hstack((np.array(unk_scores_list)[closed_mask],unk_scores_list_extd[10000*(i):10000*(i+1)]))
            gt=np.zeros(len(pred))
            gt[-10000:]=1
            auroc_other[ood_name]=roc_auc_score(y_true=gt,y_score=pred)
        print(f"#############################################################\n")
        for i,ood_name in enumerate(ood_names):
            print(f"AUROC :CIFAR10-6 against {ood_name}:{auroc_other[ood_name]*100:.2f} ")



        # Open Accuracy on Extended Test Data
        open_acc = balanced_accuracy_score(y_true, y_pred_ova)
        open_cfmat = confusion_matrix(y_true, y_pred_ova, normalize='true')
        results['o_acc_e_hq'] = open_acc
        results['o_cfmat_e_hq'] = open_cfmat

        # calculate spearman similarity between confusion matrix and cosine similarity matrix

        # normalize confusion matrix
        cf_mat = np.array(open_cfmat)
        min = cf_mat.min(axis=1, keepdims=True)
        max = cf_mat.max(axis=1, keepdims=True)
        normalized_cf_mat = (cf_mat - min) / (max - min)
        # scale to [-1,1]
        normalized_cf_mat = normalized_cf_mat * 2 - 1

        # calculate spearman similarity
        cf_mat_flatten = normalized_cf_mat.flatten()
        cosine_similarity_matrix_flatten = cosine_similarity_matrix.flatten()
        spearman_corr,spearman_p=spearmanr(cf_mat_flatten,cosine_similarity_matrix_flatten)

    print(f"#############################################################\n"
          f"Spearman correaltion:{spearman_corr}   p-value:{spearman_p}\n"
          f" AUROC on original test dataset: {results['cifar_test_auroc'] * 100:.2f}\n"
          f" Closed Accuracy on Closed Test Data: {results['c_acc_c_p'] * 100:.2f}\n"
          f" Open Accuracy on Full Test Data:     {results['o_acc_f_hq'] * 100:.2f}\n"
          f" Open Accuracy on Extended Test Data: {results['o_acc_e_hq'] * 100:.2f}\n"
          f"#############################################################\n"
          )

    return results




def evaluate_io(args, net, dataset_dict, extended_test=True):
    """
    evaluation function for open-set SSL setting
    """

    full_loader = DataLoader(dataset_dict['test']['full'], batch_size=256, drop_last=False, shuffle=False, num_workers=4)
    if extended_test:
        extended_loader = DataLoader(dataset_dict['test']['extended'], batch_size=1024, drop_last=False, shuffle=False, num_workers=4)

    total_num = 0.0
    y_true_list = []
    p_list = []
    pred_p_list = []
    pred_hat_q_list = []
    pred_q_list = []
    pred_hat_p_list = []

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

            # predictions hat_q from (closed-set + multi-binary) classifiers
            r = F.softmax(logits_mb.view(logits_mb.size(0), 2, -1), 1)
            tmp_range = torch.arange(0, logits_mb.size(0)).long().cuda()
            hat_q = torch.zeros((num_batch, args.num_classes + 1)).cuda()
            o_neg = r[tmp_range, 0, :]
            o_pos = r[tmp_range, 1, :]
            hat_q[:, :args.num_classes] = p * o_pos
            hat_q[:, args.num_classes] = torch.sum(p * o_neg, 1)
            pred_hat_q = hat_q.data.max(1)[1]
            pred_hat_q_list.extend(pred_hat_q.cpu().tolist())

            # predictions q of open-set classifier
            q = F.softmax(logits_open, 1)
            pred_q = q.data.max(1)[1]
            pred_q_list.extend(pred_q.cpu().tolist())

            # prediction hat_p of open-set classifier
            hat_p = q[:, :args.num_classes] / q[:, :args.num_classes].sum(1).unsqueeze(1)
            pred_hat_p = hat_p.data.max(1)[1]
            pred_hat_p_list.extend(pred_hat_p.cpu().tolist())

        y_true = np.array(y_true_list)
        closed_mask = y_true < args.num_classes
        open_mask = y_true >= args.num_classes
        y_true[open_mask] = args.num_classes


        pred_p = np.array(pred_p_list)
        pred_hat_p = np.array(pred_hat_p_list)
        pred_q = np.array(pred_q_list)
        pred_hat_q = np.array(pred_hat_q_list)

        # closed accuracy of p / hat_p on closed test data
        c_acc_c_p = accuracy_score(y_true[closed_mask], pred_p[closed_mask])
        c_acc_c_hp = accuracy_score(y_true[closed_mask], pred_hat_p[closed_mask])
        c_cfmat_c_p = confusion_matrix(y_true[closed_mask], pred_p[closed_mask], normalize='true')
        c_cfmat_c_hp = confusion_matrix(y_true[closed_mask], pred_hat_p[closed_mask], normalize='true')
        np.set_printoptions(precision=3, suppress=True)

        # open accuracy of q / hat_q on full test data
        o_acc_f_q = balanced_accuracy_score(y_true, pred_q)
        o_acc_f_hq = balanced_accuracy_score(y_true, pred_hat_q)
        o_cfmat_f_q = confusion_matrix(y_true, pred_q, normalize='true')
        o_cfmat_f_hq = confusion_matrix(y_true, pred_hat_q, normalize='true')





        o_acc_e_q = o_acc_e_hq = 0
        o_cfmat_e_q = None
        o_cfmat_e_hq = None

        if extended_test:
            unk_scores = []
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
                hat_q[:, :args.num_classes] = p * o_pos
                hat_q[:, args.num_classes] = torch.sum(p * o_neg, 1)
                pred_hat_q = hat_q.data.max(1)[1]
                pred_hat_q_list.extend(pred_hat_q.cpu().tolist())

                # predictions q of open-set classifier
                q = F.softmax(logits_open, 1)
                pred_q = q.data.max(1)[1]
                pred_q_list.extend(pred_q.cpu().tolist())

                # prediction hat_p of open-set classifier
                hat_p = q[:, :args.num_classes] / q[:, :args.num_classes].sum(1).unsqueeze(1)
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
              f"#############################################################\n"
            )

        return eval_dict

if __name__=='__main__':
    args = parser.parse_args(args=['--c', 'config/openset_cv/iomatch/iomatch_cifar100_200_1.yaml'])
    over_write_args_from_file(args, args.c)
    args.data_dir = 'data'
    dataset_dict = get_dataset(args, args.algorithm, args.dataset, args.num_labels, args.num_classes, args.data_dir,
                               eval_open=True)
    best_net = load_model_at('best')
    eval_dict = evaluate_io(args, best_net, dataset_dict)
    #OpenMatch
    args = parser.parse_args(args=['--c', 'config/openset_cv/openmatch/openmatch_cifar100_200_1.yaml'])
    over_write_args_from_file(args, args.c)
    args.data_dir = 'data'
    dataset_dict = get_dataset(args, args.algorithm, args.dataset, args.num_labels, args.num_classes, args.data_dir,
                               eval_open=True)
    best_net = load_model_at('best')
    eval_dict = evaluate_open(best_net, dataset_dict, num_classes=args.num_classes)

    # Confusion matrix of open-set classification (OpenMatch-CIFAR-50-200)
    fig = plt.figure()
    f, ax = plt.subplots(figsize=(12, 10))
    cf_mat = eval_dict['o_cfmat_f_hq']
    ax = sns.heatmap(cf_mat, cmap='YlGn', linewidth=0.5)
    plt.show()


