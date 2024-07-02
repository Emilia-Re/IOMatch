from evalUtils import *

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--c', type=str, default='')
    #OpenMatch
    args = parser.parse_args(args=['--c', 'config/openset_cv/openmatch/openmatch_cifar10_600_0.yaml'])
    over_write_args_from_file(args, args.c)
    args.img_save_dir=os.path.join(args.save_dir,args.save_name)
    args.data_dir = 'data'
    dataset_dict = get_dataset(args, args.algorithm, args.dataset, args.num_labels, args.num_classes, args.data_dir,
                               eval_open=True)
    best_net = load_model_at(args,'best')
    eval_dict = evaluate_open(args,best_net, dataset_dict, num_classes=args.num_classes)
    title = 'OPMatch CIFAR10-6 VS CIFAR10-4'
    id_ood_histogram(args=args, id_unk_scores=eval_dict['unk_scores_list'][eval_dict['id_mask']],
                     ood_unk_scores=eval_dict['unk_scores_list'][eval_dict['ood_mask']],
                     title=title)
    # Confusion matrix of open-set classification (OpenMatch-CIFAR-50-200)
    fig = plt.figure()
    f, ax = plt.subplots(figsize=(12, 10))

    cf_mat = eval_dict['o_cfmat_f_hq']
    ax = sns.heatmap(cf_mat, annot=True,cmap='YlGn', linewidth=0.5,fmt="d")
    plt.title('Confusion Matrix of '+title)
    plt.xlabel('Predicted Label')
    plt.ylabel('Ground Truth')

    plt.savefig(os.path.join(args.img_save_dir,'visualize',"Confusion_Matrix"))
    plt.show()

