import argparse
import torch
import pickle
import os, copy
from dataset.dataloader import get_dataloader, get_transform
from dataset.dataset import SingleDomainData, SingleClassData
from model.model2_photo import MutiClassifier_eve, MutiClassifier_, resnet18_fast, resnet18_fast_origin,resnet50_fast, ConvNet, DomainIndicator
from optimizer.optimizer import get_optimizer, get_scheduler
from loss.OVALoss import OVALoss
from train.test import eval
import torchvision
from torchvision.transforms import v2
from util.log import log, save_data
from torch.nn import functional as F
from torch.utils.data import DataLoader
from util.ROC import generate_OSCR
from util.util import ForeverDataIterator, ConnectedDataIterator, split_classes
import random
import numpy as np
def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True
setup_seed(47)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='PACS')
    parser.add_argument('--source-domain', nargs='+', default=['photo', 'art_painting', 'sketch'])
    parser.add_argument('--target-domain', nargs='+', default=['cartoon'])
    parser.add_argument('--known-classes', nargs='+', default=['elephant', 'horse', 'giraffe', 'dog', 'guitar', 'house',])
    parser.add_argument('--unknown-classes', nargs='+', default=['person'])
    parser.add_argument('--no-crossval', action='store_true')
    parser.add_argument('--gpu', default='3')
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--net-name', default='resnet18')
    parser.add_argument('--optimize-method', default="SGD")
    parser.add_argument('--schedule-method', default='StepLR')
    parser.add_argument('--num-epoch', type=int, default=10000)
    parser.add_argument('--eval-step', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.0002) 
    parser.add_argument('--meta-lr', type=float, default=0.002)
    parser.add_argument('--nesterov', action='store_true')
    parser.add_argument('--without-bcls', action='store_true')
    parser.add_argument('--share-param', action='store_true')
    parser.add_argument('--ns', type=float, default=4)
    parser.add_argument('--ova_weight', type=float, default=0.5) 
    parser.add_argument('--cls_weight', type=float, default=0.5) 
    parser.add_argument('--meta_weight', type=float, default=0.0001)
    parser.add_argument('--lamda', type=float, default=0.0002)
    parser.add_argument('--meta_test_num', type=int, default=2)
    parser.add_argument('--step', type=float, default=10)
    parser.add_argument('--save-dir', default='checkpoint/code_preparation')
    parser.add_argument('--save-name', default='demo')
    parser.add_argument('--save-best-test', action='store_true')
    parser.add_argument('--save-later', action='store_true')
    parser.add_argument('--num-epoch-before', type=int, default=0)
    parser.add_argument('--path-dataset', type=str, default="/hkfs/work/workspace/scratch/fy2374-workspace/ijcai_folders/Neurips/Homework3-PACS/")
    parser.add_argument('--rb_layer_1', type=int, default=3)
    parser.add_argument('--rb_layer_2', type=int, default=2)
    args = parser.parse_args()
    dataset = args.dataset
    source_domain = sorted(args.source_domain)
    target_domain = sorted(args.target_domain)
    known_classes = sorted(args.known_classes)
    unknown_classes = sorted(args.unknown_classes)
    crossval = not args.no_crossval   
    gpu = args.gpu
    lamda = args.lamda
    batch_size = args.batch_size
    net_name = args.net_name
    path_dataset = args.path_dataset
    optimize_method = args.optimize_method
    schedule_method = args.schedule_method
    num_epoch = args.num_epoch
    eval_step = args.eval_step
    lr = args.lr
    ns = args.ns
    step = args.step
    num_meta_test = args.meta_test_num
    meta_weight = args.meta_weight
    ova_weight = args.ova_weight
    cls_weight = args.cls_weight
    meta_lr = args.meta_lr
    nesterov = args.nesterov
    without_bcls = args.without_bcls
    share_param = args.share_param
    save_dir = args.save_dir
    save_name = args.save_name   
    save_later = args.save_later
    save_best_test = args.save_best_test
    num_epoch_before = args.num_epoch_before
    dr = 0.8
    torch.set_num_threads(1)
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if dataset == 'PACS':
        train_dir = path_dataset + '/PACS_train'
        val_dir = path_dataset + '/PACS_crossval'
        test_dir = [path_dataset + 'PACS_train', path_dataset + '/PACS_crossval']
        sub_batch_size = batch_size // 2
        small_img = False
    log_path = os.path.join(save_dir, 'log', save_name + '_train.txt')
    param_path = os.path.join(save_dir, 'param', save_name + '.pkl')
    model_val_path = os.path.join(save_dir, 'model', 'val', save_name + '.tar')
    model_test_path = os.path.join(save_dir, 'model', 'test', save_name + '.tar')
    renovate_step = int(num_epoch*0.6) if save_later else 0
    log('GPU: {}'.format(gpu), log_path)
    log('Loading path...', log_path)
    log('Save name: {}'.format(save_name), log_path)
    log('Save best test: {}'.format(save_best_test), log_path)
    log('Save later: {}'.format(save_later), log_path)
    with open(param_path, 'wb') as f: 
        pickle.dump(vars(args), f, protocol=pickle.HIGHEST_PROTOCOL)
    log('Loading dataset...', log_path)
    num_domain = len(source_domain)
    num_classes = len(known_classes)
    class_index = [i for i in range(num_classes)]
    group_length = (num_classes-1) // 10 + 1
    if dataset == "OfficeHome" and len(unknown_classes) == 0:
        group_length = 6

    log('Group length: {}'.format(group_length), log_path)
    
    group_index = [i for i in range((num_classes-1)//group_length + 1)]
    num_group = len(group_index)

    domain_specific_loader = []
    for domain in source_domain:       
        dataloader_list = []
        if num_classes <= 10:
            for i, classes in enumerate(known_classes):
                scd = SingleClassData(root_dir=train_dir, domain=domain, classes=classes, domain_label=-1, classes_label=i, transform=get_transform("train", small_img=small_img))
                loader = DataLoader(dataset=scd, batch_size=sub_batch_size, shuffle=True, drop_last=True, num_workers=1)
                dataloader_list.append(loader)
        else:
            classes_partition = split_classes(classes_list=known_classes, index_list=class_index, n=group_length)
            for classes, class_to_idx in classes_partition:
                sdd = SingleDomainData(root_dir=train_dir, domain=domain, classes=classes, domain_label=-1, get_classes_label=True, class_to_idx=class_to_idx, transform=get_transform("train", small_img=small_img))
                loader = DataLoader(dataset=sdd, batch_size=sub_batch_size, shuffle=True, drop_last=True, num_workers=1)
                dataloader_list.append(loader)

        domain_specific_loader.append(ConnectedDataIterator(dataloader_list=dataloader_list, batch_size=batch_size))
    
    if crossval:
        val_k = get_dataloader(root_dir=val_dir, domain=source_domain, classes=known_classes, batch_size=batch_size, get_domain_label=False, get_class_label=True, instr="val", small_img=small_img, shuffle=False, drop_last=False, num_workers=4)
    else:
        val_k = None
    
    test_k = get_dataloader(root_dir=test_dir, domain=target_domain, classes=known_classes, batch_size=batch_size, get_domain_label=False, get_class_label=True, instr="test", small_img=small_img, shuffle=False, drop_last=False, num_workers=4)
    if len(unknown_classes) > 0:
        test_u = get_dataloader(root_dir=test_dir, domain=target_domain, classes=unknown_classes, batch_size=batch_size, get_domain_label=False, get_class_label=False, instr="test", small_img=small_img, shuffle=False, drop_last=False, num_workers=4)   
    else:
        test_u = None

    log('DataSet: {}'.format(dataset), log_path)
    log('Source domain: {}'.format(source_domain), log_path)
    log('Target domain: {}'.format(target_domain), log_path)
    log('Known classes: {}'.format(known_classes), log_path)
    log('Unknown classes: {}'.format(unknown_classes), log_path)
    log('Batch size: {}'.format(batch_size), log_path)
    log('CrossVal: {}'.format(crossval), log_path)
    log('Loading models...', log_path)

    if share_param:
        muticlassifier = MutiClassifier_
    else:
        muticlassifier = MutiClassifier_eve
    net = MutiClassifier_eve(net=resnet18_fast(num_classes=num_classes, rb1=args.rb_layer_1, rb2=args.rb_layer_2), num_classes=num_classes)
    file_path = "model_weights/test.pth"
    w = torch.load(file_path)
    net.load_state_dict(w)
    net.eval()   
    net = net.cuda()
    best_score = 0
    best_threshold = 0
    if test_u != None:
        output_k_sum = []
        b_output_k_sum = []
        label_k_sum = []  
        with torch.no_grad():  
            for input, label, *_ in test_k:
                input = input.to(device)
                label = label.to(device)

                output, output2, b_output,b_output2,_ = net.c_forward(x=input, y = label)
                #output = (output + output2)/2
                b_output = (b_output + b_output2)/2
                output = F.softmax(output, 1)
                #b_output = net.b_forward(x=input, y=label)
                #b_output = b_output.view(output.size(0), 2, -1)
                b_output = F.softmax(b_output, 1)

                output_k_sum.append(output)
                b_output_k_sum.append(b_output)
                label_k_sum.append(label)
        output_k_sum = torch.cat(output_k_sum, dim=0)
        b_output_k_sum = torch.cat(b_output_k_sum, dim=0)
        label_k_sum = torch.cat(label_k_sum)
        output_u_sum = []
        b_output_u_sum = []
        torch.set_printoptions(profile="default")
        with torch.no_grad():
            for input, *_ in test_u:
                input = input.to(device)
                label = label.to(device)
                output, output2, b_output,b_output2,_ = net.c_forward(x=input, y = label)
                b_output = (b_output + b_output2)/2
                output = F.softmax(output, 1)
                
                b_output = F.softmax(b_output, 1)
                output_u_sum.append(output)
                b_output_u_sum.append(b_output)
        output_u_sum = torch.cat(output_u_sum, dim=0)
        b_output_u_sum = torch.cat(b_output_u_sum, dim=0)
        #################################################################################
        log('cls classifier:', log_path)
        conf_k, argmax_k = torch.max(output_k_sum, axis=1)
        conf_u, _ = torch.max(output_u_sum, axis=1)
        OSCR_C = generate_OSCR(argmax_k=argmax_k, conf_k=conf_k, label=label_k_sum, conf_u=conf_u)
        log('OSCR_cls: {:.4f}'.format(OSCR_C), log_path) 
        log('bcls classifier:', log_path)
        _, argmax_k = torch.max(output_k_sum, axis=1)
        _, argmax_u = torch.max(output_u_sum, axis=1)
        argmax_k_vertical = argmax_k.view(-1, 1)
        argmax_u_vertical = argmax_u.view(-1, 1)
        conf_u = torch.gather(b_output_u_sum[:, 1, :], dim=1, index=argmax_u_vertical).view(-1)
        conf_k = torch.gather(b_output_k_sum[:, 1, :], dim=1, index=argmax_k_vertical).view(-1)
        OSCR_B = generate_OSCR(argmax_k=argmax_k, conf_k=conf_k, label=label_k_sum, conf_u=conf_u)
        log('OSCR_bcls: {:.4f}'.format(OSCR_B), log_path) 

        acc_ = eval(net=net, loader=test_k, log_path=log_path, epoch=0, device=device, mark="Test")  
        log('ACC: {:.4f}'.format(acc_), log_path) 
        best_overall_acc = 0.0
        best_thred_acc = 0.0
        best_overall_Hscore = 0.0
        best_thred_Hscore = 0.0 #0.867
        threshold = 0.867   # notice threshold determined from crossval
        num_correct_k = num_correct_u = 0
        num_total_k = num_total_u = 0
        argmax_k = torch.argmax(output_k_sum, axis=1)
        for i in range(len(argmax_k)):
            if argmax_k[i] == label_k_sum[i] and b_output_k_sum[i][1][argmax_k[i]] >= threshold:
                num_correct_k +=1
        num_total_k += len(output_k_sum)
        argmax_u = torch.argmax(output_u_sum, axis=1)
        for i in range(len(argmax_u)):
            if b_output_u_sum[i][1][argmax_u[i]] < threshold:
                num_correct_u +=1
        num_total_u += len(output_u_sum)
        acc_k = num_correct_k / num_total_k
        acc_u = num_correct_u / num_total_u
        acc = (num_correct_k + num_correct_u) / (num_total_k + num_total_u)
        hs = 2*acc_k*acc_u/(acc_k + acc_u)
        if acc > best_overall_acc:
            best_overall_acc = acc
            best_thred_acc = threshold
        if hs > best_overall_Hscore:
            best_overall_Hscore = hs
            best_thred_Hscore = threshold
        if hs > best_score:
            best_score = hs
            best_threshold = threshold
        log('Hscore_cls:{:.4f} '.format(best_score), log_path)
        best_score = 0
        best_threshold = 0
        best_overall_acc = 0.0
        best_thred_acc = 0.0
        best_overall_Hscore = 0.0
        best_thred_Hscore = 0.0
        threshold = 0.86  # notice threshold determined from crossval
        num_correct_k = num_correct_u = 0
        num_total_k = num_total_u = 0
        argmax_k = torch.argmax(output_k_sum, axis=1)
        for i in range(len(argmax_k)):
            if argmax_k[i] == label_k_sum[i] and output_k_sum[i][argmax_k[i]] >= threshold:
                num_correct_k +=1
        num_total_k += len(output_k_sum)
        argmax_u = torch.argmax(output_u_sum, axis=1)
        for i in range(len(argmax_u)):
            if output_u_sum[i][argmax_u[i]] < threshold:
                num_correct_u +=1
        num_total_u += len(output_u_sum)
        acc_k = num_correct_k / num_total_k
        acc_u = num_correct_u / num_total_u
        acc = (num_correct_k + num_correct_u) / (num_total_k + num_total_u)
        hs = 2*acc_k*acc_u/(acc_k + acc_u)
        if acc > best_overall_acc:
            best_overall_acc = acc
            best_thred_acc = threshold
        if hs > best_overall_Hscore:
            best_overall_Hscore = hs
            best_thred_Hscore = threshold
        
        if hs > best_score:
            best_score = hs
            best_threshold = threshold
        log('Hscore_cls:{:.4f} '.format(best_score), log_path)

