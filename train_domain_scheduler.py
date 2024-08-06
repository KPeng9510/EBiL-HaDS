import argparse
import torch
import pickle
import os, copy
from dataset.dataloader import get_dataloader, get_transform
from dataset.dataset import SingleDomainData, SingleClassData
from model.model2_photo import MutiClassifier_eve, MutiClassifier_, resnet18_fast, resnet18_fast_origin,resnet50_fast, ConvNet, DomainIndicator, resnet152
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
    parser.add_argument('--source-domain', nargs='+', default=['photo', 'sketch', 'art_painting'])
    parser.add_argument('--target-domain', nargs='+', default=['cartoon'])
    parser.add_argument('--known-classes', nargs='+', default=['elephant', 'horse', 'giraffe', 'dog', 'guitar', 'house',])
    parser.add_argument('--unknown-classes', nargs='+', default=['person'])
    parser.add_argument('--no-crossval', action='store_true')
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--net-name', default='resnet18')
    parser.add_argument('--optimize-method', default="SGD")
    parser.add_argument('--schedule-method', default='StepLR')
    parser.add_argument('--num-epoch', type=int, default=20000)
    parser.add_argument('--eval-step', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.001) 
    parser.add_argument('--meta-lr', type=float, default=0.001)
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
    parser.add_argument('--path-dataset', type=str, default="Path to dataset/")

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

    net = MutiClassifier_eve(net=resnet18_fast(num_classes=num_classes), num_classes=num_classes)

    net = net.to(device)
    net_d_indicator = DomainIndicator(num_domain=1,net=resnet18_fast_origin())
    net_d_indicator = net_d_indicator.to(device)

    optimizer = get_optimizer(net=net, instr=optimize_method, lr=lr, nesterov=nesterov)
    scheduler = get_scheduler(optimizer=optimizer, instr=schedule_method, step_size=int(num_epoch*dr), gamma=0.1)

    log('Network: {}'.format(net_name), log_path)
    log('Number of epoch: {}'.format(num_epoch), log_path)
    log('Learning rate: {}'.format(lr), log_path)
    log('Meta learning rate: {}'.format(meta_lr), log_path)

    if num_epoch_before != 0:
        log('Loading state dict...', log_path)  
        if save_best_test == False:
            net.load_state_dict(torch.load(model_val_path))
        else:
            net.load_state_dict(torch.load(model_test_path))
        for epoch in range(num_epoch_before):
            scheduler.step()
        log('Number of epoch-before: {}'.format(num_epoch_before), log_path)
    log('Without binary classifier: {}'.format(without_bcls), log_path)
    log('Share Parameter: {}'.format(share_param), log_path)
    log('Start training...', log_path)  
    best_val_acc = 0
    best_val_test_acc = []
    best_test_acc = best_test_acc_ = 0
    best_test_test_acc = []
    criterion_w = torch.nn.MSELoss(size_average=None, reduce=None, reduction='mean').cuda()
    criterion = torch.nn.CrossEntropyLoss().cuda()
    criterion_reweigt = torch.nn.CrossEntropyLoss(reduction='none').cuda()
    ovaloss = OVALoss()
    if without_bcls:
        ovaloss = lambda *args: 0
    exp_domain_index = 0   
    exp_group_num = (num_group-1) // 3 + 1
    exp_group_index = random.sample(group_index, exp_group_num)

    domain_index_list = [i for i in range(num_domain)]

    fast_parameters = list(net.parameters())

    fast_parameters_indicator = list(net_d_indicator.parameters())
    for weighth in net.parameters():
        weighth.fast = None
    net.zero_grad()
    for weighth in net_d_indicator.parameters():
        weighth.fast = None
    net_d_indicator.zero_grad()
    schedule_memory = []
    for upper_epoch in range(num_epoch_before, num_epoch//step):
        net_d_indicator.eval()
        indicators = []
        for k in range(num_domain):
            if len(schedule_memory) > 1:
                dm_weight = 0.1 + lamda*(schedule_memory.count(k))/len(schedule_memory)
            else:
                dm_weight = 0.1
            domain_specific_loader[k].keep(exp_group_index)
            batch_data = []
            for i in range(ns):
                batch_data1,_ = next(domain_specific_loader[k])
                batch_data.append(batch_data1)
            batch_data = torch.cat(batch_data,0)
            pred_domain = net_d_indicator(batch_data.to(device)).squeeze()
            pred_domain = torch.exp(1 +torch.mean(pred_domain))*dm_weight
            indicators.append(pred_domain)
            domain_specific_loader[k].reset()  
        indicators = torch.stack(indicators,0)
        exp_domain_index = torch.argmin(indicators,0).long().detach()
        schedule_memory.append(exp_domain_index)
        for epoch in range(0,step):
            criterion = torch.nn.CrossEntropyLoss().cuda()
            criterion_reweigt = torch.nn.CrossEntropyLoss(reduction='none').cuda()
            net.train()
            net_d_indicator.train()
            debias_loss = meta_weight_loss = meta_train_loss = meta_val_loss = 0
            domain_index_set = set(domain_index_list) - {exp_domain_index}
            i, j = random.sample(list(domain_index_set), 2)
            domain_specific_loader[i].remove(exp_group_index)
            input, label = next(domain_specific_loader[i]) 
            domain_specific_loader[i].reset()  
            input = input.to(device)
            label = label.to(device)
            out, out2, output, output2,loss_debias = net.c_forward(x=input, y=label, train=True)
            debias_loss += loss_debias
            w = net_d_indicator(input)
            weight_gt = torch.max(out, -1)[0]
            meta_train_loss +=  ova_weight*ovaloss(output, label)
            meta_train_loss +=  ova_weight*ovaloss(output2, label)
            l_weight = criterion_w(w, weight_gt)
            meta_weight_loss += l_weight
            meta_train_loss += cls_weight*criterion(out2, label)
            meta_train_loss += cls_weight*criterion(out, label)
            domain_specific_loader[j].remove(exp_group_index)
            input, label = next(domain_specific_loader[j])
            domain_specific_loader[j].reset()
            input = input.to(device)
            label = label.to(device)
            out, out2,output,output2,loss_debias = net.c_forward(x=input, y=label, train=True)
            debias_loss += loss_debias
            w = net_d_indicator(input)
            weight_gt = torch.max(output, -1)[0]
            l_weight = criterion_w(w, weight_gt)
            meta_weight_loss += l_weight
            meta_train_loss +=  ova_weight*ovaloss(output, label)
            meta_train_loss +=  ova_weight*ovaloss(output2, label)
            meta_train_loss += torch.mean(criterion_reweigt(out, label)*(w))
            meta_train_loss += criterion(out, label)
            meta_train_loss += criterion(out2, label)
            domain_specific_loader[exp_domain_index].keep(exp_group_index)
            input, label = next(domain_specific_loader[exp_domain_index])
            domain_specific_loader[exp_domain_index].reset()
            input = input.to(device)
            label = label.to(device)
            out, out2,output,output2,loss_debias = net.c_forward(x=input, y=label, train=True)
            debias_loss += loss_debias
            w = net_d_indicator(input)
            weight_gt = torch.max(out, -1)[0]
            l_weight = criterion_w(w, weight_gt)
            meta_weight_loss += l_weight
            meta_train_loss +=  ova_weight*ovaloss(output, label)
            meta_train_loss +=  ova_weight*ovaloss(output2, label)
            meta_train_loss += torch.mean(criterion_reweigt(out, label)*(w))
            meta_train_loss += cls_weight*criterion(out2, label) +cls_weight*criterion(out, label)+ torch.mean(loss_debias)
            meta_train_loss -= meta_weight*meta_weight_loss
    ########################################################################## meta val open
            net.zero_grad()
            net_d_indicator.zero_grad()
            meta_val_loss = 0.0
            meta_weight_loss = 0.0
            loss_val_debias = 0.0
            grad = torch.autograd.grad(meta_train_loss, fast_parameters,
                                    create_graph=True, allow_unused=True)
            if (upper_epoch*step + epoch)%(dr*num_epoch):
                meta_lr = meta_lr*0.1
            for k, weighth in enumerate(net.parameters()):
                if grad[k] is not None:
                    if weighth.fast is None:
                        weighth.fast = weighth - meta_lr * grad[k]
                    else:
                        weighth.fast = weighth.fast - meta_lr * grad[
                            k]
            grad_follower = torch.autograd.grad(-meta_train_loss, fast_parameters_indicator,
                                    create_graph=True, allow_unused=True)
            for k, weight_follower in enumerate(net_d_indicator.parameters()):
                if grad_follower[k] is not None:
                    if weight_follower.fast is None:
                        weight_follower.fast = weight_follower - meta_lr * grad_follower[k]
                    else:
                        weight_follower.fast = weight_follower.fast - meta_lr * grad_follower[k]
            domain_specific_loader[i].keep(exp_group_index)
            input_1, label_1 = domain_specific_loader[i].next(batch_size=batch_size//2)
            domain_specific_loader[i].reset() 
            domain_specific_loader[j].keep(exp_group_index)
            input_2, label_2 = domain_specific_loader[j].next(batch_size=batch_size//2)
            domain_specific_loader[j].reset() 
            input = torch.cat([input_1, input_2], dim=0)
            label = torch.cat([label_1, label_2], dim=0)
            input = input.to(device)
            label = label.to(device)
            out,out2,output,output2,loss_debias = net.c_forward(x=input, y=label, train=True)
            loss_val_debias += loss_debias
            w = net_d_indicator(input)
            weight_gt = torch.max(output, -1)[0]
            l_weight = criterion_w(w, weight_gt)
            meta_weight_loss += l_weight
            meta_val_loss += torch.mean(criterion_reweigt(out, label)*(w))
            meta_val_loss += cls_weight*criterion(out2, label) + cls_weight*criterion(out, label)
            meta_val_loss += ova_weight*ovaloss(output, label)
            meta_val_loss += ova_weight*ovaloss(output2, label)
            for i in range(num_meta_test):
                domain_specific_loader[exp_domain_index].remove(exp_group_index)
                input, label = next(domain_specific_loader[exp_domain_index])
                domain_specific_loader[exp_domain_index].reset()
                input = input.to(device)
                label = label.to(device)
                out,out2, output,output2,loss_debias = net.c_forward(x=input, y=label, train=True)
                loss_val_debias += loss_debias
                w = net_d_indicator(input)
                weight_gt = torch.max(output, -1)[0]
                l_weight = criterion_w(w, weight_gt)
                meta_weight_loss += l_weight
                meta_val_loss +=torch.mean(criterion_reweigt(out, label)*(w))
                meta_val_loss += cls_weight*criterion(out2, label) + cls_weight*criterion(out, label)
                meta_val_loss +=  ova_weight*ovaloss(output, label)
                meta_val_loss +=  ova_weight*ovaloss(output2, label)
            total_loss = meta_train_loss + meta_val_loss - meta_weight* meta_weight_loss + torch.mean(loss_val_debias)
            optimizer.zero_grad()
            #optimizer_d.zero_grad()
            total_loss.backward()
            optimizer.step()
            #optimizer_d.step()
            fast_parameters = list(net.parameters())
            for weighth in net.parameters():
                weighth.fast = None
            fast_parameters_indicator = list(net_d_indicator.parameters())
            for weighth in net_d_indicator.parameters():
                weighth.fast = None
            net.zero_grad()
            meta_train_loss = 0.0
            meta_val_loss = 0.0
            total_loss = 0.0
            scheduler.step()
####################################################################
        exp_group_index = random.sample(group_index, exp_group_num)
        if (upper_epoch*step + epoch+1) % eval_step == 0:      
            net.eval()         
            if test_u != None:
                output_k_sum = []
                b_output_k_sum = []
                label_k_sum = []  
                with torch.no_grad():  
                    for input, label, *_ in test_k:
                        input = input.to(device)
                        label = label.to(device)
                        output, output2, b_output,b_output2,_ = net.c_forward(x=input, y = label)
                        b_output = (b_output + b_output2)/2
                        output = F.softmax(output, 1)
                        b_output = F.softmax(b_output, 1)
                        output_k_sum.append(output)
                        b_output_k_sum.append(b_output)
                        label_k_sum.append(label)
                output_k_sum = torch.cat(output_k_sum, dim=0)
                b_output_k_sum = torch.cat(b_output_k_sum, dim=0)
                label_k_sum = torch.cat(label_k_sum)
                output_u_sum = []
                b_output_u_sum = []
    
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
    ###################################################################################################################
                log('bcls classifier:', log_path)
                _, argmax_k = torch.max(output_k_sum, axis=1)
                _, argmax_u = torch.max(output_u_sum, axis=1)
                argmax_k_vertical = argmax_k.view(-1, 1)
                argmax_u_vertical = argmax_u.view(-1, 1)
                conf_u = torch.gather(b_output_u_sum[:, 1, :], dim=1, index=argmax_u_vertical).view(-1)
                conf_k = torch.gather(b_output_k_sum[:, 1, :], dim=1, index=argmax_k_vertical).view(-1)
                OSCR_B = generate_OSCR(argmax_k=argmax_k, conf_k=conf_k, label=label_k_sum, conf_u=conf_u)
                log('OSCR_bcls: {:.4f}'.format(OSCR_B), log_path) 
            else:
                OSCR_C = OSCR_B = 0 
                log("", log_path)
            if val_k != None:
                acc = eval(net=net, loader=val_k, log_path=log_path, epoch=upper_epoch*step + epoch+1, device=device, mark="Val") 
            
            acc_ = eval(net=net, loader=test_k, log_path=log_path, epoch=upper_epoch*step + epoch+1, device=device, mark="Test")     
            #torch.save(net.state_dict(), '/'.join(model_val_path.split('/')[:-1]) + 'results_'+'_acc_' + str(acc_.item())[:6]+'_h_score_' + str(h_score.item())[:6] + str(int(epoch))+'_OSCR_B_' + str(OSCR_B.item())[:6] +'_OSCR_C_' + str(OSCR_C.item())[:6]+'.pth')
            if val_k != None:           
                if acc > best_val_acc:
                    best_val_acc = acc
                    best_test_acc_ = acc_
                    best_val_test_acc = [{
                        "test_acc": "%.4f" % acc_.item(),
                        "OSCR_C": "%.4f" % OSCR_C,
                        "OSCR_B": "%.4f" % OSCR_B,
                    }]
                    best_val_model = copy.deepcopy(net.state_dict())
                    torch.save(best_val_model, model_val_path)
                elif acc == best_val_acc:
                    best_val_test_acc.append({
                        "test_acc": "%.4f" % acc_.item(),
                        "OSCR_C": "%.4f" % OSCR_C,
                        "OSCR_B": "%.4f" % OSCR_B,
                    })
                    if acc_ > best_test_acc_:
                        best_test_acc_ = acc_
                        best_val_model = copy.deepcopy(net.state_dict())
                        torch.save(best_val_model, model_val_path)
                log("Current best val accuracy is {:.4f} (Test: {})".format(best_val_acc, best_val_test_acc), log_path)
                
            if acc_ > best_test_acc:
                best_test_acc = acc_    
                best_test_test_acc = [{
                    "OSCR_C": "%.4f" % OSCR_C,
                    "OSCR_B": "%.4f" % OSCR_B,
                }]    
                if save_best_test:
                    best_test_model = copy.deepcopy(net.state_dict())
                    torch.save(best_test_model, model_test_path)
            log("Current best test accuracy is {:.4f} ({})".format(best_test_acc, best_test_test_acc), log_path)

        if epoch+1 == renovate_step:
                log("Reset accuracy history...", log_path)

                best_val_acc = 0
                best_val_test_acc = []
                best_test_acc = 0
                best_test_test_acc = []

        #scheduler.step()
