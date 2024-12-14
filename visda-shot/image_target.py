import argparse
import os, sys
import os.path as osp
import torchvision
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import network, loss
from torch.utils.data import DataLoader
from data_list import ImageList, ImageList_idx
import random, pdb, math, copy
from tqdm import tqdm
from scipy.spatial.distance import cdist
from sklearn.metrics import confusion_matrix
import rotation
from glob import glob
import os
from natsort import natsorted
from torch.utils.data import DataLoader, TensorDataset


def op_copy(optimizer):
    for param_group in optimizer.param_groups:
        param_group['lr0'] = param_group['lr']
    return optimizer

def lr_scheduler(optimizer, iter_num, max_iter, gamma=10, power=0.75):
    decay = (1 + gamma * iter_num / max_iter) ** (-power)
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr0'] * decay
        param_group['weight_decay'] = 1e-3
        param_group['momentum'] = 0.9
        param_group['nesterov'] = True
    return optimizer

def image_train(resize_size=256, crop_size=224, alexnet=False):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
    return  transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        transforms.RandomCrop(crop_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])

def image_test(resize_size=256, crop_size=224, alexnet=False):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
    return  transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
        normalize
    ])

def data_load(args): 
    ## prepare data
    dsets = {}
    dset_loaders = {}
    train_bs = args.batch_size
    txt_tar = open(args.t_dset_path).readlines()
    labels_tar = open(args.t_dset_path_label).readlines()
    # txt_test = open(args.test_dset_path).readlines()

    dsets["target"] = ImageList_idx(txt_tar, labels_tar, transform=image_train())
    dset_loaders["target"] = DataLoader(dsets["target"], batch_size=train_bs, shuffle=False, 
        num_workers=args.worker, drop_last=False)
    # dsets["target_te"] = ImageList_idx(txt_tar, transform=image_test())
    # dset_loaders["target_te"] = DataLoader(dsets["target_te"], batch_size=train_bs*3, shuffle=False, 
    #     num_workers=args.worker, drop_last=False)

    # dsets["test"] = ImageList_idx(txt_test, labels = [i for i in range(len(txt_test))], transform=image_test())
    # dset_loaders["test"] = DataLoader(dsets["test"], batch_size=train_bs*3, shuffle=False, 
    #     num_workers=args.worker, drop_last=False)

    return dset_loaders

def cal_acc(loader, netF, netB, netC, year, flag=False):
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader)  # Create an iterator
        for _ in range(len(loader)):
            data = next(iter_test)  # Use Python's built-in next()
            inputs, labels = data[0], data[1]
            inputs, labels = inputs.cuda(), labels.cuda()
            outputs = netC(netB(netF(inputs)))
            if start_test:
                all_output = outputs.float().cpu()
                all_label = labels.float()
                all_inputs = inputs.cpu()
                start_test = False
            else:
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)
                all_inputs = torch.cat((all_inputs, inputs.cpu()), 0)
    _, predict = torch.max(all_output, 1)
    all_output_numpy = all_output.numpy()
    predict_numpy = predict.numpy()
    
    # Save as .npy files
    np.save(f'logits_{year}.npy', all_output_numpy)
    np.save(f'predict_{year}.npy', predict_numpy)
    # np.save(f'labels_{year}.npy', all_label)
    return all_inputs, all_output_numpy, predict_numpy



def pseudo_target_synthesis (x , lam , pred_a ):
    # Random batch index .
    rand_idx = torch . randperm ( x . shape [0])
    inputs_a = x
    inputs_b = x [ rand_idx ]
    # Obtain model predictions and pseudo labels (pl ).
    pred_a = pred_a
    pl_a = pred_a
    pl_b = pl_a [ rand_idx ]
    # Select the samples with distinct labels for the mixup .
    diff_idx = ( pl_a != pl_b ). nonzero ()
    # Mixup with images and labels .
    pseudo_inputs = lam * inputs_a + (1 - lam ) * inputs_b
    if lam > 0.5:
        pseudo_labels = pl_a
    else :
        pseudo_labels = pl_b
    return pseudo_inputs [ diff_idx ] , pseudo_labels [ diff_idx ]
    # Perform supervised calibration using pseudo - target data .

def pseudoCal ( pseudo_inputs , pseudo_labels , net ):
    # Obtain predictions for the pseudo - target samples .
    pseudo_pred = net ( pseudo_inputs )
    # Apply temperature scaling to estimate the
    # pseudo - target temperature as the real temperature .
    calib_method = TempScaling ()
    pseudo_temp = calib_method ( pseudo_pred , pseudo_labels )
    return pseudo_temp

def train_target(args, year):
    dset_loaders = data_load(args)
    ## set base network
    if args.net[0:3] == 'res':
        netF = network.ResBase(res_name=args.net)
    elif args.net[0:3] == 'vgg':
        netF = network.VGGBase(vgg_name=args.net)
    feature_dim=2048
    type_bottleneck='bn'
    E_dims=256
    num_C = 12
    netB = network.feat_bottleneck(feature_dim=feature_dim, bottleneck_dim=E_dims, type=type_bottleneck)


    netC = network.feat_classifier(num_C, E_dims, type="wn")
    
    dict_to_load = torch.load("/home/dsi/rotemnizhar/dev/noisy-temperature-scaling/visda-shot/visda_SHOT/2019/last_10395.pth")
    for component in dict_to_load:
        if component == 'M':
            netF.load_state_dict(dict_to_load[component], strict=False)
        elif component == 'E':
            netB.load_state_dict(dict_to_load[component], strict=False)
        elif component=='G':
            netC.load_state_dict(dict_to_load[component], strict=False)
    
    netC.eval()
    for k, v in netC.named_parameters():
        v.requires_grad = False
    
    x, logits, predictions  = cal_acc(dset_loaders['target'], netF.cuda(), netB.cuda(), netC.cuda(), year)
    lam = 0.65
    pseudo_x , pseudo_y = pseudo_target_synthesis(x, lam, predictions)
    batch_size = 64  # Set your desired batch size
    dataset = TensorDataset(pseudo_x, torch.tensor(pseudo_y))
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # Iterate through the DataLoader to process the data in batches
    all_pseudo_pred = []
    for batch_idx, (inputs_batch, labels_batch) in enumerate(data_loader):
        # Move inputs_batch and labels_batch to GPU if needed
        inputs_batch, labels_batch = inputs_batch.cuda(), labels_batch.cuda()

        # Compute predictions for the current batch
        pseudo_pred_batch = netC(netB(netF(inputs_batch)))

        # Store the predictions (you can append them or process them here)
        all_pseudo_pred.append(pseudo_pred_batch.detach().cpu().numpy())  # Move to CPU for saving

    # Concatenate all predictions
    all_pseudo_pred = np.concatenate(all_pseudo_pred, axis=0)

    # Save the predictions and labels
    np.save(f'pseudo_labels_{year}_{lam}.npy', pseudo_y)
    np.save(f'pseudo_logits_{year}_{lam}.npy', all_pseudo_pred)
    
    
    # pseudo_pred = netC(netB(netF(pseudo_x.cuda())))
    # np.save(f'pseudo_labels_{year}_{lam}.npy', pseudo_y)
    # np.save(f'pseudo_logits_{year}_{lam}.npy', pseudo_pred.cpu())
    


def print_args(args):
    s = "==========================================\n"
    for arg, content in args.__dict__.items():
        s += "{}:{}\n".format(arg, content)
    return s

def obtain_label(loader, netF, netB, netC, args):
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader)
        for _ in range(len(loader)):
            data = iter_test.next()
            inputs = data[0]
            labels = data[1]
            inputs = inputs.cuda()
            feas = netB(netF(inputs))
            outputs = netC(feas)
            if start_test:
                all_fea = feas.float().cpu()
                all_output = outputs.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_fea = torch.cat((all_fea, feas.float().cpu()), 0)
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)

    all_output = nn.Softmax(dim=1)(all_output)
    ent = torch.sum(-all_output * torch.log(all_output + args.epsilon), dim=1)
    unknown_weight = 1 - ent / np.log(args.class_num)
    _, predict = torch.max(all_output, 1)

    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    if args.distance == 'cosine':
        all_fea = torch.cat((all_fea, torch.ones(all_fea.size(0), 1)), 1)
        all_fea = (all_fea.t() / torch.norm(all_fea, p=2, dim=1)).t()

    all_fea = all_fea.float().cpu().numpy()
    K = all_output.size(1)
    aff = all_output.float().cpu().numpy()

    for _ in range(2):
        initc = aff.transpose().dot(all_fea)
        initc = initc / (1e-8 + aff.sum(axis=0)[:,None])
        cls_count = np.eye(K)[predict].sum(axis=0)
        labelset = np.where(cls_count>args.threshold)
        labelset = labelset[0]

        dd = cdist(all_fea, initc[labelset], args.distance)
        pred_label = dd.argmin(axis=1)
        predict = labelset[pred_label]

        aff = np.eye(K)[predict]

    acc = np.sum(predict == all_label.float().numpy()) / len(all_fea)
    log_str = 'Accuracy = {:.2f}% -> {:.2f}%'.format(accuracy * 100, acc * 100)

    args.out_file.write(log_str + '\n')
    args.out_file.flush()
    print(log_str+'\n')

    return predict.astype('int')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='SHOT++')
    parser.add_argument('--gpu_id', type=str, nargs='?', default='0', help="device id to run")
    parser.add_argument('--s', type=int, default=0, help="source")
    parser.add_argument('--t', type=int, default=1, help="target")
    parser.add_argument('--max_epoch', type=int, default=15, help="max iterations")
    parser.add_argument('--interval', type=int, default=15)
    parser.add_argument('--batch_size', type=int, default=64, help="batch_size")
    parser.add_argument('--worker', type=int, default=4, help="number of workers")
    parser.add_argument('--dset', type=str, default='office-home', choices=['VISDA-C', 'office', 'office-home'])
    parser.add_argument('--lr', type=float, default=1e-2, help="learning rate")
    parser.add_argument('--net', type=str, default='resnet50', help="vgg16, resnet18, resnet50")
    parser.add_argument('--seed', type=int, default=2020, help="random seed")
 
    parser.add_argument('--gent', type=bool, default=True)
    parser.add_argument('--ent', type=bool, default=True)
    parser.add_argument('--threshold', type=int, default=0)
    parser.add_argument('--cls_par', type=float, default=0.3)
    parser.add_argument('--ent_par', type=float, default=1.0)
    parser.add_argument('--lr_decay1', type=float, default=0.1)
    parser.add_argument('--lr_decay2', type=float, default=1.0)

    parser.add_argument('--bottleneck', type=int, default=256)
    parser.add_argument('--epsilon', type=float, default=1e-5)
    parser.add_argument('--layer', type=str, default="wn", choices=["linear", "wn"])
    parser.add_argument('--classifier', type=str, default="bn", choices=["ori", "bn"])
    parser.add_argument('--distance', type=str, default='cosine', choices=["euclidean", "cosine"])  
    parser.add_argument('--output', type=str, default='san')
    parser.add_argument('--output_src', type=str, default='san')
    parser.add_argument('--da', type=str, default='uda', choices=['uda', 'pda'])
    parser.add_argument('--ssl', type=float, default=0.0) 
    parser.add_argument('--issave', type=bool, default=True)
    args = parser.parse_args()

    if args.dset == 'office-home':
        names = ['Art', 'Clipart', 'Product', 'RealWorld']
        args.class_num = 65 
    elif args.dset == 'office':
        names = ['amazon', 'dslr', 'webcam']
        args.class_num = 31
    elif args.dset == 'VISDA-C':
        names = ['train', 'validation']
        args.class_num = 12
        args.lr = 1e-3

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    SEED = args.seed
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    # torch.backends.cudnn.deterministic = True
    
    folder = '../data/'
    for i in range(1):
        # if i == args.s:
        #     continue
        args.t = i

        args.s_dset_path = folder + args.dset + '/' + names[args.s] + '_list.txt'
        args.t_dset_path = folder + args.dset + '/' + names[args.t] + '_list.txt'
        args.test_dset_path = folder + args.dset + '/' + names[args.t] + '_list.txt'
        
        # my code
        year = '2021'
        args.t_dset_path = f"/home/dsi/rotemnizhar/dev/noisy-temperature-scaling/visda-shot/image-lists/validation_images_DCPL_VISDA_{year}.txt"
        args.t_dset_path_label = f"/home/dsi/rotemnizhar/dev/noisy-temperature-scaling/visda-shot/image-lists/validation_labels_DCPL_VISDA_{year}.txt"
        # args.test_dset_path = "/home/dsi/rotemnizhar/dev/noisy-temperature-scaling/visda-shot/image-lists/validation_images_DCPL_VISDA_2020.txt"

        args.output_dir_src = osp.join(args.output_src, args.da, args.dset, names[args.s][0].upper())
        args.output_dir = osp.join(args.output, args.da, args.dset, names[args.s][0].upper()+names[args.t][0].upper())
        args.name = names[args.s][0].upper()+names[args.t][0].upper()

        if not osp.exists(args.output_dir):
            os.system('mkdir -p ' + args.output_dir)
        if not osp.exists(args.output_dir):
            os.mkdir(args.output_dir)

        args.savename = 'par_' + str(args.cls_par)
        if args.ssl > 0:
             args.savename += ('_ssl_' + str(args.ssl))
        
        args.out_file = open(osp.join(args.output_dir, 'log_' + args.savename + '.txt'), 'w')
        args.out_file.write(print_args(args)+'\n')
        args.out_file.flush()

        train_target(args, year)