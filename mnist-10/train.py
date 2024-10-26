import numpy as np


import wandb
# pytorch libraries
import torch
from torch import optim,nn
import time
import random
from configs import get_args
from models import *
from get_loaders import get_loaders
import os

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def _create_dir(dir_path):
    names= dir_path.split('/')
    cur_dir= ''
    for i in range(len(names)):
        if i==0:
            cur_dir = names[0]
        else:
            cur_dir = f'{cur_dir}/{names[i]}'
        os.makedirs(cur_dir, exist_ok=True)


def save__logits(logits_type, logits, output_dir):
    filepath = f"{output_dir}/{logits_type}_logits.pt"
    torch_logits = torch.stack([torch.from_numpy(arr) for arr in logits])
    torch.save(torch_logits, filepath)

def train_model(model, train_data_loader, valid_data_loader, optimizer,
                n_epochs, criterion, batch_size, lr, epsilon, model_name, verbose = True):
    best_acc_val = 0
    start_time = time.time()
    for epoch in range(1, n_epochs+1):
        epoch_time = time.time()
        epoch_loss = 0
        correct = 0
        total=0
        if verbose:
            print("Epoch {} / {}".format(epoch, n_epochs))

        model.train()
        
        for inputs, labels in train_data_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad() # zeroed grads
            outputs = model(inputs) # forward pass
            loss = criterion(outputs, labels) # softmax + cross entropy
            loss.backward() # back pass
            optimizer.step() # updated params
            epoch_loss += loss.item() # train loss
            _, pred = torch.max(outputs, dim=1)
            correct += (pred.cpu() == labels.cpu()).sum().item()
            total += labels.shape[0]
        acc = correct / total
        
        model.eval()
        a=0
        pred_val=0
        correct_val=0
        total_val=0
        with torch.no_grad():
            for inp_val, lab_val in valid_data_loader:
                inp_val = inp_val.to(device)
                lab_val = lab_val.to(device)
                out_val = model(inp_val)
                loss_val = criterion(out_val, lab_val)
                a += loss_val.item()
                _, pred_val = torch.max(out_val, dim=1)
                correct_val += (pred_val.cpu()==lab_val.cpu()).sum().item()
                total_val += lab_val.shape[0]
            acc_val = correct_val / total_val
        epoch_time2 = time.time()

        if verbose:                   
            print("Duration: {:.0f}s, Train Loss: {:.4f}, Train Acc: {:.4f}, Val Loss: {:.4f}, Val Acc: {:.4f}"
              .format(epoch_time2-epoch_time, epoch_loss/len(labels), acc, a/len(lab_val), acc_val))

        if acc_val > best_acc_val:
            best_acc_val = acc_val
            best_ckpt_epoch = epoch
            best_model_dir = f'mnist-10/ckpts/eps_{epsilon}'
            best_model_path = f'{best_model_dir}/new_model_{model_name}_epoch_{epoch}_{n_epochs}_lr_{lr}_bs_{batch_size}.pth'
            _create_dir(best_model_dir)
            torch.save(model, best_model_path)

    end_time = time.time()
    total_time = end_time - start_time
    # wandb.log({'total_time':total_time})

    if verbose:
        print("Total Time:{:.0f}s".format(end_time-start_time))
    return best_model_path

def test_model(model, test_data_loader, verbose = False):
    logits_list = []
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_data_loader:
            images, labels = data
            images, labels = images.to('cuda'), labels.to('cuda')
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            cur_logits = outputs.cpu().numpy()
            for i in range(len(cur_logits)):
                logits_list.append(cur_logits[i])

    test_acc = float(correct)/float(total)*100

    if verbose:
        print(f"Accuracy of the network on the test images: {test_acc:.4f}")
    
    # wandb.log({'test_acc': test_acc})

    return test_acc, logits_list


def main(args):
    # print (f'---> Noise level {args.epsilon}')

    train_data_loader, train_eval_data_loader, valid_data_loader, test_data_loader = get_loaders(batch_size=args.batch_size)
    
    model, _ = get_model(args.model_name)
    model.to(device)
    print(model)

    criterion = nn.CrossEntropyLoss().to(device)
    for lr in [0.000001, 0.000005, 0.00001, 0.00005, 0.0001, 0.0005, 0.001]:
        args.lr = lr
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        # with wandb.init(project='ham10000_with_noisy_labels'):
        #     wandb.config.update(args)
        print ('=== Start Training ===')
        best_model_path = train_model(model = model, 
                                train_data_loader = train_data_loader, 
                                valid_data_loader = valid_data_loader, 
                                optimizer = optimizer,
                                n_epochs = args.n_epochs, 
                                criterion = criterion, 
                                batch_size = args.batch_size,
                                lr=args.lr,
                                epsilon=args.epsilon,
                                model_name= args.model_name)
        print ('=== End Training ===')
        print ('=== Start Testing ===')
        model = torch.load(best_model_path)
        acc, test_logits = test_model(model, test_data_loader)
            # wandb.log({'test_accuracy': acc})
        val_acc, val_logits= test_model(model, valid_data_loader)
        output_root = '/home/dsi/rotemnizhar/dev/noisy-temperature-scaling/mnist-10/output'
        save__logits('test', test_logits, output_root)
        save__logits('valid', val_logits, output_root)
        print ('=== End Testing ===')


if __name__ == "__main__":
    print("run main")
    args = get_args()
    main(args)
