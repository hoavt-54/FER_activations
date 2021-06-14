#
#	@author Hoa Vu
#	@email hoavutrongvn at gmail.com
#	@create date 2021-06-12 09:14:46
#	@modify date 2021-06-12 09:14:46
#	@desc [description]
# ======================================================================

import os
import argparse
import pathlib
import numpy as np
from pathlib import Path
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
import utils
import data_utils
from models import VGG



def train(net, train_loader, lr =0.01, epoch=0,
        lr_decay_start=80, lr_decay_every=5,
        lr_decay_rate=0.9, use_cuda=True,
        optimizer=None, criterion=None):

    train_loss, total, correct = 0, 0, 0
    net.train()

    if epoch > lr_decay_start:
        frac = (epoch - lr_decay_start) // lr_decay_every
        decay_factor = lr_decay_rate ** frac
        current_lr = lr * decay_factor
        utils.set_lr(optimizer, current_lr)  # set the decayed rate
    else:
        current_lr = lr
    print('learning_rate: %.4f' % str(current_lr))

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        
        # outputs.data [batch_size, num_class]
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        
        #correct += predicted.eq(targets.data).cpu().sum()
        correct += (predicted == targets).sum().item()

        utils.progress_bar(batch_idx, len(train_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))


def eval(net, test_loader, use_cuda=True, criterion=None):
    net.eval()
    total, correct = 0, 0
    for batch_idx, (inputs, targets) in enumerate(test_loader):
        # print("input shape: ", np.shape(inputs))
        bs, ncrops, c, h, w = np.shape(inputs)
        inputs = inputs.view(-1, c, h, w)
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        outputs = net(inputs)
        outputs_avg = outputs.view(bs, ncrops, -1).mean(1)  # avg over crops
        loss = criterion(outputs_avg, targets)
        _, predicted = torch.max(outputs_avg.data, 1)
        total += targets.size(0)
        correct += (predicted == targets).sum().item()

        utils.progress_bar(batch_idx, len(test_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                           % (loss / (batch_idx + 1), 100. * correct / total, correct, total))
    
    return 100. * correct / total


def saving_model(accuracy, net, path, use_cuda=True, epoch=0, dataset="public"):
    print("saving")
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    print(f"best_{dataset}_acc: {accuracy}")
    state = {
        'net': net.state_dict() if use_cuda else net,
        'acc': accuracy,
        'epoch': epoch,
    }
    torch.save(state, os.path.join(path, f'{dataset}_model.t7'))


def main(args):
    trainset, publicset, privateset = data_utils.read_split("./data_dir/fer2013.csv")
    trainloader = DataLoader(trainset, batch_size=args.bs, shuffle=True, num_workers=0)
    publicloader = DataLoader(publicset, batch_size=args.bs, shuffle=False, num_workers=0)
    privateloader = DataLoader(privateset, batch_size=args.bs, shuffle=False, num_workers=0)

    if args.model == 'VGG19':
        net = VGG('VGG19', act=None)
    elif args.model  == 'Resnet18':
        net = ResNet18()

    use_cuda = torch.cuda.is_available()
    if use_cuda:
        net.cuda()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9)

    best_public_acc, best_private_acc = 0, 0
    saving_path = os.path.join("saved_models", args.model)

    for epoch in range(args.epoch):
        print(f"\nEpoch: {epoch}")
        train(
            net,
            trainloader,
            args.lr, epoch,
            use_cuda=use_cuda,
            optimizer=optimizer,
            criterion=criterion
        )
        public_acc = eval(net, publicloader, use_cuda=use_cuda, criterion=criterion)
        if public_acc > best_public_acc:
            best_public_acc = public_acc
            saving_model(
                public_acc, net, saving_path,
                use_cuda=use_cuda, epoch=epoch,
                dataset="public"
            )
            
        private_acc =  eval(net, privateloader, use_cuda=use_cuda, criterion=criterion)
        if private_acc > best_private_acc:
            best_private_acc = private_acc
            saving_model(
                private_acc, net, saving_path,
                use_cuda=use_cuda, epoch=epoch,
                dataset="private"
            )
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch Fer2013 CNN Training')
    parser.add_argument('--model', type=str, default='VGG19', help='CNN architecture')
    parser.add_argument('--dataset', type=str, default='FER2013', help='CNN architecture')
    parser.add_argument('--bs', default=20, type=int, help='learning rate')
    parser.add_argument('--epoch', default=160, type=int, help='number of epoches')
    parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
    parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
    args = parser.parse_args()
    main(args)
