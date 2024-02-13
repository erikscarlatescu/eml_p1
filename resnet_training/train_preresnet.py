from __future__ import print_function
import os
import argparse
import shutil
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import pickle

from preresnet import preresnet


# Training settings
parser = argparse.ArgumentParser(description='PyTorch Slimming CIFAR training')
parser.add_argument('--dataset', type=str, default='cifar100',
                    help='training dataset (default: cifar100)')
parser.add_argument('--data', type=str, default=None,
                    help='path to dataset')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=256, metavar='N',
                    help='input batch size for testing (default: 256)')
parser.add_argument('--epochs', type=int, default=160, metavar='N',
                    help='number of epochs to train (default: 160)')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--schedule', type=int, nargs='+', default=[],
                        help='Decrease learning rate at these epochs.')
parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                    help='learning rate (default: 0.1)')
parser.add_argument('--optim', default='sgd', type=str, 
                    help='Choice of optimizer (adam, sgd)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.9)')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--save', default='./logs', type=str, metavar='PATH',
                    help='path to save prune model (default: current directory)')

# Pruning related
parser.add_argument('--pruning_percentages', type=int, nargs='+', default=[],
                        help='Percentages at which to prune model')
parser.add_argument('--mask_path', default='', type=str, metavar='PATH',
                    help='Path where pruning masks can be found')
parser.add_argument('--model_path', default='', type=str, metavar='PATH',
                    help='Path to pretrained model (leave as blank string to start from scratch)')

args = parser.parse_args()
device = torch.device('mps')
print(device)

torch.manual_seed(args.seed)
torch.mps.manual_seed(args.seed)

train_loader = torch.utils.data.DataLoader(
    datasets.CIFAR10('./data.cifar10', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.Pad(4),
                       transforms.RandomCrop(32),
                       transforms.RandomHorizontalFlip(),
                       transforms.ToTensor(),
                       transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                   ])),
    batch_size=args.batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(
    datasets.CIFAR10('./data.cifar10', train=False, transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                   ])),
    batch_size=args.test_batch_size, shuffle=True)

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def train(model, epoch, mask=None):
    EPS = 1e-6
    model.train()
    global history_score
    avg_loss = 0.
    train_acc = 0.
    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.to(device)
        target = target.to(device)
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        avg_loss += loss.item()
        # pred = output.data.max(1, keepdim=True)[1]
        # train_acc += pred.eq(target.data.view_as(pred)).cpu().sum()
        prec1, prec5 = accuracy(output.data, target.data, topk=(1, 5))
        train_acc += prec1.item()
        loss.backward()

        if mask is not None:
            # Freezing Pruned weights by making their gradients Zero
            for name, p in model.named_parameters():
                if 'weight' in name and 'conv' in name:
                    tensor = p.data.cpu().numpy()
                    grad_tensor = p.grad.data.cpu().numpy()
                    grad_tensor = np.where(tensor < EPS, 0, grad_tensor)
                    p.grad.data = torch.from_numpy(grad_tensor).to(device)

        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.1f}%)]\tLoss: {:.6f}'.format(epoch, batch_idx * len(data), len(train_loader.dataset), 100. * batch_idx / len(train_loader), loss.item()))
    history_score[epoch][0] = avg_loss / len(train_loader)
    history_score[epoch][1] = np.round(train_acc / len(train_loader), 2)

def test(model):
    model.eval()
    test_loss = 0
    test_acc = 0
    for data, target in test_loader:
        data = data.to(device)
        target = target.to(device)
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        test_loss += F.cross_entropy(output, target, size_average=False).item() # sum up batch loss
        # pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        # correct += pred.eq(target.data.view_as(pred)).cpu().sum()
        prec1, prec5 = accuracy(output.data, target.data, topk=(1, 5))
        test_acc += prec1.item()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.1f}%)\n'.format(
        test_loss, test_acc, len(test_loader), test_acc / len(test_loader)))
    return np.round(test_acc / len(test_loader), 2)

def save_checkpoint(state, name, filepath):
    torch.save(state, os.path.join(filepath, f'checkpoint_{name}.pth.tar'))
    # if is_best:
        # shutil.copyfile(os.path.join(filepath, 'checkpoint.pth.tar'), os.path.join(filepath, 'model_best.pth.tar'))


model_masks = []
if args.mask_path != '':
    for pp in args.pruning_percentages:
        filename = f'{args.mask_path}_prune_{pp}.pkl'
        with open(filename, 'rb') as f:
            pp_mask = pickle.load(f)
            model_masks.append(pp_mask)
    # mask_checkpoint_160_prune_50.pkl
else:
    model_masks = [None]

for j, mask in enumerate(model_masks):
    model = preresnet(depth=20).to(device)
    if args.model_path != '':
        checkpoint = torch.load(args.model_path)
        model.load_state_dict(checkpoint['state_dict'])

    logdir = args.save
    if not mask is None:
        logdir = f'{logdir}/prune_{args.pruning_percentages[j]}'
        print(f"Started training model with pruning percent of: {args.pruning_percentages[j]}")
    if not os.path.exists(logdir):
        os.makedirs(logdir)

    if mask is not None: # apply mask to model, if we are masking
        i = 0
        for name, param in model.named_parameters():
            # print()
            # print(name)
            # print(mask[i].shape)
            if 'weight' in name and 'conv' in name:
                # print("Masking this layer")
                weight_dev = param.device
                param.data = torch.from_numpy(param.data.cpu().numpy() * mask[i]).to(weight_dev)
                i += 1

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    if args.optim == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    history_score = np.zeros((args.epochs, 3))


    import pdb; pdb.set_trace()

    best_prec1 = 0.
    save_checkpoint({ # save initial model for later testing with lottery ticket hypothesis
        'epoch': 0,
        'state_dict': model.state_dict(),
        'best_prec1': 0,
        'optimizer': optimizer.state_dict(),
    }, '0', filepath=logdir)
    for epoch in range(args.epochs):
        if epoch in args.schedule:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.1
        train(model, epoch, mask)
        prec1 = test(model)
        history_score[epoch][2] = prec1
        np.savetxt(os.path.join(logdir, 'record.txt'), history_score, fmt = '%10.5f', delimiter=',')
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'optimizer': optimizer.state_dict(),
        }, str(epoch+1), filepath=logdir)

    print("Best accuracy: "+str(best_prec1))
    history_score[-1][0] = best_prec1
    np.savetxt(os.path.join(logdir, 'record.txt'), history_score, fmt = '%10.5f', delimiter=',')
