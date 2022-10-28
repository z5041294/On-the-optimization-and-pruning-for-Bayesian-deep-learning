import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from model.resnet import ResNet18
#from model.wideresnet import Wide_ResNet
#from torch.utils.tensorboard import SummaryWriter
from SGHMC_SPL_CNN import SGHMC_SPL
import count_sparsity

import os
import argparse

parser = argparse.ArgumentParser(description='CIFAR10 Training')
parser.add_argument('--dir', type=str, default=None, required=True, help='path to save checkpoints (default: None)')
parser.add_argument('--data_path', type=str, default=None, required=True, metavar='PATH',
                    help='path to datasets location (default: None)')
parser.add_argument('--epochs', type=int, default=200,
                    help='number of epochs to train (default: 10)')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--beta0', type=float, default=0.1,
                    help='momentum parameter(default: 0.1')
parser.add_argument('--beta1', type=float, default=1,
                    help='momentum parameter(default: 1)')

args = parser.parse_args()


device1=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#writer = SummaryWriter('/root/tf-logs/')

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

training_data = datasets.CIFAR10(
    root=args.data_path,
    train=True,
    download=True,
    transform=transform_train
)

test_data = datasets.CIFAR10(
    root=args.data_path,
    train=False,
    download=True,
    transform=transform_test
)

train_dataloader = DataLoader(training_data, batch_size=args.batch_size, shuffle=True,num_workers=10)
test_dataloader = DataLoader(test_data, batch_size=512, shuffle=False,num_workers=10)

epochs=args.epochs
num_batch = len(train_dataloader.dataset)/args.batch_size+1
C2_0 = 1 # initial step size
M = 4 # number of cycles
T = epochs*num_batch # total number of iterations

def adjust_learning_rate(epoch, batch_idx):
    rcounter = epoch*num_batch+batch_idx
    cos_inner = np.pi * (rcounter % (T // M))
    cos_inner /= T // M
    cos_out = np.cos(cos_inner) + 1
    C2 = 0.5*cos_out*C2_0
    return C2

#net=Wide_ResNet(28,10,0,10).to(device=device1)
net=ResNet18().to(device=device1)
criterion = nn.CrossEntropyLoss()
optimizer = SGHMC_SPL(net.parameters(),N=len(train_dataloader.dataset))

def train_loop(epoch,dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    print('\nEpoch: %d' % (epoch+1))
    running_loss = 0.0
    
    model.train()
        
    for batch, (X, Y) in enumerate(dataloader):
        
        # Compute prediction and loss
        X=X.to(device1)
        Y=Y.to(device1)
        pred = model(X)
        loss = loss_fn(pred, Y)
        C = adjust_learning_rate(epoch+1,batch)**0.5
        
        if (epoch%50)+1>45:
            T=(1/size)
        else:
            T=(1/size)
                    
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step(C,epoch=epoch,batch=batch,T=T)

        running_loss += loss.item()

        #if batch % 100 == 99:    # every 100 mini-batches...
            # log the running loss
            #writer.add_scalar('training loss', running_loss / 100, epoch * len(dataloader) + batch)

def test_loop(epoch,dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0
    model.eval()
    
    with torch.no_grad():
        for X, Y in dataloader:
            X=X.to(device1)
            Y=Y.to(device1)
            pred = model(X)
            test_loss += loss_fn(pred, Y).item()
            correct += (pred.argmax(1) == Y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    #writer.add_scalar('testing loss',test_loss, epoch)
    #writer.add_scalar('testing accuracy',correct, epoch)

mt = 0
start_epoch = 0
for epoch in range(start_epoch, start_epoch+epochs):
                        
    train_loop(epoch,dataloader=train_dataloader,model=net,loss_fn=criterion,optimizer=optimizer)
    test_loop(epoch,dataloader=test_dataloader, model=net, loss_fn=criterion)
    
    sparsity_hard=count_sparsity.sparsity(net)
    #writer.add_scalar('sparse_ratio_hard',sparsity_hard, epoch)
    
    if (epoch%49)+1>46: # save 3 models per cycle
        print('save!')
        net.cpu()
        torch.save(net.state_dict(), args.dir + '/cifar10_csghmc_%i.pt'%(mt))
        mt +=1
        net.to(device=device1)
