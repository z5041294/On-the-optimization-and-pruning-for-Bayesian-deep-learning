from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets
from torch.utils.data import DataLoader
from model.wideresnet import Wide_ResNet
import torchvision.transforms as transforms
from model.resnet import ResNet34
import numpy as np
import count_sparsity


parser = argparse.ArgumentParser(description='CIFAR100 Ensemble')
parser.add_argument('--dir', type=str, default=None, required=True, help='path to checkpoints (default: None)')
parser.add_argument('--data_path', type=str, default=None, required=True, metavar='PATH',
                    help='path to datasets location (default: None)')
args = parser.parse_args()



device1=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Data
print('==> Preparing data..')
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
])


test_data = datasets.CIFAR100(
    root=args.data_path,
    train=False,
    download=True,
    transform=transform_test
)


test_dataloader = DataLoader(test_data, batch_size=512, shuffle=False,num_workers=12)

# Model
print('==> Building model..')
net=Wide_ResNet(28,10,0,10).to(device=device1)
#net=ResNet34().to(device=device1)
criterion = nn.CrossEntropyLoss()


def get_accuracy(truth, pred):
    assert len(truth)==len(pred)
    right = 0
    for i in range(len(truth)):
        if truth[i]==pred[i]:
             right += 1.0
    return right/len(truth)

def test():
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    pred_list = []
    truth_res = []
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_dataloader):
            
            inputs, targets = inputs.to(device=device1), targets.to(device=device1)
            truth_res += list(targets.data)
            outputs = net(inputs)
            pred_list.append(F.softmax(outputs,dim=1))
            loss = criterion(outputs, targets)

            test_loss += loss.data.item()
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum()

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss/len(test_dataloader), correct, total,
        100. * correct.item() / total))
    pred_list = torch.cat(pred_list,0)
    return pred_list,truth_res

pred_list = []
num_model = 12
for m in range(num_model):
    net.load_state_dict(torch.load(args.dir + '/cifar100_csghmc_%i.pt'%(m)))
    pred ,truth_res = test()
    pred_list.append(pred)
    print(count_sparsity.sparsity(net))

fake = sum(pred_list)/num_model
values, pred_label = torch.max(fake,dim = 1)
pred_res = list(pred_label.data)
acc = get_accuracy(truth_res, pred_res)
print(acc)