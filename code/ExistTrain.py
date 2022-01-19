import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torchvision import models
import os
import time
import torchvision.datasets as dset
import pickle as pk
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
best = 0

def evaluate_accuracy(data_iter, net, device=None):
    global best
    if device is None and isinstance(net, torch.nn.Module):
        device = list(net.parameters())[0].device 
    acc_sum, n = 0.0, 0
    with torch.no_grad():
        for X, y in data_iter:
            net.eval() # 评估模式, 这会关闭dropout
            acc_sum += (net(X.to(device)).argmax(dim=1) == y.to(device)).float().sum().cpu().item()
            net.train() # 改回训练模式
            n += y.shape[0]
    if acc_sum/n > best:
        best = acc_sum / n
        torch.save(net, 'ExistModel.pt')
    return acc_sum / n

def train(train_iter, test_iter, net, loss, optimizer, device, num_epochs):
    net = net.to(device)
    print("training on ", device)
    batch_count = 0
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n, start = 0.0, 0.0, 0, time.time()
        for X, y in train_iter:
            X = X.to(device)
            y = y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y) 
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            train_l_sum += l.cpu().item()
            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().cpu().item()
            n += y.shape[0]
            batch_count += 1
        test_acc = evaluate_accuracy(test_iter, net)
        print('epoch %d, loss %.4f, train acc %.4f, test acc %.4f, time %.1f sec'
              % (epoch + 1, train_l_sum / batch_count, train_acc_sum / n, test_acc, time.time() - start))

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224,0.225])
train_augs = transforms.Compose([
transforms.RandomResizedCrop(size=224),
transforms.RandomHorizontalFlip(),
transforms.ToTensor(),
normalize
])



test_augs = transforms.Compose([
transforms.Resize(size=256),
transforms.CenterCrop(size=224),
transforms.ToTensor(),
normalize
])

pretrained_net = models.resnet18(pretrained=True)
pretrained_net.fc = nn.Linear(512, 3)#因为是3分类

#学习率设置
output_params = list(map(id, pretrained_net.fc.parameters()))
feature_params = filter(lambda p: id(p) not in output_params, pretrained_net.parameters())
lr = 0.01
optimizer = optim.SGD([{'params': feature_params}, {'params': pretrained_net.fc.parameters(), 'lr': lr *10}],lr=lr, weight_decay=0.001)
loss = torch.nn.CrossEntropyLoss()

def train_fine_tuning(net, optimizer, batch_size=128, num_epochs=50):
    test_imgs = ImageFolder(r'../input/aitus-data/NumData/test', transform=test_augs)
    test_iter = DataLoader(test_imgs, 1, shuffle=False)
    train_iter = DataLoader(ImageFolder(r'../input/aitus-data/NumData/train', transform=train_augs), batch_size, shuffle=True)
    train(train_iter, test_iter, net, loss, optimizer, device, num_epochs)   
train_fine_tuning(pretrained_net, optimizer)
