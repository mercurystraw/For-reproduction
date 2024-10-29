"""
Train model on the clean training set
Author: Zhen Xiang
"""

from __future__ import absolute_import
from __future__ import print_function

import json
import sys
from tqdm import tqdm
import torch
import torch.nn as nn

from data_utils import load_data, change_label,change_label_multiattack
from model_zoo.resnet import ResNet18
from model_zoo.lenet5 import LeNet5
from model_zoo.vgg import VGG11

# Detect device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load attack configuration
with open('config_multi.json') as config_file:
    config = json.load(config_file)

DATASET = config['DATASET']
MODEL_TYPE = config['MODEL_TYPE']
EPOCH = config['EPOCH']
BATCH_SIZE = config['BATCH_SIZE']
LR = config['LR']
NUM_POISONING_SAMPLE = config['NUM_POISONING_SAMPLE']
PATIENCE = config['PATIENCE']
NUM_IMG_WARMUP = config['NUM_IMG_WARMUP']
NUM_IMG_DETECTION = config['NUM_IMG_DETECTION']
CONF_INIT = config['CONF_INIT']

ATTACKS = config['ATTACKS']

C0 = config['C0']
C1 = config['C1']
C1_str = '_'.join(map(str, C1))  # Combine C1 values into a string
data = config['DATASET']

# Load raw data and keep only two classes

# Change the labels to 0 or 1

trainset_clean, testset_clean = load_data(config)

trainset_clean = change_label_multiattack(trainset_clean, ATTACKS)
testset_clean = change_label_multiattack(testset_clean, ATTACKS)


# Load in the datasets
trainloader = torch.utils.data.DataLoader(trainset_clean, batch_size=config['BATCH_SIZE'], shuffle=True, num_workers=2)
testloader = torch.utils.data.DataLoader(testset_clean, batch_size=config['BATCH_SIZE'], shuffle=False, num_workers=2)

# Model
if config['MODEL_TYPE'] == 'resnet18':
    net = ResNet18(num_classes=2)
elif config['MODEL_TYPE'] == 'vgg11':
    net = VGG11(num_classes=2, in_channels=1)
elif config['MODEL_TYPE'] == 'lenet5':
    net = LeNet5(num_classes=2)
else:
    sys.exit("Unknown model_type!")     # Please specify other model types in advance
net = net.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=config['LR'])
# optimizer = torch.optim.SGD(net.parameters(), lr=config['LR'], momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)


# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in tqdm(enumerate(trainloader, 0), total=len(trainloader), smoothing=0.9):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    scheduler.step()

    acc = 100. * correct / total
    print('Train ACC: %.3f' % acc)
    print('Train LOSS:%.6f' % train_loss)

    return net


# Test
def eval_clean():
    global best_acc
    net.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in tqdm(enumerate(testloader, 0), total=len(testloader), smoothing=0.9):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)

            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    acc = 100. * correct / total
    print('Test ACC: %.3f' % acc)

    return acc


for epoch in range(config['EPOCH']):
    model_clean = train(epoch)
    acc = eval_clean()

print('Clean test accuracy: %.3f' % acc)


import os

# Save model
# 确保目录存在
model_dir = './trained_models/clean_models'
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

# 生成带有C0和C1标记的文件名
model_path = os.path.join(model_dir, f'train_clean_multiclasses_model_{data}_6666.pth')

# 保存模型
torch.save(model_clean.state_dict(), model_path)