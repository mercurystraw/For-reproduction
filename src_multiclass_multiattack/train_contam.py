from __future__ import absolute_import
from __future__ import print_function

import os
import json
import sys
from tqdm import tqdm
import torch
import torch.nn as nn

from data_utils import load_data, change_label,change_label_multiattack
from attack_multi_utils import poison
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
_,testset_clean = load_data(config)
testset_clean = change_label_multiattack(testset_clean, ATTACKS)

# Load in attack data
if not os.path.isdir('multiattacks'):
    print('Attack images not found, please craft attack images first!')
    sys.exit(0)

# 存储每种攻击的测试数据
attack_loaders = {}  # 使用字典来存储每个攻击的 DataLoader

# 存储攻击名称字符串的列表
attack_names = []
all_trainset_poisoned = []
for attack_name, attack_config in ATTACKS.items():

    trainset, testset = load_data(attack_config)
    print("训练前标签：", trainset.targets[:100])


    trainset = change_label_multiattack(trainset, ATTACKS)
    print("change_label后标签：", trainset.targets[:100])
    testset = change_label_multiattack(testset, ATTACKS)


    attack_names.append(attack_name)
    print(f'Loading {attack_name} attack data...')

    C0_attack = attack_config['C0']
    C1_attack = attack_config['C1']
    C1_attack_str = '_'.join(map(str, C1_attack))
    # Load crafted attack images 添加投毒图像
    train_attacks = torch.load(
        f'./multiattacks/train_multiattacks_{data}_{attack_name}_C0_{C0_attack}_C1_{C1_attack_str}')
    train_images_attacks = train_attacks['image']
    train_labels_attacks = train_attacks['label']

    test_attacks = torch.load(
        f'./multiattacks/test_multiattacks_{data}_{attack_name}_C0_{C0_attack}_C1_{C1_attack_str}')
    test_images_attacks = test_attacks['image']
    test_labels_attacks = test_attacks['label']
    
    if test_images_attacks is None or test_labels_attacks is None:
        print(f"Error: Loaded attack data for {attack_name} is None.")
        sys.exit(1)

    if not isinstance(test_images_attacks, torch.Tensor) or not isinstance(test_labels_attacks, torch.Tensor):
        print(f"Error: Loaded attack data for {attack_name} is not a valid tensor.")
        sys.exit(1)
    # 检查是否已经有数据，如果没有，则直接赋值；如果有，则拼接
    # all_test_images_attacks = test_images_attacks if all_test_images_attacks is None else torch.cat(
    #     (all_test_images_attacks, test_images_attacks), dim=0)
    # all_test_labels_attacks = test_labels_attacks if all_test_labels_attacks is None else torch.cat(
    #     (all_test_labels_attacks, test_labels_attacks), dim=0)



    # attackloader创建
    testset_attacks = torch.utils.data.TensorDataset(test_images_attacks, test_labels_attacks)
    # 创建对应的 DataLoader
    attackloader = torch.utils.data.DataLoader(testset_attacks, batch_size=config['BATCH_SIZE'], shuffle=False,
                                               num_workers=8)
    # 将当前攻击的 DataLoader 存储到字典中
    attack_loaders[attack_name] = attackloader


    # Poison the training set
    ind_train = torch.load(f'./multiattacks/ind_train_{data}_{attack_name}_C0_{C0_attack}_C1_{C1_attack_str}')
    trainset_poisoned = poison(trainset, train_images_attacks, train_labels_attacks, ind_train, data)

    # 将被污染的训练集添加到列表中
    all_trainset_poisoned.append(trainset_poisoned)

# 拼接所有的被污染的训练集
trainset_poisoned_final = torch.utils.data.ConcatDataset(all_trainset_poisoned)


# Load in the datasets with data loaders
trainloader = torch.utils.data.DataLoader(trainset_poisoned_final, batch_size=config['BATCH_SIZE'], shuffle=True, num_workers=8)
testloader = torch.utils.data.DataLoader(testset_clean, batch_size=config['BATCH_SIZE'], shuffle=False, num_workers=8)

# attackloader = torch.utils.data.DataLoader(testset_attacks, batch_size=config['BATCH_SIZE'], shuffle=False, num_workers=8)


print("DataLoater全部完成！")

# Model selection
if config['MODEL_TYPE'] == 'resnet18':
    net = ResNet18(num_classes=2)
elif config['MODEL_TYPE'] == 'vgg11':
    net = VGG11(num_classes=2, in_channels=1)  # Adapt for grayscale images (like MNIST)
elif config['MODEL_TYPE'] == 'lenet5':
    net = LeNet5(num_classes=2)
else:
    sys.exit("Unknown model_type!")     # Ensure model type is specified

    
    


# Move model to the correct device (GPU or CPU)
net = net.to(device)

# Set the loss function and optimizer
criterion = nn.CrossEntropyLoss()
# optimizer = torch.optim.Adam(net.parameters(), lr=config['LR'])
optimizer = torch.optim.SGD(net.parameters(), lr=config['LR'], momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)

# Training function
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
    print('Train LOSS:%6f' % train_loss)

    return net

# Evaluation function for clean test set
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

# Evaluation function for attack success rate (ASR)
def eval_attack(attackloader):
    net.eval()
    correct = 0
    total = 0
    all_predictions = []  # 用于存储所有预测结果
    all_targets = []      # 用于存储所有真实标签

    with torch.no_grad():
        for batch_idx, (inputs, targets) in tqdm(enumerate(attackloader, 0), total=len(attackloader), smoothing=0.9):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)

            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            # 存储所有预测和真实标签
            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

            # 打印当前批次的预测和真实标签
            print(f'Batch {batch_idx+1}/{len(attackloader)} - Predictions: {predicted.cpu().numpy()}, Targets: {targets.cpu().numpy()}')

    asr = 100. * correct / total
    print('Total Correct Predictions:', correct)
    print('Total Samples Evaluated:', total)
    print('ASR: %.3f' % asr)

    return asr
#     net.eval()
#     correct = 0
#     total = 0
#     with torch.no_grad():
#         for batch_idx, (inputs, targets) in tqdm(enumerate(attackloader, 0), total=len(attackloader), smoothing=0.9):
#             inputs, targets = inputs.to(device), targets.to(device)
#             outputs = net(inputs)

#             _, predicted = outputs.max(1)
#             total += targets.size(0)
#             correct += predicted.eq(targets).sum().item()

#     asr = 100. * correct / total
#     print('ASR: %.3f' % asr)

#     return asr
    


# Main training loop
for epoch in range(config['EPOCH']):
    model_contam = train(epoch)
    acc = eval_clean()
    for attack_name, attack_loader in attack_loaders.items():
        print(f'Evaluating attack: {attack_name}')
        attack_asr = eval_attack(attack_loader)  # 对应的attack_loader

        print(f'Attack: {attack_name}, ASR: {attack_asr:.3f}')

    # asr = eval_attack(attackloader)

print('Clean test accuracy: %.3f' % acc)

# print('Attack success rate: %.3f' % asr)

# Save the trained model with appropriate naming
attack_names_string = '_'.join(attack_names)
model_dir = './trained_models/multiattacks_contam_models'
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

model_path = os.path.join(model_dir, f'train_multi_contam_multiclasses_model_{data}_{attack_names_string}.pth')
torch.save(model_contam.state_dict(), model_path)
print(f"Model saved to {model_path}")
