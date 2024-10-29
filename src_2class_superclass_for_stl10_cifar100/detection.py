from __future__ import absolute_import
from __future__ import print_function
from datetime import datetime, timedelta
import torch
import torch.nn as nn

import json
import numpy as np
import sys
import os
from data_utils import load_data, change_label
from model_zoo.resnet import ResNet18
from model_zoo.lenet5 import LeNet5
from model_zoo.vgg import VGG11

from detection_utils import pert_est, get_MF_pert, pm_est, get_MF_patch,show_pert


device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load attack configuration
with open('config.json') as config_file:
    config = json.load(config_file)


classes = [config['C0'], config['C1']]
data=config['DATASET']

Super_C0 = '_'.join(map(str, config["SUPER_C0"]))
Super_C1 = '_'.join(map(str, config["SUPER_C1"]))

# Load raw data and keep only two classes
_, testset = load_data(config)

print("原始测试集的大小:", len(testset))
print("改变标签前的1900个测试集标签:")
print(testset.targets[:1900])  # 假设标签存储在 testset.targets 中
# print(testset.labels[:1900])  # stl10专用

# Change the labels to 0 or 1
testset = change_label(testset, config)

print("改变标签后的测试集标签:")
print(testset.targets[:1900])  # 假设标签存储在 testset.targets 中
# print(testset.labels[:1900])  # stl10

# Create test loader
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

# Model
if config['MODEL_TYPE'] == 'resnet18':
    model = ResNet18(num_classes=2)
elif config['MODEL_TYPE'] == 'vgg11':
    model = VGG11(num_classes=2, in_channels=1)
elif config['MODEL_TYPE'] == 'lenet5':
    model = LeNet5(num_classes=2)
else:
    sys.exit("Unknown model_type!")     # Please specify other model types in advance
model = model.to(device)

# Load parameters 注意训练.py文件中保存的模型路径
model.load_state_dict(torch.load(f'./trained_models/contam_models/train_contam_2_super_classes_model_{data}_SUPERC0_{Super_C0}_SUPERC1_{Super_C1}.pth'))


model.eval()

# Consider only images that are correctly classified with high confidence
keep = []
with torch.no_grad():
    for batch_idx, (inputs, targets) in enumerate(testloader):
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        _, predicted = outputs.max(1)
        posterior = nn.Softmax(dim=1)(outputs)
        keep.extend(torch.logical_and((torch.max(posterior, dim=1)[0] > config['CONF_INIT']), predicted.eq(targets)).cpu().numpy())
        # keep = np.logical_and(torch.max(posterior, dim=1)[0] > config['CONF_INIT'], predicted.eq(targets)).cpu().numpy()

        # 保留置信度大于配置文件设置的阈值的图片的索引
print("Keep array:", keep)
print("Keep array length:", len(keep), "Testset data length:", len(testset.data))
if len(keep) == 0:
    print("No images are kept, check your confidence threshold and predictions.")
testset.data = testset.data[keep]

if config['DATASET'] == 'stl10':
    testset.labels = testset.labels[keep]
else:
    testset.targets = testset.targets[keep]

print("Testset data shape:", testset.data.shape)  # 数据的形状
print("Testset targets shape:", testset.targets.shape if hasattr(testset, 'targets') else testset.labels.shape)  # 标签的形状
print("Total number of samples in testset:", len(testset.data))  # 样本总数
# Detection
NumImg = config['NUM_IMG_DETECTION']
Anchor = 0.9 # 指定置信度阈值
for s in [0, 1]:
    print(f"开始检测第{s}类")
    # Get a subset of images for detection
    if config['DATASET'] == 'stl10':
        ind = [i for i, label in enumerate(testset.labels) if label == s]
    else:
        ind = [i for i, label   in enumerate(testset.targets) if label == s]
    ind = np.random.choice(ind, np.min([len(ind), NumImg + config['NUM_IMG_WARMUP']]), replace=False)
    # NUM_IMG_WARMUP 用于指定在模式和掩码估计的预热阶段使用的图像数量 以便更好地初始化模式和掩码。
    # choice 随机选择一部分索引（抽样进行测试）
    images = testset.data[ind]
    if images.size == 0:
        print("Error: No images available for detection.")
    target = int(1 - s)
    # 二类域 一个作为受攻击的 一个为正常target
    # Warm-up for pattern and mask estimation
    if config['PATTERN_TYPE'] == 'patch' and (config['DATASET'] == 'cifar10' or config['DATASET'] == 'stl10'):
        pattern_common, mask_common, _ = pm_est(images[NumImg:], model, config=config, t=target, pattern_init=None, mask_init=None, pi=Anchor)
    else:
        pattern_common, mask_common = None, None

    rho_avg = []
    for i in range(NumImg):
        # Iteratively estimate the pattern with random initialization
        print(f"这是第{i}轮，一共{NumImg}轮")
        rho_stat_cum = np.zeros(NumImg)
        rho_stat_cum[i] = 1.
        converge = False
        stable_count = 0
        while not converge:
            transfer_prob_old = (len(np.where(rho_stat_cum > 0)[0]) - 1) / (NumImg - 1)
            print("Transfer probability(Old):", transfer_prob_old)
            rho_stat = np.zeros(NumImg)

            # Backdoor pattern reverse-engineering (can be replaced by other algorithms)
            if config['PATTERN_TYPE'] == 'perturbation':
                pert, rho = pert_est(images[[i]], model, config=config, t=target, pi=Anchor)

                show_pert(pert,config,i,s)

                # Apply the estimated perturbation to the rest parts of images
                for j in range(NumImg):
                    if j == i:
                        rho_stat[j] = rho
                        continue
                    rho_stat[j] = get_MF_pert(pert, images[[j]], model, config=config, t=target)
            else:
                pattern, mask, rho = pm_est(images[[i]], model, config=config, t=target, pattern_init=pattern_common, mask_init=mask_common, pi=Anchor)

                show_pert(pattern, config,i,s)
                print(f"图片{i}的mask：{mask}")

                # Apply the estimated perturbation to all the other images
                for j in range(NumImg):
                    if j == i:
                        rho_stat[j] = rho
                        continue
                    rho_stat[j] = get_MF_patch(pattern, mask, images[[j]], model, config=config, t=target)

            # Cumulate rho
            rho_stat_cum += rho_stat

            print("rho_stat_cum:", rho_stat_cum)
            # Decide whether to stop
            transfer_prob_new = (len(np.where(rho_stat_cum > 0)[0]) - 1) / (NumImg - 1)

            print("Transfer probability(New):", transfer_prob_new)
            
            if transfer_prob_new <= transfer_prob_old:
                stable_count += 1
            else:
                stable_count = 0
            if (stable_count >= config['PATIENCE']) or (transfer_prob_new == 1.0):
                converge = True

        rho_avg.append(transfer_prob_new)

    ET = np.mean(rho_avg)
    print('Detection stat ET: {}'.format(ET))
    
    # 获取当前时间并转换为北京时区
    utc_now = datetime.utcnow()  # 获取当前 UTC 时间
    beijing_now = utc_now + timedelta(hours=8)  # 北京时间为 UTC+8

    # 打开文件并写入 ET 和时间
    file_path = 'output.txt'

# 检查文件是否存在
    if not os.path.exists(file_path):
        print('文件不存在，将创建新文件。')

    # 打开文件并写入 ET 和时间
    with open(file_path, 'a') as file:  # 'a' 模式为追加写入
        file.write('ET: {}\n'.format(ET))
        file.write('检测的类为:{}\n'.format(s))
        file.write('时间: {}\n'.format(beijing_now.strftime('%Y-%m-%d %H:%M:%S')))
        
    if(ET>0.5):
        print('检测出受到攻击的类为：{}'.format(s))
    else:
        print('未找到受攻击的类！')

    # 打印两类的检测统计值ET，
