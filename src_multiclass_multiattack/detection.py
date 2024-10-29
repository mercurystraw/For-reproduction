from __future__ import absolute_import
from __future__ import print_function

import torch
import torch.nn as nn

import json
import numpy as np
import sys

from data_utils import load_data, change_label
from model_zoo.resnet import ResNet18
from model_zoo.lenet5 import LeNet5
from model_zoo.vgg import VGG11

from detection_utils import pert_est, get_MF_pert, pm_est, get_MF_patch,show_pert


device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load attack configuration
with open('config_multi.json') as config_file:
    config = json.load(config_file)

classes = {config['C0'], *config['C1']}  # 使用集合解包操作

print(f"detection.py中考虑的类有：{classes}")

DATASET = config['DATASET']
MODEL_TYPE = config['MODEL_TYPE']
EPOCH = config['EPOCH']
BATCH_SIZE = config['BATCH_SIZE']
# LR = config['LR']
# NUM_POISONING_SAMPLE = config['NUM_POISONING_SAMPLE']
# PATIENCE = config['PATIENCE']
# NUM_IMG_WARMUP = config['NUM_IMG_WARMUP']
# NUM_IMG_DETECTION = config['NUM_IMG_DETECTION']
# CONF_INIT = config['CONF_INIT']

ATTACKS = config['ATTACKS']

C0 = config['C0']
C1= config['C1']
C1_str = '_'.join(map(str, C1))  # Combine C1 values into a string
data = config['DATASET']

# Load raw data and keep only two classes
_, testset = load_data(config)

print("改变标签前的200个测试集标签:")
print(testset.targets[:200])  # 假设标签存储在 testset.targets 中

# Change the labels to 0 or 1
testset = change_label(testset, config)

print("改变标签后的测试集标签:")
print(testset.targets[:200])  # 打印改变后的标签

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

attack_names = []

for attack_name, attack_config in ATTACKS.items():
    attack_names.append(attack_name)
attack_names_string = '_'.join(attack_names)
# Load parameters
# model.load_state_dict(torch.load(
#     f'./trained_models/multiattacks_contam_models/train_multi_contam_multiclasses_model_{data}_{attack_names_string}.pth'))
# 干净训练测试
model.load_state_dict(torch.load(
    f'./trained_models/clean_models/train_clean_multiclasses_model_{data}_6666.pth'))
# 注意每次需要加载的模型！！
model.eval()

# Consider only images that are correctly classified with high confidence
# 筛选出那些被模型正确分类且置信度较高的图片
keep = []
with torch.no_grad():
    for batch_idx, (inputs, targets) in enumerate(testloader):
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        _, predicted = outputs.max(1)
        posterior = nn.Softmax(dim=1)(outputs)
        keep.extend(torch.logical_and((torch.max(posterior, dim=1)[0] > config['CONF_INIT']), predicted.eq(targets)).cpu().numpy())
        # 保留置信度大于配置文件设置的阈值且图片预测结果与真实标签一致的图片的索引
testset.data = testset.data[keep]

print("Keep array:", keep)
print("Keep array length:", len(keep), "Testset data length:", len(testset.data))
if len(keep) == 0:
    print("No images are kept, check your confidence threshold and predictions.")
    
if config['DATASET'] == 'stl10':
    testset.labels = testset.labels[keep]
else:
    testset.targets = testset.targets[keep]
    
print("Testset data shape:", testset.data.shape)  # 数据的形状
print("Testset targets shape:", testset.targets.shape if hasattr(testset, 'targets') else testset.labels.shape)  # 标签的形状
print("Total number of samples in testset:", len(testset.data))  # 样本总数

# Detection
NumImg = config['NUM_IMG_DETECTION']
# 指定在检测阶段要使用的图像数量
Anchor = 0.9 # 指定置信度阈值


for s in [0, 1]:
    # Get a subset of images for detection
    if config['DATASET'] == 'stl10':
        ind = [i for i, label in enumerate(testset.labels) if label == s]
    else:
        ind = [i for i, label   in enumerate(testset.targets) if label == s]
        
    print("随机选择前的ind：",ind)
    
    ind = np.random.choice(ind, np.min([len(ind), NumImg + config['NUM_IMG_WARMUP']]), replace=False)
    print("随机选择后的ind：",ind)
    # 在ind中选择 ind 的全部索引，或者 NumImg + config['NUM_IMG_WARMUP'] 的数量，以较少者为准。
    # NUM_IMG_WARMUP 用于指定在模式和掩码估计的预热阶段使用的图像数量 以便更好地初始化模式和掩码。
    # choice 随机选择一部分索引（抽样进行测试）
    images = testset.data[ind]
    target = int(1-s)
    # 二类域 一个作为受攻击的 一个为正常target

    # 先用perturbation算法估算ET1，再用patch算法估算ET2
    rho_avg = []
    for i in range(NumImg):
        # Iteratively estimate the pattern with random initialization
        rho_stat_cum = np.zeros(NumImg)
        
        print(f"pert这是第{i}轮，一共{NumImg}轮")
        
        rho_stat_cum[i] = 1.
        converge = False
        stable_count = 0
        while not converge:
            transfer_prob_old = (len(np.where(rho_stat_cum > 0)[0]) - 1) / (NumImg - 1)
            print("Transfer probability(Old):", transfer_prob_old)
            rho_stat = np.zeros(NumImg)

            # Backdoor pattern reverse-engineering (can be replaced by other algorithms)
            pert, rho = pert_est(images[[i]], model, config=config, t=target, pi=Anchor)
            
            show_pert(pert, config, i, s)
            
            
            
            # 对每张图片估计最优的扰动 对应论文中的v(xn)
            # Apply the estimated perturbation to the rest parts of images
            for j in range(NumImg):
                if j == i:
                    rho_stat[j] = rho
                    continue
                rho_stat[j] = get_MF_pert(pert, images[[j]], model, config=config, t=target)
            # Cumulate rho
            rho_stat_cum += rho_stat

            # Decide whether to stop
            transfer_prob_new = (len(np.where(rho_stat_cum > 0)[0]) - 1) / (NumImg - 1)
            
            print("Transfer probability(New):", transfer_prob_new)

            # np.where(rho_stat_cum > 0) 返回一个元组，其中包含满足条件rho_stat_cum > 0 的元素的索引。如果>0,说明在该元素在转移集中。-1则是为了去除自身
            # [0]表示取元组的第一个元素，即满足条件的索引数组。
            # 事实上如果rho_stat_cum是一个多维数组（例如二维）那么>0的式子会返回满足条件元素的行索引和列索引，以两个一维数组的形式，所以选取[0]或者其他通过len求数量

            if transfer_prob_new <= transfer_prob_old:
                stable_count += 1
            else:
                stable_count = 0
            if (stable_count >= config['PATIENCE']) or (transfer_prob_new == 1.0):
                converge = True

        rho_avg.append(transfer_prob_new)

    ET1 = np.mean(rho_avg)

    print(f"perturbation优化算法估计扰动下得到的ET1: {ET1}")

#     rho_avg = [] # 重新初始化 为了ET2的计算

#     # 以下是对patch的扰动估计算法
#     # Warm-up for pattern and mask estimation
#     # cifar10 stl10两个数据集相对多样化 预热可以具有较好的初始化效果
#     if (config['DATASET'] == 'cifar10' or config['DATASET'] == 'stl10'):
#         pattern_common, mask_common, _ = pm_est(images[NumImg:], model, config=config, t=target, pattern_init=None, mask_init=None, pi=Anchor)
#     else:
#         pattern_common, mask_common = None, None

#     for i in range(NumImg):
#         # Iteratively estimate the pattern with random initialization
#         print(f"patch这是第{i}轮，一共{NumImg}轮")
#         rho_stat_cum = np.zeros(NumImg)
#         rho_stat_cum[i] = 1.
#         converge = False
#         stable_count = 0
#         while not converge:
#             transfer_prob_old = (len(np.where(rho_stat_cum > 0)[0]) - 1) / (NumImg - 1)
#             print("Transfer probability(Old):", transfer_prob_old)
#             rho_stat = np.zeros(NumImg)
#             pattern, mask, rho = pm_est(images[[i]], model, config=config, t=target, pattern_init=pattern_common, mask_init=mask_common, pi=Anchor)

#             show_pert(pattern, config, i, s)


#             # Apply the estimated perturbation to all the other images
#             for j in range(NumImg):
#                 if j == i:
#                     rho_stat[j] = rho
#                     continue
#                 rho_stat[j] = get_MF_patch(pattern, mask, images[[j]], model, config=config, t=target)

#             # Cumulate rho
#             rho_stat_cum += rho_stat

#             # Decide whether to stop
#             transfer_prob_new = (len(np.where(rho_stat_cum > 0)[0]) - 1) / (NumImg - 1)
#             print("Transfer probability(New):", transfer_prob_new)
#             # np.where(rho_stat_cum > 0) 返回一个元组，其中包含满足条件rho_stat_cum > 0 的元素的索引。如果>0,说明在该元素在转移集中。-1则是为了去除自身
#             # [0]表示取元组的第一个元素，即满足条件的索引数组。
#             # 事实上如果rho_stat_cum是一个多维数组（例如二维）那么>0的式子会返回满足条件元素的行索引和列索引，以两个一维数组的形式，所以选取[0]或者其他通过len求数量

#             if transfer_prob_new <= transfer_prob_old:
#                 stable_count += 1
#             else:
#                 stable_count = 0
#             if (stable_count >= config['PATIENCE']) or (transfer_prob_new == 1.0):
#                 converge = True

#         rho_avg.append(transfer_prob_new)

    ET2 = np.mean(rho_avg)

    print(f"patch优化算法估计扰动下得到的ET2: {ET2}")

    ET = max(ET1, ET2)

    print(f"对{s}类进行检测，检测出的ET最大值为：{ET}")

    if(ET>0.5):
        print(f'检测出受到攻击的类为：{s}')
    else:
        print(f'类{s}未被检测出受到攻击')
        # 打印两类的检测统计值ET，
