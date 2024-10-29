"""
Create backdoor training samples
Author: Zhen Xiang
"""

from __future__ import absolute_import
from __future__ import print_function

import torch
import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt

from data_utils import load_data
from attack_utils import create_pattern, pattern_save, backdoor_embedding


# Create attack dir
if not os.path.isdir('attacks'):
    os.mkdir('attacks')

# Load attack configuration
with open('config.json') as config_file:
    config = json.load(config_file)


C0 = config['C0']
C1 = config['C1']
PATTERN_TYPE = config['PATTERN_TYPE']
data=config['DATASET']
attack=config['PATTERN_TYPE']
# Load raw data and keep only two classes
trainset, testset = load_data(config)

# Create the backdoor patterns
backdoor_pattern = create_pattern(im_size=trainset.__getitem__(0)[0].size(), config=config)

# Save a visualization of the backdoor pattern
pattern_save(pattern=backdoor_pattern, config=config, path='attacks')

# Create backdoor training images (for poisoning the training set) and backdoor test images (for attack effectiveness evaluation)
train_images_attacks = None
train_labels_attacks = None
test_images_attacks = None
test_labels_attacks = None
ind_train = None
if config['DATASET'] in ['cifar10', 'fmnist', 'mnist']:
    ind_train = [i for i, label in enumerate(trainset.targets) if label == config['C0']]
    ind_test = [i for i, label in enumerate(testset.targets) if label == config['C0']]
elif config['DATASET'] in ['stl10']:
    ind_train = [i for i, label in enumerate(trainset.labels) if label in config['SUPER_C0']]
    ind_test = [i for i, label in enumerate(testset.labels) if label in config['SUPER_C0']]
elif config['DATASET'] in ['cifar100']:
    ind_train = [i for i, label in enumerate(trainset.targets) if label in config['SUPER_C0']]
    ind_test = [i for i, label in enumerate(testset.targets) if label in config['SUPER_C0']]
else:
    sys.exit("Unknown dataset!")
ind_train = np.random.choice(ind_train, config['NUM_POISONING_SAMPLE'], False)

for i in ind_train:
    if train_images_attacks is not None:
        train_images_attacks = torch.cat([train_images_attacks, backdoor_embedding(image=trainset.__getitem__(i)[0], pattern=backdoor_pattern, config=config).unsqueeze(0)], dim=0)
        train_labels_attacks = torch.cat([train_labels_attacks, torch.tensor([1], dtype=torch.long)], dim=0)
    else:
        train_images_attacks = backdoor_embedding(image=trainset.__getitem__(i)[0], pattern=backdoor_pattern, config=config).unsqueeze(0)
        train_labels_attacks = torch.tensor([1], dtype=torch.long)

for i in ind_test:
    if test_images_attacks is not None:
        test_images_attacks = torch.cat([test_images_attacks, backdoor_embedding(image=testset.__getitem__(i)[0], pattern=backdoor_pattern, config=config).unsqueeze(0)], dim=0)
        test_labels_attacks = torch.cat([test_labels_attacks, torch.tensor([1], dtype=torch.long)], dim=0)
    else:
        test_images_attacks = backdoor_embedding(image=testset.__getitem__(i)[0], pattern=backdoor_pattern, config=config).unsqueeze(0)
        test_labels_attacks = torch.tensor([1], dtype=torch.long)


# Save created backdoor image
train_attacks = {'image': train_images_attacks, 'label': train_labels_attacks}
test_attacks = {'image': test_images_attacks, 'label': test_labels_attacks}
torch.save(train_attacks, f'./attacks/train_attacks_{data}_{attack}_C0_{C0}_C1_{C1}')
torch.save(test_attacks, f'./attacks/test_attacks_{data}_{attack}_C0_{C0}_C1_{C1}')
torch.save(ind_train, f'./attacks/ind_train_{data}_{attack}_C0_{C0}_C1_{C1}')

# Save example backdoor images for visualization



image_clean = trainset.__getitem__(ind_train[0])[0]
image_clean = image_clean.numpy()
image_clean = np.transpose(image_clean, [1, 2, 0])
if config['DATASET'] in ['cifar10', 'cifar100', 'stl10']:
    plt.imshow(image_clean)
else:
    plt.imshow(image_clean[:, :, 0], cmap='gray', vmin=0., vmax=1.)
plt.savefig(f'./attacks/image_clean_{data}_{attack}_C0_{C0}_C1_{C1}.png')
image_poisoned = train_images_attacks[0]
image_poisoned = image_poisoned.numpy()
image_poisoned = np.transpose(image_poisoned, [1, 2, 0])
if config['DATASET'] in ['cifar10', 'cifar100', 'stl10']:
    plt.imshow(image_poisoned)
else:
    plt.imshow(image_poisoned[:, :, 0], cmap='gray', vmin=0., vmax=1.)
plt.savefig(f'./attacks/image_poisoned_{data}_{attack}_C0_{C0}_C1_{C1}.png')

