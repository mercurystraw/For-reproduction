
from __future__ import absolute_import
from __future__ import print_function

import torch
import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt

from data_utils import load_data
from attack_multi_utils import create_pattern, pattern_save, backdoor_embedding

# Create attack dir
if not os.path.isdir('multiattacks'):
    os.mkdir('multiattacks')

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

# Load raw data
trainset, testset = load_data(config)


print("这是训练集的长度:",len(trainset))


# Iterate over each attack configuration
for attack_name, attack_config in ATTACKS.items():
    print(f"Processing attack: {attack_name}")

    C0 = attack_config['C0']
    C1 = attack_config['C1']
    C1_str = '_'.join(map(str, C1))  # For filename purposes


    # Create the backdoor patterns
    backdoor_pattern = create_pattern(im_size=trainset.__getitem__(0)[0].size(), config=attack_config)

    # Save a visualization of the backdoor pattern
    pattern_save(pattern=backdoor_pattern, config=attack_config, path='multiattacks', attack_name=attack_name)

    # Initialize containers for attacks
    train_images_attacks = None
    train_labels_attacks = None
    test_images_attacks = None
    test_labels_attacks = None

    # Select indices based on dataset and C0
    if DATASET in ['cifar10', 'fmnist', 'mnist']:
        ind_train = [i for i, label in enumerate(trainset.targets) if label == C0]
        ind_test = [i for i, label in enumerate(testset.targets) if label == C0]
    elif DATASET in ['stl10', 'cifar100']:
        ind_train = [i for i, label in enumerate(trainset.labels if DATASET == 'stl10' else trainset.targets) if
                     label in attack_config['SUPER_C0']]
        ind_test = [i for i, label in enumerate(testset.labels if DATASET == 'stl10' else testset.targets) if
                    label in attack_config['SUPER_C0']]
    else:
        sys.exit("Unknown dataset!")

    # Randomly select poisoning samples
    ind_train = np.random.choice(ind_train, NUM_POISONING_SAMPLE, False)

    # Embed backdoor pattern and modify labels for training samples
    for i in ind_train:
        poisoned_image = backdoor_embedding(
            image=trainset.__getitem__(i)[0],
            pattern=backdoor_pattern,
            config=attack_config
        ).unsqueeze(0)

        poisoned_label = torch.tensor([1], dtype=torch.long)  # 攻击设置标签变为1
        # 对于每个攻击 目标标签都为1，如果其不受检测，那么最终还是会change_label到1，如果其受到检测，那么它就受到攻击了（因为标签改为1）

        if train_images_attacks is not None:
            train_images_attacks = torch.cat([train_images_attacks, poisoned_image], dim=0)
            train_labels_attacks = torch.cat([train_labels_attacks, poisoned_label], dim=0)
        else:
            train_images_attacks = poisoned_image
            train_labels_attacks = poisoned_label

    # Similarly, embed backdoor pattern and modify labels for test samples
    for i in ind_test:
        poisoned_image = backdoor_embedding(
            image=testset.__getitem__(i)[0],
            pattern=backdoor_pattern,
            config=attack_config
        ).unsqueeze(0)

        poisoned_label = torch.tensor([1], dtype=torch.long)  # Assuming single target per attack

        if test_images_attacks is not None:
            test_images_attacks = torch.cat([test_images_attacks, poisoned_image], dim=0)
            test_labels_attacks = torch.cat([test_labels_attacks, poisoned_label], dim=0)
        else:
            test_images_attacks = poisoned_image
            test_labels_attacks = poisoned_label

    # Save created backdoor images
    train_attacks = {'image': train_images_attacks, 'label': train_labels_attacks}
    test_attacks = {'image': test_images_attacks, 'label': test_labels_attacks}
    torch.save(train_attacks, f'./multiattacks/train_multiattacks_{DATASET}_{attack_name}_C0_{C0}_C1_{C1_str}')
    torch.save(test_attacks, f'./multiattacks/test_multiattacks_{DATASET}_{attack_name}_C0_{C0}_C1_{C1_str}')
    torch.save(ind_train, f'./multiattacks/ind_train_{DATASET}_{attack_name}_C0_{C0}_C1_{C1_str}')

    # Save example backdoor images for visualization
    image_clean = trainset.__getitem__(ind_train[0])[0].numpy()
    image_clean = np.transpose(image_clean, [1, 2, 0])

    if DATASET in ['cifar10', 'cifar100', 'stl10']:
        plt.imshow(image_clean)
    else:
        plt.imshow(image_clean[:, :, 0], cmap='gray', vmin=0., vmax=1.)
    plt.savefig(f'./multiattacks/image_clean_{attack_name}_C0_{C0}_C1_{C1_str}.png')

    image_poisoned = train_images_attacks[0].numpy()
    image_poisoned = np.transpose(image_poisoned, [1, 2, 0])

    if DATASET in ['cifar10', 'cifar100', 'stl10']:
        plt.imshow(image_poisoned)
    else:
        plt.imshow(image_poisoned[:, :, 0], cmap='gray', vmin=0., vmax=1.)
    plt.savefig(f'./multiattacks/image_poisoned_{attack_name}_C0_{C0}_C1_{C1_str}.png')

    print(f"Attack {attack_name} processed and saved.\n")

print("All attacks have been processed.")
