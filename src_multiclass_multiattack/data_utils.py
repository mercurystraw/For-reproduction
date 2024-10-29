
import sys
import numpy as np
import torchvision
import torchvision.transforms as transforms


def load_data(config):

    classes = {config['C0'], *config['C1']} # 普通类

    print(f"load_data中考虑的类classes有: {classes}")
    if 'SUPER_C0' in config:
        super_classes = [config['SUPER_C0'], config['SUPER_C1']] # 超类

    # if config['PATTERN_TYPE'] == 'perturbation': 默认使用perturbation
    transform_train = transforms.Compose([transforms.ToTensor()])   # Training set augmentation may be harmful to the success of perturbation backdoor pattern
    # else:
    #     transform_train = transforms.Compose([transforms.RandomHorizontalFlip(),
    #                                           transforms.RandomRotation(30),
    #                                           transforms.ToTensor()])
    transform_test = transforms.Compose([transforms.ToTensor()])

    if config['DATASET'] == 'cifar10':
        trainset = torchvision.datasets.CIFAR10(root='./data/cifar10', train=True, download=True, transform=transform_train)
        testset = torchvision.datasets.CIFAR10(root='./data/cifar10', train=False, download=True, transform=transform_test)
    elif config['DATASET'] == 'cifar100':
        trainset = torchvision.datasets.CIFAR100(root='./data/cifar100', train=True, download=True, transform=transform_train)
        testset = torchvision.datasets.CIFAR100(root='./data/cifar100', train=False, download=True, transform=transform_test)
    elif config['DATASET'] == 'stl10':
        trainset = torchvision.datasets.STL10(root='./data/stl10', split='train', download=True, transform=transform_train)
        testset = torchvision.datasets.STL10(root='./data/stl10', split='test', download=True, transform=transform_test)
    elif config['DATASET'] == 'fmnist':
        trainset = torchvision.datasets.FashionMNIST(root='./data/fmnist', train=True, download=True, transform=transform_train)
        testset = torchvision.datasets.FashionMNIST(root='./data/fmnist', train=False, download=True, transform=transform_test)
    elif config['DATASET'] == 'mnist':
        trainset = torchvision.datasets.MNIST(root='./data/mnist', train=True, download=True, transform=transform_train)
        testset = torchvision.datasets.MNIST(root='./data/mnist', train=False, download=True, transform=transform_test)
    else:
        sys.exit("Unknown dataset!")
        
    
    
    if config['DATASET'] in ['cifar10', 'cifar100', 'fmnist', 'mnist']:
        print("保留类:",classes)
        # print(f"训练集所有标签: {set(trainset.targets.numpy())}") 
        ind = [i for i, label in enumerate(trainset.targets) if int(label) not in classes]
        print(len(ind))
        print(ind)
        trainset.data = np.delete(trainset.data, ind, axis=0)
        trainset.targets = np.delete(trainset.targets, ind, axis=0)
        # print(f"测试集所有标签: {set(trainset.targets.numpy())}") 
        ind = [i for i, label in enumerate(testset.targets) if int(label) not in classes]
        print(len(ind))
        print(ind)
        testset.data = np.delete(testset.data, ind, axis=0)
        testset.targets = np.delete(testset.targets, ind, axis=0)
    elif config['DATASET'] == 'stl10':
        ind = [i for i, label in enumerate(trainset.labels) if label not in super_classes[0] + super_classes[1]]
        trainset.data = np.delete(trainset.data, ind, axis=0)
        trainset.labels = np.delete(trainset.labels, ind, axis=0)
        ind = [i for i, label in enumerate(testset.labels) if label not in super_classes[0] + super_classes[1]]
        testset.data = np.delete(testset.data, ind, axis=0)
        testset.labels = np.delete(testset.labels, ind, axis=0)

    return trainset, testset


def change_label(dataset, config):
    print("攻击设置的C0：", config['C0'])
    print("攻击设置的C1：", config['C1'])

    if config['DATASET'] in ['cifar10', 'cifar100', 'fmnist', 'mnist']:
        # ind_0 = [i for i, label in enumerate(dataset.targets) if label == config['C0']]
        # ind_1 = [i for i, label in enumerate(dataset.targets) if label == config['C1']]
        # dataset.targets[ind_0] = 0
        # dataset.targets[ind_1] = 1
        ind_0 = [i for i, label in enumerate(dataset.targets) if int(label) == config['C0']]
        ind_1 = [i for i, label in enumerate(dataset.targets) if int(label) in config['C1']]
        dataset.targets[ind_0] = 0
        dataset.targets[ind_1] = 1

    elif config['DATASET'] == 'stl10':
        ind_0 = [i for i, label in enumerate(dataset.labels) if int(label) in config['SUPER_C0']]
        ind_1 = [i for i, label in enumerate(dataset.labels) if int(label) in config['SUPER_C1']]
        dataset.labels[ind_0] = 0
        dataset.labels[ind_1] = 1

    return dataset

def change_label_multiattack(dataset, ATTACKS):
    c0_list = []  # 初始化一个空列表
    for _,attack_config in ATTACKS.items():
        if 'C0' in attack_config:
            c0_list.append(attack_config['C0'])
        data=attack_config['DATASET']
    print("训练设置的类0为C0——list：",c0_list)

    if data in ['cifar10', 'cifar100', 'fmnist', 'mnist']:
        # ind_0 = [i for i, label in enumerate(dataset.targets) if label == config['C0']]
        # ind_1 = [i for i, label in enumerate(dataset.targets) if label == config['C1']]
        # dataset.targets[ind_0] = 0
        # dataset.targets[ind_1] = 1
        ind_0 = [i for i, label in enumerate(dataset.targets) if int(label) in c0_list]
        ind_1 = [i for i, label in enumerate(dataset.targets) if int(label) not in c0_list]
        dataset.targets[ind_0] = 0
        dataset.targets[ind_1] = 1

    # elif data == 'stl10':
    #     ind_0 = [i for i, label in enumerate(dataset.labels) if label in config['SUPER_C0']]
    #     ind_1 = [i for i, label in enumerate(dataset.labels) if label in config['SUPER_C1']]
    #     dataset.labels[ind_0] = 0
    #     dataset.labels[ind_1] = 1

    return dataset