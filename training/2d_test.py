import os
import sys
from torchvision.models import resnet18
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from glob import glob
import torch
import random
from training import train
from parameters import Parameters
from dataset import AdniDataset
import time


def count_classes(list):
    ad = 0
    for element in list:
        if 'AD' in element:
            ad += 1
    return ad, len(list) - ad


if __name__ == '__main__':
    save_path = './../results/Run_{}'.format(time.strftime("%Y%m%d_%H%M%S"))
    os.makedirs(save_path)
    param = Parameters()
    param.save(save_path, 'Parameters')
    net2d = resnet18(weights='DEFAULT')
    net2d.fc = torch.nn.Linear(512, 2)
    random.seed(111)
    x = glob('D:/preprocessed_OR/2d/*/*.npy')
    random.shuffle(x)

    train_proc = int(len(x) * param.training_dataset)
    val_proc = int(len(x) * param.validation_dataset)
    test_proc = int(len(x) * param.test_dataset)

    ad, cn = count_classes(x[:train_proc])
    print('Training - {} AD and {} CN'.format(ad, cn))
    ad, cn = count_classes(x[train_proc:(train_proc + val_proc)])
    print('Validation - {} AD and {} CN'.format(ad, cn))
    ad, cn = count_classes(x[(train_proc + val_proc):])
    print('Test - {} AD and {} CN'.format(ad, cn))

    dataset_train = AdniDataset(x[:train_proc], (224, 224, 3))
    dataloader_train = DataLoader(dataset_train, batch_size=param.batch_size, shuffle=True)
    dataset_validation = AdniDataset(x[train_proc:(train_proc + val_proc)], (224, 224, 3))
    dataloader_validation = DataLoader(dataset_validation, batch_size=param.batch_size, shuffle=True)
    dataset_test = AdniDataset(x[(train_proc + val_proc):], (224, 224, 3))
    dataloader_test = DataLoader(dataset_test, batch_size=1, shuffle=True)

    optimizer = None
    if param.optim_fcn == 'adam':
        optimizer = torch.optim.Adam([
            {'params': net2d.parameters()}
        ], lr=param.learning_rate, weight_decay=param.weight_decay)
    elif param.optim_fcn == 'sgd':
        optimizer = torch.optim.SGD([
            {'params': net2d.parameters()}
        ], lr=param.learning_rate, weight_decay=param.weight_decay)
    elif param.optim_fcn == 'adagrad':
        optimizer = torch.optim.Adagrad([
            {'params': net2d.parameters()}
        ], lr=param.learning_rate, weight_decay=param.weight_decay)
    else:
        print('Wrong optim function!')
        sys.exit()

    scheduler = lr_scheduler.StepLR(optimizer, step_size=param.scheduler_step_size, gamma=param.scheduler_gama)

    loss_function = None
    if param.loss_fcn == 'cross_entropy':
        loss_function = torch.nn.CrossEntropyLoss
    elif param.loss_fcn == 'mse':
        loss_function = torch.nn.MSELoss
    else:
        print('Wrong loss function!')
        sys.exit()

    train(net2d, dataloader_train, dataloader_validation, dataloader_test, optimizer, param.num_of_epochs,
          loss_function, scheduler,
          save_path)
