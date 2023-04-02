import os
from glob import glob
import numpy as np
import sys
from torchvision.models import resnet18
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
import torch
import random
import time
from training import train
from parameters import Parameters
from dataset import AdniDataset
from image_processing import *


def count_classes(list):
    ad = 0
    for element in list:
        if 'AD' in element:
            ad += 1
    return ad, len(list) - ad


if __name__ == '__main__':
    save_path = './../results/Run_{}'.format(time.strftime("%Y%m%d_%H%M%S"))
    os.mkdir(save_path)
    param = Parameters()
    param.save(save_path, 'Parameters')
    net2d = resnet18(weights="DEFAULT")
    net2d.fc = torch.nn.Linear(512, 2)
    net2d.load_state_dict(torch.load('./../results/Run_20230329_155858/model_6.pth'))
    random.seed(111)
    data_path = 'D:/preprocessed_OR/2d/*/*.npy'
    #  x = np.asarray(range(len(glob(data_path))))
    x = glob(data_path)
    AD_list_ids = []
    NC_list_ids = []
    for filename in x:
        image_id = get_image_id_from_filename(filename)
        if 'AD_rgb' in filename:
            if len(AD_list_ids) == 0 or AD_list_ids[len(AD_list_ids) - 1] != image_id:
                AD_list_ids.append(image_id)
        else:
            if len(NC_list_ids) == 0 or NC_list_ids[len(NC_list_ids) - 1] != image_id:
                NC_list_ids.append(image_id)

    train_AD_list_ids = AD_list_ids[:int(0.7*len(AD_list_ids))]
    validation_AD_list_ids = AD_list_ids[int(0.7 * len(AD_list_ids)):int(0.9 * len(AD_list_ids))]
    test_AD_list_ids = AD_list_ids[int(0.9 * len(AD_list_ids)):]

    train_NC_list_ids = NC_list_ids[:int(0.7 * len(NC_list_ids))]
    validation_NC_list_ids = NC_list_ids[int(0.7 * len(NC_list_ids)):int(0.9 * len(NC_list_ids))]
    test_NC_list_ids = NC_list_ids[int(0.9 * len(NC_list_ids)):]
    """
    print("Number of ids in train NC ", len(train_NC_list_ids))
    print("Number of ids in val NC ", len(validation_NC_list_ids))
    print("Number of ids in test NC ", len(test_NC_list_ids))
    print("Number of ids in train AD ", len(train_AD_list_ids))
    print("Number of ids in val AD ", len(validation_AD_list_ids))
    print("Number of ids in test AD ", len(test_AD_list_ids))

    train_proc = int(len(x) * param.training_dataset)
    val_proc = int(len(x) * param.validation_dataset)
    test_proc = int(len(x) * param.test_dataset)
    """

    train_AD_images = []
    validation_AD_images = []
    test_AD_images = []
    train_NC_images = []
    validation_NC_images = []
    test_NC_images = []

    file_cnt = 0
    for filename in x:
        file_cnt += 1
        img_id = get_image_id_from_filename(filename)
        if 'AD_rgb' in filename:
            if img_id in train_AD_list_ids:
                train_AD_images.append(filename)
            if img_id in validation_AD_list_ids:
                validation_AD_images.append(filename)
            if img_id in test_AD_list_ids:
                test_AD_images.append(filename)
        if 'CN_rgb' in filename:
            if img_id in train_NC_list_ids:
                train_NC_images.append(filename)
            if img_id in validation_NC_list_ids:
                validation_NC_images.append(filename)
            if img_id in test_NC_list_ids:
                test_NC_images.append(filename)

    # ad, cn = count_classes(x[:train_proc])
    print('Training - {} AD and {} CN'.format(len(train_AD_images), len(train_NC_images)))
    # ad, cn = count_classes(x[train_proc:(train_proc + val_proc)])
    print('Validation - {} AD and {} CN'.format(len(validation_AD_images), len(validation_NC_images)))
    # ad, cn = count_classes(x[(train_proc + val_proc):])
    print('Test - {} AD and {} CN'.format(len(test_AD_images), len(test_NC_images)))

    training_list = train_AD_images + train_NC_images
    validation_list = validation_AD_images + validation_NC_images
    test_list = test_AD_images + test_NC_images

    dataset_train = AdniDataset(training_list, (224, 224))
    dataloader_train = DataLoader(dataset_train, batch_size=param.batch_size, shuffle=True, num_workers=4)
    dataset_validation = AdniDataset(validation_list, (224, 224))
    dataloader_validation = DataLoader(dataset_validation, batch_size=param.batch_size, shuffle=True, num_workers=4)
    dataset_test = AdniDataset(test_list, (224, 224))
    dataloader_test = DataLoader(dataset_test, batch_size=59, shuffle=False, num_workers=4)

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
        loss_function = torch.nn.CrossEntropyLoss()
    elif param.loss_fcn == 'mse':
        loss_function = torch.nn.MSELoss()
    else:
        print('Wrong loss function!')
        sys.exit()

    train(net2d, dataloader_train, dataloader_validation, dataloader_test, optimizer, param.num_of_epochs,
          loss_function, scheduler,
          save_path)
