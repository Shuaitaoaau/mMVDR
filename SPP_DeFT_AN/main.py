#!/usr/bin/env python
# coding=utf-8
# Author: Shuai Tao
# Mail: stao@create.aau.dk
# Create Time: Thu 14 Jul 2022 11:11:00 AM CEST

import torch

from hyperparameter import hyperparameter
from data_module import Train_Loader
from train_module import  Model_Fit, get_no_params
from KL_Loss import KL_Loss
from DeFT_AN import Network

if __name__ == "__main__":
    torch.manual_seed(0)

    para = hyperparameter()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    criterion = KL_Loss()

    model = Network().to(device)

    get_no_params(model)

    optimizer = torch.optim.Adam(params=model.parameters(),
                                   lr=1e-3,
                                   betas=(0.9, 0.999),
                                   eps=1e-8,
                                   weight_decay=0.00001,
                                   amsgrad=False)


    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)
    train_loader = Train_Loader(para.train_file_path, para.train_label_file_path, Train=True)
    val_loader = Train_Loader(para.validation_file_path, para.val_label_file_path, Train=True)
    Model_Fit(model, criterion, optimizer, train_loader, val_loader, scheduler, para, device)

