#!/usr/bin/env python
# coding=utf-8
# Author: Shuai Tao

import torch
import torch.nn as nn
import os
import matplotlib.pyplot as plt
import numpy as np
import random


def Train(model, criterion, optimizer, train_loader, para, device):
    loss_sum = 0
    cnt = 0

    model.train()

    for i, data in enumerate(train_loader):
        inputs, labels = data

        inputs = inputs.float().to(device)
        labels = labels.float().to(device)

        n_samples = inputs.shape[3] // para.init_frames
        arr = list(range(n_samples))
        random.shuffle(arr)

        for j in range(n_samples):
            t = arr[j]
            train_input = inputs[:, :, :, t * para.init_frames : (t + 1) * para.init_frames, :]
            train_label = labels[:, :, t * para.init_frames : (t + 1) * para.init_frames]

            y_pred = model(train_input)
            loss = criterion(y_pred, train_label)
            optimizer.zero_grad()
            loss_sum += loss.item()
            loss.backward()
            optimizer.step()

            cnt += 1

    return loss_sum / cnt


def Validation(model, criterion, val_loader, para, device):

    model.eval()
    loss_sum = 0
    cnt = 0

    with torch.no_grad():
        for k, data in enumerate(val_loader):
            inputs_val, labels_val = data

            if torch.isnan(labels_val).any():
                continue

            n_samples = inputs_val.shape[3] // para.init_frames
            arr = list(range(n_samples))
            random.shuffle(arr)

            for l in range(n_samples):
                t = arr[l]
                inputs_val = inputs_val.float().to(device)
                labels_val = labels_val.float().to(device)

                val_input = inputs_val[:, :, :, t * para.init_frames : (t + 1) * para.init_frames, :]
                val_label = labels_val[:, :, t * para.init_frames : (t +  1) * para.init_frames]


                val_pre = model(val_input)
                val_loss = criterion(val_pre, val_label)
                loss_sum += val_loss.item()

                cnt += 1


    return loss_sum / cnt


def Model_Fit(model, criterion, optimizer, train_loader, val_loader, scheduler, para, device):
    y_temp1 = []
    y_temp2 = []

    for epoch in range(para.epoch):
        print('Epoch = %d' % epoch)

        #Model_1
        # Training
        train_loss_avrg = Train(model, criterion, optimizer, train_loader, para, device)

        #Validation
        val_loss_avrg = Validation(model, criterion, val_loader, para, device)

        y_temp1.append(train_loss_avrg)
        y_temp2.append(val_loss_avrg)
        print('train loss = %.4f' % train_loss_avrg)
        print('validation loss = %.4f' % val_loss_avrg)
        scheduler.step()
        print(optimizer.state_dict()['param_groups'][0]['lr'])

        if epoch == 0:
            best_val_loss = val_loss_avrg
            model_name = os.path.join(para.save_path, 'model.pth')
            torch.save(model, model_name)
        else:
            if best_val_loss >= val_loss_avrg:
                best_val_loss = val_loss_avrg
                model_name = os.path.join(para.save_path, 'model.pth')
                torch.save(model, model_name)

    #Save the loss data
    np.save(para.train_loss_path, y_temp1)
    np.save(para.val_loss_path, y_temp2)

    #Figure

    import matplotlib as mpl
    
    mpl.use('Agg')

    # Figure
    fig, ax = plt.subplots()
    x_temp = np.arange(para.epoch)
    ax.plot(x_temp, y_temp1, label='Training Loss')
    ax.plot(x_temp, y_temp2, label='Validation Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Error')
    ax.set_title('Loss Error (Adam)')
    ax.legend()
    figure = 'loss_' + '.jpg'
    plt.savefig(figure)


def get_no_params(model):
    nop = 0
    for param in list(model.parameters()):
        nn = 1
        for s in list(param.size()):
            nn = nn * s
        nop += nn

    print("nop = %d"%nop)
