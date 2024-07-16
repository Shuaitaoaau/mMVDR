#!/usr/bin/env python
# coding=utf-8
# Author: Shuai Tao
# Mail: stao@es.aau.dk
# Create Time: Thu 14 Jul 2022 09:57:41 AM CEST

import torch

class hyperparameter():
    def __init__(self):

        self.save_path = '/home/ts/Github_Pages/mMVDR/SPP_DeFT_AN/model_save'

        # Train
        self.train_file_path = '/home/ts/Github_Pages/mMVDR/SPP_DeFT_AN/Data/No_Reb/train.csv'
        self.train_label_file_path = '/home/ts/Github_Pages/mMVDR/SPP_DeFT_AN/Data/No_Reb/train_label.csv'

        # Validation
        self.validation_file_path = '/home/ts/Github_Pages/mMVDR/SPP_DeFT_AN/Data/No_Reb/val.csv'
        self.val_label_file_path = '/home/ts/Github_Pages/mMVDR/SPP_DeFT_AN/Data/No_Reb/val_label.csv'

        # Test
        # self.test_file_path = '/home/ts/Project_5/Data/Test/test.csv'

        # Epoch
        self.epoch = 100
        
        #Number of frames
        self.num_frames = 249

        # Init frame length
        self.init_frames = int(5)

        #STFT
        self.n_fft = 256
        self.hop_len = 128
        self.window_len = 256
        self.window = torch.hamming_window(self.n_fft)


        #Loss data save path
        self.train_loss_path = '/home/ts/Github_Pages/mMVDR/SPP_DeFT_AN/Results/Training_loss.npy'
        self.val_loss_path = '/home/ts/Github_Pages/mMVDR/SPP_DeFT_AN/Results/Val_loss.npy'
