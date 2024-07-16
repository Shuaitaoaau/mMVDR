#!/usr/bin/env python
# coding=utf-8
# Author: Shuai Tao
# Mail: stao@es.aau.dk
# Create Time: Thu 23 Nov 2023 11:44:54 AM CET

import torch
import torch.nn as nn
import soundfile as sf
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torchaudio
from scipy.io import loadmat

class Train_Dataset(Dataset):
    def __init__(self, noisy_file_path, label_file_path, Train=True):
        self.noisy_file = np.loadtxt(noisy_file_path, dtype='str')
        self.label_file = np.loadtxt(label_file_path, dtype='str')
        self.noisy_path = self.noisy_file[:, 0].tolist()
        self.label_path = self.label_file.tolist()

        self.Train = Train


        self.file_len = len(self.noisy_path)

        #STFT
        self.n_fft = 256
        self.hop_len = 128
        self.window_len = 256
        self.window = torch.hamming_window(self.n_fft)

    def __len__(self):
        return self.file_len

    def __getitem__(self, index):
        noisy_data, fs = torchaudio.load(self.noisy_path[index])
        noisy_stft = torch.stft(noisy_data, n_fft=self.n_fft, hop_length=self.hop_len, win_length=self.window_len,
                                window=self.window, center=False, return_complex=True)

        noisy_stft = torch.view_as_real(noisy_stft)
        #Label
        mc_spp = loadmat(self.label_path[index])
        mc_spp = torch.tensor(mc_spp['MC_SPP'])

        return noisy_stft, mc_spp


def Train_Loader(input_file, label_file, Train):

    dataset = Train_Dataset(input_file, label_file, Train)

    loader = DataLoader(dataset = dataset,
                        batch_size = 64,
                        shuffle = True,
                        num_workers = 2,
                        )
    return loader




