#!/usr/bin/env python
# coding=utf-8
# Author: Shuai Tao
# Mail: stao@es.aau.dk
# Create Time: Mon 04 Mar 2024 11:32:38 AM CET

import numpy as np
from hyperparameter import hyperparameter
import torch
import torchaudio
import os
import matplotlib.pyplot as plt
import librosa.display
import soundfile as sf
from tqdm import tqdm

def compute_noisecov(noise_cov):
    C, C = noise_cov.size()
    eyem = torch.eye(C).type(torch.cfloat)
    # cov_eye = torch.real(torch.mul(noise_cov,eyem))
    cov_eye = torch.mul(noise_cov,eyem)
    cov_eye_inv = inverse_or_pseudo_inverse(cov_eye)
    return cov_eye, cov_eye_inv


def Initial_Noise(noisy_stft):
    init_noise_psd = torch.zeros((129, 6, 6), dtype=torch.complex64)
    for i in range(129):
        for j in range(5):
            single_ft = noisy_stft[:, i, j].unsqueeze(1)
            single_psd = torch.matmul(single_ft, single_ft.conj().transpose(0, 1))
            if j == 0:
                temp = single_psd
            else:
                temp += single_psd

        init_noise_psd[i, :, :] = temp / 5

    return init_noise_psd



def inverse_or_pseudo_inverse(matrix):
    try:
        inverse = torch.inverse(matrix)
        return inverse
    except torch.linalg.LinAlgError:
        pseudo_inverse = torch.pinverse(matrix)
        return pseudo_inverse


def MVDR_Beamforming(noisy_file, model, para):
    noisy_data, fs = torchaudio.load(noisy_file)
    noisy_stft = torch.stft(noisy_data, n_fft=para.n_fft, hop_length=para.hop_len, win_length=para.window_len,
                            window=para.window, center=False, return_complex=True)

    noisy_stft_real = torch.view_as_real(noisy_stft)

    val_inputs = noisy_stft_real
    val_inputs = torch.unsqueeze(val_inputs, 0)
    num_frames = val_inputs.shape[3]
    temp = torch.zeros((val_inputs.shape[0], val_inputs.shape[1], val_inputs.shape[2], para.init_frames, val_inputs.shape[4]))
    e_matrix = torch.zeros(6, 1, dtype=torch.complex64)
    e_matrix[0] = 1

    p_previous = torch.empty(129)
    alpha = 0.99
    alpha_c = 0.1

    output_mvdr = torch.zeros(129, num_frames, dtype=torch.complex64)

    init_noise_psd = Initial_Noise(noisy_stft)

    for t in range(num_frames):
        #SPP Estimation
        temp[:, :, :, -1, :] = val_inputs[:, :, :, t, :]
        PH1 = model(temp)
        PH1 = PH1[..., -1]
        if t < num_frames:
            temp[:, :, :, 0 : (para.init_frames-1), :] = temp[:, :, :, 1:, :]

        #Noise PSD Matrix Estimation
        noisy_frame = noisy_stft[..., t].unsqueeze(1)
        PH1 = PH1.squeeze(0)

        for f in range(129):
            noisy_psd = torch.matmul(noisy_frame[...,f], noisy_frame[...,f].conj().transpose(0, 1))


            if t == 0:
                noise_psd = init_noise_psd[f, :, :]
                clean_psd = noisy_psd - noise_psd
                a = torch.matmul(clean_psd, e_matrix) / torch.matmul(
                    (torch.matmul(e_matrix.conj().transpose(0, 1), (clean_psd + 1e-6 + 1e-6j))), e_matrix)
                _, noise_psd_inv = compute_noisecov(noise_psd)
            else:
                noise_psd = alpha * noise_psd + (1 - alpha) * (1 - PH1[f]) * noisy_psd
                clean_psd = alpha_c * clean_psd + (1 - alpha_c) * PH1[f] * noisy_psd
                noise_psd_inv = inverse_or_pseudo_inverse(noise_psd)

                a = torch.matmul(clean_psd, e_matrix) / torch.matmul(
                    (torch.matmul(e_matrix.conj().transpose(0, 1), clean_psd)), e_matrix)


            h_w = torch.matmul(noise_psd_inv, a) / torch.matmul(
                torch.matmul(a.conj().transpose(0, 1), noise_psd_inv), a) 

            h_w = PH1[f] * h_w

            output_mvdr[f, t] = torch.squeeze(torch.matmul(h_w.conj().transpose(0, 1), noisy_frame[...,f]))


    enhanced = torch.istft(output_mvdr, n_fft=para.n_fft, hop_length=para.hop_len, win_length=para.window_len,
                            window=para.window, center=False)

    return output_mvdr, enhanced, fs



if __name__ == "__main__":
    test_file_path = '/home/ts/Github_Pages/mMVDR/Data/test.csv'
    test_file = np.loadtxt(test_file_path, dtype='str')
    noisy_file = test_file[:, 0].tolist()
    file_len = len(noisy_file)
    para = hyperparameter()

    enhanced_file_path = '/home/ts/Github_Pages/mMVDR/Enhanced'

    # Loading Model
    model_name = '/home/ts/Github_Pages/mMVDR/model_save/model.pth'
    trained_model = torch.load(model_name, map_location=torch.device('cpu')).float()
    trained_model.eval()
    with torch.no_grad():
        for i in tqdm(range(file_len)):
            output, enhanced_wav, fs = MVDR_Beamforming(noisy_file[i], trained_model, para)
            # output = np.array(output, dtype=np.float64)


            #Waveform saving
            enhanced_wav = np.array(enhanced_wav)

            enhance_name = 'Enhanced_' + str(i) + '.wav'
            enhance_name = os.path.join(enhanced_file_path, enhance_name)
            sf.write(enhance_name, enhanced_wav, fs)


