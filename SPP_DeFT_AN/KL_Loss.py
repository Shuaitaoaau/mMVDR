#!/usr/bin/env python
# coding=utf-8
# Author: Shuai Tao
# Mail: stao@es.aau.dk
# Create Time: Fri 24 Nov 2023 02:34:02 PM CET

import torch
import torch.nn as nn
import torch.nn.functional as F


class KL_Loss(nn.Module):
    def __init__(self):
        super(KL_Loss, self).__init__()

    def forward(self, inputs, target):
        kl_divergence1 = F.kl_div(F.log_softmax(target, dim=0), F.softmax(inputs, dim=0), reduction='sum')

        return kl_divergence1
