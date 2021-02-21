#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Reimplementation of On-chip Memory Based Binarized Convolutional
Deep Neural Network Applying Batch Normalization Free Technique on an FPGA By Haruyoshi Yonekawa.
@author: Ninnart Fuengfusin
"""
import torch
import torch.nn as nn


class YonekawaBatchNorm1d(nn.BatchNorm1d):
    r"""Recreated interger batch normalization for binarized neural network.
    GUINNESS implements this with int20 or int16 datatype.
    Note that by using `forward` method will act as a common batch normalization.
    However, by using `forward_with_int_bias` method will act as Yonekawa et al. or
    using only integer addition as the Batch Normalization.

    Modified from:
        https://github.com/HirokiNakahara/GUINNESS/blob/master/conv_npz2txt_v2.py
    """
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def get_int_bias(self, bias: torch.Tensor = None) -> torch.Tensor:
        """Get integer bias which can be used to replace BatchNorm process by using
        this integer to add with.
        """
        if self.training:
            raise Exception(
                'This Model is in training model, this method supports only eval mode.')
        beta, gamma, eps = self.bias, self.weight, self.eps
        mean, var = self.running_mean, self.running_var

        if bias is None:
            integer_bias = torch.round(-mean + (beta*(torch.sqrt(var + eps))/gamma))
        else:
            integer_bias = torch.round(bias - mean + (beta*(torch.sqrt(var + eps))/gamma))
        return integer_bias

    def forward_with_int_bias(self, x: torch.Tensor) -> torch.Tensor:
        """Adding integer bias to the tensor.
        Designed to add after binary layer and before Hardtanh and binary layer.
        """
        return x + self.get_int_bias()


class YonekawaBatchNorm2d(nn.BatchNorm2d):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def get_int_bias(self, bias: torch.Tensor = None) -> torch.Tensor:
        r"""Get integer bias which can be used to replace BatchNorm process by using
        this integer to add with.
        """
        if self.training:
            raise Exception(
                'This Model is in training model, this method supports only eval mode.')
        beta, gamma, eps = self.bias, self.weight, self.eps
        mean, var = self.running_mean, self.running_var
        if bias is None:
            int_bias = torch.round(-mean + (beta*(torch.sqrt(var + eps))/gamma))
        else:
            int_bias = torch.round(bias - mean + (beta*(torch.sqrt(var + eps))/gamma))
        return int_bias

    def forward_with_int_bias(self, x: torch.Tensor) -> torch.Tensor:
        r""" Adding integer bias to the tensor.
        Designed to add after binary layer and before Hardtanh and binary layer.
        """
        return x + self.get_int_bias()
