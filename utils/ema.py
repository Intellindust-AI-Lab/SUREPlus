#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author  : AI Partner
# @Email   : ai.partner.cool@outlook.com

import math
from copy import deepcopy
import torch
import torch.nn as nn
from tqdm import tqdm


class ModelEMA:
    """
    Model Exponential Moving Average from https://github.com/rwightman/pytorch-image-models
    Keep a moving average of everything in the model state_dict (parameters and buffers).
    This is intended to allow functionality like
    https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage
    A smoothed version of the weights is necessary for some training schemes to perform well.
    This class is sensitive where it is initialized in the sequence of model init,
    GPU assignment and distributed training wrappers.
    """

    def __init__(self, model, decay=0.9998, updates=0):
        """
        Args:
            model (nn.Module): model to apply EMA.
            decay (float): ema decay reate.
            updates (int): counter of EMA updates.
        """
        # Create EMA(FP32)
        self.ema = deepcopy(model).eval()
        self.updates = updates
        # decay exponential ramp (to help early epochs)
        self.decay = lambda x: decay * (1 - math.exp(-x / 2000))
        for p in self.ema.parameters():
            p.requires_grad_(False)

    def update(self, model):
        # Update EMA parameters
        with torch.no_grad():
            self.updates += 1
            decay_value = self.decay(self.updates)

            state_dict = model.state_dict()
            for k, v in self.ema.state_dict().items():
                ## in dpp mode, the weight are start with module.
                if v.dtype.is_floating_point:
                    v *= decay_value
                    v += (1.0 - decay_value) * state_dict[k].detach()

    @torch.no_grad()
    def update_bn(self, loader, device=None):
        """
        Update BatchNorm statistics for the EMA model using a data loader.
        Reference: torch.optim.swa_utils.update_bn
        """
        momenta = {}
        model = self.ema
        model.train()

        for module in model.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                module.running_mean.zero_()
                module.running_var.fill_(1)
                momenta[module] = module.momentum

        if not momenta:
            return

        # Backup original training mode
        was_training = model.training

        # Temporarily set momentum to None
        for module in momenta.keys():
            module.momentum = None
            module.num_batches_tracked *= 0

        for input in tqdm(loader, leave=False):
            if isinstance(input, (list, tuple)):
                input = input[0]
            if device is not None:
                input = input.to(device)
            model(input)

        # Restore original momentum
        for module in momenta.keys():
            module.momentum = momenta[module]
        model.train(was_training)