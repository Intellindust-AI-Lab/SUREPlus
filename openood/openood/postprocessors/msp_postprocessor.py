from typing import Any
import torch
import torch.nn as nn
from .base_postprocessor import BasePostprocessor


class MSPPostprocessor(BasePostprocessor):
    def __init__(self, config):
        super().__init__(config)

    @torch.no_grad()
    def postprocess(self, net: nn.Module, data: Any):
        """
        输入:
            net: 神经网络模型
            data: 一个 batch 的输入数据
        输出:
            pred: 模型预测类别 (tensor, shape [N])
            conf: 最大 softmax 概率作为置信度 (tensor, shape [N])
        """
        output = net(data)
        score = torch.softmax(output, dim=1)
        conf, pred = torch.max(score, dim=1)
        return pred, conf

    def set_hyperparam(self, hyperparam: list):
        pass

    def get_hyperparam(self):
        return None
