from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from .base_postprocessor import BasePostprocessor
from tqdm import tqdm

class SIRCPostprocessor(BasePostprocessor):
    def __init__(self, config):
        super(SIRCPostprocessor, self).__init__(config)
        self.variant = self.config.postprocessor.variant
        self.stats = None   # 在 setup 时计算 ID 数据统计量

    @torch.no_grad()
    def setup(self, net: nn.Module, id_loader_dict, ood_loader):
        """
        计算 ID 数据特征统计量 (S2 mean, std, mu_feat)
        """
        id_features = []
        net.eval()
        for batch in tqdm(id_loader_dict['val'],
                            desc='Setup: ',
                            position=0,
                            leave=True):
            data = batch['data'].cuda().float()
            _, feats = net(data, return_feature=True)
            id_features.append(feats.cpu())

        id_features = torch.cat(id_features, dim=0)
        S2 = torch.norm(id_features, p=1, dim=-1)

        self.stats = {
            "S2_mean": S2.mean().item(),
            "S2_std": S2.std().item(),
            "mu_feat": id_features.mean(dim=0)
        }

    @torch.no_grad()
    def postprocess(self, net: nn.Module, data: Any):
        """
        前向 + SIRC 分数计算
        """
        if self.stats is None:
            raise ValueError("请先调用 setup(net, id_loader) 来计算统计量")

        logits, feats = net(data, return_feature=True)
        sftmax = F.softmax(logits, dim=-1)

        # 分类预测
        conf, pred = sftmax.max(dim=-1)

        # baseline MSP
        msp_score = conf

        # --- 计算 SIRC ---
        if self.variant.startswith("MSP"):
            S1 = conf
            Smax1 = 1.0
        else:  # -H
            entropy_val = -torch.sum(sftmax * torch.log(sftmax + 1e-12), dim=-1)
            S1 = -entropy_val
            Smax1 = 0.0

        if self.variant.endswith("L1"):
            S2 = torch.norm(feats, p=1, dim=-1)
        else:  # Residual
            mu_feat = self.stats["mu_feat"].to(feats.device)
            S2 = torch.norm(feats - mu_feat, p=1, dim=-1)

        S2_mean, S2_std = self.stats["S2_mean"], self.stats["S2_std"]
        a = S2_mean - 3 * S2_std
        b = 1.0 / (S2_std + 1e-12)
        
        eps = 1e-12
        soft = (Smax1 - S1 + eps).log()
        additional = torch.logaddexp(
            torch.zeros(len(S2), device=S2.device),
            -b * (S2 - a)
        )
        sirc_score = -soft - additional

        return pred, sirc_score
