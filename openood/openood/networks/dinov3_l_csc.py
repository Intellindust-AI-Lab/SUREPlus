import torch
import torch.nn as nn
import os
import torch.nn.functional as F

class Classifier(torch.nn.Module):
    def __init__(self,
                 feat_dim,
                 nb_cls,
                 cos_temp):
        super(Classifier, self).__init__()

        fc = torch.nn.Linear(feat_dim, nb_cls)        
        self.weight = torch.nn.Parameter(fc.weight.t(), requires_grad=True)
        self.bias = torch.nn.Parameter(fc.bias, requires_grad=True)
        self.cos_temp = torch.nn.Parameter(torch.FloatTensor(1).fill_(cos_temp), requires_grad=False)


    def apply_cosine(self, feature, weight, bias):
        
        feature = F.normalize(feature, p=2, dim=1, eps=1e-12) ## Attention: normalized along 2nd dimension!!!
        weight = F.normalize(weight, p=2, dim=0, eps=1e-12)## Attention: normalized along 1st dimension!!!

        cls_score = self.cos_temp * (torch.mm(feature, weight))
        return cls_score


    def forward(self, feature):
        # weight, bias = self.get_weight()
        cls_score = self.apply_cosine(feature, self.weight, self.bias)

        return cls_score
    
class DINOv3VitL16_CSC(nn.Module):
    def __init__(self, repo_path='/data1032/liyang/git/dinov3', weight_path='/data/dinov3/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth', num_classes=1000):
        super().__init__()

        # =========================
        # Load pretrained DINOv3 ViT-L/16
        # =========================
        self.backbone = torch.hub.load(
            repo_path,
            'dinov3_vitl16',
            source='local',
            weights=weight_path
        )

        # =========================
        # Replace classification head
        # =========================
        self.feature_size = self.backbone.embed_dim  # 1024 for ViT-L/16
        self.head = Classifier(self.feature_size, num_classes, 64.0)

        # Save feature size
        

    def forward(self, x, return_feature=False, return_feature_list=False):
        """
        Forward pass in DeiT style
        """
        feats_dict = self.backbone.forward_features(x)
        cls_feat = feats_dict["x_norm_clstoken"]  # [B, C]

        logits = self.head(cls_feat)

        if return_feature_list:
            feature_list = [
                blk_out[:, 0]  # CLS token
                for blk_out in self.backbone._get_intermediate_layers_not_chunked(x, n=len(self.backbone.blocks))
            ]
            return logits, feature_list
        elif return_feature:
            return logits, cls_feat
        else:
            return logits

    def forward_threshold(self, x, threshold):
        """
        Forward pass with feature clipping (for OOD detection)
        """
        # Step 1: get final CLS token from backbone
        feats_dict = self.backbone.forward_features(x)
        cls_feat = feats_dict["x_norm_clstoken"]  # [B, embed_dim]

        # Step 2: clip features
        cls_feat = cls_feat.clip(max=threshold)

        # Step 3: compute logits
        logits = self.head(cls_feat)
        return logits

    def get_fc(self):
        """
        Return numpy arrays of fc weights and bias
        """
        fc = self.head
        return fc.weight.detach().cpu().numpy(), fc.bias.detach().cpu().numpy()

    def get_fc_layer(self):
        """
        Return the nn.Linear layer
        """
        return self.head


