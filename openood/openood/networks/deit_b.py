import torch
import torch.nn as nn
import timm
import os

class DeiTBasePatch16_224(nn.Module):
    def __init__(self):
        super().__init__()
        if os.path.exists('/data/pretrained_model/DEIT/deit_base_patch16_224-b5f2ef4d.pth'):
            self.backbone = timm.create_model(
                'deit_base_patch16_224',
                checkpoint_path='/data/pretrained_model/DEIT/deit_base_patch16_224-b5f2ef4d.pth'
            )
        else:
            self.backbone = timm.create_model(
                'deit_base_patch16_224',
                checkpoint_path='/data9022/pretrained_model/DEIT/deit_base_patch16_224-b5f2ef4d.pth'
            )
        self.feature_size = self.backbone.num_features
        
    def forward(self, x, return_feature=False, return_feature_list=False):
        # forward_features: [B, 197, 768]
        feats = self.backbone.patch_embed(x)                # patch embedding
        cls_token = self.backbone.cls_token.expand(feats.shape[0], -1, -1)
        feats = torch.cat((cls_token, feats), dim=1)
        feats = feats + self.backbone.pos_embed
        feats = self.backbone.pos_drop(feats)

        feature_list = []
        for blk in self.backbone.blocks:
            feats = blk(feats)
            if return_feature_list:
                feature_list.append(feats[:, 0])  # collect cls_token of each block

        feats = self.backbone.norm(feats)
        cls_feat = feats[:, 0]                    # final cls token
        logits = self.backbone.head(cls_feat)

        if return_feature:
            return logits, cls_feat
        elif return_feature_list:
            return logits, feature_list
        else:
            return logits


    def forward_threshold(self, x, threshold):
        feats = self.backbone.forward_features(x)   # [B, 197, 768]
        cls_feat = feats[:, 0]                      # [B, 768]
        cls_feat = cls_feat.clip(max=threshold)     # 截断特征
        logits = self.backbone.head(cls_feat)
        return logits

    def get_fc(self):

        fc = self.backbone.head
        return fc.weight.detach().cpu().numpy(), fc.bias.detach().cpu().numpy()

    def get_fc_layer(self):
        return self.backbone.head


if __name__=='__main__':
    net = DeiTBasePatch16_224(1000)
    net.get_fc()
    net.get_fc_layer()