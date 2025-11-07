import torch
import torch.nn as nn
import os

class DINOv3VitL16(nn.Module):
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
        in_dim = self.backbone.embed_dim  # 1024 for ViT-L/16
        self.head = nn.Linear(in_dim, num_classes)

        # Save feature size
        self.feature_size = in_dim

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



if __name__ == "__main__":
    # Test
    x = torch.randn(2, 3, 224, 224)
    net = DINOv3VitL16(num_classes=1000)

    #  forward
    logits = net(x)
    print("logits.shape:", logits.shape)  # [2, 1000]

    # return_feature
    logits, cls_feat = net(x, return_feature=True)
    print("cls_feat.shape:", cls_feat.shape)  # [2, 1024]

    # return_feature_list
    logits, feat_list = net(x, return_feature_list=True)
    print("len(feat_list):", len(feat_list))  # = num_blocks
    print("feat_list[0].shape:", feat_list[0].shape)  # [2, 1024]

