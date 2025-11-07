# import mmcv
from copy import deepcopy
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
# from mmcls.apis import init_model

import openood.utils.comm as comm

from .bit import KNOWN_MODELS
from .conf_branch_net import ConfBranchNet
from .csi_net import get_csi_linear_layers, CSINet
from .cider_net import CIDERNet
from .t2fnorm_net import T2FNormNet
from .de_resnet18_256x256 import AttnBasicBlock, BN_layer, De_ResNet18_256x256
from .densenet import DenseNet3
from .draem_net import DiscriminativeSubNetwork, ReconstructiveSubNetwork
from .dropout_net import DropoutNet
from .dsvdd_net import build_network
from .godin_net import GodinNet
from .lenet import LeNet
from .mcd_net import MCDNet
from .npos_net import NPOSNet
from .openmax_net import OpenMax
from .patchcore_net import PatchcoreNet
from .projection_net import ProjectionNet
from .react_net import ReactNet
from .resnet18_32x32 import ResNet18_32x32
from .resnet18_64x64 import ResNet18_64x64
from .resnet18_224x224 import ResNet18_224x224
from .resnet18_256x256 import ResNet18_256x256
from .resnet18_32x32_csc import ResNet18_32x32_csc
from .resnet50 import ResNet50
from .rot_net import RotNet
from .udg_net import UDGNet
from .vit_b_16 import ViT_B_16
from .deit_b_csc import DeiTBasePatch16_CSC
from .deit_b import DeiTBasePatch16_224
from .dinov3_l import DINOv3VitL16
from .dinov3_l_csc import DINOv3VitL16_CSC
from .wrn import WideResNet
from .rts_net import RTSNet
from .palm_net import PALMNet
from .ascood_net import ASCOODNet
from .vit_t_16 import make_vit_tiny

def _strip_common_prefixes(state_dict, prefixes=('module.', 'model.')):
    """
    去掉常见前缀，如 module., model.
    """
    new_sd = {}
    for k, v in state_dict.items():
        new_k = k
        for p in prefixes:
            if k.startswith(p):
                new_k = k[len(p):]
                break
        new_sd[new_k] = v
    return new_sd

def get_network(network_config):

    num_classes = network_config.num_classes

    if network_config.name == 'resnet18_32x32':
        net = ResNet18_32x32(num_classes=num_classes)

    elif  network_config.name == 'resnet18_32x32_csc':
        net = ResNet18_32x32_csc(num_classes=num_classes)
        
    elif network_config.name == 'resnet18_256x256':
        net = ResNet18_256x256(num_classes=num_classes)

    elif network_config.name == 'resnet18_64x64':
        net = ResNet18_64x64(num_classes=num_classes)

    elif network_config.name == 'resnet18_224x224':
        net = ResNet18_224x224(num_classes=num_classes)

    elif network_config.name == 'resnet50':
        net = ResNet50(num_classes=num_classes)

    elif network_config.name == 'lenet':
        net = LeNet(num_classes=num_classes, num_channel=3)
        
    elif network_config.name == 'deit':
        net = DeiTBasePatch16_224()
    
    elif network_config.name == 'vit_tiny':
        net = make_vit_tiny()
    
    elif network_config.name == 'dinov3_l':
        net = DINOv3VitL16(network_config.repo_path, network_config.weight_path, num_classes=num_classes)  
    
    elif network_config.name == 'dinov3_l_csc':
        net = DINOv3VitL16_CSC(network_config.repo_path, network_config.weight_path, num_classes=num_classes)
        
    elif network_config.name == 'deit_csc':
        net  = DeiTBasePatch16_CSC()

    elif network_config.name == 'wrn':
        net = WideResNet(depth=28,
                         widen_factor=10,
                         dropRate=0.0,
                         num_classes=num_classes)

    elif network_config.name == 'densenet':
        net = DenseNet3(depth=100,
                        growth_rate=12,
                        reduction=0.5,
                        bottleneck=True,
                        dropRate=0.0,
                        num_classes=num_classes)

    elif network_config.name == 'patchcore_net':
        # path = '/home/pengyunwang/.cache/torch/hub/vision-0.9.0'
        # module = torch.hub._load_local(path,
        #                                'wide_resnet50_2',
        #                                pretrained=True)
        backbone = get_network(network_config.backbone)
        net = PatchcoreNet(backbone)
    elif network_config.name == 'wide_resnet_50_2':
        module = torch.hub.load('pytorch/vision:v0.9.0',
                                'wide_resnet50_2',
                                pretrained=True)
        net = PatchcoreNet(module)

    elif network_config.name == 'godin_net':
        # don't wrap ddp here cuz we need to modify
        # backbone
        network_config.backbone.num_gpus = 1
        backbone = get_network(network_config.backbone)
        feature_size = backbone.feature_size
        # remove fc otherwise ddp will
        # report unused params
        backbone.fc = nn.Identity()

        net = GodinNet(backbone=backbone,
                       feature_size=feature_size,
                       num_classes=num_classes,
                       similarity_measure=network_config.similarity_measure)

    elif network_config.name == 'cider_net':
        # don't wrap ddp here cuz we need to modify
        # backbone
        network_config.backbone.num_gpus = 1
        backbone = get_network(network_config.backbone)
        # remove fc otherwise ddp will
        # report unused params
        backbone.fc = nn.Identity()

        net = CIDERNet(backbone=backbone,
                       head=network_config.head,
                       feat_dim=network_config.feat_dim,
                       num_classes=num_classes)

    elif network_config.name == 't2fnorm_net':
        network_config.backbone.num_gpus = 1
        backbone = get_network(network_config.backbone)

        net = T2FNormNet(backbone=backbone, num_classes=num_classes)

    elif network_config.name == 'palm_net':
        # don't wrap ddp here cuz we need to modify
        # backbone
        network_config.backbone.num_gpus = 1
        backbone = get_network(network_config.backbone)
        # remove fc otherwise ddp will
        # report unused params
        backbone.fc = nn.Identity()

        net = PALMNet(backbone=backbone,
                      head=network_config.head,
                      feat_dim=network_config.feat_dim,
                      num_classes=num_classes)

    elif network_config.name == 'npos_net':
        # don't wrap ddp here cuz we need to modify
        # backbone
        network_config.backbone.num_gpus = 1
        backbone = get_network(network_config.backbone)
        # remove fc otherwise ddp will
        # report unused params
        backbone.fc = nn.Identity()

        net = NPOSNet(backbone=backbone,
                      head=network_config.head,
                      feat_dim=network_config.feat_dim,
                      num_classes=num_classes)

    elif network_config.name == 'ascood_net':
        network_config.backbone.num_gpus = 1
        backbone = get_network(network_config.backbone)
        net = ASCOODNet(backbone=backbone)

    elif network_config.name == 'rts_net':
        backbone = get_network(network_config.backbone)
        try:
            feature_size = backbone.feature_size
        except AttributeError:
            feature_size = backbone.module.feature_size
        net = RTSNet(backbone=backbone,
                     feature_size=feature_size,
                     num_classes=num_classes,
                     dof=network_config.dof)

    elif network_config.name == 'react_net':
        backbone = get_network(network_config.backbone)
        net = ReactNet(backbone)

    elif network_config.name == 'csi_net':
        # don't wrap ddp here cuz we need to modify
        # backbone
        network_config.backbone.num_gpus = 1
        backbone = get_network(network_config.backbone)
        feature_size = backbone.feature_size
        # remove fc otherwise ddp will
        # report unused params
        backbone.fc = nn.Identity()

        net = get_csi_linear_layers(feature_size, num_classes,
                                    network_config.simclr_dim,
                                    network_config.shift_trans_type)
        net['backbone'] = backbone

        dummy_net = CSINet(deepcopy(backbone),
                           feature_size=feature_size,
                           num_classes=num_classes,
                           simclr_dim=network_config.simclr_dim,
                           shift_trans_type=network_config.shift_trans_type)
        net['dummy_net'] = dummy_net

    elif network_config.name == 'draem':
        model = ReconstructiveSubNetwork(in_channels=3,
                                         out_channels=3,
                                         base_width=int(
                                             network_config.image_size / 2))
        model_seg = DiscriminativeSubNetwork(
            in_channels=6,
            out_channels=2,
            base_channels=int(network_config.image_size / 4))

        net = {'generative': model, 'discriminative': model_seg}

    elif network_config.name == 'openmax_network':
        backbone = get_network(network_config.backbone)
        net = OpenMax(backbone=backbone, num_classes=num_classes)

    elif network_config.name == 'mcd':
        # don't wrap ddp here cuz we need to modify
        # backbone
        network_config.backbone.num_gpus = 1
        backbone = get_network(network_config.backbone)
        feature_size = backbone.feature_size
        # remove fc otherwise ddp will
        # report unused params
        backbone.fc = nn.Identity()

        net = MCDNet(backbone=backbone, num_classes=num_classes)

    elif network_config.name == 'udg':
        # don't wrap ddp here cuz we need to modify
        # backbone
        network_config.backbone.num_gpus = 1
        backbone = get_network(network_config.backbone)
        feature_size = backbone.feature_size
        # remove fc otherwise ddp will
        # report unused params
        backbone.fc = nn.Identity()

        net = UDGNet(backbone=backbone,
                     num_classes=num_classes,
                     num_clusters=network_config.num_clusters)

    elif network_config.name == 'opengan':
        from .opengan import Discriminator, Generator
        backbone = get_network(network_config.backbone)
        netG = Generator(in_channels=network_config.nz,
                         feature_size=network_config.ngf,
                         out_channels=network_config.nc)
        netD = Discriminator(in_channels=network_config.nc,
                             feature_size=network_config.ndf)

        net = {'netG': netG, 'netD': netD, 'backbone': backbone}

    elif network_config.name == 'arpl_gan':
        from .arpl_net import (resnet34ABN, Generator, Discriminator,
                               Generator32, Discriminator32, ARPLayer)
        feature_net = resnet34ABN(num_classes=num_classes, num_bns=2)
        dim_centers = feature_net.fc.weight.shape[1]
        feature_net.fc = nn.Identity()

        criterion = ARPLayer(feat_dim=dim_centers,
                             num_classes=num_classes,
                             weight_pl=network_config.weight_pl,
                             temp=network_config.temp)

        assert network_config.image_size == 32 \
            or network_config.image_size == 64, \
            'ARPL-GAN only supports 32x32 or 64x64 images!'

        if network_config.image_size == 64:
            netG = Generator(1, network_config.nz, network_config.ngf,
                             network_config.nc)  # ngpu, nz, ngf, nc
            netD = Discriminator(1, network_config.nc,
                                 network_config.ndf)  # ngpu, nc, ndf
        else:
            netG = Generator32(1, network_config.nz, network_config.ngf,
                               network_config.nc)  # ngpu, nz, ngf, nc
            netD = Discriminator32(1, network_config.nc,
                                   network_config.ndf)  # ngpu, nc, ndf

        net = {
            'netF': feature_net,
            'criterion': criterion,
            'netG': netG,
            'netD': netD
        }

    elif network_config.name == 'arpl_net':
        from .arpl_net import ARPLayer
        # don't wrap ddp here because we need to modify
        # feature_net
        network_config.feat_extract_network.num_gpus = 1
        feature_net = get_network(network_config.feat_extract_network)
        try:
            if isinstance(feature_net, nn.parallel.DistributedDataParallel):
                dim_centers = feature_net.module.fc.weight.shape[1]
                feature_net.module.fc = nn.Identity()
            else:
                dim_centers = feature_net.fc.weight.shape[1]
                feature_net.fc = nn.Identity()
        except Exception:
            if isinstance(feature_net, nn.parallel.DistributedDataParallel):
                dim_centers = feature_net.module.classifier[0].weight.shape[1]
                feature_net.module.classifier = nn.Identity()
            else:
                dim_centers = feature_net.classifier[0].weight.shape[1]
                feature_net.classifier = nn.Identity()

        criterion = ARPLayer(feat_dim=dim_centers,
                             num_classes=num_classes,
                             weight_pl=network_config.weight_pl,
                             temp=network_config.temp)

        net = {'netF': feature_net, 'criterion': criterion}

    elif network_config.name == 'bit':
        net = KNOWN_MODELS[network_config.model](
            head_size=network_config.num_logits,
            zero_head=True,
            num_block_open=network_config.num_block_open)

    elif network_config.name == 'vit-b-16':
        net = ViT_B_16(num_classes=num_classes)

    elif network_config.name == 'conf_branch_net':
        # don't wrap ddp here cuz we need to modify
        # backbone
        network_config.backbone.num_gpus = 1
        backbone = get_network(network_config.backbone)
        feature_size = backbone.feature_size
        # remove fc otherwise ddp will
        # report unused params
        backbone.fc = nn.Identity()

        net = ConfBranchNet(backbone=backbone, num_classes=num_classes)

    elif network_config.name == 'rot_net':
        # don't wrap ddp here cuz we need to modify
        # backbone
        network_config.backbone.num_gpus = 1
        backbone = get_network(network_config.backbone)
        feature_size = backbone.feature_size
        # remove fc otherwise ddp will
        # report unused params
        backbone.fc = nn.Identity()

        net = RotNet(backbone=backbone, num_classes=num_classes)

    elif network_config.name == 'dsvdd':
        net = build_network(network_config.type)

    elif network_config.name == 'projectionNet':
        backbone = get_network(network_config.backbone)
        net = ProjectionNet(backbone=backbone, num_classes=2)

    elif network_config.name == 'dropout_net':
        backbone = get_network(network_config.backbone)
        net = DropoutNet(backbone=backbone, dropout_p=network_config.dropout_p)

    elif network_config.name == 'simclr_net':
        # backbone = get_network(network_config.backbone)
        # net = SimClrNet(backbone, out_dim=128)
        from .temp import SSLResNet
        net = SSLResNet()
        net.encoder = nn.DataParallel(net.encoder).cuda()

    elif network_config.name == 'rd4ad_net':
        encoder = get_network(network_config.backbone)
        bn = BN_layer(AttnBasicBlock, 2)
        decoder = De_ResNet18_256x256()
        net = {'encoder': encoder, 'bn': bn, 'decoder': decoder}
    else:
        raise Exception('Unexpected Network Architecture!')

    if network_config.pretrained:
        ckpt = torch.load(network_config.checkpoint, map_location="cpu")
        if 'model' in ckpt:
            ckpt = ckpt['model']

        new_state_dict = _strip_common_prefixes(
            ckpt, prefixes=('module.module.', 'module.', 'model.', 'backbone.', '0.', '1.')
        )

        if "n_averaged" in new_state_dict:
            print("Removing n_averaged from checkpoint...")
            del new_state_dict["n_averaged"]

        if network_config.name.startswith('deit') or network_config.name.startswith('dino'):
            renamed_state_dict = {}
            for k, v in new_state_dict.items():
                if k.endswith('head.weight'):
                    renamed_state_dict['head.weight'] = v
                elif k.endswith('head.bias'):
                    renamed_state_dict['head.bias'] = v
                else:
                    renamed_state_dict[k] = v
            backbone_dict = {k: v for k, v in renamed_state_dict.items() if not k.startswith('head')}
            net.backbone.load_state_dict(backbone_dict, strict=False)
            if 'head.weight' in renamed_state_dict and 'head.bias' in renamed_state_dict:
                net.head.weight.data.copy_(renamed_state_dict['head.weight'])
                net.head.bias.data.copy_(renamed_state_dict['head.bias'])
        else:
            net.load_state_dict(new_state_dict, strict=True)

        print(f"Model loading for {network_config.name} completed!")





    if network_config.num_gpus > 1:
        if type(net) is dict:
            for key, subnet in zip(net.keys(), net.values()):
                net[key] = torch.nn.parallel.DistributedDataParallel(
                    subnet.cuda(),
                    device_ids=[comm.get_local_rank()],
                    broadcast_buffers=True,
                    find_unused_parameters=True)
        else:
            net = torch.nn.parallel.DistributedDataParallel(
                net.cuda(),
                device_ids=[comm.get_local_rank()],
                broadcast_buffers=True,
                find_unused_parameters=True)

    if network_config.num_gpus > 0:
        if type(net) is dict:
            for subnet in net.values():
                subnet.cuda()
        else:
            net.cuda()

    cudnn.benchmark = True
    return net
