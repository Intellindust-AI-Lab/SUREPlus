import model.resnet18
import model.resnet32
import model.resnet50
import model.densenet_BC
import model.vgg
import model.wrn
import model.classifier
import timm
import torch

import copy
import torch.nn as nn
import torch.nn.functional as F

def reparameterize_model(model: torch.nn.Module) -> nn.Module:
    """Method returns a model where a multi-branched structure
        used in training is re-parameterized into a single branch
        for inference.

    Args:
        model: MobileOne model in train mode.

    Returns:
        MobileOne model in inference mode.
    """
    # Avoid editing original graph
    model = copy.deepcopy(model)
    for module in model.modules():
        if hasattr(module, "reparameterize"):
            module.reparameterize()
    return model

def get_model(model_name, nb_cls, logger, args):
    if model_name == 'resnet18':
        net = model.resnet18.ResNet18(num_classes=nb_cls, use_cos=args.use_cosine, cos_temp=args.cos_temp, activation=args.activation).cuda()
    elif model_name == 'resnet32':
        net = model.resnet32.ResNet32(num_classes=nb_cls, use_cos=args.use_cosine, cos_temp=args.cos_temp).cuda()
    elif model_name == 'resnet50':
        net = model.resnet50.ResNet50(num_classes=nb_cls, use_cos=args.use_cosine, cos_temp=args.cos_temp).cuda()
    elif model_name == 'densenet':
        net = model.densenet_BC.DenseNet3(depth=100,
                                          num_classes=nb_cls,
                                          use_cos=args.use_cosine,
                                          cos_temp=args.cos_temp,
                                          growth_rate=12,
                                          reduction=0.5,
                                          bottleneck=True,
                                          dropRate=0.0).cuda()
    elif model_name == 'vgg':
        net = model.vgg.vgg16(num_classes=nb_cls, use_cos=args.use_cosine, cos_temp=args.cos_temp).cuda()
    elif model_name == 'vgg19bn':
        net = model.vgg.vgg19bn(num_classes=nb_cls, use_cos=args.use_cosine, cos_temp=args.cos_temp).cuda()
    elif model_name == 'wrn':
        net = model.wrn.WideResNet(28, nb_cls, args.use_cosine, args.cos_temp, 10).cuda()
    elif model_name == "FastViT-SA24":
        net = timm.create_model('fastvit_sa24.apple_in1k',checkpoint_path='/data/pretrained_model/FastViT/fastvit_sa24.bin').cuda()
        # checkpoint = torch.load('/data9022/pretrained_model/FastViT/fastvit_sa24.pth.tar')
        # net.load_state_dict(checkpoint['state_dict'])

    elif model_name.startswith("deit"):
        if 'base_patch16_224' in args.deit_path : 
            net = timm.create_model('deit_base_patch16_224', checkpoint_path=args.deit_path).cuda()
        elif 'base_patch16_384' in args.deit_path : 
            net = timm.create_model('deit_base_patch16_384', checkpoint_path=args.deit_path).cuda()
        elif 'base_distilled_patch16_224' in args.deit_path : 
            net = timm.create_model('deit_base_distilled_patch16_224', checkpoint_path=args.deit_path).cuda()
        elif 'base_distilled_patch16_384' in args.deit_path : 
            net = timm.create_model('deit_base_distilled_patch16_384', checkpoint_path=args.deit_path).cuda()
        num_ftrs = net.head.in_features
        if args.use_cosine:
            net.head = model.classifier.Classifier(num_ftrs, nb_cls, args.cos_temp).cuda()
            if 'distilled' in args.deit_path : 
                net.head_dist = model.classifier.Classifier(num_ftrs, nb_cls, args.cos_temp).cuda()
    # msg = 'Using {} ...'.format(model_name)
    # logger.info(msg)
    return net
