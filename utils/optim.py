import torch 
import torch.nn 

import utils.sam
import utils.fsam
import torch.optim
import torch.optim.lr_scheduler
import torch.optim.swa_utils
import numpy as np


def _cosine_annealing(step, total_steps, lr_max, lr_min):
    return lr_min + (lr_max -
                     lr_min) * 0.5 * (1 + np.cos(step / total_steps * np.pi))


def get_cosine_annealing_scheduler(optimizer, epochs, steps_per_epoch, base_lr):
    lr_min = 0.0
    total_steps = epochs * steps_per_epoch

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: _cosine_annealing(
            step,
            total_steps,
            1,  # since lr_lambda computes multiplicative factor
            lr_min / base_lr))

    return scheduler   

def get_optimizer_scheduler(model_name,
                            optim_name,
                            net,
                            lr,
                            momentum,
                            weight_decay,
                            max_epoch_cos = 200,
                            swa_lr = 0.05,
                            args=None,
                            train_loader=None) :

    ## sgd + sam
    sgd_optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    sam_sgd = utils.sam.SAM(net.parameters(), torch.optim.SGD, lr=lr, momentum=momentum, weight_decay=weight_decay)

    ## adamw + sam
    adamw_optimizer = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=weight_decay)
    sam_adamw = utils.sam.SAM(net.parameters(), torch.optim.AdamW, lr=lr, weight_decay=weight_decay)

    ## sgd + fsam
    sgd_optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    fsam_sgd = utils.fsam.FriendlySAM(net.parameters(), torch.optim.SGD, 
                                rho=args.rho, sigma=args.sigma, lmbda=args.lmbda, adaptive=0, 
                                lr=lr, momentum=momentum, weight_decay=weight_decay,
                                nesterov=False)
    
    ## adamw + fsam
    adamw_optimizer = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=weight_decay)
    fsam_adamw = utils.fsam.FriendlySAM(net.parameters(), torch.optim.AdamW, 
                                rho=args.rho, sigma=args.sigma, lmbda=args.lmbda, adaptive=0, 
                                lr=lr, weight_decay=weight_decay)

    
    ## convmixer uses adamw optimzer while cnn backbones uses sgd
    if model_name in ["dinov3_l16"] : 
        if optim_name in ['sam', 'fmfp'] : 
            optimizer = sam_adamw
        elif optim_name in ['fsam', 'fmfpfsam'] : 
            optimizer = fsam_adamw
        else :
            optimizer = adamw_optimizer
        # print("-----Using Adamw-----")
            
    else: 
        if optim_name in ['sam', 'fmfp'] : 
            optimizer = sam_sgd
        elif optim_name in ['fsam', 'fmfpfsam'] : 
            optimizer = fsam_sgd
        else :
            optimizer = sgd_optimizer
        # print("-----Using SGD-----")

    if args.per_epoch_scheduler:
        cos_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epoch_cos)
    else:
        cos_scheduler = get_cosine_annealing_scheduler(optimizer, args.epochs, len(train_loader), args.lr)
    
    ## swa model
    swa_model = torch.optim.swa_utils.AveragedModel(net)
    swa_scheduler = torch.optim.swa_utils.SWALR(optimizer, swa_lr=swa_lr)

    return optimizer, cos_scheduler, swa_model, swa_scheduler
