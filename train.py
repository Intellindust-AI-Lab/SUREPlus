import utils.utils
import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F
import random
from torch.nn.modules.batchnorm import _BatchNorm


class Mixup_Criterion(nn.Module):
    def __init__(self, beta, cls_criterion):
        super().__init__()
        self.beta = beta
        self.cls_criterion =  cls_criterion

    def get_mixup_data(self, image, target) :
        beta = np.random.beta(self.beta, self.beta)
        index = torch.randperm(image.size()[0]).to(image.device)
        shuffled_image, shuffled_target = image[index], target[index]
        mixed_image = beta * image + (1 - beta) * shuffled_image
        return mixed_image, shuffled_target, beta

    def forward(self, image, target, net):
        mixed_image, shuffled_target, beta = self.get_mixup_data(image, target)
        pred_mixed = net(mixed_image)
        loss_mixup = beta * self.cls_criterion(pred_mixed, target) + (1 - beta) * self.cls_criterion(pred_mixed, shuffled_target)
        return loss_mixup

class Correctness_Log(object):
    def __init__(self, n_data):
        self.correctness = np.zeros((n_data))
        self.max_correctness = 1

    # correctness update
    def update(self, data_idx, correctness):
        self.correctness[data_idx] += correctness.cpu().numpy()

    def max_correctness_update(self, epoch):
        if epoch > 1:
            self.max_correctness += 1

    # correctness normalize (0 ~ 1) range
    def _normalize(self, data):
        data_min = self.correctness.min()
        data_max = float(self.max_correctness)

        return (data - data_min) / (data_max - data_min)

    # get target & margin
    def get_target_margin(self, idx1, idx2):
        idx1 = idx1.cpu().numpy()
        idx2 = idx2.cpu().numpy()
        correctness_norm = self._normalize(self.correctness)

        target1, target2 = correctness_norm[idx1], correctness_norm[idx2]

        # 1 for idx1 > idx2, 0 for idx1 = idx2, -1 for idx1 < idx2
        target = np.array(target1 > target2, dtype='float') + np.array(target1 < target2, dtype='float') * (-1)
        target = torch.from_numpy(target).float().cuda()

        # calc margin
        margin = abs(target1 - target2)
        margin = torch.from_numpy(margin).float().cuda()

        return target, margin


class CRL_Criterion(nn.Module):
    '''
    Confidence-Aware Learning for Deep Neural Networks
    ICML 2020
    http://proceedings.mlr.press/v119/moon20a/moon20a.pdf
    code borrows from: https://github.com/daintlab/confidence-aware-learning/blob/master/crl_utils.py
    '''

    def __init__(self, args):
        super().__init__()
        self.rank_criterion = torch.nn.MarginRankingLoss(margin=0)
        self.loss_type = args.loss

    def forward(self, output, image_idx, correct_log):
        if self.loss_type == 'ce':
            conf = torch.softmax(output, dim=1)
        else:
            conf = torch.sigmoid(output)  # bs x num_class
        conf,_ = conf.max(dim=1)  # bs x 1
        conf_roll, image_idx_roll = torch.roll(conf, -1), torch.roll(image_idx, -1)
        rank_target, rank_margin = correct_log.get_target_margin(image_idx, image_idx_roll)
        conf_roll = conf_roll + rank_margin / (rank_target + 1e-7)
    
        ranking_loss = self.rank_criterion(conf, conf_roll, rank_target)
        
        return ranking_loss

def compute_loss(epoch,
                 args,
                 net,
                 image,
                 pixmix_image,
                 target,
                 image_idx,
                 correct_log,
                 cls_criterion,
                 mixup_criterion,
                 rank_criterion
                 ):
    """
    Compute total loss with multiple components:
    - Classification loss (CLS)
    - Mixup loss
    - CRL loss
    - PixMix loss
    - CutMix loss
    """

    # ================== Forward: Normal ==================
    output = net(image)

    # ================== CLS Loss ==================
    loss_cls = cls_criterion(output, target)

    # ================== Mixup Loss ==================
    if args.mixup_weight > 0:
        loss_mixup = mixup_criterion(image, target, net)
    else:
        loss_mixup = torch.tensor(0.0, device=image.device)

    # ================== CRL Loss ==================
    if args.crl_weight > 0:
        loss_crl = rank_criterion(output, image_idx, correct_log)
    else:
        loss_crl = torch.tensor(0.0, device=image.device)

    # ================== PixMix Loss ==================
    pix_output = net(pixmix_image)
    loss_pixmix = cls_criterion(pix_output, target)

    # ================== CutMix Loss ==================
    if args.cutmix_weight > 0 and args.cutmix_beta > 0 and torch.rand(1).item() < args.cutmix_prob:
        # Sample lambda from Beta distribution
        lam = torch.distributions.Beta(args.cutmix_beta, args.cutmix_beta).sample().item()

        # Generate shuffled indices
        rand_index = torch.randperm(image.size(0), device=image.device)

        # Define two sets of targets
        target_a = target
        target_b = target[rand_index]

        # Random bounding box
        bbx1, bby1, bbx2, bby2 = rand_bbox(image.size(), lam)

        # Apply CutMix
        mixed_image = image.clone()
        mixed_image[:, :, bbx1:bbx2, bby1:bby2] = image[rand_index, :, bbx1:bbx2, bby1:bby2]

        # Adjust lambda according to the patch size
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (image.size(-1) * image.size(-2)))

        # Forward and compute loss
        cutmix_output = net(mixed_image)
        loss_cutmix = (F.cross_entropy(cutmix_output, target_a) * lam +
                       F.cross_entropy(cutmix_output, target_b) * (1. - lam))
    else:
        loss_cutmix = torch.tensor(0.0, device=image.device)

    # ================== Total Loss ==================
    loss = (loss_cls
            + args.mixup_weight * loss_mixup
            + args.crl_weight * loss_crl
            + args.cutmix_weight * loss_cutmix
            + args.pixmix_weight * loss_pixmix)

    return loss, loss_cls, loss_mixup, loss_crl, loss_cutmix, loss_pixmix, output


def rand_bbox(size, lam):
    """
    Generate random bounding box for CutMix.
    size: (B, C, H, W)
    lam: lambda parameter
    """
    H, W = size[2], size[3]
    cut_rat = (1. - lam) ** 0.5
    cut_h = int(H * cut_rat)
    cut_w = int(W * cut_rat)

    # uniform center
    cx = torch.randint(W, (1,))
    cy = torch.randint(H, (1,))

    bbx1 = torch.clamp(cx - cut_w // 2, 0, W)
    bby1 = torch.clamp(cy - cut_h // 2, 0, H)
    bbx2 = torch.clamp(cx + cut_w // 2, 0, W)
    bby2 = torch.clamp(cy + cut_h // 2, 0, H)

    return bbx1.item(), bby1.item(), bbx2.item(), bby2.item()

def disable_running_stats(model):
    def _disable(module):
        if isinstance(module, _BatchNorm):
            module.backup_momentum = module.momentum
            module.momentum = 0

    model.apply(_disable)

def enable_running_stats(model):
    def _enable(module):
        if isinstance(module, _BatchNorm) and hasattr(module, "backup_momentum"):
            module.momentum = module.backup_momentum
    model.apply(_enable)


def train(train_loader, net, net_ema, optimizer, epoch, correct_log, logger, writer, cos_scheduler, args):
    net.train()

    ## define criterion

    cls_criterion = torch.nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    mixup_criterion = Mixup_Criterion(beta=args.mixup_beta, cls_criterion=cls_criterion)
    rank_criterion = CRL_Criterion(args)

    cutout = None


    train_log = {
        'Top1 Acc.': utils.utils.AverageMeter(),
        'CLS Loss': utils.utils.AverageMeter(),
        'Mixup Loss': utils.utils.AverageMeter(),
        'CRL Loss': utils.utils.AverageMeter(),
        'CutMix Loss': utils.utils.AverageMeter(),
        'PixMix Loss': utils.utils.AverageMeter(),
        'Tot. Loss': utils.utils.AverageMeter(),
        'LR': utils.utils.AverageMeter(),
    }

    msg = '####### --- Training Epoch {:d} --- #######'.format(epoch)
        
    if logger is not None:
        logger.info(msg)
        
    for i, (image, pixmix_image, target, image_idx) in enumerate(train_loader):
        image, pixmix_image, target = image.cuda(), pixmix_image.cuda(), target.cuda()
        
        enable_running_stats(net)
        
        loss, loss_cls, loss_mixup, loss_crl, loss_cutmix, loss_pixmix, output = compute_loss(epoch,
                                                                                    args,
                                                                                    net,
                                                                                    image,
                                                                                    pixmix_image,
                                                                                    target,
                                                                                    image_idx,
                                                                                    correct_log,
                                                                                    cls_criterion,
                                                                                    mixup_criterion,
                                                                                    rank_criterion
                                                                                    )
        optimizer.zero_grad()
        loss.backward()
        if args.optim_name in ['sam', 'fmfp', 'fsam', 'fmfpfsam', 'csam', 'cfsam']:

            optimizer.first_step(zero_grad=True)

            disable_running_stats(net)

            compute_loss(epoch, args, net, image, pixmix_image, target, image_idx, correct_log, cls_criterion, mixup_criterion,
                         rank_criterion)[0].backward()
            optimizer.second_step(zero_grad=True)

            if args.optim_name in ['fmfp', 'fmfpfsam']:
                if epoch <= args.swa_epoch_start and (not args.per_epoch_scheduler):
                    cos_scheduler.step()                    
            elif (not args.per_epoch_scheduler):
                cos_scheduler.step()     

        else:
            optimizer.step()
            if (not args.per_epoch_scheduler):
                cos_scheduler.step()  
        
        if not args.use_cosine:
            net_ema.update(net)


        prec, correct = utils.utils.accuracy(output, target)

        correct_log.update(image_idx, correct)
        for param_group in optimizer.param_groups:
            lr = param_group["lr"]
            break

        train_log['Tot. Loss'].update(loss.item(), image.size(0))
        train_log['CLS Loss'].update(loss_cls.item(), image.size(0))
        train_log['Mixup Loss'].update(loss_mixup.item(), image.size(0))
        train_log['CRL Loss'].update(loss_crl.item(), image.size(0))
        train_log['CutMix Loss'].update(loss_cutmix.item(), image.size(0))
        train_log['PixMix Loss'].update(loss_pixmix.item(), image.size(0))
        train_log['Top1 Acc.'].update(prec.item(), image.size(0))
        train_log['LR'].update(lr, image.size(0))

        if i % 100 == 99:
            log = ['LR : {:.5f}'.format(train_log['LR'].avg)] + [key + ': {:.3f}'.format(train_log[key].avg) for key in
                                                                 train_log if key != 'LR']
            msg = 'Epoch {:d} \t Batch {:d}\t'.format(epoch, i) + '\t'.join(log)
                
            if logger is not None:
                logger.info(msg)
            for key in train_log:
                train_log[key] = utils.utils.AverageMeter()
    correct_log.max_correctness_update(epoch)
    if writer:
        for key in train_log:
            writer.add_scalar('./Train/' + key, train_log[key].avg, epoch)



