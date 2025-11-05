import utils.utils
import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F
from torch.nn.modules.batchnorm import _BatchNorm


class RegMixup_Criterion(nn.Module):
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

    def __init__(self):
        super().__init__()
        self.rank_criterion = torch.nn.MarginRankingLoss(margin=0)
        
    def forward(self, output, image_idx, correct_log):
        conf = torch.softmax(output, dim=1)
        conf,_ = conf.max(dim=1)  # bs x 1
        conf_roll, image_idx_roll = torch.roll(conf, -1), torch.roll(image_idx, -1)
        rank_target, rank_margin = correct_log.get_target_margin(image_idx, image_idx_roll)
        conf_roll = conf_roll + rank_margin / (rank_target + 1e-7)
    
        ranking_loss = self.rank_criterion(conf, conf_roll, rank_target)
        
        return ranking_loss
    

def compute_loss(args,
                 net,
                 image,
                 pixmix_image,
                 target,
                 image_idx,
                 correct_log,
                 cls_criterion,
                 regmixup_criterion,
                 rank_criterion):
    """
    Compute total loss with multiple components:
    - Classification loss (CLS)
    - Mixup loss
    - CRL loss
    - PixMix loss
    """

    # ================== Forward: Normal ==================
    output = net(image)

    # ================== CLS Loss ==================
    loss_cls = cls_criterion(output, target)

    # ================== RegMixup Loss ==================
    if args.regmixup_weight > 0:
        loss_mixup = regmixup_criterion(image, target, net)
    else:
        loss_mixup = torch.tensor(0.0, device=image.device)

    # ================== CRL Loss ==================
    if args.crl_weight > 0:
        loss_crl = rank_criterion(output, image_idx, correct_log)
    else:
        loss_crl = torch.tensor(0.0, device=image.device)

    # ================== RegPixMix Loss ==================
    if args.pixmix_weight > 0:
        pix_output = net(pixmix_image)
        loss_pixmix = cls_criterion(pix_output, target)
    else:
        loss_pixmix = torch.tensor(0.0, device=image.device)

    # ================== Total Loss ==================
    loss =  loss_cls + args.regmixup_weight * loss_mixup + args.crl_weight * loss_crl + args.pixmix_weight * loss_pixmix

    return loss, loss_cls, loss_mixup, loss_crl, loss_pixmix, output



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


def train_epoch(train_loader, net, net_ema, optimizer, epoch, correct_log, logger, writer, cos_scheduler, args):
    net.train()

    ## define criterion

    cls_criterion = torch.nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    regmixup_criterion = RegMixup_Criterion(beta=args.mixup_beta, cls_criterion=cls_criterion)
    rank_criterion = CRL_Criterion()

    train_log = {
        'Top1 Acc.': utils.utils.AverageMeter(),
        'CLS Loss': utils.utils.AverageMeter(),
        'Mixup Loss': utils.utils.AverageMeter(),
        'CRL Loss': utils.utils.AverageMeter(),
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
        
        loss, loss_cls, loss_mixup, loss_crl, loss_pixmix, output = compute_loss(args,
                                                                                net,
                                                                                image,
                                                                                pixmix_image,
                                                                                target,
                                                                                image_idx,
                                                                                correct_log,
                                                                                cls_criterion,
                                                                                regmixup_criterion,
                                                                                rank_criterion)
        optimizer.zero_grad()
        loss.backward()
        if args.optim_name in ['sam', 'fmfp', 'fsam', 'fmfpfsam']:

            optimizer.first_step(zero_grad=True)

            disable_running_stats(net)

            compute_loss(args, net, image, pixmix_image, target, image_idx, correct_log, cls_criterion, regmixup_criterion,
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



