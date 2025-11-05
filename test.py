import torch
import utils.valid as valid
import os
import utils.test_option
import data.dataset
import utils.utils
import model.get_model
from torch.optim.swa_utils import AveragedModel
from utils.ema import ModelEMA
import csv
from torch.utils.data import DataLoader
import torchvision.transforms
import utils.ood_metric
import utils.svhn_loader as svhn
from torchvision.datasets import ImageFolder
from torch.optim.swa_utils import AveragedModel
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
import numpy as np
from tqdm import tqdm
from model.get_model import reparameterize_model
to_np = lambda x: x.detach().cpu().numpy()

from collections import OrderedDict

def load_model(model, model_path):
    """
    加载模型权重，并自动处理 DDP/DataParallel 保存的权重文件中带有的 'module.' 前缀。
    
    Args:
        model: 模型对象
        model_path: 权重文件路径
        strict: 是否启用严格模式（默认 True）

    Returns:
        model: 加载后的模型
    """
    print(f"[INFO] Loading model from {model_path}")
    state_dict = torch.load(model_path, map_location='cpu')

    # 自动检测是否是 DDP 保存的模型
    new_state_dict = OrderedDict()
    for key in state_dict.keys():
        if key.startswith('module.'):
            # 去掉 'module.' 前缀
            new_state_dict[key[7:]] = state_dict[key]
        else:
            new_state_dict[key] = state_dict[key]


    model.load_state_dict(new_state_dict, strict=True)


    return model

def process_results(loader, model, metrics, logger, method_name, results_storage, loss):
    if loss == 'ce':
        res = valid.validation(loader, model)
    elif loss == 'bce':
        res = valid.validation_sigmoid(loader, model)
    else:
        raise
    for metric in metrics:
        results_storage[metric].append(res[metric])
    log = [f"{key}: {res[key]:.2f}" for key in res]
    logger.info(f'ID Results:\n' + '\t'.join(log))
    return res


def test_cifar10c_corruptions(model, test_dir, transform_test, batch_size, metrics, logger):
    cor_results_storage = {corruption: {severity: {metric: [] for metric in metrics} for severity in range(1, 6)} for
                           corruption in data.CIFAR10C.CIFAR10C.cifarc_subsets}

    for corruption in data.CIFAR10C.CIFAR10C.cifarc_subsets:
        for severity in range(1, 6):
            logger.info(f"Testing on corruption: {corruption}, severity: {severity}")
            corrupted_test_dataset = data.CIFAR10C.CIFAR10C(root=test_dir, transform=transform_test, subset=corruption,
                                                            severity=severity, download=True)
            corrupted_test_loader = DataLoader(dataset=corrupted_test_dataset, batch_size=batch_size, shuffle=False,
                                               num_workers=4, drop_last=False)
            res = valid.validation(corrupted_test_loader, model)
            for metric in metrics:
                cor_results_storage[corruption][severity][metric].append(res[metric])

    return cor_results_storage

@torch.no_grad()
def get_output(model, test_loader, loss='ce'):

    model.eval()
    id_preds = []       # Store class preds
    osr_preds = []      # Stores OSR preds

    # First extract all features
    if loss=='ce':
        for data in tqdm(test_loader, leave=False):

            images = data[0].cuda()
            logits = model(images)
            # logits = model(images)
            sftmax = torch.nn.functional.softmax(logits, dim=-1)

            id_preds.extend(to_np(sftmax.argmax(dim=-1)))
            osr_preds.extend(to_np(sftmax.max(dim=-1)[0]))
            
            
        id_preds = np.array(id_preds)
        osr_preds = np.array(osr_preds)
        
        return id_preds, osr_preds
    elif loss=='bce':
        for data in tqdm(test_loader, leave=False):

            images = data[0].cuda()
            logits = model(images)
            # logits = model(images)
            smoid = torch.sigmoid(logits)
            pred_osr, pred_cls = smoid.max(1)

            id_preds.extend(to_np(pred_cls))
            osr_preds.extend(to_np(pred_osr))
            
        id_preds = np.array(id_preds)
        osr_preds = np.array(osr_preds)
        
        return id_preds, osr_preds
    elif loss=='energy':
        for data in tqdm(test_loader, leave=False):

            images = data[0].cuda()
            logits = model(images)
            sftmax = torch.nn.functional.softmax(logits, dim=-1)

            id_preds.extend(to_np(sftmax.argmax(dim=-1)))
            osr_preds.extend(to_np(torch.logsumexp(logits, dim=-1)))
            # print(osr_preds)
            
        id_preds = np.array(id_preds)
        osr_preds = np.array(osr_preds)
        
        return id_preds, osr_preds
    else:
        raise


def test_ood(model, id_loader, ood_dir, batch_size=64, logger=None, loss='ce', model_name='resnet18', data_name='cifar100'):
    logger.info('OOD Results:')
    if model_name == 'deit' and data_name == 'imagenet1k':
        mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
        img_size = 224
    if model_name == 'FastViT-SA24':
        mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
        img_size = 256
    else:
        mean, std = (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)
        img_size = 32
    id_labels = np.array([x[1] for x in id_loader.dataset.samples])
    id_class_pred, id_ood_pred = get_output(model, id_loader, loss)
    auroc_list, fpr_list,  acc_list = [], [], []

    if data_name == 'imagenet1k':
        logger.info('----------Openood Detection----------')
        auroc_list, fpr_list,  acc_list = [], [], []
        test_data_list = ['ninco_data',  'ssb_hard_data', 'openimage_o_data', 'inaturalist_data', 'textures_data']
        # test_data_list = ['ssb_hard']
        for test_data in test_data_list:
            logger.info(test_data)
            ood_data = ImageFolder(root=os.path.join(ood_dir, test_data),
                                        transform=Compose([
                                            Resize(int(img_size/0.875)), 
                                            CenterCrop(img_size),
                                            ToTensor(), 
                                            Normalize(mean, std)]))
            ood_loader = torch.utils.data.DataLoader(ood_data, batch_size=batch_size, shuffle=True,
                                                    num_workers=8, pin_memory=True)
            _, ood_ood_pred = get_output(model, ood_loader, loss)
            fpr, auroc, acc = utils.ood_metric.get_metric(
                id_labels, id_class_pred, id_ood_pred, ood_ood_pred, 1, logger)

            fpr_list.append(fpr); auroc_list.append(auroc); acc_list.append(acc)
        near_ood_auroc = np.mean(auroc_list[:2])
        far_ood_auroc = np.mean(auroc_list[2:])
        near_ood_fpr = np.mean(fpr_list[:2])
        far_ood_fpr = np.mean(fpr_list[2:])
        
        logger.info('Near-OOD AUROC: {:.2f} | Near-OOD FPR95: {:.2f}'.format(near_ood_auroc * 100, near_ood_fpr * 100))
        logger.info('Far-OOD AUROC: {:.2f} | Far-OOD FPR95: {:.2f}'.format(far_ood_auroc * 100, far_ood_fpr * 100))
        return 100*np.mean(acc_list), 100*np.mean(auroc_list), 100*np.mean(fpr_list)

    else:
        # /////////////// Textures ///////////////
        logger.info('----------Texture Detection----------')
        ood_data = ImageFolder(root=os.path.join(ood_dir, 'dtd/images'),
                                    transform=Compose([
                                        Resize(int(img_size/0.875)), 
                                        CenterCrop(img_size),
                                        ToTensor(), 
                                        Normalize(mean, std)]))
        ood_loader = torch.utils.data.DataLoader(ood_data, batch_size=batch_size, shuffle=True,
                                                num_workers=8, pin_memory=True)
        _, ood_ood_pred = get_output(model, ood_loader, loss)
        fpr, auroc, acc  = utils.ood_metric.get_metric(
            id_labels, id_class_pred, id_ood_pred, ood_ood_pred, 1, logger)
        fpr_list.append(fpr); auroc_list.append(auroc); acc_list.append(acc)
        
        # /////////////// SVHN /////////////// # cropped and no sampling of the test set
        logger.info('----------SVHN Detection----------')
        ood_data = svhn.SVHN(root=ood_dir, split="test",
                                transform=Compose([
                                    Resize(img_size), 
                                    ToTensor(), 
                                    Normalize(mean, std)
                                    ]), download=False)
        ood_loader = torch.utils.data.DataLoader(ood_data, batch_size=batch_size, shuffle=True,
                                                num_workers=8, pin_memory=True)
        _, ood_ood_pred = get_output(model, ood_loader, loss)
        fpr, auroc, acc  = utils.ood_metric.get_metric(
            id_labels, id_class_pred, id_ood_pred, ood_ood_pred, 1, logger)
        fpr_list.append(fpr); auroc_list.append(auroc); acc_list.append(acc)

        # /////////////// Places365 ///////////////
        logger.info('----------Places365 Detection----------')
        ood_data = ImageFolder(root=os.path.join(ood_dir,'Places365'),
                                    transform=Compose([
                                        Resize(int(img_size/0.875)), 
                                        CenterCrop(img_size),
                                        ToTensor(), 
                                        Normalize(mean, std)]))
        ood_loader = torch.utils.data.DataLoader(ood_data, batch_size=batch_size, shuffle=True,
                                                num_workers=8, pin_memory=True)
        _, ood_ood_pred = get_output(model, ood_loader, loss)
        fpr, auroc, acc  = utils.ood_metric.get_metric(
            id_labels, id_class_pred, id_ood_pred, ood_ood_pred, 1, logger)
        fpr_list.append(fpr); auroc_list.append(auroc); acc_list.append(acc)

        # /////////////// LSUN-C ///////////////
        logger.info('----------LSUN_C Detection----------')
        ood_data = ImageFolder(root=os.path.join(ood_dir,'LSUN_C'),
                                    transform=Compose([
                                        Resize(img_size), 
                                        ToTensor(), 
                                        Normalize(mean, std)]))
        ood_loader = torch.utils.data.DataLoader(ood_data, batch_size=batch_size, shuffle=True,
                                                num_workers=8, pin_memory=True)
        _, ood_ood_pred = get_output(model, ood_loader, loss)
        fpr, auroc, acc  = utils.ood_metric.get_metric(
            id_labels, id_class_pred, id_ood_pred, ood_ood_pred, 1, logger)
        fpr_list.append(fpr); auroc_list.append(auroc); acc_list.append(acc)

        # /////////////// LSUN-R ///////////////
        logger.info('----------LSUN_Resize Detection----------')
        ood_data = ImageFolder(root=os.path.join(ood_dir,'LSUN_resize'),
                                    transform=Compose([
                                        Resize(img_size), 
                                        ToTensor(), 
                                        Normalize(mean, std)]))
        ood_loader = torch.utils.data.DataLoader(ood_data, batch_size=batch_size, shuffle=True,
                                                num_workers=8, pin_memory=True)
        _, ood_ood_pred = get_output(model, ood_loader, loss)
        fpr, auroc, acc  = utils.ood_metric.get_metric(
            id_labels, id_class_pred, id_ood_pred, ood_ood_pred, 1, logger)
        fpr_list.append(fpr); auroc_list.append(auroc); acc_list.append(acc)

        # /////////////// iSUN ///////////////
        logger.info('----------iSUN Detection----------')
        ood_data = ImageFolder(root=os.path.join(ood_dir,'iSUN'),
                                    transform=Compose([
                                        Resize(img_size), 
                                        ToTensor(), 
                                        Normalize(mean, std)]))
        ood_loader = torch.utils.data.DataLoader(ood_data, batch_size=batch_size, shuffle=True,
                                                num_workers=8, pin_memory=True)
        _, ood_ood_pred = get_output(model, ood_loader, loss)
        fpr, auroc, acc  = utils.ood_metric.get_metric(
            id_labels, id_class_pred, id_ood_pred, ood_ood_pred, 1, logger)
        fpr_list.append(fpr); auroc_list.append(auroc); acc_list.append(acc)

        logger.info('----------Mean Test Results!!!!!----------')
        logger.info('----------OOD Detection----------')
        logger.info('AUROC | FPR95 | ACC ')
        logger.info('{:.2f} | {:.2f} | {:.2f}'.format(
            100*np.mean(auroc_list), 100*np.mean(fpr_list),  100*np.mean(acc_list)))
        return 100*np.mean(acc_list), 100*np.mean(auroc_list), 100*np.mean(fpr_list)



def test(args):

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu[0])
    metrics = ['Acc.', 'AUROC', 'AUPR Succ.', 'AUPR', 'FPR', 'AURC', 'EAURC', 'ECE', 'NLL', 'Brier']
    results_storage = {metric: [] for metric in metrics}

    save_path = os.path.join(args.save_dir,
                               f"{args.data_name}_{args.model_name}_{args.loss}_{args.optim_name}-mixup_{args.mixup_weight}-crl_{args.crl_weight}-pixmix_{args.pixmix_weight}-mosaic_{args.mosaic_weight}_usecos_{args.use_cosine}_act_{args.activation}")
    logger = utils.utils.get_logger(save_path)
    
    if args.data_name in ['cifar10', 'cifar100'] :
        ood_dir = '/data4022/ood_datasets' 
    elif args.data_name == 'imagenet1k':
        ood_dir = '/data/Private/wushengliang/openood_data' 

    # Original Model
    for posthoc in ['ce']:
        for r in range(args.nb_run):
            logger.info(f'Testing Orginal model_{r + 1} ...')
            train_loader, _, test_loader, nb_cls = data.dataset.get_loader(args, args.data_name, args.train_dir, args.val_dir,
                                                                        args.test_dir, args.batch_size, args.imb_factor, args.model_name, args.gpu)
            
            # net = model.get_model.get_model(args.model_name, nb_cls, logger, args)
            # if args.optim_name == 'fmfp' or args.optim_name == 'swa':
            #     net = AveragedModel(net)
            # net = load_model(net, os.path.join(save_path, f'best_acc_net_orig_{r + 1}.pth'))

            # net.eval()
            # net = reparameterize_model(net)
            # # net.load_state_dict(torch.load(os.path.join(save_path, f'best_acc_net_orig_{r + 1}.pth'), map_location='cpu'), strict=True)
            
            # res = process_results(test_loader, net, metrics, logger, posthoc, results_storage, args.loss)
            # aurc = res['AURC']
            # acc, auroc, fpr95 = test_ood(net, test_loader, ood_dir, 64, logger, posthoc, args.model_name, args.data_name)
            # logger.info(f'\n>>>>>>>>>> Orig Model({posthoc}): Acc: {acc:.2f}, AURC: {aurc:.2f}, AUROC: {auroc:.2f}, FPR95: {fpr95:.2f} <<<<<<<<<<')


        # EMA Model
            logger.info(f'Testing EMA model_{r + 1} ...')
            net = model.get_model.get_model(args.model_name, nb_cls, logger, args)
            if args.optim_name == 'fmfp' or args.optim_name == 'swa':
                net = AveragedModel(net)
            net = load_model(net, os.path.join(save_path, f'best_acc_net_ema_{r + 1}.pth'))
            net = net.cuda()
            ema_model = ModelEMA(net)
            ema_model.update_bn(train_loader, "cuda:0")
        # ✅ 保存 EMA 模型的权重
            torch.save(ema_model.ema.state_dict(), os.path.join(save_path, f'best_acc_net_ema_rebn_{r + 1}.pth'))
            
            # ema_model.eval()
            net = reparameterize_model(ema_model.ema)
            # net.load_state_dict(torch.load(os.path.join(save_path, f'best_acc_net_ema_{r + 1}.pth'), map_location='cpu'), strict=True)
            net = net.cuda()
            net.eval()
            res = process_results(test_loader, net, metrics, logger, posthoc, results_storage, args.loss)
            aurc = res['AURC']
            acc, auroc, fpr95 = test_ood(net, test_loader, ood_dir, 64, logger, posthoc, args.model_name, args.data_name)
            logger.info(f'\n>>>>>>>>>> ReBN EMA Model({posthoc}): Acc: {acc:.2f}, AURC: {aurc:.2f}, AUROC: {auroc:.2f}, FPR95: {fpr95:.2f} <<<<<<<<<<')


if __name__ == '__main__':
    args = utils.test_option.get_args_parser()
    test(args)

