import torch.backends.cudnn
import torch.utils.tensorboard

import os
import json 

import train
import valid
from torch.optim.swa_utils import AveragedModel
import model.get_model
import optim

import data.dataset
import utils.utils
import utils.option
import utils.ema
from test import test

import resource
# the multi GPU setting support
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

def synchronize():
    """
    Helper function to synchronize (barrier) among all processes when using distributed training
    """
    if not dist.is_available():
        return
    if not dist.is_initialized():
        return
    world_size = dist.get_world_size()
    if world_size == 1:
        return
    dist.barrier()

def find_free_port():
    """
    Find an available port of current machine / node.
    """
    import socket

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # Binding to port 0 will cause the OS to find an available port for us
    sock.bind(("", 0))
    port = sock.getsockname()[1]
    sock.close()
    # NOTE: there is still a chance the port could be taken by other processes.
    return str(port)

def main(proc_idx, args):
    gpu_id = proc_idx
    rank = int(gpu_id)
    torch.cuda.set_device(proc_idx)

    dist.init_process_group('nccl', rank=rank, world_size=len(args.gpu))

    torch.backends.cudnn.benchmark = True

    utils.utils.fix_seed(seed=42)
    if rank == 0:
        save_path = os.path.join(args.save_dir, f"{args.data_name}_{args.model_name}_{args.loss}_{args.optim_name}-mixup_{args.mixup_weight}-crl_{args.crl_weight}-pixmix_{args.pixmix_weight}-cutmix_{args.cutmix_weight}_usecos_{args.use_cosine}_act_{args.activation}")
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        writer = torch.utils.tensorboard.SummaryWriter(save_path)
        logger = utils.utils.get_logger(save_path)
        logger.info(json.dumps(vars(args), indent=4, sort_keys=True))
    else:
        writer, logger = None, None

    # os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    train_loader, valid_loader, _, nb_cls = data.dataset.get_loader(args, args.data_name, args.train_dir, args.val_dir,args.test_dir,
                                                                        args.batch_size, args.imb_factor, args.model_name, args.gpu)

    for r in range(args.nb_run):
        prefix = '{:d} / {:d} Running'.format(r + 1, args.nb_run)
        if rank ==0:logger.info(100*'#' + '\n' + prefix)
        
        ## define model, optimizer 
        net = model.get_model.get_model(args.model_name, nb_cls, logger, args)
        net = net.to(rank)
        # multi GPU support
        net = DDP(net, device_ids=[rank], output_device=rank, broadcast_buffers=False, find_unused_parameters=True)
        net_ema = utils.ema.ModelEMA(net)
        # print(net)
        if args.resume:
            if args.use_cosine: 
                net.module.load_state_dict(torch.load(args.resume), strict=False)
            else:
                net.module.load_state_dict(torch.load(args.resume), strict=True)
            if logger :logger.info("Successfully Loaded checkpoints")
            if args.optim_name == 'fmfp' or args.optim_name == 'swa':
                net = AveragedModel(net)
        optimizer, cos_scheduler, swa_model, swa_scheduler = optim.get_optimizer_scheduler(args.model_name,
                                                                                        args.optim_name,
                                                                                        net,
                                                                                        args.lr,
                                                                                        args.momentum,
                                                                                        args.weight_decay,
                                                                                        max_epoch_cos=args.epochs,
                                                                                        swa_lr=args.swa_lr,
                                                                                        args=args,
                                                                                        train_loader=train_loader)


        # make logger
        correct_log, best_acc, best_acc_ema = train.Correctness_Log(len(train_loader.dataset)), 0, 0

        # start Train
        for epoch in range(1, args.epochs + 2):
            train.train(train_loader, net, net_ema, optimizer, epoch, correct_log, logger, writer, cos_scheduler, args)

            if args.optim_name in ['swa', 'fmfp'] :
                if epoch > args.swa_epoch_start:
                    if len(args.gpu) > 1:
                        swa_model.update_parameters(net.module)
                    else:
                        swa_model.update_parameters(net)
                    swa_scheduler.step()
                elif args.per_epoch_scheduler:
                    cos_scheduler.step()
            elif args.per_epoch_scheduler:
                cos_scheduler.step()

            # net_ema ReBN
            if args.rebn:
                net_ema.update_bn(train_loader, device='cuda')

            # validation
            if rank == 0:
                if args.loss == 'ce':
                    if args.optim_name in ['swa', 'fmfp', 'fmfpfsam'] and epoch > args.swa_epoch_start:
                        # SWA validation
                        torch.optim.swa_utils.update_bn(train_loader, swa_model, device='cuda')
                        net_val = swa_model.cuda()
                        res = valid.validation(valid_loader, net_val)
                        log = [key + ': {:.3f}'.format(res[key]) for key in res]
                        msg = '################## \n ---> Validation Epoch {:d} (SWA Model)\t'.format(epoch) + '\t'.join(log)
                        logger.info(msg)
                        if r < 1:
                            for key in res:
                                writer.add_scalar('./Val/SWA_' + key, res[key], epoch)
                    else:
                        # Original model validation
                        net_val = net
                        res_orig = valid.validation(valid_loader, net_val)
                        log_orig = [key + ': {:.3f}'.format(res_orig[key]) for key in res_orig]
                        msg_orig = '################## \n ---> Validation Epoch {:d} (Original Model)\t'.format(epoch) + '\t'.join(log_orig)
                        logger.info(msg_orig)

                        # EMA model validation if enabled
                        res_ema = valid.validation(valid_loader, net_ema.ema)
                        log_orig = [key + ': {:.3f}'.format(res_ema[key]) for key in res_ema]
                        msg_orig = '################## \n ---> Validation Epoch {:d} (EMA Model)\t'.format(epoch) + '\t'.join(log_orig)
                        logger.info(msg_orig)

                else:
                    if args.optim_name in ['swa', 'fmfp', 'fmfpfsam'] and epoch > args.swa_epoch_start:
                        # SWA validation
                        torch.optim.swa_utils.update_bn(train_loader, swa_model, device='cuda')
                        net_val = swa_model.cuda()
                        res = valid.validation_sigmoid(valid_loader, net_val)
                        log = [key + ': {:.3f}'.format(res[key]) for key in res]
                        msg = '################## \n ---> Validation Epoch {:d} (SWA Model)\t'.format(epoch) + '\t'.join(log)
                        logger.info(msg)
                        if r < 1:
                            for key in res:
                                writer.add_scalar('./Val/SWA_' + key, res[key], epoch)
                    else:
                        # Original model validation
                        net_val = net
                        res_orig = valid.validation_sigmoid(valid_loader, net_val)
                        log_orig = [key + ': {:.3f}'.format(res_orig[key]) for key in res_orig]
                        msg_orig = '################## \n ---> Validation Epoch {:d} (Original Model)\t'.format(epoch) + '\t'.join(log_orig)
                        logger.info(msg_orig)

                        # EMA model validation if enabled
                        res_ema = valid.validation_sigmoid(valid_loader, net_ema.ema)
                        log_orig = [key + ': {:.3f}'.format(res_ema[key]) for key in res_ema]
                        msg_orig = '################## \n ---> Validation Epoch {:d} (EMA Model)\t'.format(epoch) + '\t'.join(log_orig)
                        logger.info(msg_orig)


                if r < 1:
                    for key in res_orig:
                        writer.add_scalar('./Val/Original_' + key, res_orig[key], epoch)
                        if net_ema is not None and key in res_ema:
                            writer.add_scalar('./Val/EMA_' + key, res_ema[key], epoch)

                # Track best accuracy for original model
                if res_orig['Acc.'] > best_acc:
                    acc = res_orig['Acc.']
                    msg = f'Original Model Accuracy improved from {best_acc:.2f} to {acc:.2f}!!!'
                    logger.info(msg)
                    best_acc = acc
                    torch.save(net.state_dict(), os.path.join(save_path, f'best_acc_net_orig_{r+1}.pth'))

                # Track best accuracy for EMA model if enabled
                if net_ema is not None and res_ema['Acc.'] > best_acc_ema:
                    acc_ema = res_ema['Acc.']
                    msg = f'EMA Model Accuracy improved from {best_acc_ema:.2f} to {acc_ema:.2f}!!!'
                    logger.info(msg)
                    best_acc_ema = acc_ema
                    torch.save(net_ema.ema.state_dict(), os.path.join(save_path, f'best_acc_net_ema_{r+1}.pth'))

                # Optionally save SWA model if it outperforms
                if args.optim_name in ['swa', 'fmfp', 'fmfpfsam'] and epoch > args.swa_epoch_start and res['Acc.'] > best_acc:
                    logger.info(f'SWA Model Accuracy improved from {best_acc:.2f} to {res["Acc."]:.2f}!!!')
                    best_acc = res['Acc.']
                    torch.save(swa_model.state_dict(), os.path.join(save_path, f'best_acc_net_swa_{r+1}.pth'))
        synchronize()



if __name__ == "__main__":
    rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (2048, rlimit[1]))

    args = utils.option.get_args_parser()
    torch.backends.cudnn.benchmark = True
    
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, args.gpu))

    ## fix avilabel port
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = find_free_port()
    if len(args.gpu) == 1:
        main(0, args)
    else:
        mp.spawn(main, nprocs=len(args.gpu), args=(args,))
