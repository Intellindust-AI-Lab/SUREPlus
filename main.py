#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main training script.
Author: Yang Li
"""

import os
import json
import torch
import torch.backends.cudnn
import torch.utils.tensorboard
import torch.multiprocessing as mp
import torch.distributed as dist

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.swa_utils import AveragedModel
import resource

# Internal modules
import train
import model.get_model
import utils.optim as optim
import data.dataset
import utils.utils
import utils.valid as valid
import utils.option
import utils.ema


# ==============================
# Utility Functions
# ==============================
def synchronize():
    """
    Synchronize (barrier) among all distributed processes.
    Ensures all GPUs are at the same state before continuing.
    """
    if not dist.is_available() or not dist.is_initialized():
        return
    if dist.get_world_size() > 1:
        dist.barrier()


def find_free_port():
    """
    Automatically find an available TCP port on the current machine.

    Returns:
        str: Available port number.
    """
    import socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("", 0))
    port = sock.getsockname()[1]
    sock.close()
    return str(port)


# ==============================
# Main Training Function
# ==============================
def main(proc_idx, args):
    """
    Main training loop for each GPU process (used in DDP mode).

    Args:
        proc_idx (int): Process index (GPU ID).
        args (argparse.Namespace): Command-line arguments.
    """
    # --------------------------
    # Distributed initialization
    # --------------------------
    gpu_id = proc_idx
    rank = int(gpu_id)
    torch.cuda.set_device(proc_idx)
    dist.init_process_group('nccl', rank=rank, world_size=len(args.gpu))
    torch.backends.cudnn.benchmark = True

    # --------------------------
    # Environment setup
    # --------------------------
    utils.utils.fix_seed(seed=42)
    if rank == 0:
        os.makedirs(args.save_dir, exist_ok=True)
        writer = torch.utils.tensorboard.SummaryWriter(args.save_dir)
        logger = utils.utils.get_logger(args.save_dir)
        logger.info(json.dumps(vars(args), indent=4, sort_keys=True))
    else:
        writer, logger = None, None

    # --------------------------
    # Dataset and Dataloader
    # --------------------------
    train_loader, valid_loader, _, nb_cls = data.dataset.get_loaders(args.train_dir, 
                                                                    args.val_dir,
                                                                    args.test_dir, 
                                                                    args.pixmix_path, 
                                                                    args.data_name,
                                                                    args.batch_size, 
                                                                    num_workers=8,
                                                                    model_name=args.model_name)

    # ==========================
    # Multiple experiment runs
    # ==========================
    for run_idx in range(args.nb_run):
        if rank == 0:
            logger.info(f"\n{'#' * 100}\n>>> Running Experiment {run_idx + 1} / {args.nb_run}")

        # --------------------------
        # Model setup
        # --------------------------
        net = model.get_model.get_model(args.model_name, nb_cls, logger, args)
        net = net.to(rank)
        net = DDP(net, device_ids=[rank], output_device=rank,
                  broadcast_buffers=False, find_unused_parameters=True)

        # EMA model
        net_ema = utils.ema.ModelEMA(net)

        optimizer, cos_scheduler, swa_model, swa_scheduler = optim.get_optimizer_scheduler(
            args.model_name, args.optim_name, net,
            args.lr, args.momentum, args.weight_decay,
            max_epoch_cos=args.epochs,
            swa_lr=args.swa_lr,
            args=args,
            train_loader=train_loader
        )

        correct_log = train.Correctness_Log(len(train_loader.dataset))
        best_acc, best_acc_ema = 0.0, 0.0

        # ==========================
        # Training Loop
        # ==========================
        for epoch in range(1, args.epochs + 1):

            # Train one 
            train.train_epoch(
                train_loader, net, net_ema, optimizer,
                epoch, correct_log, logger, writer,
                cos_scheduler, args
            )
            #  Learning rate scheduling
            if args.optim_name in ['swa', 'fmfp']:
                if epoch > args.swa_epoch_start:
                    swa_model.update_parameters(net.module if len(args.gpu) > 1 else net)
                    swa_scheduler.step()
                elif args.per_epoch_scheduler:
                    cos_scheduler.step()
            elif args.per_epoch_scheduler:
                cos_scheduler.step()

            # Step 4: EMA update for batchnorm
            net_ema.update_bn(train_loader, device='cuda')

            # --------------------------
            # Validation phase
            # --------------------------
            if rank == 0:
                report_dict = {}

                # SWA Model
                if args.optim_name in ['swa', 'fmfp', 'fmfpfsam'] and epoch > args.swa_epoch_start:

                    torch.optim.swa_utils.update_bn(train_loader, swa_model, device='cuda')
                    res_swa = valid.validation(valid_loader, swa_model.cuda())

                    msg = f"[Epoch {epoch}] Validation (SWA) " + ' | '.join([f"{k}: {v:.3f}" for k, v in res_swa.items()])
                    logger.info(msg)

                    for k, v in res_swa.items():
                        writer.add_scalar(f'Val/SWA_{k}', v, epoch)

                    if res_swa['Acc.'] > best_acc:
                        logger.info(f"SWA Model Accuracy ↑ {best_acc:.2f} → {res_swa['Acc.']:.2f}")
                        best_acc = res_swa['Acc.']
                        best_aurc = res_swa.get('AURC', None)
                        torch.save(swa_model.state_dict(), os.path.join(args.save_dir, f'best_acc_net_swa_{run_idx+1}.pth'))

                    report_dict = {'Acc.': best_acc, 'AURC': best_aurc}

                # EMA Model
                else:
                    if args.rebn:
                        res_ema = valid.validation(valid_loader, net_ema.ema)
                        msg = f"[Epoch {epoch}] Validation (EMA) " + ' | '.join([f"{k}: {v:.3f}" for k, v in res_ema.items()])
                        logger.info(msg)

                        for k, v in res_ema.items():
                            writer.add_scalar(f'Val/EMA_{k}', v, epoch)

                        if res_ema['Acc.'] > best_acc_ema:
                            logger.info(f"EMA Model Accuracy ↑ {best_acc_ema:.2f} → {res_ema['Acc.']:.2f}")
                            best_acc_ema = res_ema['Acc.']
                            best_aurc_ema = res_ema.get('AURC', None)
                            torch.save(net_ema.ema.state_dict(), os.path.join(args.save_dir, f'best_acc_net_ema_{run_idx+1}.pth'))

                        report_dict = {'Acc.': best_acc_ema, 'AURC': best_aurc_ema}

                    # Original Model
                    else:
                        res_orig = valid.validation(valid_loader, net)
                        msg = f"[Epoch {epoch}] Validation (Original) " + ' | '.join([f"{k}: {v:.3f}" for k, v in res_orig.items()])
                        logger.info(msg)

                        for k, v in res_orig.items():
                            writer.add_scalar(f'Val/Original_{k}', v, epoch)
                            
                        if res_orig['Acc.'] > best_acc:
                            logger.info(f"Original Model Accuracy ↑ {best_acc:.2f} → {res_orig['Acc.']:.2f}")
                            best_acc = res_orig['Acc.']
                            best_aurc = res_orig.get('AURC', None)
                            torch.save(net.state_dict(), os.path.join(args.save_dir, f'best_acc_net_orig_{run_idx+1}.pth'))

                        report_dict = {'Acc.': best_acc, 'AURC': best_aurc}

                logger.info(f"[Epoch {epoch}] Best Report: Acc={report_dict['Acc.']:.3f}, AURC={report_dict['AURC']:.3f}")


        synchronize()


# ==============================
# Entry Point
# ==============================
if __name__ == "__main__":
    # Increase system file descriptor limit
    rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (2048, rlimit[1]))

    # Parse arguments
    args = utils.option.get_args_parser()
    torch.backends.cudnn.benchmark = True

    # GPU configuration
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, args.gpu))
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = find_free_port()

    # Launch single or multi-GPU training
    if len(args.gpu) == 1:
        main(0, args)
    else:
        mp.spawn(main, nprocs=len(args.gpu), args=(args,))
