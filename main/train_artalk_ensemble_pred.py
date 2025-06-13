#!/usr/bin/env python
import os
import pdb
import time
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.multiprocessing as mp
import torch.distributed as dist
from base.baseTrainer import load_state_dict

import math
import cv2
from base.baseTrainer import poly_learning_rate, reduce_tensor, save_checkpoint
from base.utilities import get_parser, get_logger, main_process, AverageMeter
from models import get_model
from torch.optim.lr_scheduler import StepLR, LambdaLR

# cv2.ocl.setUseOpenCL(False)
# cv2.setNumThreads(0)

import warnings
warnings.filterwarnings("ignore")
import wandb

def main():
    args = get_parser()

    #os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(x) for x in args.train_gpu)
    cudnn.benchmark = True

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])
    args.distributed = args.world_size > 1 or args.multiprocessing_distributed
    args.ngpus_per_node = len(args.train_gpu)
    if len(args.train_gpu) == 1:
        args.train_gpu = args.train_gpu[0]
        args.sync_bn = False
        args.distributed = False
        args.multiprocessing_distributed = False

    #initialize wandb
    wandb.init(project="multitalk_new_s2", name=args.save_path.split("/")[-1],dir="logs")
    wandb.config.update(args)

    if args.multiprocessing_distributed:
        args.world_size = args.ngpus_per_node * args.world_size
        mp.spawn(main_worker, nprocs=args.ngpus_per_node, args=(args.ngpus_per_node, args))
    else:
        main_worker(args.train_gpu, args.ngpus_per_node, args)

def main_worker(gpu, ngpus_per_node, args):
    cfg = args
    cfg.gpu = gpu

    if cfg.distributed:
        if cfg.dist_url == "env://" and cfg.rank == -1:
            cfg.rank = int(os.environ["RANK"])
        if cfg.multiprocessing_distributed:
            cfg.rank = cfg.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=cfg.dist_backend, init_method=cfg.dist_url, world_size=cfg.world_size,
                                rank=cfg.rank)
    # ####################### Model ####################### #
    global logger
    logger = get_logger()

    model = get_model(cfg)
    if cfg.sync_bn:
        logger.info("using DDP synced BN")
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    if main_process(cfg):
        logger.info(cfg)
        logger.info("=> creating model ...")

    if cfg.distributed:
        torch.cuda.set_device(gpu)
        cfg.batch_size = int(cfg.batch_size / ngpus_per_node)
        cfg.batch_size_val = int(cfg.batch_size_val / ngpus_per_node)
        cfg.workers = int(cfg.workers / ngpus_per_node)
        model = torch.nn.parallel.DistributedDataParallel(model.cuda(gpu), device_ids=[gpu])
    else:
        torch.cuda.set_device(gpu)
        model = model.cuda()

    
    # ####################### Loss ############################# #
    loss_fn = nn.MSELoss()
    
    # ####################### Optimizer ######################## #
    if cfg.use_sgd:
        optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad,model.parameters()), lr=cfg.base_lr, momentum=cfg.momentum,
                                    weight_decay=cfg.weight_decay)
    else:
        optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad,model.parameters()), lr=cfg.base_lr)

    if cfg.StepLR:
        scheduler = StepLR(optimizer, step_size=cfg.step_size, gamma=cfg.gamma)
    else:
        scheduler = None
    
    # ####################### Data Loader ####################### #
    from dataset.data_loader_ensemble import get_dataloaders
    dataset = get_dataloaders(cfg)
    train_loader = dataset['train']

    if cfg.evaluate:
        val_loader = dataset['valid']
        test_loader = dataset['test']

    val_loss_log = 1000

    # =========== Load checkpoint ===========
    #checkpoint_path = cfg.noninteractive_pretrained_s2_path
    checkpoint_path = "/root/Projects/fasttalk/logs/joint_data/joint_data_1k_s2/model_100/model.pth.tar"
    print("=> loading checkpoint '{}'".format(checkpoint_path))
    checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage.cpu())
    load_state_dict(model, checkpoint['state_dict'], strict=False)
    print("=> loaded checkpoint '{}'".format(checkpoint_path))
    # =========== End load checkpoint ===========


    # ####################### Train ############################# #
    for epoch in range(cfg.start_epoch, cfg.epochs):
        loss_train, motion_loss_train, blendshapes_loss_train, reg_loss_train, feat_loss_meter = train(train_loader, model, loss_fn, optimizer, epoch, cfg)
        epoch_log = epoch + 1
        if cfg.StepLR:
            scheduler.step()
            print("Updated learning rate to {}".format(scheduler.get_last_lr()[0]))
        if main_process(cfg):
            logger.info('TRAIN Epoch: {} '
                        'loss_train: {} '
                        'motion_loss_train: {} ' 
                        'blendshapes_loss_train: {} '
                        'reg_loss_train: {} '
                        .format(epoch_log, loss_train, motion_loss_train, blendshapes_loss_train, reg_loss_train)
                        )

        wandb.log({"loss_train": loss_train, "motion_loss_train": motion_loss_train, "blendshapes_loss_train": blendshapes_loss_train, "feat_loss_meter": feat_loss_meter, "reg_loss_train": reg_loss_train}, epoch_log)

        if cfg.evaluate and (epoch_log % cfg.eval_freq == 0):
            loss_val = validate(val_loader, model, loss_fn, cfg)
            if main_process(cfg):
                logger.info('VAL Epoch: {} '
                            'loss_val: {} '
                            .format(epoch_log, loss_val)
                            )
            wandb.log({"loss_val": loss_val}, epoch_log)
            save_checkpoint(model,
                            sav_path=os.path.join(cfg.save_path, 'model_'+str(epoch_log)),
                            stage=2
                            )

def train(train_loader, model, loss_fn, optimizer, epoch, cfg):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    loss_meter = AverageMeter()
    loss_motion_meter = AverageMeter()
    loss_blendshapes_meter = AverageMeter()
    loss_reg_meter = AverageMeter()
    loss_feat_meter = AverageMeter()

    model.train()
    model.autoencoder.eval()

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"----> Total trainable parameters: {trainable_params}")

    end = time.time()
    max_iter = cfg.epochs * len(train_loader)

    for i, (audio, audio_features, vertice, blendshapes, template, filename) in enumerate(train_loader):

        ####################
        current_iter = epoch * len(train_loader) + i + 1
        data_time.update(time.time() - end)

        #################### cpu to gpu
        audio          = audio.cuda(cfg.gpu, non_blocking=True)
        audio_features = audio_features.cuda(cfg.gpu, non_blocking=True)
        vertice        = vertice.cuda(cfg.gpu, non_blocking=True)
        blendshapes    = blendshapes.cuda(cfg.gpu, non_blocking=True)
        template       = template.cuda(cfg.gpu, non_blocking=True)

        loss, loss_detail = model(filename, audio, audio_features, vertice, blendshapes, template, criterion=loss_fn)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        ######################
        batch_time.update(time.time() - end)
        end = time.time()
        for m, x in zip([loss_meter, loss_motion_meter, loss_blendshapes_meter, loss_reg_meter, loss_feat_meter],
                        [loss, loss_detail[0],  loss_detail[1], loss_detail[2], loss_detail[3]]):
            m.update(x.item(), 1)

        if cfg.poly_lr:
            current_lr = poly_learning_rate(cfg.base_lr, current_iter, max_iter, power=cfg.power)
            for param_group in optimizer.param_groups:
                param_group['lr'] = current_lr
        else:
            current_lr = optimizer.param_groups[0]['lr']

        # calculate remain time
        remain_iter = max_iter - current_iter
        remain_time = remain_iter * batch_time.avg
        t_m, t_s = divmod(remain_time, 60)
        t_h, t_m = divmod(t_m, 60)
        remain_time = '{:02d}:{:02d}:{:02d}'.format(int(t_h), int(t_m), int(t_s))

        if (i + 1) % cfg.print_freq == 0 and main_process(cfg):
            logger.info('Epoch: [{}/{}][{}/{}] '
                        'Data: {data_time.val:.3f} ({data_time.avg:.3f}) '
                        'Batch: {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                        'Remain: {remain_time} '
                        'Loss: {loss_meter.val:.4f} '
                        'loss_motion_meter: {loss_motion_meter.val:.4f} '
                        'loss_blendshapes_meter: {loss_blendshapes_meter.val:.4f} '
                        'loss_reg_meter: {loss_reg_meter.val:.4f} '
                        .format(epoch + 1, cfg.epochs, i + 1, len(train_loader),
                                batch_time=batch_time, data_time=data_time,
                                remain_time=remain_time,
                                loss_meter=loss_meter,
                                loss_motion_meter=loss_motion_meter,
                                loss_blendshapes_meter=loss_blendshapes_meter,
                                loss_reg_meter=loss_reg_meter
                                ))

    return loss_meter.avg, loss_motion_meter.avg, loss_blendshapes_meter.avg, loss_reg_meter.avg, loss_feat_meter.avg

def validate(val_loader, model, loss_fn, cfg):
    loss_meter = AverageMeter()
    model.eval()

    with torch.no_grad():
        for i, (audio, audio_features, vertice, blendshapes, template, filename) in enumerate(val_loader):
            audio   = audio.cuda(cfg.gpu, non_blocking=True)
            audio_features = audio_features.cuda(cfg.gpu, non_blocking=True)
            vertice = vertice.cuda(cfg.gpu, non_blocking=True)
            blendshapes     = blendshapes.cuda(cfg.gpu, non_blocking=True)
            template = template.cuda(cfg.gpu, non_blocking=True)

            loss, _ = model(filename, audio, audio_features, vertice, blendshapes, template, criterion=loss_fn)
            loss_meter.update(loss.item(), 1)

    return loss_meter.avg

if __name__ == '__main__':
    main()
