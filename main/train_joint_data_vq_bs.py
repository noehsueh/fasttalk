#!/usr/bin/env python
import os
import time
import torch
import torch.backends.cudnn as cudnn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.multiprocessing as mp
import torch.distributed as dist
import cv2

from base.baseTrainer import poly_learning_rate, reduce_tensor, save_checkpoint
from base.utilities import get_parser, get_logger, main_process, AverageMeter
from models import get_model
from metrics.loss import calc_vq_loss
from torch.optim.lr_scheduler import StepLR

cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(0)

import wandb

def main():
    args = get_parser()
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(x) for x in args.train_gpu)
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

    # initialize wandb
    wandb.init(project="multitalk_custom_vq", name=args.save_path.split("/")[-1],dir="logs")
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

    # ####################### Load Pretrained ####################### #
    #pretrained_checkpoint_path = "/root/Projects/fasttalk/logs/joint_data/joint_data_custom_s1/model_170_50k/model.pth.tar"
    #pretrained_vq_checkpoint = torch.load(pretrained_checkpoint_path)

    #if "state_dict" in pretrained_vq_checkpoint:
    #    model.load_state_dict(pretrained_vq_checkpoint["state_dict"])
    #else:
    #    model.load_state_dict(pretrained_vq_checkpoint,map_location=lambda storage, loc: storage.cpu())

    #print("Loaded pretrained model from: ", pretrained_checkpoint_path)
    #print("Starting fine-tuning...")

    # ####################### Optimizer ####################### #
    if cfg.use_sgd:
        optimizer = torch.optim.SGD(model.parameters(), lr=cfg.base_lr, momentum=cfg.momentum,
                                    weight_decay=cfg.weight_decay)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.base_lr)

    if cfg.StepLR:
        scheduler = StepLR(optimizer, step_size=cfg.step_size, gamma=cfg.gamma)
    else:
        scheduler = None

    # ####################### Data Loader ####################### #
    from dataset.data_loader_joint_data_batched import get_dataloaders
    dataset = get_dataloaders(cfg)
    train_loader = dataset['train']
    if cfg.evaluate:
        val_loader = dataset['valid']

    val_blendshapes_loss_log = 1000
    val_quan_loss_log = 1000

    # ####################### Train ############################# #
    for epoch in range(cfg.start_epoch, cfg.epochs):
        blendshapes_loss_train, quant_loss_train, pp_train = train(train_loader, model, calc_vq_loss, optimizer, epoch, cfg)
        epoch_log = epoch + 1
        if cfg.StepLR:
            scheduler.step()
        if main_process(cfg):
            logger.info('TRAIN Epoch: {} '
                        'blendshapes_loss_train: {} '
                        'pp_train: {} '
                        .format(epoch_log, blendshapes_loss_train, pp_train)
                        )

        wandb.log({"blendshapes_loss_train": blendshapes_loss_train, "quan_loss_train": quant_loss_train}, epoch_log)

        if cfg.evaluate and (epoch_log % cfg.eval_freq == 0):
            blendshapes_loss_val, quant_loss_val, pp_val = validate(val_loader, model, calc_vq_loss, epoch, cfg)
            if main_process(cfg):
                logger.info('VAL Epoch: {} '
                            'blendshapes_loss_val: {} '
                            'pp_val: {} '
                            .format(epoch_log, blendshapes_loss_val, pp_val)
                            )

            wandb.log({ "blendshapes_loss_val": blendshapes_loss_val, "quan_loss_val": quant_loss_val}, epoch_log)

            #save_checkpoint
            #if blendshapes_loss_val<val_blendshapes_loss_log:
            #    val_blendshapes_loss_log, val_quan_loss_log = blendshapes_loss_val, quant_loss_val
            save_checkpoint(model,sav_path=os.path.join(cfg.save_path, 'model_'+str(epoch_log)))


def train(train_loader, model, loss_fn, optimizer, epoch, cfg):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    blendshapes_loss_meter = AverageMeter()
    quant_loss_meter = AverageMeter()
    pp_meter = AverageMeter()

    model.train()
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"----> Total trainable parameters: {trainable_params}")

    end = time.time()
    max_iter = cfg.epochs * len(train_loader)
    for i, (blendshapes, mask) in enumerate(train_loader):
        current_iter = epoch * len(train_loader) + i + 1
        data_time.update(time.time() - end)

        blendshapes  = blendshapes.cuda(cfg.gpu, non_blocking=True)
        mask         = mask.cuda(cfg.gpu, non_blocking=True)
        blendshapes_out, quant_loss = model(blendshapes,mask)

        # LOSS
        loss, loss_details = loss_fn(blendshapes_out, blendshapes, quant_loss, quant_loss_weight=cfg.quant_loss_weight)

        optimizer.zero_grad()
        loss.backward()
        grad_norm_first_layer = model.encoder_proj.weight.grad.norm()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()
        for m, x in zip([blendshapes_loss_meter, quant_loss_meter],
                        [loss_details[0], loss_details[1]]):  # info[0] is perplexity
            m.update(x.item(), 1)

        # Adjust lr
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
                        'Grad norm: {grad_norm_first_layer:.4f} '
                        'Loss blendshapes: {loss_meter_blendshapes.val:.4f} '
                        .format(epoch + 1, cfg.epochs, i + 1, len(train_loader),
                                batch_time=batch_time, data_time=data_time,
                                remain_time=remain_time,
                                grad_norm_first_layer=grad_norm_first_layer,
                                loss_meter_blendshapes=blendshapes_loss_meter
                                ))

    return blendshapes_loss_meter.avg, quant_loss_meter.avg, pp_meter.avg


def validate(val_loader, model, loss_fn, epoch, cfg):
    blendshapes_loss_meter = AverageMeter()
    quant_loss_meter = AverageMeter()
    pp_meter = AverageMeter()
    model.eval()

    with torch.no_grad():
        for i, (blendshapes, mask) in enumerate(val_loader):
            blendshapes  = blendshapes.cuda(cfg.gpu, non_blocking=True)
            mask         = mask.cuda(cfg.gpu, non_blocking=True)
            blendshapes_out, quant_loss =  model(blendshapes, mask)

            # LOSS
            loss, loss_details = loss_fn(blendshapes_out, blendshapes, quant_loss, quant_loss_weight=cfg.quant_loss_weight)

            if cfg.distributed:
                loss = reduce_tensor(loss, cfg)

            for m, x in zip([blendshapes_loss_meter, quant_loss_meter],
                            [loss_details[0], loss_details[1]]):
                m.update(x.item(), 1)  # batch_size = 1 for validation

    return blendshapes_loss_meter.avg, quant_loss_meter.avg, pp_meter.avg


if __name__ == '__main__':
    main()
