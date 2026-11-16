#!/usr/bin/env python3
"""Train PlanktonNet on known classes with the paper's CAC objective."""

import argparse
import time
import logging
import os
import torch
from contextlib import suppress
from datetime import datetime, timedelta

from timm import utils
from timm.data import create_dataset, create_loader, resolve_data_config
from timm.optim import create_optimizer_v2, optimizer_kwargs
from timm.scheduler import create_scheduler_v2, scheduler_kwargs

from planktonnet_features import PlanktonNet
from cacUtils import CACModel, PaperCACLoss

_logger = logging.getLogger('train_cac')


def add_file_handler(logger, log_file):
    file_handler = logging.FileHandler(log_file)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)


def parse_args():
    parser = argparse.ArgumentParser(description='Train PlanktonNet with CAC')

    # Dataset paths, output, and runtime settings.
    parser.add_argument('--data-dir', metavar='DIR', required=True)
    parser.add_argument('--dataset', default='ZooplanktonBench50_OpenSet')
    parser.add_argument('--output', default='./output/train_cac', type=str)
    parser.add_argument('--experiment', default='', type=str)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', default='cuda', type=str)

    # Backbone projection and CAC loss settings.
    parser.add_argument('--model', default='plankton_net', choices=('plankton_net',))
    parser.add_argument('--num-classes', type=int, default=50)
    parser.add_argument('--feature-dim', default=448, type=int)
    parser.add_argument('--cac-magnitude', default=10.0, type=float, help='Anchor magnitude')
    parser.add_argument('--cac-scale', default=0.1, type=float, help='Anchor loss weight')

    # Training schedule.
    parser.add_argument('--batch-size', type=int, default=26)
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--start-epoch', type=int, default=None)
    parser.add_argument('--log-interval', type=int, default=50)

    # Optimizer and gradient clipping.
    parser.add_argument('--opt', type=str, default='adamw')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--opt-eps', default=None, type=float)
    parser.add_argument('--opt-betas', default=None, type=float, nargs='+')
    parser.add_argument('--clip-grad', type=float, default=None)
    parser.add_argument('--clip-mode', type=str, default='norm')

    # Learning-rate scheduler.
    parser.add_argument('--sched', default='cosine', type=str)
    parser.add_argument('--min-lr', type=float, default=1e-4)
    parser.add_argument('--warmup-lr', type=float, default=1e-4)
    parser.add_argument('--warmup-epochs', type=int, default=5)
    parser.add_argument('--cooldown-epochs', type=int, default=0)
    parser.add_argument('--decay-epochs', type=float, default=30)
    parser.add_argument('--decay-rate', type=float, default=0.1)
    parser.add_argument('--patience-epochs', type=int, default=10)

    # Dataset splits and input loading.
    parser.add_argument('--train-split', default='train')
    parser.add_argument('--val-split', default='validation')
    parser.add_argument('--no-prefetcher', action='store_true')
    parser.add_argument('--pin-mem', action='store_true')
    parser.add_argument('--input-img-mode', default='RGB')
    parser.add_argument('--train-interpolation', default='bicubic')

    # Input sizing, regularization, and image augmentation.
    parser.add_argument('--img-size', type=int, default=224)
    parser.add_argument('--drop', type=float, default=0.0)
    parser.add_argument('--drop-path', type=float, default=0.0)
    parser.add_argument('--aa', type=str, default=None)
    parser.add_argument('--reprob', type=float, default=0.0)
    parser.add_argument('--remode', type=str, default='pixel')
    parser.add_argument('--recount', type=int, default=1)
    parser.add_argument('--resplit', action='store_true')

    # Keep these CLI options for compatibility. CAC requires hard labels.
    parser.add_argument('--mixup', type=float, default=0.0)
    parser.add_argument('--cutmix', type=float, default=0.0)
    parser.add_argument('--mixup-prob', type=float, default=1.0)
    parser.add_argument('--mixup-switch-prob', type=float, default=0.5)
    parser.add_argument('--mixup-mode', type=str, default='batch')
    parser.add_argument('--smoothing', type=float, default=0.0)

    # Random crop and color augmentation.
    parser.add_argument('--scale', type=float, nargs='+', default=[0.08, 1.0])
    parser.add_argument('--ratio', type=float, nargs='+', default=[3. / 4., 4. / 3.])
    parser.add_argument('--hflip', type=float, default=0.5)
    parser.add_argument('--vflip', type=float, default=0.)
    parser.add_argument('--color-jitter', type=float, default=0.4)
    parser.add_argument('--aug-repeats', type=float, default=0)
    parser.add_argument('--aug-splits', type=int, default=0)

    # Checkpointing, mixed precision, and distributed options.
    parser.add_argument('--resume', default='')
    parser.add_argument('--no-resume-opt', action='store_true')
    parser.add_argument('--checkpoint-hist', type=int, default=5)
    parser.add_argument('--amp', action='store_true', default=False)
    parser.add_argument('--distributed', action='store_true')
    parser.add_argument('--world-size', default=1)
    parser.add_argument('--rank', default=0)
    parser.add_argument('--local_rank', default=0)

    args = parser.parse_args()
    if any((args.mixup, args.cutmix, args.smoothing)):
        parser.error('Paper CAC training requires hard labels. Use zero Mixup, CutMix, and smoothing.')
    return args


def main():
    args = parse_args()
    utils.setup_default_logging()
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    utils.random_seed(args.seed)

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    exp_name = args.experiment or 'cac_paper'
    final_output_dir = os.path.join(args.output, f"{timestamp}_{exp_name}")
    os.makedirs(final_output_dir, exist_ok=True)
    add_file_handler(_logger, os.path.join(final_output_dir, 'train.log'))

    _logger.info(f"Dataset Root: {args.data_dir}")
    _logger.info(f"Output Directory: {final_output_dir}")

    _logger.info(f"Creating PlanktonNet (Backbone)...")
    backbone = PlanktonNet(pretrained=False, drop_rate=args.drop, drop_path_rate=args.drop_path)

    _logger.info(f"Wrapping with CACModel (Map {args.feature_dim} -> {args.num_classes})...")
    model = CACModel(backbone, feature_dim=args.feature_dim, num_classes=args.num_classes)
    model.to(device)

    loss_fn = PaperCACLoss(args.num_classes, args.cac_magnitude, args.cac_scale).to(device)
    optimizer = create_optimizer_v2(model, **optimizer_kwargs(cfg=args))

    amp_autocast = suppress
    loss_scaler = None
    if args.amp:
        from torch.cuda.amp import autocast
        from timm.utils import NativeScaler
        amp_autocast = autocast
        loss_scaler = NativeScaler()

    data_config = resolve_data_config(vars(args), model=model, verbose=True)
    # Set the RGB input size explicitly from the command line.
    data_config['input_size'] = (3, args.img_size, args.img_size)

    train_path = os.path.join(args.data_dir, args.dataset, args.train_split)
    val_path = os.path.join(args.data_dir, args.dataset, args.val_split)

    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Train path not found: {train_path}")

    dataset_train = create_dataset(name='', root=train_path, split='', is_training=True, batch_size=args.batch_size)
    _logger.info(f"Train samples: {len(dataset_train)}")

    loader_train = create_loader(
        dataset_train,
        input_size=data_config['input_size'],
        batch_size=args.batch_size,
        is_training=True,
        use_prefetcher=not args.no_prefetcher,
        no_aug=False,
        re_prob=args.reprob,
        re_mode=args.remode,
        scale=args.scale,
        ratio=args.ratio,
        hflip=args.hflip,
        vflip=args.vflip,
        color_jitter=args.color_jitter,
        auto_augment=args.aa,
        mean=data_config['mean'],
        std=data_config['std'],
        num_workers=args.workers,
        persistent_workers=args.workers > 0,
        device=device
    )

    # Use the training class indices for known validation images.
    class_map = dataset_train.reader.class_to_idx if hasattr(dataset_train, 'reader') else dataset_train.class_to_idx

    dataset_eval = create_dataset(name='', root=val_path, split='', is_training=False, batch_size=args.batch_size,
                                  class_map=class_map)
    loader_eval = create_loader(
        dataset_eval,
        input_size=data_config['input_size'],
        batch_size=args.batch_size,
        is_training=False,
        mean=data_config['mean'],
        std=data_config['std'],
        num_workers=args.workers,
        persistent_workers=args.workers > 0,
        device=device
    )

    lr_scheduler, _ = create_scheduler_v2(optimizer, **scheduler_kwargs(args), updates_per_epoch=len(loader_train))

    saver = utils.CheckpointSaver(
        model=model, optimizer=optimizer, args=args,
        checkpoint_dir=final_output_dir, recovery_dir=final_output_dir,
        max_history=args.checkpoint_hist, decreasing=False
    )

    _logger.info(f"Start CAC training ({args.epochs} epochs).")
    total_start_time = time.time()

    for epoch in range(args.start_epoch or 0, args.epochs):
        ep_start = time.time()

        model.train()
        avg_loss = utils.AverageMeter()
        avg_anchor = utils.AverageMeter()

        for batch_idx, (input, target) in enumerate(loader_train):
            input, target = input.to(device), target.to(device)

            with amp_autocast():
                output = model(input)
                loss, anchor_loss = loss_fn(output, target)

            optimizer.zero_grad()
            if loss_scaler:
                loss_scaler(loss, optimizer, clip_grad=args.clip_grad, clip_mode=args.clip_mode,
                            parameters=model.parameters())
            else:
                loss.backward()
                if args.clip_grad:
                    utils.dispatch_clip_grad(model.parameters(), args.clip_grad, mode=args.clip_mode)
                optimizer.step()

            avg_loss.update(loss.item(), input.size(0))
            avg_anchor.update(anchor_loss.item(), input.size(0))

            if lr_scheduler: lr_scheduler.step_update(epoch * len(loader_train) + batch_idx)

        # Measure top-1 accuracy on known validation classes for checkpoint selection.
        model.eval()
        top1 = utils.AverageMeter()
        val_loss_m = utils.AverageMeter()

        with torch.no_grad():
            for input, target in loader_eval:
                input, target = input.to(device), target.to(device)
                output = model(input)

                prediction = output.argmax(dim=1)
                acc = (prediction == target).sum() / input.size(0)

                v_loss, _ = loss_fn(output, target)

                top1.update(acc.item() * 100, input.size(0))
                val_loss_m.update(v_loss.item(), input.size(0))

        saver.save_checkpoint(epoch, metric=top1.avg)

        lrl = [param_group['lr'] for param_group in optimizer.param_groups]
        lr = sum(lrl) / len(lrl)
        duration = time.time() - ep_start

        _logger.info(
            f"Epoch {epoch}: "
            f"Loss {avg_loss.avg:.4f} (Anchor {avg_anchor.avg:.4f}) | "
            f"Val Loss {val_loss_m.avg:.4f} | "
            f"Val Acc {top1.avg:.2f}% | "
            f"LR {lr:.2e} | Time {duration:.1f}s"
        )

    total_time = time.time() - total_start_time
    total_time_str = str(timedelta(seconds=int(total_time)))
    _logger.info(f"Training completed in {total_time_str}")
    _logger.info(f"Models saved to: {final_output_dir}")


if __name__ == '__main__':
    main()
