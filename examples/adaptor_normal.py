# Copyright (c) 2021-2022, InterDigital Communications, Inc
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted (subject to the limitations in the disclaimer
# below) provided that the following conditions are met:

# * Redistributions of source code must retain the above copyright notice,
#   this list of conditions and the following disclaimer.
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
# * Neither the name of InterDigital Communications, Inc nor the names of its
#   contributors may be used to endorse or promote products derived from this
#   software without specific prior written permission.

# NO EXPRESS OR IMPLIED LICENSES TO ANY PARTY'S PATENT RIGHTS ARE GRANTED BY
# THIS LICENSE. THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND
# CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT
# NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
# PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
# OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
# OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
# ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

from cmath import exp
import tqdm
import argparse
import math
import random
import shutil
import sys
import os
import time
import logging
from datetime import datetime, timedelta

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from torchvision import transforms

from collections import OrderedDict
import pickle


import yaml

from models.tic_sfma import *
from models.tic import TIC

## General
from detectron2.config import get_cfg
from detectron2.layers import ShapeSpec
from detectron2.modeling.backbone.fpn import build_resnet_fpn_backbone
from utils.dataloader import MSCOCO, Kodak, PASCALContext, get_transformations

## Test
from detectron2.evaluation import COCOEvaluator
from detectron2.data.datasets import register_coco_instances
from detectron2.data import build_detection_test_loader
from detectron2.data.detection_utils import read_image

from contextlib import ExitStack, contextmanager
from utils.predictor import ModPredictor
from utils.alignment import Alignment
from task_loss import *
from utils.logger import create_logger
from utils.padding import *
from utils.lr_scheduler import build_scheduler, configure_optimizers
import cv2
from evaluation.evaluate_utils import PerformanceMeter, get_output

from SwinMT import SwinTransformerMTLoRA
from swin_mtl import MultiTaskSwin
from utils.load_mtl import load_checkpoint
from torch.nn.utils import clip_grad_norm_
from utils.save_tools import save_imgs_mtl

## Function for model to eval
@contextmanager
def inference_context(model):
    training_mode = model.training
    model.eval()
    yield
    model.train(training_mode)

class AverageMeter:
    """Compute running average."""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class CustomDataParallel(nn.DataParallel):
    """Custom DataParallel to access the module methods."""

    def __getattr__(self, key):
        try:
            return super().__getattr__(key)
        except AttributeError:
            return getattr(self.module, key)


def init(args):
    base_dir = f'{args.root}/{args.exp_name}/lmbdabpp{args.lmbda_bpp}'
    os.makedirs(base_dir, exist_ok=True)
    return base_dir


def setup_logger(log_dir):
    log_formatter = logging.Formatter("%(asctime)s [%(levelname)-5.5s]  %(message)s")
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    log_file_handler = logging.FileHandler(log_dir, encoding='utf-8')
    log_file_handler.setFormatter(log_formatter)
    root_logger.addHandler(log_file_handler)

    log_stream_handler = logging.StreamHandler(sys.stdout)
    log_stream_handler.setFormatter(log_formatter)
    root_logger.addHandler(log_stream_handler)

    logging.info('Logging file is %s' % log_dir)

def save_checkpoint(state, base_dir, filename="checkpoint.pth.tar"):
    logging.info(f"Saving checkpoint: {base_dir+filename}")
    torch.save(state, f'{base_dir}/epoch{state['epoch']}_{filename}')

def parse_args(argv):
    from yacs.config import CfgNode as CN
    parser = argparse.ArgumentParser(description="Example training script.")
    parser.add_argument(
        "-c",
        "--config",
        default="config/vpt_default.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        '--name',
        default=datetime.now().strftime('%Y-%m-%d_%H_%M_%S'),
        type=str,
        help='Result dir name',
    )
    given_configs, remaining = parser.parse_known_args(argv)
    with open(given_configs.config) as file:
        yaml_data= yaml.safe_load(file)
        parser.set_defaults(**yaml_data)
        cfg = CN(yaml_data)

    parser.add_argument("-T", "--TEST", action='store_true', help='Testing')

    args = parser.parse_args(remaining)

    return args, cfg


def train_one_epoch(
  train_dataloader, optimizer, model, rdcriterion, epoch,scaler,lr_scheduler,device,
  task_model, seg_loss_ft, human_los_ft, normals_loss_ft, sal_loss_ft, loss_weights,
  config
  ):
    model.train()
    device = next(model.parameters()).device
    num_steps = len(train_dataloader)
    data_times =AverageMeter()
    model_times =AverageMeter()
    bpps =  AverageMeter()
    mse_loss =  AverageMeter()
    rd_loss =  AverageMeter()
    total_losses = AverageMeter()
    task_losses = AverageMeter()
    seg_losses = AverageMeter()
    human_losses = AverageMeter()
    sal_losses = AverageMeter()
    normal_losses = AverageMeter()
    time1 = time.time()
    for i, d in enumerate(train_dataloader):
        data_times.update(time.time()-time1)
        samples = d['image'].cuda(device, non_blocking=True)
        targets = {task: d[task].cuda(device, non_blocking=True) for task in config.TASKS}

        optimizer.zero_grad()
        time2= time.time()

        # pad to 512 & compress
        x_padded, padding = pad(samples, p=512)
        x_compress_padded = model(x_padded)
        x_compress = remove_padding_dict(x_compress_padded, padding)

        out_criterion = rdcriterion(x_compress_padded, x_padded)

        outputs = {}
        # outputs['semseg'] = task_model(x_compress['x_hat'], ['semseg'])['semseg']
        # outputs['human_parts'] = task_model(x_compress['x_hat'], ['human_parts'])['human_parts']
        # outputs['sal'] = task_model(x_compress['sal'], ['sal'])['sal']
        outputs['normals'] = task_model(x_compress['x_hat'], ['normals'])['normals']
        # task loss
        # seg_loss = seg_loss_ft(outputs['semseg'], targets['semseg'])
        # human_loss = human_los_ft(outputs['human_parts'], targets['human_parts']) +1e-7
        # sal_loss = sal_loss_ft(outputs['sal'], targets['sal'])
        normal_loss = normals_loss_ft(outputs['normals'], targets['normals'])
        rdloss = config.lmbda_bpp*out_criterion['bpp_loss'] + out_criterion['mse_loss']
        task_loss = config.task_lmbda*normal_loss

        total_loss = task_loss + rdloss

        total_loss.backward()
        clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        lr_scheduler.step_update(
            (epoch * num_steps + i))

        bpps.update(out_criterion['bpp_loss'].item())
        mse_loss.update(out_criterion['mse_loss'].item())
        rd_loss.update(rdloss.item())
        model_times.update(time.time()-time2)
        # seg_losses.update(seg_loss.item())
        # sal_losses.update(sal_loss.item())
        # normal_losses.update(normal_loss.item())
        # human_losses.update(human_loss.item())
        total_losses.update(total_loss.item())
        task_losses.update(task_loss.item())
        # wandb log
        metrics = {
                "train/epoch_ndx": epoch,
                "train/batch_ndx": i,
                "train/train_loss": total_losses.val,
                "train/train_loss_avg": total_losses.avg,
                "train/train_bpp": bpps.val,
                "train/train_bpp_avg": bpps.avg,
                "train/seg_loss": seg_losses.val,
                "train/seg_loss_avg": seg_losses.avg,
                "train/human_loss": human_losses.val,
                "train/human_loss_avg": human_losses.avg,
                "train/sal_loss": sal_losses.val,
                "train/sal_loss_avg": sal_losses.avg,
                "train/normal_loss": normal_losses.val,
                "train/normal_loss_avg": normal_losses.avg,
                "train/task_loss": task_losses.val,
                "train/ntask_loss_avg": task_losses.avg,
            }
        wandb.log(metrics)
        if i%10==0:
            lr = optimizer.param_groups[0]['lr']
            etas = model_times.avg * (num_steps - i)
            logging.info(
                f'Train: [{epoch}/{config.epochs}][{i}/{num_steps}]\t'
                f'eta {timedelta(seconds=int(etas))} lr {lr:.7f}\t'
                f'time(model) {model_times.val:.4f} ({model_times.avg:.4f})\t'
                f'time(data) {data_times.val:.4f} ({data_times.avg:.4f})\t'
                f'loss {total_losses.val:.4f} ({total_losses.avg:.4f})\t'
                f'bpp {bpps.val:.4f} ({bpps.avg:.4f})\t'
                f'rd_loss {rd_loss.val:.4f} ({rd_loss.avg:.4f})\t'
                f'task_loss {task_losses.val:.4f} ({task_losses.avg:.4f})\t'
                # f'seg_loss {seg_losses.val:.4f} ({seg_losses.avg:.4f})\t'
                # f'human_loss {human_losses.val:.4f} ({human_losses.avg:.4f})\t'
                # f'sal_loss {sal_losses.val:.4f} ({sal_losses.avg:.4f})\t'
                # f'normal_loss {normal_losses.val:.4f} ({normal_losses.avg:.4f})\t'
                )
            print(datetime.now())
        time1 = time.time()

def validation_epoch(epoch, val_dataloader, model, criterion_rd,device, base_dir,
                     task_model, seg_loss_ft, human_los_ft, normals_loss_ft, sal_loss_ft, loss_weights,
                     config):
    model.eval()
    device = next(model.parameters()).device

    performance_meter = PerformanceMeter(config, config.DATA['DBNAME'])
    bpp_loss = AverageMeter()
    mse_loss = AverageMeter()
    rdlosses = AverageMeter()
    psnr = AverageMeter()
    totalloss = AverageMeter()
    task_losses = AverageMeter()
    seg_losses = AverageMeter()
    human_losses = AverageMeter()
    sal_losses = AverageMeter()
    normal_losses = AverageMeter()

    with torch.no_grad():
        tqdm_meter = enumerate(val_dataloader)
        for i, d in tqdm_meter:
            samples = d['image'].cuda(device, non_blocking=True)
            targets = {task: d[task].cuda(device, non_blocking=True) for task in config.TASKS}

            # pad to 512 & compress
            x_padded, padding = pad(samples, p=512)
            x_compress_padded = model(x_padded)
            x_compress = remove_padding_dict(x_compress_padded, padding)

            out_criterion = criterion_rd(x_compress_padded, x_padded)

            outputs = {}
            # outputs['semseg'] = task_model(x_compress['semseg'], ['semseg'])['semseg']
            # outputs['human_parts'] = task_model(x_compress['x_hat'], ['human_parts'])['human_parts']
            # outputs['sal'] = task_model(x_compress['sal'], ['sal'])['sal']
            outputs['normals'] = task_model(x_compress['x_hat'], ['normals'])['normals']

            # seg_loss = seg_loss_ft(outputs['semseg'], targets['semseg'])
            # human_loss = human_los_ft(outputs['human_parts'], targets['human_parts'])
            # sal_loss = sal_loss_ft(outputs['sal'], targets['sal'])
            normal_loss = normals_loss_ft(outputs['normals'], targets['normals'])

            task_loss = normal_loss
            rdloss = config.lmbda_bpp*out_criterion['bpp_loss'] + out_criterion['mse_loss']

            total_loss = config.task_lmbda*task_loss + rdloss

            bpp_loss.update(out_criterion["bpp_loss"])
            mse_loss.update(out_criterion["mse_loss"])
            psnr.update(out_criterion['psnr'])
            rdlosses.update(rdloss)
            # seg_losses.update(seg_loss)
            # human_losses.update(human_loss)
            # sal_losses.update(sal_loss)
            # normal_losses.update(normal_loss)
            task_losses.update(task_loss)
            totalloss.update(total_loss)

            #  Measure performance
            processed_output = {t: get_output(
            outputs[t], t) for t in config.TASKS}
            performance_meter.update(processed_output, targets)
                
            metrics = {
                "val/epoch_ndx": epoch,
                "val/batch_ndx": i,
                "val/bpp_loss": bpp_loss.val,
                "val/bpp_loss_avg": bpp_loss.avg,
                "val/psnr": psnr.val,
                "val/psnr_avg": psnr.avg,
                "val/val_loss": totalloss.val,
                "val/val_loss_avg": totalloss.avg,
                "val/seg_loss": seg_losses.val,
                "val/seg_loss_avg": seg_losses.avg,
                "val/human_loss": human_losses.val,
                "val/human_loss_avg": human_losses.avg,
                "val/sal_loss": sal_losses.val,
                "val/sal_loss_avg": sal_losses.avg,
                "val/normal_loss": normal_losses.val,
                "val/normal_loss_avg": normal_losses.avg,
                "val/task_loss": task_losses.val,
                "val/task_loss_avg": task_losses.avg,
            }
            wandb.log(metrics)

        txt = f"Loss: {totalloss.avg:.3f} | MSE loss: {mse_loss.avg:.5f} | bpp loss: {bpp_loss.avg:.4f} | rdloss: {rdlosses.avg:.4f} | Psnr: {psnr.avg:.4f} | task loss: {task_losses.avg:.4f}"
    print(txt)

    eval_results = performance_meter.get_score(verbose=True)
    scores_logs = {
            "val/epoch": epoch,
        }
    if 'semseg' in eval_results:
        scores_logs["val/tasks/semseg/mIoU"] = eval_results['semseg']['mIoU']
    if 'normals' in eval_results:
        scores_logs["val/tasks/normals/mean"] = eval_results['normals']['mean']
        scores_logs["val/tasks/normals/rmse"] = eval_results['normals']['rmse']
        scores_logs["val/tasks/normals/mean_v2"] = eval_results['normals']['mean_v2']
        scores_logs["val/tasks/normals/rmse_v2"] = eval_results['normals']['rmse_v2']
    if 'human_parts' in eval_results:
        scores_logs["val/tasks/human_parts/mIoU"] = eval_results['human_parts']['mIoU']
    if 'sal' in eval_results:
        scores_logs["val/tasks/sal/maxF"] = eval_results['sal']['maxF']
        scores_logs["val/tasks/sal/Beta maxF"] = eval_results['sal']['Beta maxF']
        scores_logs["val/tasks/sal/mIoU"] = eval_results['sal']['mIoU']
    if 'edge' in eval_results:
        scores_logs["val/tasks/sal/loss"] = eval_results['edge']['loss']
    if 'depth' in eval_results:
        scores_logs["val/tasks/depth/rmse"] = eval_results['depth']['rmse']
        scores_logs["val/tasks/depth/log_rmse"] = eval_results['depth']['log_rmse']

    wandb.log(scores_logs)

    model.train()
    return eval_results, processed_output

def main(argv):
    args, config = parse_args(argv)
    base_dir = init(args)

    if args.seed is not None:
        torch.manual_seed(args.seed)
        random.seed(args.seed)

    setup_logger(base_dir + '/' + time.strftime('%Y%m%d_%H%M%S') + '.log')
    msg = f'======================= {args.name} ======================='
    logging.info(msg)
    for k in args.__dict__:
        logging.info(k + ':' + str(args.__dict__[k]))
    logging.info('=' * len(msg))

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    device = f"cuda:{args.gpu_id}" if args.cuda and torch.cuda.is_available() else "cpu"

    if args.dataset=='PASCALContext':
        # task_train_transform, task_val_transform = get_transformations(args.dataset, args)
        from utils import custom_transforms as tr
        resize_flagvals = {
        # 'image': cv2.INTER_CUBIC,
        'image': cv2.INTER_LINEAR,
        'semseg': cv2.INTER_NEAREST,
        'human_parts': cv2.INTER_NEAREST,
        'sal': cv2.INTER_NEAREST,
        # 'normals': cv2.INTER_CUBIC,
        'normals': cv2.INTER_LINEAR,
        }
        transforms_tr = [tr.RandomHorizontalFlip()]
        transforms_tr.extend([tr.ScaleNRotate(rots=(-20, 20), scales=(.75, 1.25),
                                              flagvals={x: resize_flagvals[x] for x in resize_flagvals})])
        transforms_tr.extend([tr.FixedResize(resolutions={x: tuple(args.DATA['RESIZE']) for x in resize_flagvals},
                                            flagvals={x: resize_flagvals[x] for x in resize_flagvals})])
        transforms_tr.extend([tr.AddIgnoreRegions(), tr.ToTensor()])
        transforms_tr = transforms.Compose(transforms_tr)

        transforms_ts = []
        transforms_ts.extend([tr.FixedResize(resolutions={x: tuple(args.DATA['RESIZE']) for x in resize_flagvals},
                                            flagvals={x: resize_flagvals[x] for x in resize_flagvals})])
        transforms_ts.extend([tr.AddIgnoreRegions(), tr.ToTensor()])
        transforms_ts = transforms.Compose(transforms_ts)

        dataset_train = PASCALContext(root=args.dataset_path, split=['train'],
                                      transform=transforms_tr,
                                      retname=True,
                                      do_semseg='semseg' in args.TASKS,
                                      do_edge='edge' in args.TASKS,
                                      do_normals='normals' in args.TASKS,
                                      do_sal='sal' in args.TASKS,
                                      do_human_parts='human_parts' in args.TASKS,
                                      overfit=False)

        dataset_val = PASCALContext(root=args.dataset_path, split=['val'],
                                      transform=transforms_ts,
                                      retname=True,
                                      do_semseg='semseg' in args.TASKS,
                                      do_edge='edge' in args.TASKS,
                                      do_normals='normals' in args.TASKS,
                                      do_sal='sal' in args.TASKS,
                                      do_human_parts='human_parts' in args.TASKS,
                                      overfit=False)


        train_dataloader = DataLoader(dataset_train,
                                      batch_size=args.batch_size,
                                      num_workers=args.num_workers,
                                      shuffle=True,
                                      pin_memory=True)
        val_dataloader = DataLoader(dataset_val,
                                    batch_size=args.test_batch_size,
                                    num_workers=args.num_workers,
                                    shuffle=False,
                                    pin_memory=True)


    net = TIC_SFMA(N=128,M=192)
    net = net.to(device)
    print('total paramaters:',sum(p.numel() for p in net.parameters() )/1e6)

    for k, p in net.named_parameters():
        if "sfma" not in k:
            p.requires_grad = False
    print('tuning paramaters:',sum(p.numel() for p in net.parameters() if p.requires_grad)/1e6)

    optimizer = configure_optimizers(net,args)
    lr_scheduler = build_scheduler(config, optimizer, len(train_dataloader))

    task_backbone = SwinTransformerMTLoRA(img_size=448,
                                          patch_size=4,
                                          in_chans=3,
                                          num_classes=0,
                                          embed_dim=96,
                                          depths=[2,2,6,2],
                                          num_heads=[3,6,12,24],
                                          window_size=7,
                                          mlp_ratio=4.0,
                                          qkv_bias=True,
                                          qk_scale=None,
                                          drop_rate=0.0,
                                          drop_path_rate=0.2,
                                          ape=False,
                                          norm_layer=nn.LayerNorm,
                                          patch_norm=True,
                                          use_checkpoint=False,
                                          fused_window_process=False,
                                          tasks=config.TASKS,
                                          mtlora=config.MODEL.MTLORA)
    task_model = MultiTaskSwin(task_backbone, config)
    task_model = task_model.to(device)
    task_model.freeze_all()
    task_model.eval()
    max_accuracy = 0.0
    max_accuracy = load_checkpoint(config, task_model)

    loss_weights = {
            'semseg': 1.0,
            'human_parts': 2.0,
            'sal': 5.0,
            'normals': 10.0,
        }
    seg_loss_ft = SoftMaxwithLoss(ignore_index=255)
    human_los_ft = SoftMaxwithLoss(ignore_index=255)
    normals_loss_ft = NormalsLoss(normalize=True, size_average=True, norm=1)
    sal_loss_ft = BalancedCrossEntropyLoss(size_average=True)

    rdcriterion = RateDistortionLoss(config.lmbda_rd)

    last_epoch = 0
    if args.checkpoint:
        logging.info("Loading "+str(args.checkpoint))
        checkpoint = torch.load(args.checkpoint, map_location=device)

        if list(checkpoint["state_dict"].keys())[0][:7]=='module.':
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in checkpoint["state_dict"].items():
                name = k[7:]
                new_state_dict[name] = v
        else:
            new_state_dict = checkpoint['state_dict']
        net.load_state_dict(new_state_dict, strict=True if args.TEST else False)

    scaler = torch.cuda.amp.GradScaler()
    if args.TEST:
        config.save = True
        validation_epoch(last_epoch, val_dataloader, net, rdcriterion,device,base_dir,
                        task_model, seg_loss_ft, human_los_ft, normals_loss_ft, sal_loss_ft, loss_weights,
                        config)
        return
    tqrange = tqdm.trange(last_epoch, args.epochs)
    for epoch in tqrange:
        print('/')
        train_one_epoch(train_dataloader, optimizer, net, rdcriterion, epoch,scaler,lr_scheduler,device,
                        task_model, seg_loss_ft, human_los_ft, normals_loss_ft, sal_loss_ft, loss_weights,
                        config)
        # lr_scheduler.step()

        if epoch==0 or (epoch+1) % config.save_freq == 0:
            validation_epoch(epoch, val_dataloader, net, rdcriterion,device,base_dir,
                        task_model, seg_loss_ft, human_los_ft, normals_loss_ft, sal_loss_ft, loss_weights,
                        args)
            save_checkpoint(
                {
                    "epoch": epoch,
                    "state_dict": net.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "lr_scheduler": lr_scheduler.state_dict(),
                },
                base_dir,
                filename='checkpoint.pth.tar'
            )
            config_path = os.path.join(base_dir, "config.yaml")
            with open(config_path, 'w') as f:
                yaml.dump(vars(args), f)
        torch.cuda.empty_cache()

if __name__ == "__main__":
    try:
        import wandb
        if not os.getenv("WANDB_API_KEY"):
            wandb.login()
        else:
            wandb.login(key=os.getenv("WANDB_API_KEY"))
        config_name = f"LOG_PATH/multi_task"
        wandb.init(project='MTLoRA',name=config_name)
    except wandb.exc.LaunchError:
            logging.info("Could not initialize wandb. Logging is disabled.")
    main(sys.argv[1:])
