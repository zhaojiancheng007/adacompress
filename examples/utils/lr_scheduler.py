# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------

import bisect

import torch
from timm.scheduler.cosine_lr import CosineLRScheduler
from timm.scheduler.step_lr import StepLRScheduler
from timm.scheduler.scheduler import Scheduler
import torch.optim as optim

def configure_optimizers_prompt(net, args):
    """Set optimizer for only the parameters for propmts"""

    if args.TRANSFER_TYPE == "prompt":
        parameters = {
        k
        for k, p in net.named_parameters()
        if "prompt" in k
    }

    params_dict = dict(net.named_parameters())

    optimizer = optim.Adam(
        (params_dict[n] for n in sorted(parameters)),
        lr=args.learning_rate,
    )

    return optimizer

def configure_optimizers(net, args):
    """Set optimizer for only the parameters for propmts"""

    parameters = {
        k
        for k, p in net.named_parameters()
        if "sfma" in k
    }

    params_dict = dict(net.named_parameters())

    optimizer = optim.Adam(
        (params_dict[n] for n in sorted(parameters)),
        lr=args.learning_rate
    )

    return optimizer

def build_scheduler(config, optimizer, n_iter_per_epoch):
    num_steps = int(config.epochs * n_iter_per_epoch)
    warmup_steps = int(config.LR_SCHEDULER['warm_up_epoch'] * n_iter_per_epoch)
    decay_steps = int(
        config.LR_SCHEDULER['decay_steps'] * n_iter_per_epoch)
    multi_steps = [
        i * n_iter_per_epoch for i in config.LR_SCHEDULER['multisteps']]
    lr_scheduler = None
    if config.LR_SCHEDULER['name'] == 'cosine':
        lr_scheduler = CosineLRScheduler(
            optimizer,
            t_initial=(
                num_steps - warmup_steps),
            # t_mul=1.,
            lr_min=config.LR_SCHEDULER['min_lr'],
            warmup_lr_init=config.LR_SCHEDULER['warmup_lr'],
            warmup_t=warmup_steps,
            cycle_limit=1,
            t_in_epochs=False,
            warmup_prefix=True,
        )
    elif config.LR_SCHEDULER['name'] == 'linear':
        lr_scheduler = LinearLRScheduler(
            optimizer,
            t_initial=num_steps,
            lr_min_rate=0.01,
            warmup_lr_init=config.LR_SCHEDULER['warmup_lr'],
            warmup_t=warmup_steps,
            t_in_epochs=False,
        )
    elif config.LR_SCHEDULER['name'] == 'step':
        lr_scheduler = StepLRScheduler(
            optimizer,
            decay_t=decay_steps,
            decay_rate=0.1,
            warmup_lr_init=config.LR_SCHEDULER['warmup_lr'],
            warmup_t=warmup_steps,
            t_in_epochs=False,
        )
    elif config.LR_SCHEDULER['name'] == 'multistep':
        lr_scheduler = MultiStepLRScheduler(
            optimizer,
            milestones=multi_steps,
            gamma=0.1,
            # warmup_lr_init=config.LR_SCHEDULER['warmup_lr'],
            # warmup_t=warmup_steps,
            # t_in_epochs=False,
        )

    return lr_scheduler


class LinearLRScheduler(Scheduler):
    def __init__(self,
                 optimizer: torch.optim.Optimizer,
                 t_initial: int,
                 lr_min_rate: float,
                 warmup_t=0,
                 warmup_lr_init=0.,
                 t_in_epochs=True,
                 noise_range_t=None,
                 noise_pct=0.67,
                 noise_std=1.0,
                 noise_seed=42,
                 initialize=True,
                 ) -> None:
        super().__init__(
            optimizer, param_group_field="lr",
            noise_range_t=noise_range_t, noise_pct=noise_pct, noise_std=noise_std, noise_seed=noise_seed,
            initialize=initialize)

        self.t_initial = t_initial
        self.lr_min_rate = lr_min_rate
        self.warmup_t = warmup_t
        self.warmup_lr_init = warmup_lr_init
        self.t_in_epochs = t_in_epochs
        if self.warmup_t:
            self.warmup_steps = [(v - warmup_lr_init) /
                                 self.warmup_t for v in self.base_values]
            super().update_groups(self.warmup_lr_init)
        else:
            self.warmup_steps = [1 for _ in self.base_values]

    def _get_lr(self, t):
        if t < self.warmup_t:
            lrs = [self.warmup_lr_init + t * s for s in self.warmup_steps]
        else:
            t = t - self.warmup_t
            total_t = self.t_initial - self.warmup_t
            lrs = [v - ((v - v * self.lr_min_rate) * (t / total_t))
                   for v in self.base_values]
        return lrs

    def get_epoch_values(self, epoch: int):
        if self.t_in_epochs:
            return self._get_lr(epoch)
        else:
            return None

    def get_update_values(self, num_updates: int):
        if not self.t_in_epochs:
            return self._get_lr(num_updates)
        else:
            return None


class MultiStepLRScheduler(Scheduler):
    def __init__(self, optimizer: torch.optim.Optimizer, milestones, gamma=0.1, warmup_t=0, warmup_lr_init=0, t_in_epochs=True) -> None:
        super().__init__(optimizer, param_group_field="lr")

        self.milestones = milestones
        self.gamma = gamma
        self.warmup_t = warmup_t
        self.warmup_lr_init = warmup_lr_init
        self.t_in_epochs = t_in_epochs
        if self.warmup_t:
            self.warmup_steps = [(v - warmup_lr_init) /
                                 self.warmup_t for v in self.base_values]
            super().update_groups(self.warmup_lr_init)
        else:
            self.warmup_steps = [1 for _ in self.base_values]

        assert self.warmup_t <= min(self.milestones)

    def _get_lr(self, t):
        if t < self.warmup_t:
            lrs = [self.warmup_lr_init + t * s for s in self.warmup_steps]
        else:
            lrs = [v * (self.gamma ** bisect.bisect_right(self.milestones, t))
                   for v in self.base_values]
        return lrs

    def get_epoch_values(self, epoch: int):
        if self.t_in_epochs:
            return self._get_lr(epoch)
        else:
            return None

    def get_update_values(self, num_updates: int):
        if not self.t_in_epochs:
            return self._get_lr(num_updates)
        else:
            return None
