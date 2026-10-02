import math
import torch
from torch import nn
from .model import RMSNorm


def make_optimizer(model, config, device):
    decay, no_decay, seen = [], [], set()
    for module in model.modules():
        for name, p in module.named_parameters(recurse=False):
            if not p.requires_grad or id(p) in seen:
                continue
            seen.add(id(p))
            if isinstance(module, (RMSNorm, nn.LayerNorm)) or name == 'bias':
                no_decay.append(p)
            elif isinstance(module, (nn.Linear, nn.Embedding)) and name == 'weight':
                decay.append(p)
            else:
                raise TypeError(f'Unclassified optimizer parameter {type(module).__name__}.{name}')
    if seen != {id(p) for p in model.parameters() if p.requires_grad}:
        raise RuntimeError('Optimizer parameter partition is incomplete')
    if config.fused and device.type != 'cuda':
        raise ValueError('Fused AdamW must be benchmarked on CUDA; disable it for CPU tests')
    return torch.optim.AdamW([
        {'params': decay, 'weight_decay': config.weight_decay},
        {'params': no_decay, 'weight_decay': 0.0}],
        lr=0.0, betas=(config.beta1, config.beta2), eps=config.eps,
        fused=config.fused, foreach=False if config.fused else None)


class WarmupCosine:
    """Explicit 1-based optimizer-update schedule; completed_steps persists on resume."""
    def __init__(self, optimizer, config):
        self.optimizer, self.config, self.completed_steps = optimizer, config, 0

    def lr_for(self, step):
        c = self.config
        if c.warmup_steps and step <= c.warmup_steps:
            return c.max_lr * step / c.warmup_steps
        progress = min(1.0, max(0.0, (step - c.warmup_steps) / (c.total_steps - c.warmup_steps)))
        return c.min_lr + 0.5 * (c.max_lr - c.min_lr) * (1 + math.cos(math.pi * progress))

    def prepare(self, step):
        if step != self.completed_steps + 1:
            raise ValueError('Non-contiguous scheduler update')
        lr = self.lr_for(step)
        for group in self.optimizer.param_groups:
            group['lr'] = lr
        return lr

    def commit(self, step):
        self.completed_steps = step

    def state_dict(self):
        return {'completed_steps': self.completed_steps, 'config': vars(self.config).copy()}

    def load_state_dict(self, state):
        if state['config'] != vars(self.config):
            raise ValueError('LR schedule changed on exact resume')
        self.completed_steps = state['completed_steps']
