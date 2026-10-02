import os
from datetime import timedelta
import random
import signal
import numpy as np
import torch
import torch.distributed as dist


class Runtime:
    def __init__(self, config):
        self.rank = int(os.environ.get('RANK', '0'))
        self.local_rank = int(os.environ.get('LOCAL_RANK', '0'))
        self.world = int(os.environ.get('WORLD_SIZE', '1'))
        expected = config.distributed.expected_world_size
        if expected and self.world != expected:
            raise RuntimeError(f'Expected world_size={expected}, got {self.world}; use the matching config/launcher')
        if self.world > 1 and 'LOCAL_RANK' not in os.environ:
            raise RuntimeError('Use torchrun: LOCAL_RANK is missing')
        if config.training.device == 'cuda':
            if not torch.cuda.is_available():
                raise RuntimeError('CUDA required. CPU smoke: training.device=cpu, precision.dtype=fp32, distributed.backend=gloo')
            if self.local_rank >= torch.cuda.device_count():
                raise RuntimeError('LOCAL_RANK exceeds visible CUDA devices')
            torch.cuda.set_device(self.local_rank)
            self.device = torch.device('cuda', self.local_rank)
            if config.precision.dtype == 'bf16' and not torch.cuda.is_bf16_supported():
                raise RuntimeError('This GPU/PyTorch does not support BF16. Check driver/build; explicitly select fp32 only for diagnosis.')
        else:
            self.device = torch.device('cpu')
        if self.world > 1:
            dist.init_process_group(config.distributed.backend,
                                    timeout=timedelta(seconds=config.distributed.timeout_seconds))

    @property
    def primary(self):
        return self.rank == 0

    def reduce(self, tensor, op=dist.ReduceOp.SUM):
        if self.world > 1:
            dist.all_reduce(tensor, op=op)
        return tensor

    def gather(self, obj):
        if self.world == 1:
            return [obj]
        result = [None] * self.world if self.primary else None
        dist.gather_object(obj, result, dst=0)
        return result

    def broadcast(self, obj):
        values = [obj]
        if self.world > 1:
            dist.broadcast_object_list(values, src=0)
        return values[0]

    def close(self):
        # Never enter a barrier in finally: another rank may already have failed.
        if dist.is_initialized():
            dist.destroy_process_group()


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed % 2**32)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


class StopSignal:
    def __init__(self):
        self.requested = False
        self.old = {}

    def __enter__(self):
        def handler(signum, frame):
            self.requested = True  # no CUDA, filesystem or collectives inside signal handler
        for sig in (signal.SIGINT, signal.SIGTERM):
            self.old[sig] = signal.signal(sig, handler)
        return self

    def __exit__(self, *args):
        for sig, handler in self.old.items():
            signal.signal(sig, handler)
