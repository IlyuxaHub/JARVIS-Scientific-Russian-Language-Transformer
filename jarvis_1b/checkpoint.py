"""Atomic rank-0 checkpoints at committed optimizer boundaries, with per-rank RNG."""
import errno
import json
import os
from pathlib import Path
import random
import shutil
import uuid
import numpy as np
import torch


FORMAT = 'jarvis-scientific-checkpoint-v1'


def rng_state(device):
    return {'python': random.getstate(), 'numpy': np.random.get_state(),
            'torch_cpu': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state(device) if device.type == 'cuda' else None}


def restore_rng(state, device):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch_cpu'].cpu())
    if device.type == 'cuda':
        torch.cuda.set_rng_state(state['cuda'].cpu(), device)


def resume_contract(config, world, data_identity):
    c = config.to_dict()
    return {'world_size': world, 'data_identity': data_identity,
            'model': c['model'], 'optimizer': c['optimizer'], 'scheduler': c['scheduler'],
            'precision': c['precision'],
            'training': {k: c['training'][k] for k in ('seed', 'micro_batch_size', 'gradient_accumulation_steps', 'compile', 'compile_mode', 'device')},
            'sampling': {k: c['data'][k] for k in ('sampling', 'shuffle_block')},
            'backend': c['distributed']['backend']}


def _fsync_directory(path):
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def tensor_bytes(obj):
    seen = set()
    def visit(x):
        if isinstance(x, torch.Tensor):
            storage = x.untyped_storage()
            key = (str(x.device), storage.data_ptr(), storage.nbytes())
            if key in seen:
                return 0
            seen.add(key)
            return storage.nbytes()
        if isinstance(x, dict):
            return sum(visit(v) for v in x.values())
        if isinstance(x, (list, tuple)):
            return sum(visit(v) for v in x)
        return 0
    return visit(obj)


def _atomic_link(source, dest):
    temp = dest.with_name('.' + dest.name + '.' + uuid.uuid4().hex + '.link.tmp')
    try:
        os.link(source, temp)  # same filesystem; never copy a 12+ GiB checkpoint for aliases
        os.replace(temp, dest)
    finally:
        temp.unlink(missing_ok=True)


def atomic_save(payload, directory, step, keep, margin_gib, best=False, emergency=False):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    needed = int(tensor_bytes(payload) * 1.10) + int(margin_gib * 2**30) + 1024**2
    free = shutil.disk_usage(directory).free
    if free < needed:
        raise OSError(errno.ENOSPC, f'Checkpoint needs {needed / 2**30:.2f} GiB free, found {free / 2**30:.2f}; last valid checkpoint is preserved')
    dest = directory / f'step_{step:09d}.pt'
    temp = directory / f'.step_{step:09d}.{uuid.uuid4().hex}.tmp'
    try:
        with temp.open('wb') as f:
            torch.save(payload, f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp, dest)
        _fsync_directory(directory)
        _atomic_link(dest, directory / 'latest.pt')
        if best:
            _atomic_link(dest, directory / 'best.pt')
        if emergency:
            _atomic_link(dest, directory / 'emergency.pt')
        else:
            # Keep at most one extra emergency alias; it is superseded by a newer save.
            (directory / 'emergency.pt').unlink(missing_ok=True)
        _fsync_directory(directory)
        # Prune only after the new checkpoint and latest alias are durable.
        for old in sorted(directory.glob('step_*.pt'))[:-keep]:
            old.unlink()
        _fsync_directory(directory)
        return str(dest)
    finally:
        temp.unlink(missing_ok=True)


def save_checkpoint(runtime, model, optimizer, scheduler, config, state, metadata, data_identity,
                    best=False, emergency=False):
    states = runtime.gather(rng_state(runtime.device))
    result = None
    if runtime.primary:
        try:
            payload = {'format': FORMAT, 'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                       'scheduler': scheduler.state_dict(), 'state': state.copy(), 'rng_by_rank': states,
                       'config': config.to_dict(), 'metadata': metadata,
                       'contract': resume_contract(config, runtime.world, data_identity)}
            c = config.checkpointing
            path = atomic_save(payload, c.directory, state['step'], c.keep_numbered,
                               c.free_space_margin_gib, best, emergency)
            result = {'path': path, 'error': None}
        except Exception as exc:
            result = {'path': None, 'error': f'{type(exc).__name__}: {exc}'}
    # Other ranks wait here, not in DDP backward. IO errors reach all peers.
    result = runtime.broadcast(result)
    if result['error']:
        raise RuntimeError(result['error'])
    return result['path']


def load_checkpoint(path, runtime, model, optimizer, scheduler, config, data_identity):
    # Trusted local training artifacts contain Python/NumPy RNG state, not just tensors.
    payload = torch.load(path, map_location='cpu', weights_only=False, mmap=True)
    if payload.get('format') != FORMAT:
        raise ValueError('Unsupported checkpoint format; legacy artifacts are not auto-converted')
    if payload['metadata'].get('pytorch') != torch.__version__ or payload['metadata'].get('cuda') != torch.version.cuda:
        raise ValueError('Runtime version changed on exact resume; restore the recorded PyTorch/CUDA build')
    if payload['contract'] != resume_contract(config, runtime.world, data_identity):
        raise ValueError('Exact-resume contract changed (world/batch/model/data/tokenizer/optimizer/LR/precision/compile). Start a new run instead.')
    if len(payload['rng_by_rank']) != runtime.world:
        raise ValueError('Per-rank RNG states are missing')
    model.load_state_dict(payload['model'])
    optimizer.load_state_dict(payload['optimizer'])
    scheduler.load_state_dict(payload['scheduler'])
    state = payload['state']
    expected_tokens = state['step'] * config.training.micro_batch_size * config.training.gradient_accumulation_steps * runtime.world * config.model.context
    if scheduler.completed_steps != state['step'] or state['tokens'] != expected_tokens:
        raise ValueError('Checkpoint step/token/scheduler counters disagree')
    return state.copy(), payload['rng_by_rank'][runtime.rank]


class RunLock:
    """POSIX advisory lock also protects stale-temp cleanup from concurrent writers."""
    def __init__(self, directory):
        import fcntl
        self.fcntl = fcntl
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.file = (self.directory / '.run.lock').open('a+')
        try:
            fcntl.flock(self.file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            self.file.close()
            raise RuntimeError(f'Another trainer owns {directory}')
        for p in self.directory.glob('.*.tmp'):
            p.unlink()  # incomplete saves from killed processes, never valid .pt artifacts

    def close(self):
        self.fcntl.flock(self.file, self.fcntl.LOCK_UN)
        self.file.close()
