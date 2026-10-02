"""Memory-mapped tokens and O(1)-memory deterministic distributed sampling."""
import hashlib
import json
import math
from pathlib import Path
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Sampler


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_binary(root, name, info, vocab_size, verify_hashes):
    file = (root / info['path']).resolve()
    if info['dtype'] not in ('<u2', '<u4'):
        raise ValueError(f'{name}: use explicit little-endian uint16/uint32')
    if file.stat().st_size != info['tokens'] * np.dtype(info['dtype']).itemsize:
        raise ValueError(f'{name}: binary length does not match manifest')
    if not 0 <= info['max_token_id'] < vocab_size:
        raise ValueError(f'{name}: token IDs exceed vocabulary')
    if verify_hashes and sha256_file(file) != info['sha256']:
        raise ValueError(f'{name}: content hash mismatch')
    return file


def read_manifest(path, vocab_size, verify_hashes=False, allow_synthetic=False,
                  required_validation_slices=()):
    path = Path(path).resolve()
    raw = json.loads(path.read_text())
    if raw.get('format') != 'jarvis-tokens-v1' or raw.get('vocab_size') != vocab_size:
        raise ValueError('Dataset format/vocabulary mismatch')
    if raw.get('synthetic') and not allow_synthetic:
        raise ValueError('Synthetic data are allowed only in smoke/benchmark/server-smoke mode')
    if not raw.get('disjoint_documents'):
        raise ValueError('Prepare document-disjoint train/validation before tokenization')
    tok = raw.get('tokenizer', {})
    if not raw.get('synthetic'):
        frozen = path.parent / tok['path']
        if sha256_file(frozen) != tok['sha256']:
            raise ValueError('Frozen tokenizer hash mismatch')
    files = {
        'train': _validate_binary(path.parent, 'train', raw['train'], vocab_size, verify_hashes),
        'validation': _validate_binary(path.parent, 'validation', raw['validation'], vocab_size, verify_hashes),
        'validation_sets': {},
    }
    validation_sets = raw.get('validation_sets', {})
    missing = sorted(set(required_validation_slices) - set(validation_sets))
    if missing:
        raise ValueError(f'Missing required validation slices: {missing}')
    hashes = {raw['validation']['sha256']}
    for name, info in validation_sets.items():
        if type(name) is not str or not name:
            raise ValueError('Invalid validation slice name')
        files['validation_sets'][name] = _validate_binary(
            path.parent, f'validation_sets.{name}', info, vocab_size, verify_hashes)
        hashes.add(info['sha256'])
    if files['train'].samefile(files['validation']) or raw['train']['sha256'] in hashes:
        raise ValueError('Training data overlap a validation binary')
    identity = hashlib.sha256(json.dumps(raw, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    return raw, files, identity


class TokenWindows(Dataset):
    def __init__(self, path, dtype, context):
        self.path, self.dtype, self.context = str(path), dtype, context
        self.tokens = Path(path).stat().st_size // np.dtype(dtype).itemsize
        self.n = (self.tokens - 1) // context
        self._mapping = None
        if self.n < 1:
            raise ValueError('Dataset must contain at least context+1 tokens')

    def __len__(self):
        return self.n

    def __getstate__(self):
        state = self.__dict__.copy()
        state['_mapping'] = None
        return state

    def __getitem__(self, index):
        if not 0 <= index < self.n:
            raise IndexError(index)
        if self._mapping is None:
            self._mapping = np.memmap(self.path, dtype=self.dtype, mode='r')
        start = index * self.context
        return torch.from_numpy(np.array(self._mapping[start:start+self.context+1], dtype=np.int64))


def _mix(x):
    mask = (1 << 64) - 1
    x = (x + 0x9E3779B97F4A7C15) & mask
    x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & mask
    x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & mask
    return x ^ (x >> 31)


def _permute(i, n, key):
    if n <= 1:
        return i
    a = 1 + _mix(key) % (n - 1)
    while math.gcd(a, n) != 1:
        a = a % (n - 1) + 1
    return (a * i + _mix(key + 1) % n) % n


def sample_index(position, size, usable, seed, sampling, block_size):
    epoch, offset = divmod(position, usable)
    if sampling == 'sequential':
        return offset
    key = _mix(seed + epoch)
    full, remainder = divmod(size, block_size)
    shift = _mix(key + 4) % size
    if offset < full * block_size:
        block, within = divmod(offset, block_size)
        mapped_block = _permute(block, full, key)
        value = mapped_block * block_size + _permute(within, block_size, _mix(key + mapped_block + 2))
    else:
        value = full * block_size + _permute(offset - full * block_size, remainder, key + 3)
    return (value + shift) % size


class StepBatchSampler(Sampler):
    def __init__(self, size, micro_batch, accumulation, rank, world, start_step, end_step,
                 seed, sampling='random', block_size=4096):
        self.size, self.micro, self.accum = size, micro_batch, accumulation
        self.rank, self.world, self.start, self.end = rank, world, start_step, end_step
        self.seed, self.sampling, self.block = seed, sampling, block_size
        self.global_batch = micro_batch * accumulation * world
        self.usable = size // self.global_batch * self.global_batch
        if self.usable < self.global_batch:
            raise ValueError(f'Training needs at least {self.global_batch} distinct windows; found {size}')
        if not 0 <= rank < world or end_step < start_step:
            raise ValueError('Invalid sampler range/rank')

    def __len__(self):
        return (self.end - self.start) * self.accum

    def __iter__(self):
        start = self.start * self.global_batch
        for microstep in range(len(self)):
            base = start + microstep * self.world * self.micro + self.rank * self.micro
            yield [sample_index(base + j, self.size, self.usable, self.seed, self.sampling, self.block)
                   for j in range(self.micro)]


def seed_worker(worker_id):
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)


def make_loader(dataset, config, seed, *, batch_sampler=None, indices=None, batch_size=None):
    generator = torch.Generator().manual_seed(seed)
    kwargs = dict(num_workers=config.workers, pin_memory=config.pin_memory,
                  worker_init_fn=seed_worker, generator=generator)
    if config.workers:
        kwargs.update(persistent_workers=config.persistent_workers,
                      prefetch_factor=config.prefetch_factor,
                      multiprocessing_context='spawn')
    if batch_sampler is not None:
        return DataLoader(dataset, batch_sampler=batch_sampler, **kwargs)
    return DataLoader(dataset, sampler=indices, batch_size=batch_size, drop_last=False, **kwargs)
