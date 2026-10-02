"""Strict config: unknown fields fail rather than silently ignoring misspellings."""
from dataclasses import asdict, dataclass, field, fields
import json
from pathlib import Path


@dataclass
class ModelConfig:
    vocab_size: int = 32768
    n_layers: int = 20
    d_model: int = 2048
    n_heads: int = 16
    d_ff: int = 5504
    context: int = 2048
    rope_theta: float = 10000.0
    norm_eps: float = 1e-5
    dropout: float = 0.0
    init_std: float = 0.02
    gradient_checkpointing: bool = True
    loss_chunk_tokens: int = 512

    def validate(self):
        for k in ('vocab_size', 'n_layers', 'd_model', 'n_heads', 'd_ff', 'context', 'loss_chunk_tokens'):
            if type(getattr(self, k)) is not int or getattr(self, k) <= 0:
                raise ValueError(f'model.{k} must be a positive integer')
        if self.d_model % self.n_heads or (self.d_model // self.n_heads) % 2:
            raise ValueError('RoPE requires an even, integral head_dim')
        if not 0 <= self.dropout < 1 or self.norm_eps <= 0 or self.rope_theta <= 0:
            raise ValueError('Invalid dropout/norm_eps/rope_theta')


@dataclass
class DataConfig:
    manifest: str = 'data/scientific/manifest.json'
    sampling: str = 'random'
    shuffle_block: int = 4096
    workers: int = 4
    pin_memory: bool = True
    persistent_workers: bool = True
    prefetch_factor: int = 2
    verify_hashes: bool = True
    # Empty keeps v1 manifests compatible. The PRO 6000 config requires these
    # deterministic, document-disjoint validation views.
    validation_slices: list = field(default_factory=list)


@dataclass
class OptimizerConfig:
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-8
    fused: bool = False
    grad_clip: float = 1.0


@dataclass
class SchedulerConfig:
    max_lr: float = 3e-4
    min_lr: float = 3e-5
    warmup_steps: int = 2000
    total_steps: int = 100000


@dataclass
class DistributedConfig:
    backend: str = 'nccl'
    timeout_seconds: int = 1800
    bucket_cap_mb: int = 25
    # 0 accepts direct execution or torchrun. A positive value prevents an
    # accidental change in world size and therefore in the effective batch.
    expected_world_size: int = 0


@dataclass
class PrecisionConfig:
    dtype: str = 'bf16'
    tf32: bool = True
    deterministic: bool = False


@dataclass
class TrainingConfig:
    device: str = 'cuda'
    seed: int = 20260903
    micro_batch_size: int = 2
    gradient_accumulation_steps: int = 32
    max_steps: int = 100000
    compile: bool = False
    compile_fallback: bool = True
    compile_mode: str = 'default'
    cpu_threads: int = 4


@dataclass
class CheckpointConfig:
    directory: str = 'runs/scientific-1b/checkpoints'
    every_steps: int = 500
    keep_numbered: int = 3
    free_space_margin_gib: float = 2.0
    enabled: bool = True


@dataclass
class LoggingConfig:
    every_steps: int = 10
    eval_every_steps: int = 500
    eval_batches: int = 32  # global batches, independent of world_size; 0 = full validation
    jsonl: str = 'runs/scientific-1b/metrics.jsonl'


@dataclass
class BenchmarkConfig:
    warmup_steps: int = 10
    measured_steps: int = 50
    output: str = 'runs/benchmark.json'
    gpu_poll_seconds: float = 1.0
    vram_headroom_gib: float = 12.0


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    distributed: DistributedConfig = field(default_factory=DistributedConfig)
    precision: PrecisionConfig = field(default_factory=PrecisionConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    checkpointing: CheckpointConfig = field(default_factory=CheckpointConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    benchmark: BenchmarkConfig = field(default_factory=BenchmarkConfig)

    def validate(self):
        for section in fields(self):
            obj = getattr(self, section.name)
            for entry in fields(obj):
                value = getattr(obj, entry.name)
                expected = entry.type
                valid = type(value) in (int, float) if expected is float else type(value) is expected
                if not valid:
                    raise TypeError(f'{section.name}.{entry.name} expects {expected.__name__}')
        self.model.validate()
        for section, keys in [(self.training, ('micro_batch_size', 'gradient_accumulation_steps', 'max_steps', 'cpu_threads')),
                              (self.checkpointing, ('every_steps', 'keep_numbered')),
                              (self.logging, ('every_steps', 'eval_every_steps')),
                              (self.data, ('shuffle_block', 'prefetch_factor')),
                              (self.distributed, ('timeout_seconds', 'bucket_cap_mb')),
                              (self.benchmark, ('measured_steps',))]:
            for key in keys:
                if type(getattr(section, key)) is not int or getattr(section, key) <= 0:
                    raise ValueError(f'{type(section).__name__}.{key} must be a positive integer')
        s = self.scheduler
        if not 0 <= s.warmup_steps < s.total_steps or not 0 <= s.min_lr <= s.max_lr or s.max_lr <= 0:
            raise ValueError('Invalid warmup/cosine schedule')
        if self.training.max_steps > s.total_steps:
            raise ValueError('training.max_steps exceeds scheduler.total_steps; explicitly set the intended LR horizon')
        if self.data.sampling not in ('random', 'sequential') or self.data.workers < 0:
            raise ValueError('Invalid data sampling/workers')
        if any(type(name) is not str or not name for name in self.data.validation_slices):
            raise ValueError('data.validation_slices must contain nonempty names')
        if len(set(self.data.validation_slices)) != len(self.data.validation_slices):
            raise ValueError('data.validation_slices contains duplicates')
        if self.distributed.expected_world_size < 0:
            raise ValueError('distributed.expected_world_size must be zero or positive')
        if self.precision.dtype not in ('bf16', 'fp32') or self.training.device not in ('cuda', 'cpu'):
            raise ValueError('Supported precision/device: bf16|fp32, cuda|cpu')
        if self.training.device == 'cpu' and (self.precision.dtype != 'fp32' or self.distributed.backend != 'gloo'):
            raise ValueError('CPU smoke tests require fp32 and gloo explicitly')
        if self.training.device == 'cuda' and self.distributed.backend != 'nccl':
            raise ValueError('CUDA training requires NCCL')
        if self.optimizer.grad_clip <= 0 or self.optimizer.eps <= 0 or self.optimizer.weight_decay < 0:
            raise ValueError('Invalid optimizer configuration')
        if not 0 <= self.optimizer.beta1 < 1 or not 0 <= self.optimizer.beta2 < 1:
            raise ValueError('Invalid AdamW betas')
        if (self.logging.eval_batches < 0 or self.benchmark.warmup_steps < 0
                or self.benchmark.gpu_poll_seconds <= 0 or self.benchmark.vram_headroom_gib < 0):
            raise ValueError('Invalid eval/benchmark configuration')
        if self.checkpointing.free_space_margin_gib < 0:
            raise ValueError('Negative checkpoint margin')
        if self.precision.deterministic and self.precision.tf32:
            raise ValueError('Set precision.tf32=false for deterministic runs')

    def to_dict(self):
        return asdict(self)


def from_dict(raw):
    result = Config()
    known = {f.name for f in fields(result)}
    if raw.keys() - known:
        raise ValueError(f'Unknown config sections: {raw.keys() - known}')
    for section, values in raw.items():
        obj = getattr(result, section)
        setattr(result, section, type(obj)(**values))
    result.validate()
    return result


def load_config(path, overrides=()):
    raw = json.loads(Path(path).read_text())
    for override in overrides:
        key, value = override.split('=', 1)
        section, name = key.split('.', 1)
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            pass  # unquoted paths/string enums are convenient on the shell
        raw.setdefault(section, {})[name] = value
    return from_dict(raw)
