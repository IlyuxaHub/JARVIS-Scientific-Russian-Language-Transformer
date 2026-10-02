"""Run with torchrun --standalone --nproc_per_node=2 -m jarvis_scientific.train."""
import argparse
from contextlib import nullcontext
import json
import math
import os
from pathlib import Path
import time
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from .checkpoint import RunLock, load_checkpoint, restore_rng, rng_state, save_checkpoint
from .config import load_config
from .data import TokenWindows, StepBatchSampler, make_loader, read_manifest
from .distributed import Runtime, StopSignal, seed_all
from .model import ScientificLM, parameter_count
from .optim import make_optimizer, WarmupCosine
from .telemetry import GPUMonitor, Logger, device_info, git_metadata, memory, sdpa_backend_info


def autocast(config, device):
    return torch.autocast(device_type=device.type, dtype=torch.bfloat16) if config.precision.dtype == 'bf16' else nullcontext()


@torch.no_grad()
def evaluate(model, loader, config, runtime):
    previous = model.training
    model.eval()
    sums = torch.zeros(2, device=runtime.device, dtype=torch.float64)
    try:
        for batch in loader:
            batch = batch.to(runtime.device, non_blocking=True)
            with autocast(config, runtime.device):
                _, loss = model(batch[:, :-1], batch[:, 1:])
            n = batch.shape[0] * (batch.shape[1] - 1)
            sums[0].add_(loss.detach().double() * n)
            sums[1].add_(n)
        # Raw model, not DDP forward: ranks can have unequal numbers of validation batches.
        runtime.reduce(sums)
        loss_sum, count = sums.tolist()
        if count <= 0 or not math.isfinite(loss_sum):
            raise FloatingPointError('Empty/nonfinite validation; last checkpoint is preserved')
        return loss_sum / count
    finally:
        model.train(previous)


def compile_model(model, config, runtime):
    if not config.training.compile:
        return model
    snapshot = rng_state(runtime.device)
    error, candidate = None, None
    try:
        candidate = torch.compile(model, mode=config.training.compile_mode)
        x = torch.zeros((config.training.micro_batch_size, config.model.context), dtype=torch.long, device=runtime.device)
        with autocast(config, runtime.device):
            _, loss = candidate(x, x)
        loss.backward()  # trigger both forward and backward compilation before DDP collectives
    except Exception as exc:
        error = f'{type(exc).__name__}: {exc}'
    finally:
        model.zero_grad(set_to_none=True)
        restore_rng(snapshot, runtime.device)
    errors = runtime.gather(error)
    errors = runtime.broadcast(errors)
    if any(errors):
        if not config.training.compile_fallback:
            raise RuntimeError(f'Compile failed: {errors}')
        if runtime.primary:
            print(f'Compile failed; all ranks switch to eager: {errors}', flush=True)
        config.training.compile = False  # persist actual mode in resume contract
        return model
    return candidate


def run(config, mode='train', resume=None):
    config.validate()
    if mode == 'smoke' and (parameter_count(config.model) > 10_000_000 or config.training.max_steps > 20):
        raise ValueError('Smoke is restricted to <=10M parameters and <=20 steps')
    if mode == 'server-smoke' and config.training.max_steps > 2:
        raise ValueError('Server smoke is restricted to at most two production-model updates')
    if mode == 'benchmark' and resume:
        raise ValueError('Benchmark must start fresh')
    if config.precision.deterministic:
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    torch.set_num_threads(config.training.cpu_threads)
    torch.use_deterministic_algorithms(config.precision.deterministic)
    torch.set_float32_matmul_precision('high' if config.precision.tf32 else 'highest')
    # cuDNN is not used by this Transformer; do not enable convolution autotuning globally.
    runtime = Runtime(config)
    lock, monitor = None, None
    started = time.perf_counter()
    try:
        if config.checkpointing.enabled and mode != 'benchmark':
            error = None
            if runtime.primary:
                try:
                    lock = RunLock(config.checkpointing.directory)
                    if not resume and any(Path(config.checkpointing.directory).glob('*.pt')):
                        raise FileExistsError('Checkpoint directory is not empty. Use --resume or a new run directory')
                except Exception as exc:
                    error = str(exc)
            error = runtime.broadcast(error)
            if error:
                raise RuntimeError(error)
        seed_all(config.training.seed)
        # Full content hashes are streamed once by rank 0, before any model allocation.
        # Other ranks receive the outcome instead of duplicating disk scans.
        data_error = None
        if runtime.primary:
            try:
                read_manifest(config.data.manifest, config.model.vocab_size,
                              config.data.verify_hashes, mode in ('smoke', 'benchmark', 'server-smoke'),
                              config.data.validation_slices)
            except Exception as exc:
                data_error = f'{type(exc).__name__}: {exc}'
        data_error = runtime.broadcast(data_error)
        if data_error:
            raise ValueError(data_error)
        manifest, files, identity = read_manifest(config.data.manifest, config.model.vocab_size,
                                                  False, mode in ('smoke', 'benchmark', 'server-smoke'),
                                                  config.data.validation_slices)
        train_data = TokenWindows(files['train'], manifest['train']['dtype'], config.model.context)
        validation_data = {'overall': TokenWindows(
            files['validation'], manifest['validation']['dtype'], config.model.context)}
        for name in config.data.validation_slices:
            info = manifest['validation_sets'][name]
            validation_data[name] = TokenWindows(
                files['validation_sets'][name], info['dtype'], config.model.context)
        model = ScientificLM(config.model).to(runtime.device)
        compiled = compile_model(model, config, runtime)
        wrapped = DDP(compiled, device_ids=[runtime.local_rank] if runtime.device.type == 'cuda' else None,
                      broadcast_buffers=False, gradient_as_bucket_view=True,
                      bucket_cap_mb=config.distributed.bucket_cap_mb, find_unused_parameters=False) if runtime.world > 1 else compiled
        optimizer = make_optimizer(model, config.optimizer, runtime.device)
        scheduler = WarmupCosine(optimizer, config.scheduler)
        seed_all(config.training.seed + runtime.rank)
        state = {'step': 0, 'tokens': 0, 'best_validation': None, 'last_validation': None, 'elapsed_seconds': 0.0}
        saved_rng = None
        if resume:
            path = Path(config.checkpointing.directory) / 'latest.pt' if resume == 'latest' else Path(resume)
            state, saved_rng = load_checkpoint(path, runtime, model, optimizer, scheduler, config, identity)
        start_step = state['step']
        end_step = config.training.max_steps if mode != 'benchmark' else config.benchmark.warmup_steps + config.benchmark.measured_steps
        if end_step > config.scheduler.total_steps or start_step > end_step:
            raise ValueError('Invalid step bounds for the configured scheduler/resume')
        micro, accum = config.training.micro_batch_size, config.training.gradient_accumulation_steps
        global_batch = micro * accum * runtime.world
        tokens_per_update = global_batch * config.model.context
        sampler = StepBatchSampler(len(train_data), micro, accum, runtime.rank, runtime.world,
                                   start_step, end_step, config.training.seed, config.data.sampling, config.data.shuffle_block)
        train_loader = make_loader(train_data, config.data, config.training.seed + runtime.rank, batch_sampler=sampler)
        validation_loaders, validation_counts = {}, {}
        for offset, (name, dataset) in enumerate(validation_data.items()):
            count = min(len(dataset), config.logging.eval_batches * micro) if config.logging.eval_batches else len(dataset)
            validation_counts[name] = count
            validation_loaders[name] = make_loader(
                dataset, config.data, config.training.seed + 100000 + offset * 1000 + runtime.rank,
                indices=range(runtime.rank, count, runtime.world), batch_size=micro)
        iterator = iter(train_loader)
        # Worker generator is separate from the model RNG; prefetch progress is not a checkpoint cursor.
        if saved_rng is not None:
            restore_rng(saved_rng, runtime.device)
        devices = runtime.gather(device_info(runtime))
        metadata = dict(git_metadata(), pytorch=torch.__version__, cuda=torch.version.cuda,
                        cuda_available=torch.cuda.is_available(), devices=devices,
                        sdpa_backends=sdpa_backend_info(), tf32=config.precision.tf32,
                        data_identity=identity, startup_seconds=time.perf_counter() - started)
        logger = Logger(config.logging.jsonl, runtime.primary)
        if runtime.primary:
            logger.emit(dict(event='startup', mode=mode, parameters=sum(p.numel() for p in model.parameters()),
                             trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
                             world_size=runtime.world, context=config.model.context, micro_batch=micro, accumulation=accum,
                             effective_global_batch=global_batch, tokens_per_update=tokens_per_update,
                             max_lr=config.scheduler.max_lr, min_lr=config.scheduler.min_lr,
                             train_tokens=train_data.tokens, validation_tokens=validation_data['overall'].tokens,
                             validation_windows=validation_counts, dropped_train_windows_per_epoch=len(train_data) - sampler.usable,
                             steps_per_epoch=sampler.usable // global_batch, estimated_steps=end_step,
                             planned_tokens=end_step * tokens_per_update, start_step=start_step,
                             checkpoint_directory=str(Path(config.checkpointing.directory).resolve()),
                             bf16_autocast=config.precision.dtype == 'bf16', fp32_master_weights=True,
                             gradient_checkpointing=config.model.gradient_checkpointing,
                             compile=config.training.compile, fused_adamw=config.optimizer.fused,
                             precision=config.precision.dtype, **metadata))
            if metadata['git_dirty']:
                print('WARNING: working tree contains uncommitted changes', flush=True)
            if runtime.device.type == 'cuda' and any(d['vram_gib'] < 40 for d in devices) and parameter_count(config.model) > 1e9:
                print('WARNING: less than 40 GiB per GPU; this run may not have enough headroom. Run the hardware-specific memory benchmark first.', flush=True)
        model.train()
        elapsed_before = state['elapsed_seconds']
        training_started = time.perf_counter()
        interval_seconds, interval_data_wait, interval_steps = 0.0, 0.0, 0
        interval_loss = torch.zeros((), device=runtime.device)
        bench_started, bench_steps, data_wait, events = None, 0, 0.0, []
        bench_initial_peak = None
        last_eval = state['last_validation']
        if last_eval is not None and not isinstance(last_eval, dict):
            last_eval = {'overall': last_eval}  # v1 checkpoint compatibility
        stop_requested = False
        with StopSignal() as signals:
            for step in range(start_step + 1, end_step + 1):
                measuring = mode == 'benchmark' and step > config.benchmark.warmup_steps
                if measuring and bench_started is None:
                    if runtime.device.type == 'cuda':
                        torch.cuda.synchronize(runtime.device)  # benchmark boundary only
                        bench_initial_peak = memory(runtime)
                        torch.cuda.reset_peak_memory_stats(runtime.device)
                    if runtime.world > 1:
                        dist.barrier()  # one measured-window alignment, never per update
                    bench_started = time.perf_counter()
                    if runtime.primary and runtime.device.type == 'cuda':
                        monitor = GPUMonitor(config.benchmark.gpu_poll_seconds)
                        monitor.start()
                update_started = time.perf_counter()
                begin_event = end_event = None
                if measuring and runtime.device.type == 'cuda':
                    begin_event, end_event = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    begin_event.record()
                optimizer.zero_grad(set_to_none=True)
                total_loss = torch.zeros((), device=runtime.device)
                for substep in range(accum):
                    wait_start = time.perf_counter()
                    batch = next(iterator)
                    waited = time.perf_counter() - wait_start
                    interval_data_wait += waited
                    if measuring:
                        data_wait += waited
                    batch = batch.to(runtime.device, non_blocking=True)
                    sync = wrapped.no_sync() if runtime.world > 1 and substep < accum - 1 else nullcontext()
                    # no_sync must enclose forward AND backward.
                    with sync:
                        with autocast(config, runtime.device):
                            _, loss = wrapped(batch[:, :-1], batch[:, 1:])
                        (loss / accum).backward()
                    total_loss.add_(loss.detach() / accum)
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.optimizer.grad_clip)
                flags = torch.stack((~(torch.isfinite(total_loss) & torch.isfinite(grad_norm)),
                                     torch.full((), signals.requested, device=runtime.device, dtype=torch.bool))).to(torch.int32)
                runtime.reduce(flags, dist.ReduceOp.MAX)
                bad, stop_requested = flags.tolist()  # one necessary finite/signal rendezvous per optimizer update
                if bad:
                    optimizer.zero_grad(set_to_none=True)
                    raise FloatingPointError(f'Nonfinite loss/gradient at update {step}; optimizer not advanced; resume last valid checkpoint')
                lr = scheduler.prepare(step)
                optimizer.step()
                scheduler.commit(step)
                state['step'], state['tokens'] = step, step * tokens_per_update
                interval_loss.add_(total_loss)
                interval_steps += 1
                interval_seconds += time.perf_counter() - update_started
                if measuring:
                    bench_steps += 1
                    if end_event:
                        end_event.record()
                        events.append((begin_event, end_event))
                if step % config.logging.every_steps == 0 or step == end_step or stop_requested:
                    reduced = runtime.reduce(interval_loss.detach().clone()) / (runtime.world * interval_steps)
                    train_loss = reduced.item()
                    now = time.perf_counter()
                    peaks = runtime.gather(memory(runtime))
                    logger.emit(dict(event='train', step=step, train_loss=train_loss, validation_loss=last_eval,
                                     lr=lr, tokens_per_second=interval_steps * tokens_per_update / max(interval_seconds, 1e-9),
                                     seconds_per_update=interval_seconds / interval_steps,
                                     data_wait_seconds_per_update=interval_data_wait / interval_steps,
                                     data_wait_fraction=interval_data_wait / max(interval_seconds, 1e-9),
                                     micro_batch=micro, accumulation=accum,
                                     effective_global_batch=global_batch, tokens_per_update=tokens_per_update,
                                     gradient_checkpointing=config.model.gradient_checkpointing,
                                     fused_adamw=config.optimizer.fused, compile=config.training.compile,
                                     tokens=state['tokens'], elapsed_seconds=elapsed_before + now - training_started,
                                     gpu_memory=peaks))
                    interval_loss.zero_()
                    interval_seconds, interval_data_wait, interval_steps = 0.0, 0.0, 0
                improved = False
                if mode != 'benchmark' and not stop_requested and (step % config.logging.eval_every_steps == 0 or step == end_step):
                    last_eval = {name: evaluate(model, loader, config, runtime)
                                 for name, loader in validation_loaders.items()}
                    overall = last_eval['overall']
                    improved = state['best_validation'] is None or overall < state['best_validation']
                    state['last_validation'] = last_eval
                    if improved:
                        state['best_validation'] = overall
                    logger.emit(dict(event='validation', step=step, losses=last_eval,
                                     windows=validation_counts))
                state['elapsed_seconds'] = elapsed_before + time.perf_counter() - training_started
                if config.checkpointing.enabled and mode != 'benchmark' and (improved or stop_requested or step % config.checkpointing.every_steps == 0 or step == end_step):
                    checkpoint_started = time.perf_counter()
                    saved = save_checkpoint(runtime, model, optimizer, scheduler, config, state, metadata, identity,
                                            best=improved, emergency=bool(stop_requested))
                    checkpoint_seconds = time.perf_counter() - checkpoint_started
                    size = Path(saved).stat().st_size if runtime.primary else None
                    size = runtime.broadcast(size)
                    logger.emit(dict(event='checkpoint', step=step, path=saved,
                                     emergency=bool(stop_requested), seconds=checkpoint_seconds,
                                     size_gib=size / 2**30,
                                     estimated_required_free_gib=size * 1.10 / 2**30 + config.checkpointing.free_space_margin_gib))
                if stop_requested:
                    break
        if mode == 'benchmark':
            if runtime.device.type == 'cuda':
                torch.cuda.synchronize(runtime.device)  # end of measured window, not hot path
            elapsed = time.perf_counter() - bench_started if bench_started is not None else 0.0
            timing = torch.tensor([elapsed, data_wait], dtype=torch.float64, device=runtime.device)
            runtime.reduce(timing, dist.ReduceOp.MAX)
            elapsed, slowest_data_wait = timing.tolist()
            per_rank = runtime.gather(dict(rank=runtime.rank, memory=memory(runtime), startup_peak=bench_initial_peak,
                                          mean_cuda_step_ms=sum(a.elapsed_time(b) for a, b in events) / len(events) if events else None))
            telemetry = monitor.finish() if monitor else None
            monitor = None
            result = dict(event='benchmark', world_size=runtime.world, measured_steps=bench_steps,
                          tokens_per_second=bench_steps * tokens_per_update / elapsed if elapsed > 0 else None,
                          seconds_per_update=elapsed / bench_steps if bench_steps else None,
                          seconds=elapsed, slowest_rank_data_wait_seconds=slowest_data_wait,
                          data_wait_fraction=slowest_data_wait / elapsed if elapsed else None,
                          per_rank=per_rank, gpu_telemetry=telemetry, interrupted=bool(stop_requested),
                          config=config.to_dict(), data_identity=identity, metadata=metadata)
            if runtime.primary:
                target = Path(config.benchmark.output)
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
                logger.emit(result)
        return state
    finally:
        if monitor:
            monitor.finish()
        if lock:
            lock.close()
        runtime.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', default='configs/scientific_1b.json')
    p.add_argument('--set', action='append', default=[], metavar='SECTION.KEY=VALUE')
    p.add_argument('--mode', choices=('train', 'smoke', 'server-smoke', 'benchmark'), default='train')
    p.add_argument('--resume', nargs='?', const='latest')
    args = p.parse_args()
    try:
        run(load_config(args.config, args.set), args.mode, args.resume)
    except torch.OutOfMemoryError:
        diagnostic = {'event': 'fatal_cuda_oom', 'policy': 'terminate_without_checkpoint'}
        if torch.cuda.is_available():
            device = torch.cuda.current_device()
            diagnostic.update(allocated_gib=torch.cuda.memory_allocated(device) / 2**30,
                              reserved_gib=torch.cuda.memory_reserved(device) / 2**30,
                              peak_allocated_gib=torch.cuda.max_memory_allocated(device) / 2**30,
                              peak_reserved_gib=torch.cuda.max_memory_reserved(device) / 2**30)
        print(json.dumps(diagnostic), flush=True)
        raise


if __name__ == '__main__':
    main()
