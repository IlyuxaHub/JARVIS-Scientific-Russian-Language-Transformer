import json
from pathlib import Path
import subprocess
import threading
import torch


def git_metadata():
    def run(args):
        try:
            return subprocess.check_output(['git', *args], text=True, stderr=subprocess.DEVNULL).strip()
        except (OSError, subprocess.CalledProcessError):
            return None
    commit = run(['rev-parse', 'HEAD'])
    status = run(['status', '--porcelain'])
    return {'git_commit': commit, 'git_dirty': bool(status) if commit else None}


def sdpa_backend_info():
    if not torch.cuda.is_available():
        return {'available': False}
    backend = torch.backends.cuda
    result = {'available': True}
    for name in ('flash_sdp_enabled', 'mem_efficient_sdp_enabled', 'math_sdp_enabled',
                 'cudnn_sdp_enabled'):
        fn = getattr(backend, name, None)
        if fn is not None:
            try:
                result[name] = bool(fn())
            except RuntimeError as exc:
                result[name] = f'unavailable: {exc}'
    result['note'] = 'Enabled flags are policy, not proof of the kernel selected for every shape.'
    return result


def device_info(runtime):
    if runtime.device.type == 'cpu':
        return {'rank': runtime.rank, 'name': 'CPU', 'visible_gpu_count': 0,
                'vram_gib': 0, 'bf16': False, 'capability': None}
    prop = torch.cuda.get_device_properties(runtime.device)
    return {'rank': runtime.rank, 'local_rank': runtime.local_rank, 'name': prop.name,
            'visible_gpu_count': torch.cuda.device_count(),
            'vram_gib': prop.total_memory / 2**30,
            'uuid': str(getattr(prop, 'uuid', 'unknown')),
            'bf16': torch.cuda.is_bf16_supported(),
            'capability': [prop.major, prop.minor]}


def memory(runtime):
    if runtime.device.type == 'cpu':
        return {'allocated_gib': 0.0, 'reserved_gib': 0.0,
                'peak_allocated_gib': 0.0, 'peak_reserved_gib': 0.0}
    return {key: fn(runtime.device) / 2**30 for key, fn in [
        ('allocated_gib', torch.cuda.memory_allocated),
        ('reserved_gib', torch.cuda.memory_reserved),
        ('peak_allocated_gib', torch.cuda.max_memory_allocated),
        ('peak_reserved_gib', torch.cuda.max_memory_reserved)]}


class Logger:
    def __init__(self, path, enabled=True):
        self.path = Path(path)
        self.enabled = enabled
        if enabled:
            self.path.parent.mkdir(parents=True, exist_ok=True)

    def emit(self, record):
        if not self.enabled:
            return
        serialized = json.dumps(record, ensure_ascii=False, allow_nan=False)
        print(serialized, flush=True)
        with self.path.open('a', encoding='utf-8') as f:
            f.write(serialized + '\n')


class GPUMonitor:
    """Coarse nvidia-smi telemetry sampled outside the training thread."""
    def __init__(self, interval=1.0):
        self.interval, self.samples, self.error = interval, {}, None
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self.run, daemon=True)

    def start(self):
        self.thread.start()

    def run(self):
        while not self.stop_event.is_set():
            try:
                text = subprocess.check_output(
                    ['nvidia-smi', '--query-gpu=uuid,index,utilization.gpu,memory.used',
                     '--format=csv,noheader,nounits'], text=True, timeout=3,
                    stderr=subprocess.DEVNULL)
                for line in text.strip().splitlines():
                    uuid, index, util, used = [x.strip() for x in line.split(',')]
                    self.samples.setdefault(uuid, []).append((float(util), float(used), index))
            except (OSError, ValueError, subprocess.SubprocessError) as exc:
                self.error = str(exc)
                return
            self.stop_event.wait(self.interval)

    def finish(self):
        self.stop_event.set()
        self.thread.join(timeout=4)
        result = {uuid: {'index': samples[0][2], 'samples': len(samples),
                         'mean_utilization_percent': sum(x[0] for x in samples) / len(samples),
                         'peak_memory_used_mib': max(x[1] for x in samples)}
                  for uuid, samples in self.samples.items()}
        return {'all_physical_gpus': result, 'error': self.error,
                'note': 'Device-wide nvidia-smi telemetry can include other processes.'}
