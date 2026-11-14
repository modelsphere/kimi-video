"""Assign a distinct preprocessing GPU to each tokenizer worker process."""

from __future__ import annotations

import atexit
import copy
import fcntl
import hashlib
import logging
import os
import tempfile
import threading
from contextlib import nullcontext
from pathlib import Path

GPU_ENV = "KIMI_K3_TOKENIZER_GPUS"
logger = logging.getLogger(__name__)


def tokenizer_gpus(environ=None):
    environ = os.environ if environ is None else environ
    value = environ.get(GPU_ENV)
    if value is None:
        return None
    parts = value.split(",")
    if any(not part.isascii() or not part.isdecimal() for part in parts):
        raise ValueError(f"{GPU_ENV} must be comma-separated nonnegative GPU indices")
    devices = tuple(map(int, parts))
    if len(set(devices)) != len(devices):
        raise ValueError(f"{GPU_ENV} must contain distinct GPU indices")
    return devices


class GpuLease:
    """Hold a slot until process exit; the kernel releases it even after SIGKILL.

    Files must not be unlinked: another worker may already hold the same inode.
    Closing an inherited descriptor, rather than LOCK_UN, preserves the parent's
    lock when SGLang forks a CPU preprocessing subprocess.
    """

    def __init__(self, namespace, devices, *, root=None):
        root = Path(tempfile.gettempdir() if root is None else root)
        directory = root / f"kimi-k3-tokenizer-gpus-{os.getuid()}"
        directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        key = hashlib.sha256(namespace.encode()).hexdigest()
        self.fd = None
        for slot, gpu in enumerate(devices):
            fd = os.open(
                directory / f"{key}-{slot}.lock", os.O_CREAT | os.O_RDWR, 0o600
            )
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                os.close(fd)
                continue
            except BaseException:
                os.close(fd)
                raise
            self.fd, self.gpu, self.slot = fd, gpu, slot
            return
        raise RuntimeError(f"No free tokenizer GPU slot in {GPU_ENV}={devices}")

    def close(self):
        if self.fd is not None:
            os.close(self.fd)
            self.fd = None


_lease = None
_lease_key = None
_lease_pid = None
_lock = threading.Lock()


def _release_lease():
    global _lease, _lease_key, _lease_pid
    if _lease is not None:
        _lease.close()
    _lease = _lease_key = _lease_pid = None


def _after_fork():
    global _lock
    _release_lease()
    _lock = threading.Lock()


atexit.register(_release_lease)
os.register_at_fork(after_in_child=_after_fork)


def _worker_gpu(server_args, devices):
    import torch
    from sglang.srt.managers.multi_tokenizer_mixin import get_main_process_id

    global _lease, _lease_key, _lease_pid
    if len(devices) != server_args.tokenizer_worker_num:
        raise ValueError(f"{GPU_ENV} needs exactly --tokenizer-worker-num GPU indices")
    if not torch.cuda.is_available():
        raise RuntimeError(f"{GPU_ENV} requires CUDA")
    count = torch.cuda.device_count()
    if any(gpu >= count for gpu in devices):
        raise ValueError(
            f"{GPU_ENV}={devices} exceeds the {count} visible CUDA devices; "
            "use indices after CUDA_VISIBLE_DEVICES remapping"
        )
    main_pid = get_main_process_id() or os.getpid()
    namespace = f"{main_pid}:{server_args.host}:{server_args.port}"
    key = namespace, devices
    with _lock:
        if _lease is None:
            _lease = GpuLease(namespace, devices)
            _lease_key, _lease_pid = key, os.getpid()
            logger.info(
                "Kimi-K3 tokenizer worker PID %d slot %d uses cuda:%d",
                _lease_pid,
                _lease.slot,
                _lease.gpu,
            )
        elif _lease_key != key or _lease_pid != os.getpid():
            raise RuntimeError("Tokenizer GPU placement changed within one process")
        return _lease.gpu


def init_tokenizer_processor(original, self, *args, **kwargs):
    devices = tokenizer_gpus()
    if devices is None or self.model_config.hf_config.model_type != "kimi_k31":
        return original(self, *args, **kwargs)
    import torch

    gpu = _worker_gpu(self.server_args, devices)
    # All later tokenizer-side transports see this copy too. The scheduler's
    # original launch configuration (including TP placement) is untouched.
    self.server_args = copy.copy(self.server_args)
    self.server_args.base_gpu_id = gpu
    torch.cuda.set_device(gpu)
    return original(self, *args, **kwargs)


def worker_cuda_scope():
    if _lease is None or _lease_pid != os.getpid():
        return nullcontext()
    import torch

    return torch.cuda.device(_lease.gpu)


def on_worker_gpu(original, *args, **kwargs):
    # Native K3 image operators use bare "cuda" and IO/executor threads do not
    # inherit the main thread's current device. Bind each synchronous GPU call.
    with worker_cuda_scope():
        return original(*args, **kwargs)
