"""Process ownership, restart recovery and tokenizer-local placement."""

from __future__ import annotations

import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from functools import partial
from types import SimpleNamespace

import pytest
import torch

from kimi_k3_video import compat, placement


@pytest.mark.parametrize("value", ["", "0,", "-1", "+1", "0,0", "0, 1", "１", "1.0"])
def test_invalid_gpu_list(value):
    with pytest.raises(ValueError, match=placement.GPU_ENV):
        placement.tokenizer_gpus({placement.GPU_ENV: value})


def test_gpu_list_is_opt_in_and_uses_visible_indices():
    assert placement.tokenizer_gpus({}) is None
    assert placement.tokenizer_gpus({placement.GPU_ENV: "0,2,4,6"}) == (0, 2, 4, 6)


def test_four_processes_keep_distinct_slots_and_reclaim_after_kill(tmp_path):
    code = f"""
import sys
from kimi_k3_video.placement import GpuLease
lease = GpuLease('test-service', (0, 2, 4, 6), root={str(tmp_path)!r})
print(lease.gpu, flush=True)
sys.stdin.readline()
"""
    workers = []

    def launch():
        worker = subprocess.Popen(
            [sys.executable, "-c", code],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env={**os.environ, "PYTHONPATH": os.pathsep.join(sys.path)},
        )
        workers.append(worker)
        return worker

    try:
        with ThreadPoolExecutor(max_workers=4) as pool:
            active = list(pool.map(lambda _: launch(), range(4)))
        assigned = [int(worker.stdout.readline()) for worker in active]
        assert set(assigned) == {0, 2, 4, 6}
        with pytest.raises(RuntimeError, match="No free"):
            placement.GpuLease("test-service", (0, 2, 4, 6), root=tmp_path)
        # A different service does not share this service's slot ownership.
        other = placement.GpuLease("another-service", (0, 2, 4, 6), root=tmp_path)
        other.close()
        active[1].kill()
        active[1].wait(timeout=10)
        replacement = launch()
        assert int(replacement.stdout.readline()) == assigned[1]
    finally:
        for worker in workers:
            if worker.poll() is None:
                worker.communicate("\n", timeout=10)
            else:
                worker.communicate(timeout=10)


@pytest.fixture
def fake_worker_cuda(monkeypatch, tmp_path, source_modules):
    placement._release_lease()
    monkeypatch.setattr(
        placement, "GpuLease", partial(placement.GpuLease, root=tmp_path)
    )
    source_modules.module(
        "sglang.srt.managers.multi_tokenizer_mixin"
    ).get_main_process_id = lambda: 1234
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 8)
    selected = []
    monkeypatch.setattr(torch.cuda, "set_device", selected.append)
    yield selected
    placement._release_lease()


def manager(model_type="kimi_k31", workers=4):
    return SimpleNamespace(
        model_config=SimpleNamespace(hf_config=SimpleNamespace(model_type=model_type)),
        server_args=SimpleNamespace(
            base_gpu_id=0, tokenizer_worker_num=workers, host="0.0.0.0", port=31031
        ),
    )


def test_local_args_are_set_before_processor_and_transport_init(
    fake_worker_cuda, monkeypatch
):
    monkeypatch.setenv(placement.GPU_ENV, "2,3,4,5")
    worker = manager()
    scheduler_args = worker.server_args
    initialized = []

    def initialize(self):
        initialized.append(self.server_args.base_gpu_id)
        return "processor-and-pool"

    assert (
        placement.init_tokenizer_processor(initialize, worker) == "processor-and-pool"
    )
    assert initialized == [2] and fake_worker_cuda == [2]
    assert worker.server_args is not scheduler_args and scheduler_args.base_gpu_id == 0
    # Reinitialization in the same process keeps its GPU, without a second lease.
    placement.init_tokenizer_processor(initialize, worker)
    assert initialized == [2, 2]


def test_disabled_or_other_model_delegates_without_cuda_init(
    fake_worker_cuda, monkeypatch
):
    monkeypatch.delenv(placement.GPU_ENV, raising=False)
    worker = manager()
    original_args = worker.server_args
    assert placement.init_tokenizer_processor(lambda self: 42, worker) == 42
    assert worker.server_args is original_args
    monkeypatch.setenv(placement.GPU_ENV, "0,1,2,3")
    assert placement.init_tokenizer_processor(lambda self: 42, manager("other")) == 42
    assert not fake_worker_cuda and placement._lease is None


@pytest.mark.parametrize(
    "devices,available,message",
    [
        ("0,1", True, "exactly"),
        ("0,1,2,8", True, "visible"),
        ("0,1,2,3", False, "CUDA"),
    ],
)
def test_invalid_placement_fails_before_initialization(
    fake_worker_cuda, monkeypatch, devices, available, message
):
    monkeypatch.setenv(placement.GPU_ENV, devices)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: available)

    def unexpected(self):
        pytest.fail("Invalid placement must not allocate a processor or feature pool")

    with pytest.raises((ValueError, RuntimeError), match=message):
        placement.init_tokenizer_processor(unexpected, manager())
    assert placement._lease is None and not fake_worker_cuda


def test_native_image_threads_bind_worker_device_and_restore_after_failure(
    fake_worker_cuda, monkeypatch
):
    monkeypatch.setenv(placement.GPU_ENV, "3,4,5,6")
    placement.init_tokenizer_processor(lambda self: None, manager())
    events = []

    @contextmanager
    def device_scope(device):
        events.append(("enter", device))
        try:
            yield
        finally:
            events.append(("exit", device))

    monkeypatch.setattr(torch.cuda, "device", device_scope)

    def fail():
        raise ValueError("preprocessing failed")

    with ThreadPoolExecutor(max_workers=1) as pool:
        with pytest.raises(ValueError, match="preprocessing failed"):
            pool.submit(placement.on_worker_gpu, fail).result()
    assert events == [("enter", 3), ("exit", 3)]


def test_registry_preserves_image_loader_classmethod_and_detects_hook(
    registry_source, source_modules
):
    module = source_modules.module("isolated_image_loader")

    class Loader:
        @classmethod
        def load(cls, data, modality):
            return cls, data, modality

    module.Loader = Loader
    registry_source.HookRegistry.register(
        "isolated_image_loader.Loader.load",
        placement.on_worker_gpu,
        registry_source.HookType.AROUND,
    )
    registry_source.HookRegistry.apply_hooks()
    assert compat._contains_hook(Loader.load, placement.on_worker_gpu)
    assert Loader.load("image", "IMAGE") == (Loader, "image", "IMAGE")


@pytest.mark.skipif(not hasattr(os, "fork"), reason="SGLang CPU workers use fork")
def test_forked_cpu_child_does_not_keep_parent_slot_alive(tmp_path):
    placement._release_lease()
    placement._lease = placement.GpuLease("fork-service", (2,), root=tmp_path)
    ready_r, ready_w = os.pipe()
    stop_r, stop_w = os.pipe()
    pid = os.fork()
    if pid == 0:
        try:
            os.close(ready_r)
            os.close(stop_w)
            os.write(ready_w, b"1" if placement._lease is None else b"0")
            os.read(stop_r, 1)
        finally:
            os._exit(0)
    os.close(ready_w)
    os.close(stop_r)
    try:
        assert os.read(ready_r, 1) == b"1"
        # The child is still alive; closing the parent lease must free the slot.
        placement._release_lease()
        replacement = placement.GpuLease("fork-service", (2,), root=tmp_path)
        replacement.close()
    finally:
        placement._release_lease()
        os.write(stop_w, b"1")
        os.close(stop_w)
        os.close(ready_r)
        os.waitpid(pid, 0)
