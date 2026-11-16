"""Run on the supported CUDA SGLang image; never substitute CPU results."""

from __future__ import annotations

import os
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch

from kimi_k3_video.config import VideoConfig
from kimi_k3_video.preprocess import preprocess_visual_items, resize_config
from kimi_k3_video.sampling import VisualItem
from kimi_k3_video.vision import position_embeddings

pytestmark = [
    pytest.mark.cuda,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="Real CUDA is required"),
]


@pytest.mark.parametrize("t", [1, 2, 3, 4])
def test_cuda_temporal_preprocess_positions_pool_and_projector(config_values, t):
    from sglang.srt.models.kimi_k3_vl import (
        KimiK3MultiModalProjector,
        Learnable2DInterpPosEmbDividedFixed,
    )
    from sglang.srt.models.kimi_vl_moonvit import tpool_patch_merger
    from sglang.srt.multimodal.processors.kimi_k25 import navit_resize_config

    config = VideoConfig.from_mapping(config_values)
    frames = tuple(
        torch.full((3, 28, 28), index * 40, dtype=torch.uint8) for index in range(t)
    )
    item = VisualItem("video", 0, frames, 28, 28, 16)
    features, grids = preprocess_visual_items(
        [item],
        config,
        [resize_config(item, config, navit_resize_config)],
        device="cuda",
    )
    assert features[0].is_cuda and grids == [(t, 2, 2)]
    for index in range(t):
        torch.testing.assert_close(
            features[0][index * 4 : (index + 1) * 4],
            torch.full((4, 3, 14, 14), index * 40 / 127.5 - 1, device="cuda"),
        )
    embed = Learnable2DInterpPosEmbDividedFixed(2, 2, 4, 8).cuda()
    # Explicitly leave time buffer on the host to cover the device correction.
    embed.time_weight = embed.time_weight.cpu()
    positions = position_embeddings(
        type(embed).position_embeddings, embed, torch.tensor(grids)
    )
    assert positions.is_cuda and positions.shape == (t * 4, 8)
    merged = tpool_patch_merger(
        torch.ones(t * 4, 8, device="cuda"), torch.tensor(grids)
    )
    assert merged[0].shape == (1, 4, 8)
    projector = KimiK3MultiModalProjector(
        SimpleNamespace(
            mm_hidden_size=8,
            merge_kernel_size=(2, 2),
            text_hidden_size=16,
            projector_ln_eps=1e-5,
        )
    ).cuda()
    assert projector(merged).shape == (1, 16)


def test_cuda_video_preprocessing_bounds_source_resolution_peak(config_values):
    from sglang.srt.multimodal.processors.kimi_k25 import navit_resize_config

    config = VideoConfig.from_mapping(config_values)
    # All-at-once float conversion alone needs 384 MiB for these 128 frames.
    # Bounded batches must preserve the output with a much smaller live peak.
    frames = torch.full((128, 3, 512, 512), 127, dtype=torch.uint8)
    items = [
        VisualItem("video", 0, tuple(chunk.unbind()), 512, 512, 16)
        for chunk in frames.split(4)
    ]
    configs = [resize_config(item, config, navit_resize_config) for item in items]
    torch.cuda.synchronize()
    baseline = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    features, grids = preprocess_visual_items(items, config, configs, device="cuda")
    torch.cuda.synchronize()
    assert grids == [(4, 4, 4)] * 32
    assert len(features) == 32
    for feature in features:
        torch.testing.assert_close(feature, torch.full_like(feature, 127 / 127.5 - 1))
    peak = torch.cuda.max_memory_allocated() - baseline
    print(f"preprocess_peak_allocated={peak} bytes ({peak / 1024**2:.3f} MiB)")
    assert peak < 128 * 1024 * 1024


def test_cuda_four_tokenizer_workers_place_images_video_and_pools(
    config_values, tmp_path
):
    if torch.cuda.device_count() < 4:
        pytest.skip("Four visible CUDA devices are required")
    # Use spawned processes just like the HTTP workers, with the real plugin
    # loader and native image/video operators. No model weights are loaded.
    script = tmp_path / "four_gpu_workers.py"
    script.write_text(
        """
import multiprocessing as mp
import os
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

def worker(queue, stop):
    import torch
    import kimi_k3_video.http_app
    from PIL import Image
    from kimi_k3_video import placement
    from kimi_k3_video.config import VideoConfig
    from kimi_k3_video.preprocess import preprocess_visual_items
    from kimi_k3_video.sampling import VisualItem
    from sglang.srt.managers.schedule_batch import MultimodalDataItem, Modality
    from sglang.srt.multimodal.processors.kimi_k3 import (
        KimiK3GPUProcessorWrapper, KimiK3ImageProcessor,
    )
    from sglang.srt.multimodal.transport.cuda_ipc import MmItemMemoryPool

    values = __CONFIG__
    manager = SimpleNamespace(
        model_config=SimpleNamespace(hf_config=SimpleNamespace(model_type='kimi_k31')),
        server_args=SimpleNamespace(base_gpu_id=0, tokenizer_worker_num=4,
                                    host='test-only', port=os.getppid()),
    )
    def initialize(owner):
        gpu = owner.server_args.base_gpu_id
        torch.cuda.set_per_process_memory_fraction(0.005, gpu)
        owner.pool = MmItemMemoryPool(1024 * 1024, 60, gpu, consumer_count=1)
    placement.init_tokenizer_processor(initialize, manager)
    gpu = manager.server_args.base_gpu_id
    try:
        wrapper = KimiK3GPUProcessorWrapper(
            SimpleNamespace(image_processor=None, tokenizer=None, media_processor=None),
            '<|media_pad|>', 99, SimpleNamespace(**values),
        )
        native = KimiK3ImageProcessor.__new__(KimiK3ImageProcessor)
        native._processor = wrapper
        native.mm_feature_transport = 'cuda_ipc'
        def preprocess():
            # A new thread defaults to cuda:0. Native image code uses bare cuda;
            # the plugin hook must bind the worker GPU in this thread.
            with Image.new('RGB', (28, 28)) as image:
                image_features = wrapper.prepare_image_features([image])[0]
                artifacts = native.prepare_artifact_batch([SimpleNamespace(
                    media=image, content_digest='test-image',
                    artifact_key='sha256:' + 'a' * 64,
                )])
            # Feature hashing calls bare .cuda() upstream. It must not copy
            # nonzero-worker tensors to the new thread's default GPU 0.
            mm_item = MultimodalDataItem(modality=Modality.IMAGE,
                                         feature=image_features[0])
            native._precompute_hashes_before_cpu_transfer([mm_item])
            assert isinstance(mm_item.hash, int)
            assert artifacts[0].feature.device.index == gpu
            if gpu != 0:
                assert torch.cuda.memory_reserved(0) == 0
            item = VisualItem('video', 0, (torch.zeros(3, 28, 28, dtype=torch.uint8),),
                              28, 28, 16)
            config = VideoConfig.from_mapping(values)
            resize = dict(new_width=28, new_height=28, pad_width=0, pad_height=0,
                          num_tokens=1)
            video_features, _ = preprocess_visual_items([item], config, [resize],
                                                        device=f'cuda:{gpu}')
            return image_features[0].device.index, video_features[0].device.index
        with ThreadPoolExecutor(max_workers=1) as executor:
            image_gpu, video_gpu = executor.submit(preprocess).result()
        queue.put((gpu, manager.pool.memory_pool.device.index, image_gpu, video_gpu))
        if not stop.wait(90):
            raise TimeoutError('Parent did not release workers')
    finally:
        manager.pool.shutdown()

if __name__ == '__main__':
    os.environ['KIMI_K3_TOKENIZER_GPUS'] = '0,1,2,3'
    os.environ['SGLANG_PLUGINS'] = 'kimi_k3_video'
    ctx = mp.get_context('spawn')
    queue, stop = ctx.Queue(), ctx.Event()
    workers = [ctx.Process(target=worker, args=(queue, stop)) for _ in range(4)]
    try:
        for process in workers:
            process.start()
        results = [queue.get(timeout=120) for _ in workers]
        assert {row[0] for row in results} == {0, 1, 2, 3}, results
        assert all(len(set(row)) == 1 for row in results), results
        print('Four worker GPU placements:', sorted(results), flush=True)
    finally:
        stop.set()
        for process in workers:
            if process.pid is not None:
                process.join(timeout=15)
                if process.is_alive():
                    process.kill()
                    process.join(timeout=15)
    assert all(process.exitcode == 0 for process in workers)
""".replace("__CONFIG__", repr(config_values))
    )
    result = subprocess.run(
        [sys.executable, str(script)],
        env={**os.environ, "PYTHONPATH": os.pathsep.join(sys.path)},
        capture_output=True,
        text=True,
        timeout=200,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    print(result.stdout)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__]))
