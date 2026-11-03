# SPDX-FileCopyrightText: 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Isolated pack-thread bench (single core, fp32 DEST, HiFi2, approx off): per subblock, math only acquire/commits, the
pack thread runs pt::gelu_pack_subblock (GELU + pack of sb tiles to a bf16 CB). Perf-only (DEST holds zeros).
Profile: PT_VARIANTS=... ./run.sh --profile test_bench.py ; pack ns/subblock = slope over PT_NSB."""
import os
import shutil
import sys
from pathlib import Path

import pytest
import ttnn

sys.path.insert(0, str(Path(__file__).parent))
from test_e2e import VARIANTS, _collecting, _log_call  # noqa: E402

HERE = Path(__file__).parent
GEN = HERE / "gen" / "bench"
SEL = os.environ.get("PT_VARIANTS", "base,nr_nostall").split(",")
NSB = [int(v) for v in os.environ.get("PT_NSB", "32,160").split(",")]
SB = [int(v) for v in os.environ.get("PT_SB", "4,1").split(",")]
TILE = 32


def _make(name):
    d = GEN / name
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True)
    for f in ("pt_pack.hpp", "linear_gelu_gelu_sfpu.h"):
        shutil.copy(HERE / "kernels" / f, d)
    text = (HERE / "bench" / "pt_bench_compute.cpp").read_text()
    (d / "pt_bench_compute.cpp").write_text("".join(f"#define {k} {v}\n" for k, v in VARIANTS[name].items()) + text)
    return d / "pt_bench_compute.cpp"


def _core():
    return ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))])


def _mem(tiles):
    return ttnn.create_sharded_memory_config(
        shape=(tiles * TILE, TILE),
        core_grid=_core(),
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )


@pytest.mark.parametrize("sb", SB)
@pytest.mark.parametrize("nsb", NSB)
@pytest.mark.parametrize("variant", SEL)
def test_bench(device, variant, nsb, sb):
    import torch

    src = _make(variant)
    xd = ttnn.from_torch(
        torch.zeros(TILE, TILE), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=_mem(1)
    )
    out = ttnn.allocate_tensor_on_device(ttnn.Shape([4 * TILE, TILE]), ttnn.bfloat16, ttnn.TILE_LAYOUT, device, _mem(4))
    cfg = ttnn.ComputeConfigDescriptor(
        math_fidelity=ttnn.MathFidelity.HiFi2, fp32_dest_acc_en=True, math_approx_mode=False
    )
    kern = ttnn.KernelDescriptor(
        kernel_source=str(src),
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=_core(),
        compile_time_args=[],
        runtime_args=[(ttnn.CoreCoord(0, 0), [nsb, sb])],
        config=cfg,
    )
    cbs = [ttnn.cb_descriptor_from_sharded_tensor(0, xd), ttnn.cb_descriptor_from_sharded_tensor(16, out)]
    device.disable_and_clear_program_cache()
    ttnn.generic_op([xd, out], ttnn.ProgramDescriptor(kernels=[kern], semaphores=[], cbs=cbs))
    ttnn.synchronize_device(device)
    _log_call(variant, f"nsb={nsb}:sb={sb}")
    if not _collecting():
        print(f"PT_BENCH variant={variant} nsb={nsb} sb={sb}")
