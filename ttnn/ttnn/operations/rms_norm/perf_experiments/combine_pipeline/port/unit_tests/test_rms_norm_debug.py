# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""rms_norm debug / regression tests — DO NOT DELETE.

Each shape pins a kernel path the acceptance suite did not exercise when the first build went in
(found by probes, see the [ttnn-implementer] FAIL commit):

  * multi-row RM blocks          -> back-to-back tilize calls (reconfig after fast_tilize_init zeroed rows 1..31)
  * multi-chunk RM tile-rows     -> chunks_this > 1, per-chunk normalize + untilize, chunk-major writer
  * TILE core_w % out_unit != 0  -> writer batch straddling the cb_out ring end
  * TILE W tail on tile-row >= 1 -> zero-fill offset relative to the advanced CB write pointer (it hit cb_gamma)
  * multi-block cores (x_block_depth 2 ping-pong) and the k > 1 combine lookahead
  * L1 budget at the fit boundary (verifier): the CB budget was get_max_worker_l1_unreserved_size(),
    which overshoots the allocator's CB region by the kernel-config ring buffer (~69 KB on BH), so a
    footprint in that gap threw "circular buffers grow to N B which is beyond max L1 size"; and
    L1-resident input/output tensors shrink the region the CBs may use
"""

import pytest
import ttnn

from ttnn.operations.rms_norm import rms_norm


def _check(
    device, shape, layout, *, gamma=True, dtype=ttnn.bfloat16, pcc_min=0.999, memory_config=ttnn.DRAM_MEMORY_CONFIG
):
    import torch  # function-local: the ttnn tree forbids global torch imports

    torch.manual_seed(0)
    tdt = torch.bfloat16 if dtype == ttnn.bfloat16 else torch.float32
    x = torch.randn(shape).to(tdt)
    tx = ttnn.from_torch(x, dtype=dtype, layout=layout, device=device, memory_config=memory_config)
    g = tg = None
    if gamma:
        g = torch.randn(shape[-1]).to(tdt)
        tg = ttnn.from_torch(g.reshape(1, 1, 1, -1), dtype=dtype, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    y = ttnn.to_torch(rms_norm(tx, gamma=tg)).float()
    xf = x.float()
    ref = xf / torch.sqrt((xf * xf).mean(-1, keepdim=True) + 1e-6)
    if g is not None:
        ref = ref * g.float()
    assert torch.isfinite(y).all()
    pcc = torch.corrcoef(torch.stack([y.flatten().double(), ref.flatten().double()]))[0, 1].item()
    assert pcc >= pcc_min, f"PCC {pcc} for {shape} {layout}"


@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT], ids=["tile", "row_major"])
@pytest.mark.parametrize(
    "shape",
    [
        (1, 1, 6400, 64),  # 2 tile-rows per core, 1 RM chunk
        (1, 1, 4160, 1280),  # 1 tile-row per core, 2 RM chunks of 20 tiles; TILE: 40 % 16 != 0
        (1, 1, 6400, 1280),  # 2 tile-rows x 2 chunks
        (1, 1, 6400, 1000),  # W tail on every tile-row of a 2-row block
        (2, 3, 1000, 4100),  # H tail per image, W tail, 2 chunks, multi-row blocks
    ],
    ids=lambda s: "x".join(map(str, s)),
)
def test_rms_norm_debug_paths(device, shape, layout):
    _check(device, shape, layout)


@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT], ids=["tile", "row_major"])
@pytest.mark.parametrize(
    "shape",
    [
        (1, 1, 8192, 8192),  # k = 1 multi-block at the fit boundary
        (1, 1, 6144, 12288),  # k = 2 by fit, >= 3 blocks per group: ping-pong + combine lookahead
        (1, 1, 2048, 16416),  # k by fit with a ragged last member (W tail)
    ],
    ids=lambda s: "x".join(map(str, s)),
)
def test_rms_norm_debug_multi_block(device, shape, layout):
    _check(device, shape, layout, pcc_min=0.995)


@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT], ids=["tile", "row_major"])
@pytest.mark.parametrize(
    "memory_config", [ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG], ids=["dram", "l1_interleaved"]
)
def test_rms_norm_debug_l1_budget_boundary(device, layout, memory_config):
    # 17 tile-rows x 344 W tiles (Llama FFN width): k = 1 by occupancy; bf16 footprint ~1.48 MB per core
    # is above the allocator's CB region (~1.46 MB on BH) but was below the overshooting budget.
    _check(device, (1, 1, 544, 11008), layout, memory_config=memory_config, pcc_min=0.995)
