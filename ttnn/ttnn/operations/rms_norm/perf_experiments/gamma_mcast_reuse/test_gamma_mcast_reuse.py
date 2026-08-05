# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
#
# Isolated bake-off for rms_norm's GAMMA STAGING stage.  Correctness is the ONLY
# pass/fail; perf is measured with `scripts/run_safe_pytest.sh --profile ...` and read
# out of the profiler CSV / the per-stage zones — never asserted here.
#
#   delivery   the gate: every variant must land BIT-IDENTICAL bytes in cb_gamma on
#              every core (the whole staged region is dumped and compared).
#   perf       one op launch per (variant, n_cores, wt_chunk, gamma layout).
#   contention does the broadcast steal NoC0/DRAM bandwidth from the x reads?  Same
#              program, plus each core's own distinct x block, zone-split.

import pytest
import ttnn

from ttnn.operations.rms_norm.perf_experiments.gamma_mcast_reuse.gamma_mcast_reuse import (
    VARIANTS,
    gamma_stage,
    rect_for,
)

TILE = 32

CORE_COUNTS = [8, 32, 64, 110]
WT_CHUNKS = [4, 32, 72, 224]  # W = 128, 1024, 2304, 7168 — the live rms_norm profiles


def _gamma(device, wt_chunk, layout, dtype=ttnn.bfloat16):
    """gamma exactly as rms_norm sees it: a (1,1,1,W) weight vector, DRAM interleaved."""
    import torch

    W = wt_chunk * TILE
    torch.manual_seed(1234)
    t = torch.randn(W, dtype=torch.bfloat16).reshape(1, 1, 1, W)
    return ttnn.from_torch(t, dtype=dtype, layout=layout, device=device), t


def _out_sum(device, n_cores):
    import torch

    return ttnn.from_torch(
        torch.zeros(n_cores * TILE, TILE, dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )


def _out_dump(device, n_cores, wt_chunk, layout, dtype=ttnn.bfloat16):
    import torch

    if layout == ttnn.TILE_LAYOUT:
        t = torch.zeros(n_cores * TILE, wt_chunk * TILE, dtype=torch.bfloat16)
        return ttnn.from_torch(t, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    t = torch.zeros(n_cores, wt_chunk * TILE, dtype=torch.bfloat16)
    return ttnn.from_torch(t, dtype=dtype, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)


# ======================================================================================
# THE GATE — bit-exact delivery.  A pure-dataflow lever: identical bytes or nothing.
# ======================================================================================
@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("n_cores", CORE_COUNTS)
@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT], ids=["gamma_tile", "gamma_rm"])
def test_delivery(device, variant, n_cores, layout):
    import torch

    wt_chunk = 4  # small: this test measures nothing, it only proves the bytes
    g, g_torch = _gamma(device, wt_chunk, layout)
    out = _out_dump(device, n_cores, wt_chunk, layout)
    gamma_stage(g, out, variant=variant, n_cores=n_cores, mode="dump")
    got = ttnn.to_torch(out)

    if layout == ttnn.ROW_MAJOR_LAYOUT:
        want = g_torch.reshape(1, -1).to(torch.bfloat16)
        for c in range(n_cores):
            assert torch.equal(got[c : c + 1, :], want), f"core {c} got wrong gamma sticks"
    else:
        ref = got[0:TILE, :]
        # row 0 of the staged tile-row IS gamma (the only row the op's row-broadcast reads).
        assert torch.equal(ref[0:1, :], g_torch.reshape(1, -1).to(torch.bfloat16))
        for c in range(n_cores):
            assert torch.equal(got[c * TILE : (c + 1) * TILE, :], ref), f"core {c} block differs from core 0"


@pytest.mark.parametrize("variant", VARIANTS)
def test_delivery_wide(device, variant):
    """Same gate at the real operating point (110 cores, W = 1024 -> 32 tiles), so the
    bit-exactness claim is not made only at a toy width."""
    import torch

    n_cores, wt_chunk, layout = 110, 32, ttnn.TILE_LAYOUT
    g, g_torch = _gamma(device, wt_chunk, layout)
    out = _out_dump(device, n_cores, wt_chunk, layout)
    gamma_stage(g, out, variant=variant, n_cores=n_cores, mode="dump")
    got = ttnn.to_torch(out)
    ref = got[0:TILE, :]
    assert torch.equal(ref[0:1, :], g_torch.reshape(1, -1).to(torch.bfloat16))
    for c in range(n_cores):
        assert torch.equal(got[c * TILE : (c + 1) * TILE, :], ref), f"core {c} block differs from core 0"


@pytest.mark.parametrize("variant", VARIANTS)
def test_delivery_fp32_gamma(device, variant):
    """gamma_dtype independent of the activation dtype (the op supports that): fp32 gamma."""
    import torch

    n_cores, wt_chunk, layout = 32, 4, ttnn.TILE_LAYOUT
    W = wt_chunk * TILE
    torch.manual_seed(7)
    t = torch.randn(W, dtype=torch.float32).reshape(1, 1, 1, W)
    g = ttnn.from_torch(t, dtype=ttnn.float32, layout=layout, device=device)
    out = ttnn.from_torch(
        torch.zeros(n_cores * TILE, W, dtype=torch.float32),
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )
    gamma_stage(g, out, variant=variant, n_cores=n_cores, mode="dump")
    got = ttnn.to_torch(out)
    ref = got[0:TILE, :]
    assert torch.equal(ref[0:1, :], t.reshape(1, -1))
    for c in range(n_cores):
        assert torch.equal(got[c * TILE : (c + 1) * TILE, :], ref), f"core {c} block differs from core 0"


# ======================================================================================
# PERF — one launch per cell.  Every launch still checksums (baseline == candidate is
# implied by the delivery gate; the checksum catches gross corruption in the perf shape).
# ======================================================================================
@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("wt_chunk", WT_CHUNKS)
@pytest.mark.parametrize("n_cores", CORE_COUNTS)
def test_perf_tile(device, variant, n_cores, wt_chunk):
    import torch

    g, g_torch = _gamma(device, wt_chunk, ttnn.TILE_LAYOUT)
    out = _out_sum(device, n_cores)
    gamma_stage(g, out, variant=variant, n_cores=n_cores, mode="sum")
    got = ttnn.to_torch(out)
    # gamma's TILE padding is zero, so summing the tile-row collapses to gamma's row 0.
    want = g_torch.reshape(-1, TILE).to(torch.float32).sum(dim=0)
    for c in range(n_cores):
        assert torch.allclose(got[c * TILE, :].to(torch.float32), want, atol=0.05, rtol=0.02), f"core {c} checksum"


@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("wt_chunk", [4, 32, 224])
@pytest.mark.parametrize("n_cores", [110])
def test_perf_rm(device, variant, n_cores, wt_chunk):
    g, _ = _gamma(device, wt_chunk, ttnn.ROW_MAJOR_LAYOUT)
    out = _out_sum(device, n_cores)
    gamma_stage(g, out, variant=variant, n_cores=n_cores, mode="sum")
    ttnn.to_torch(out)  # drain; the byte-level gate is test_delivery


# ======================================================================================
# CONTENTION — gamma stage + each core's OWN distinct x block, in the op's order.
# Zone-split (`reader_read_gamma` vs `reader_read_x`) answers whether the broadcast
# steals bandwidth from the x reads, which are already at the DRAM roofline.
# ======================================================================================
@pytest.mark.parametrize(
    "variant", ["baseline", "mcast_1inj_noc0", "mcast_1inj_noc1", "mcast_perrow_noc0", "mcast_percol_noc0"]
)
@pytest.mark.parametrize("rep", [0, 1], ids=["rep0", "rep1"])
def test_contention(device, variant, rep):
    """Mirror of the primary target (1,1,8192,1024): 110 cores, Wt = 32, ~3 tile-rows of
    distinct x per core (256 tile-rows / 110 cores, rounded up = the busiest core's share)."""
    import torch

    n_cores, wt_chunk, x_rows = 110, 32, 3
    g, _ = _gamma(device, wt_chunk, ttnn.TILE_LAYOUT)
    x = ttnn.from_torch(
        torch.randn(n_cores * x_rows * TILE, wt_chunk * TILE, dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )
    out = _out_sum(device, n_cores)
    gamma_stage(g, out, variant=variant, n_cores=n_cores, mode="sum", x=x, x_rows=x_rows)
    ttnn.to_torch(out)


@pytest.mark.parametrize("variant", ["baseline", "mcast_1inj_noc0", "mcast_percol_noc0"])
@pytest.mark.parametrize("n_cores", [2, 4, 8, 12, 16, 20, 24, 48])
@pytest.mark.parametrize("wt_chunk", [32])
def test_perf_crossover(device, variant, n_cores, wt_chunk):
    """Where does the broadcast start paying?  The sweep above shows 8 cores LOSING and 32
    cores winning; this pins the crossover (and covers a group of ONE / TWO cores)."""
    g, _ = _gamma(device, wt_chunk, ttnn.TILE_LAYOUT)
    out = _out_sum(device, n_cores)
    gamma_stage(g, out, variant=variant, n_cores=n_cores, mode="sum")
    ttnn.to_torch(out)


@pytest.mark.parametrize("variant", ["baseline", "mcast_1inj_noc0"])
@pytest.mark.parametrize("x_rows", [0, 1, 3])
def test_x_presence(device, variant, x_rows):
    """Why is the gamma stage 3x dearer once a big x tensor is in the program?  Same
    geometry, x ALLOCATED in all three, but read 0 / 1 / 3 tile-rows per core.  x_rows=0
    separates "x exists in DRAM" from "x is being read"."""
    import torch

    n_cores, wt_chunk = 110, 32
    g, _ = _gamma(device, wt_chunk, ttnn.TILE_LAYOUT)
    x = ttnn.from_torch(
        torch.randn(n_cores * 3 * TILE, wt_chunk * TILE, dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )
    out = _out_sum(device, n_cores)
    gamma_stage(g, out, variant=variant, n_cores=n_cores, mode="sum", x=x, x_rows=x_rows)
    ttnn.to_torch(out)


def test_grid(device):
    grid = device.compute_with_storage_grid_size()
    print(f"\nGRID: {grid.x} x {grid.y} = {grid.x * grid.y}")
    for n in CORE_COUNTS:
        print(f"  {n} cores -> rect {rect_for(device, n)}")
