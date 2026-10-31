# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""PERF EXPERIMENT direct_exchange: bit-identity of this dir's op copy (every exchange schedule: relay / direct /
mcast) against the REAL op, per device, on the perf shapes and the regimes the transport touches (Ring / Linear,
1 / 2 links, cluster_axis 0 / 1, fp32 / bf8b, bounded landing ring). Each call is run twice on the copy (re-arm of the
persistent counters across calls).

    scripts/run_safe_pytest.sh --no-precompile --run-all \\
        ttnn/ttnn/operations/distributed_rms_norm/perf_experiments/direct_exchange/test_direct.py
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
import ttnn

from eval.golden_tests.distributed_rms_norm.conftest import _fabric_mesh_cache  # noqa: F401 (fixture)
import ttnn.operations.distributed_rms_norm.distributed_rms_norm_program_descriptor as real_pd
from ttnn.operations.distributed_rms_norm import distributed_rms_norm as real_op
import direct_exchange.distributed_rms_norm_program_descriptor as pd
from direct_exchange import distributed_rms_norm as copy_op

TORCH_DTYPE = {ttnn.float32: torch.float32, ttnn.bfloat16: torch.bfloat16, ttnn.bfloat8_b: torch.bfloat16}
import os

MODES = (os.environ.get("DRMS_TEST_MODES", "") or "relay,direct,mcast").split(",")


def _to_mesh(stacked, mesh, dtype):
    rows, cols = stacked.shape[0], stacked.shape[1]
    g = torch.cat([torch.cat(list(stacked[r]), dim=1) for r in range(rows)], dim=0)
    return ttnn.from_torch(
        g,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=mesh,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh, dims=(0, 1), mesh_shape=tuple(mesh.shape)),
    )


def _cfg(fp32=True):
    cfg = ttnn.ComputeConfigDescriptor()
    cfg.math_fidelity = ttnn.MathFidelity.HiFi4
    cfg.fp32_dest_acc_en = fp32
    cfg.math_approx_mode = False
    return cfg


def _outs(t, mesh):
    return [ttnn.to_torch(d) for d in ttnn.get_device_tensors(t)]


def _pcc(a, b):
    a, b = a.flatten().double(), b.flatten().double()
    a, b = a - a.mean(), b - b.mean()
    return float((a * b).sum() / (a.norm() * b.norm()))


# (mesh_shape, cluster_axis, topology, num_links, shape, dtype, bounded)
R, L = ttnn.Topology.Ring, ttnn.Topology.Linear
CASES = [
    ((1, 4), 1, R, 2, (1, 1, 640, 1792), ttnn.bfloat16, False),
    ((1, 4), 1, R, 2, (1, 1, 32, 1792), ttnn.bfloat16, False),
    ((1, 4), 1, R, 2, (1, 1, 2048, 1280), ttnn.bfloat16, False),
    ((1, 4), 1, R, 1, (1, 1, 640, 1792), ttnn.bfloat16, False),
    ((1, 4), 1, R, 2, (1, 1, 640, 1792), ttnn.float32, False),
    ((1, 4), 1, R, 2, (1, 1, 640, 1792), ttnn.bfloat8_b, False),
    ((1, 4), 1, R, 2, (1, 1, 64, 1000), ttnn.bfloat16, False),
    ((1, 4), 1, R, 2, (4, 1, 4096, 1792), ttnn.bfloat16, True),
    ((1, 4), 1, R, 1, (4, 1, 4096, 1792), ttnn.bfloat16, True),
    ((1, 4), 1, R, 2, (1, 1, 16384, 512), ttnn.float32, True),
    ((1, 4), 1, L, 2, (1, 1, 640, 1792), ttnn.bfloat16, False),
    ((1, 4), 1, L, 1, (1, 1, 32, 1792), ttnn.bfloat16, False),
    ((1, 4), 1, L, 2, (4, 1, 4096, 1792), ttnn.bfloat16, True),
    ((4, 1), 0, R, 2, (1, 1, 640, 1792), ttnn.bfloat16, False),
    ((4, 1), 0, L, 1, (1, 1, 640, 1792), ttnn.bfloat16, False),
    ((2, 2), 1, L, 1, (1, 1, 640, 1792), ttnn.bfloat16, False),
    ((2, 2), 0, R, 1, (1, 1, 640, 1792), ttnn.bfloat16, False),
]


def _id(c):
    ms, ax, topo, nl, shape, dt, bounded = c
    return (
        f"{ms[0]}x{ms[1]}-ax{ax}-{'ring' if topo == R else 'line'}-L{nl}-{'x'.join(map(str, shape))}-"
        f"{str(dt).split('.')[-1]}{'-bounded' if bounded else ''}"
    )


# sorted by mesh shape so each mesh opens once
@pytest.mark.parametrize("case", sorted(CASES, key=lambda c: c[0]), ids=_id)
def test_bit_identical(case, _fabric_mesh_cache, monkeypatch):  # noqa: F811
    ms, ax, topo, nl, shape, dt, bounded = case
    mesh = _fabric_mesh_cache.get(ms)
    if bounded:
        for m in (real_pd, pd):
            monkeypatch.setattr(m, "LANDING_FULL_MAX_BYTES", 0)
            monkeypatch.setattr(m, "LANDING_RING_WAVES", 2)
            monkeypatch.setattr(m, "MIN_BLOCKS_PER_CORE", 8 if shape[0] == 4 else 4)
    torch.manual_seed(7)
    rows, cols = ms
    x = torch.randn((rows, cols, *shape), dtype=torch.float32).to(TORCH_DTYPE[dt])
    gdt = ttnn.bfloat16 if dt == ttnn.bfloat8_b else dt
    gm = torch.randn((rows, cols, shape[-1]), dtype=torch.float32).to(TORCH_DTYPE[gdt])
    tx = _to_mesh(x, mesh, dt)
    tg = _to_mesh(gm.reshape(rows, cols, 1, 1, 1, shape[-1]), mesh, gdt)
    kw = dict(cluster_axis=ax, gamma=tg, epsilon=1e-6, topology=topo, num_links=nl, compute_kernel_config=_cfg())
    ref = _outs(real_op(tx, **kw), mesh)
    if bounded:
        assert real_pd.LAST_EXCHANGE_LAYOUT["landing_waves"] < real_pd.LAST_EXCHANGE_LAYOUT["num_waves"]
    for mode in MODES:
        monkeypatch.setattr(pd, "EXCHANGE_MODE", mode)
        for rep in range(2):
            try:
                got = _outs(copy_op(tx, **kw), mesh)
            except NotImplementedError as exc:
                print(f"{_id(case)} {mode}: INEXPRESSIBLE: {exc}")
                break
            if bounded:
                assert pd.LAST_EXCHANGE_LAYOUT["landing_waves"] < pd.LAST_EXCHANGE_LAYOUT["num_waves"]
            for i, (a, b) in enumerate(zip(ref, got)):
                assert torch.equal(a, b), f"{mode} call {rep}: device {i} differs (pcc {_pcc(a.float(), b.float())})"
        else:
            print(f"{_id(case)} {mode}: BIT-IDENTICAL x2")
