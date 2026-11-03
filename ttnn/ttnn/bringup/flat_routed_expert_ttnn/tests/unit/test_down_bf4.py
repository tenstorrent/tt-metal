# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""ttnn.bringup.flat_routed_expert's down_weights_bf4 (deepseek_v41_flash_d_p P.27, EXPERTS_IMPL=bfp8_down4):
FlatRoutedExpert(wdtype="bf8d4") lays out gate/up bfp8 tiles and down (and reader-tail down) bfp4 tiles; the op takes
the down format from the down tensors' dtype (flat_routed_expert_plan(down_weights_bf4=True) for the plan).

* test_bf8d4_bit_identical: weights_nk, DeepSeek-V4.1's per-chip shapes at a small count (emb 5120, intermediate 2304),
  w1 / w3 e2m1 x 2^e per 32 along K (exact in bfp8), w2 integers in [-7, 7] x 2^e per 32 along K (exact in bfp4 and
  in bfp8): the bf8d4 op (bfp4 down tiles) gives the same bits as the bf8 op (bfp8 down tiles) on the same values,
  and both match a float64 reference within the bfp8 x / h tiles' rounding.
* test_plan_down_bf4: the plan with down_weights_bf4 packs the weights where the bf8 plan does (same weight layout,
  down cores and coordinators) and the down regions are bfp4-sized; it refuses bfp4 gate/up weights.

Runs on MESH_PARAMS (2x2 FABRIC_2D; BRINGUP_TEST_MESH=8x4 on the Galaxy); each chip its own experts.
"""

import pytest
import torch
from loguru import logger

import ttnn
from ttnn.bringup._testing.mesh import MESH_PARAMS
from ttnn.bringup.flat_routed_expert_ttnn.flat_expert import FlatRoutedExpert, _weight_packing

TILE = 32
HIFI4 = 4
LIM = 10.0
H, I, EPC = 5120, 2304, 2


def _put(mesh, t, dtype, layout):
    return ttnn.from_torch(
        t.contiguous(), dtype=dtype, layout=layout, device=mesh, memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=0),
    )  # fmt: skip


def _expert64(x, w1, w3, w2):
    g, u = x.double() @ w1.double().T, x.double() @ w3.double().T
    h = torch.nn.functional.silu(g.clamp(max=LIM)) * u.clamp(-LIM, LIM)
    return (h @ w2.double().T).float()


def _weights(n_dev, seed):
    g = torch.Generator().manual_seed(seed)
    e2m1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
    out = []
    for _ in range(n_dev):
        dev = []
        for _ in range(EPC):
            mats = []
            for n, k, kind in ((I, H, "e2m1"), (I, H, "e2m1"), (H, I, "int7")):
                e = torch.randint(-9, -5, (n, k // 32), generator=g).float().repeat_interleave(32, dim=1)
                if kind == "e2m1":
                    v = e2m1[torch.randint(0, 8, (n, k), generator=g)] * (
                        torch.randint(0, 2, (n, k), generator=g) * 2 - 1
                    )
                else:  # (bfp4-exact: |q| <= 7 units of 2^e, one e per 32 along K)
                    v = torch.randint(-7, 8, (n, k), generator=g).float()
                mats.append(v * torch.exp2(e))
            dev.append(tuple(mats))
        out.append(dev)
    return out


@pytest.mark.timeout(3600)
@MESH_PARAMS
def test_bf8d4_bit_identical(mesh_device, device_params):
    n_dev = mesh_device.get_num_devices()
    E = EPC * n_dev
    rows, cap = 256, 256
    counts_local = [150, 33]
    w = _weights(n_dev, seed=27)
    g = torch.Generator().manual_seed(28)
    x = torch.randn(n_dev * rows, H, generator=g).to(torch.bfloat16)
    cnt = torch.zeros(n_dev, E, dtype=torch.int32)
    reg = torch.zeros(n_dev, E, dtype=torch.int32)
    for d in range(n_dev):
        off = 0
        for le in range(EPC):
            n = counts_local[(d + le) % len(counts_local)]
            cnt[d, d * EPC + le], reg[d, d * EPC + le] = n, off
            off += (n + TILE - 1) // TILE * TILE
    tt_x = _put(mesh_device, x, ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT)
    tt_c = _put(mesh_device, cnt, ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)
    tt_r = _put(mesh_device, reg, ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)
    gids = [[d * EPC + le for le in range(EPC)] for d in range(n_dev)]
    ys = {}
    for wd in ("bf8", "bf8d4"):
        fe = FlatRoutedExpert(
            mesh_device, w, m=cap, H=H, I=I, gids=gids, n_global=E, wdtype=wd, act="clamped_silu", pin=1,
            weights_nk=True, check_exact=True,
        )  # fmt: skip
        assert fe.w_gu.dtype == ttnn.bfloat8_b
        assert fe.w_d.dtype == (ttnn.bfloat4_b if wd == "bf8d4" else ttnn.bfloat8_b)
        out = fe(tt_x, tt_c, tt_r, y_bf16=True, down_fp32=True, math_fidelity=HIFI4)
        ys[wd] = [ttnn.to_torch(t).float().reshape(rows, H) for t in ttnn.get_device_tensors(out)]
        for t in (fe.w_gu, fe.w_d, fe.w_rd, out):
            if t is not None:
                ttnn.deallocate(t)
    worst = 0.0
    for d in range(n_dev):
        off = 0
        for le in range(EPC):
            n = int(cnt[d, d * EPC + le])
            r0 = int(reg[d, d * EPC + le])
            if not n:
                continue
            a, b = ys["bf8d4"][d][r0 : r0 + n], ys["bf8"][d][r0 : r0 + n]
            assert torch.equal(a, b), f"dev {d} le {le}: bfp4 down tiles != bfp8 down tiles on bfp4-exact weights"
            ref = _expert64(x[d * rows + r0 : d * rows + r0 + n], *w[d][le])
            worst = max(worst, float((a - ref).norm() / ref.norm()))
    logger.info(f"bf8d4 == bf8 bit for bit on every chip; worst rel vs float64 {worst:.4g} (bfp8 x / h tiles)")
    assert worst < 0.05, worst  # (bfp8 x and h tiles round ~2^-8 per value: rel ~1e-2; a layout bug is O(1))


@pytest.mark.timeout(600)
@MESH_PARAMS
def test_plan_down_bf4(mesh_device, device_params, expect_error):
    plan = ttnn._ttnn.operations.bringup.flat_routed_expert_plan
    a = dict(plan(mesh_device, H, I, EPC, EPC * mesh_device.get_num_devices(), 256, weights_bf8=True))
    b = dict(
        plan(mesh_device, H, I, EPC, EPC * mesh_device.get_num_devices(), 256, weights_bf8=True, down_weights_bf4=True)
    )
    assert _weight_packing(a, nk=True) == _weight_packing(b, nk=True)
    for k in ("down", "coords", "nd", "pcds"):
        assert a[k] == b[k], k
    with expect_error(RuntimeError, "down_weights_bf4 needs bfp8 gate/up weights"):
        plan(mesh_device, H, I, EPC, EPC * mesh_device.get_num_devices(), 256, weights_bf8=False, down_weights_bf4=True)
