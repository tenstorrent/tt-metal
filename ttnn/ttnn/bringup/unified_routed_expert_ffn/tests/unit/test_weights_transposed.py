# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""ttnn.bringup.unified_routed_expert_moe's weights_transposed option (deepseek_v41_flash_d_p P.2): gate / up / down
given as stored by nn.Linear, [N, K], read and multiplied transposed inside the op.

Kept from P.2's test_ds41_h_options.py when FIX.9 removed the DeepSeek-only h options (routing_weights,
fp8_act_quant) it also covered. Every test runs on the box's 2x2 FABRIC_2D mesh; each chip runs its own experts on
its own buffer.

* test_transposed_bit_identical: [N, K] weights with weights_transposed give the same bits as the [K, N] weights
  without it (bf16 weights; and bfp8 [N, K] of e2m1 x 2^e values, exact in bfp8 along K, vs their bf16 [K, N]).
* test_transposed_refuses_kn: [K, N] tensors passed as [N, K] do not fit x and are refused.
"""

import pytest
import torch
from loguru import logger

import ttnn

TILE = 32
MESH = (2, 2)
N_DEV = MESH[0] * MESH[1]
DEVICE_PARAMS = {
    "fabric_config": ttnn.FabricConfig.FABRIC_2D,
    "l1_small_size": 24576,
    "require_exact_physical_num_devices": True,
}
PARAMS = dict(argnames="mesh_device, device_params", argvalues=[(MESH, DEVICE_PARAMS)], indirect=True)


def _cfg(fp32=True):
    return ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=fp32,
        packer_l1_acc=True,
        dst_full_sync_en=False,
    )


def _mapper(mesh):
    return ttnn.ShardTensorToMesh(mesh, dim=0)  # device d (row-major) gets block d


def _put(mesh, t, dtype, layout):
    return ttnn.from_torch(
        t.contiguous(),
        dtype=dtype,
        layout=layout,
        device=mesh,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=_mapper(mesh),
    )


def _routing(epc, counts_local, rows):
    """Per device: counts / regions [E] over global ids (chip d holds d*epc .. +epc-1), tile-aligned regions in
    local-slot order, and the dispatched buffer's row count."""
    E = epc * N_DEV
    counts, regions = [], []
    for d in range(N_DEV):
        cnt = torch.zeros(E, dtype=torch.int64)
        reg = torch.zeros(E, dtype=torch.int64)
        off = 0
        for le in range(epc):
            n = int(counts_local[(d + le) % len(counts_local)])
            cnt[d * epc + le] = n
            reg[d * epc + le] = off
            off += (n + TILE - 1) // TILE * TILE
        assert off <= rows, (off, rows)
        counts.append(cnt)
        regions.append(reg)
    return counts, regions


class Case:
    """Random inputs on the mesh: x [rows, H] bf16 ROW_MAJOR per chip, per local expert W1 / W3 [I, H] and W2 [H, I]
    (stored, [N, K]) in `wdtype`, and the same values as bf16 [K, N] (the op's default orientation)."""

    def __init__(self, mesh, *, H, I, epc, counts_local, rows, cap, seed, wdtype=ttnn.bfloat8_b, w_kind="normal"):
        g = torch.Generator().manual_seed(seed)
        self.mesh, self.H, self.I, self.epc, self.rows, self.cap = mesh, H, I, epc, rows, cap
        self.counts, self.regions = _routing(epc, counts_local, rows)
        self.x = torch.randn(N_DEV * rows, H, generator=g).to(torch.bfloat16)
        compose = ttnn.ConcatMeshToTensor(mesh, dim=0)
        self.w = {"w1": [], "w3": [], "w2": []}  # device tensors, stored [N, K]
        self.wt = {"w1": [], "w3": [], "w2": []}  # device tensors, [K, N]
        for le in range(epc):
            for nm, (n, k) in (("w1", (I, H)), ("w3", (I, H)), ("w2", (H, I))):
                if w_kind == "e2m1":
                    # DeepSeek fp4 experts: e2m1 values times a power of two shared by 32 consecutive values along K
                    vals = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
                    code = torch.randint(0, 8, (N_DEV * n, k), generator=g)
                    sign = torch.randint(0, 2, (N_DEV * n, k), generator=g) * 2 - 1
                    e = torch.randint(-9, -5, (N_DEV * n, k // 32), generator=g).float()
                    w = vals[code] * sign * torch.exp2(e).repeat_interleave(32, dim=1)
                else:
                    w = torch.randn(N_DEV * n, k, generator=g) * k**-0.5
                t = ttnn.from_torch(w, dtype=wdtype, layout=ttnn.TILE_LAYOUT, mesh_mapper=_mapper(mesh))
                h = ttnn.to_torch(t, mesh_composer=compose).float().reshape(N_DEV, n, k)
                self.w[nm].append(ttnn.to_device(t, mesh, memory_config=ttnn.DRAM_MEMORY_CONFIG))
                # the [K, N] copy in bf16: exactly the values the [N, K] tensor holds (a bfp8 W^T would re-round)
                wt = h.transpose(1, 2).reshape(N_DEV * k, n)
                self.wt[nm].append(_put(mesh, wt, ttnn.bfloat16, ttnn.TILE_LAYOUT))
        E = epc * N_DEV
        self.tt_x = _put(mesh, self.x, ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT)
        self.tt_reg = _put(mesh, torch.stack(self.regions).to(torch.int32), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)
        self.tt_cnt = _put(mesh, torch.stack(self.counts).to(torch.int32), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)
        gidx = torch.cat([d * epc + torch.arange(epc) for d in range(N_DEV)]).to(torch.int32)
        self.tt_gidx = _put(mesh, gidx, ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)
        assert list(self.tt_reg.shape) == [1, E] and list(self.tt_gidx.shape) == [epc]

    def call(self, *, transposed=True):
        ws = self.w if transposed else self.wt
        return ttnn.bringup.unified_routed_expert_moe(
            self.tt_x,
            self.tt_reg,
            self.tt_cnt,
            self.tt_gidx,
            ws["w1"],
            ws["w3"],
            ws["w2"],
            max_dispatched_tokens_per_expert=self.cap,
            compute_kernel_config=_cfg(),
            activation=ttnn.bringup.RoutedExpertActivation.ClampedSiluGlu,
            high_precision=True,
            weights_transposed=transposed,
        )

    def regions_of(self, d):
        """[(local slot, first row, count)] of device d."""
        return [
            (le, int(self.regions[d][d * self.epc + le]), int(self.counts[d][d * self.epc + le]))
            for le in range(self.epc)
        ]

    def got(self, out):
        return [ttnn.to_torch(t).float().reshape(self.rows, self.H) for t in ttnn.get_device_tensors(out)]


@pytest.mark.timeout(1800)
@pytest.mark.parametrize(**PARAMS)
@pytest.mark.parametrize("w_kind", ["bf16", "e2m1_bfp8"])
def test_transposed_bit_identical(mesh_device, device_params, w_kind):
    if w_kind == "bf16":
        c = Case(
            mesh_device, H=512, I=320, epc=2, counts_local=[70, 33], rows=192, cap=96, seed=3, wdtype=ttnn.bfloat16
        )
    else:  # bfp8 [N, K] holds e2m1 x 2^e (shared along K) exactly; the [K, N] side is the same values in bf16
        c = Case(
            mesh_device,
            H=512,
            I=320,
            epc=2,
            counts_local=[70, 33],
            rows=192,
            cap=96,
            seed=3,
            wdtype=ttnn.bfloat8_b,
            w_kind="e2m1",
        )
    a = c.got(c.call(transposed=True))
    b = c.got(c.call(transposed=False))
    for d in range(N_DEV):
        for le, r0, n in c.regions_of(d):
            assert torch.equal(a[d][r0 : r0 + n], b[d][r0 : r0 + n]), f"{w_kind} dev {d} le {le}"


@pytest.mark.timeout(600)
@pytest.mark.parametrize(**PARAMS)
def test_transposed_refuses_kn(mesh_device, device_params):
    c = Case(mesh_device, H=512, I=320, epc=2, counts_local=[40, 9], rows=128, cap=64, seed=1)
    c.w, c.wt = c.wt, c.w  # [K, N] tensors passed as [N, K] (weights_transposed) do not fit x
    try:
        c.call(transposed=True)
    except Exception as e:  # noqa: BLE001 (the op's TT_FATAL surfaces as RuntimeError)
        logger.info(f"refused: {str(e).splitlines()[0][:160]}")
        return
    raise AssertionError("not refused: [K, N] weights as [N, K]")
