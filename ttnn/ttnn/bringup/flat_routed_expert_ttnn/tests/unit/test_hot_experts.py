# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Hot experts through flat_routed_expert_with_program + the weighted combine (DeepSeek-V4.1 K.1, 2026-10-27): routings
where a few experts get (almost) every token of their column, built as the model wires the call (tests/weighted_case.py)
at the model's per-chip shape (8x4: 384 experts, 12 a chip, 640 home tokens a chip, top 6; the 2x2 seed: 96 experts, 24
a chip, 1024 tokens), one random expert (bfp4-exact values) in every slot, the call's arena and the words below it
filled with random bits first (L1 as a model leaves it).

Why: the K.1 stand-in decode hung on its first MoE call, whose 5120-token chunk had 128 real tokens and PAD rows that
all route to the same 6 experts (54, 99, 153, 318, 355, 379 on 8x4: ~4900 of a column's 5120 tokens each). The routing
was not the cause (this file passes it on a correctly opened mesh): that mesh was opened without the model's 10880 B
fabric payload (combine_rf tests/unit/test_fabric_payload.py; deepseek_v41_flash_d_p DS41-127). Kept as the regression
test that every routing completes and sums right, up to every token of a column on one expert and a chip at rows_cap.
Checked against the CPU model: with one expert in every slot, chip (r, c)'s combine output for its home token t is
(sum of t's routing weights on column c's experts) x ffn(x_t).

  - test_hot_experts[<case>]: synthetic routings (pad6: the model's pattern; one_chip: every token on 6 experts of
    one chip, its rows at rows_cap; full_column: every token of every column on the same expert of that column);
  - test_hot_capture: a model's captured routing (HOT_CAPTURE=<moe dump .pt: idx / wts per chip, mesh>; skipped
    without it).

    BRINGUP_TEST_MESH=8x4 scripts/run_safe_pytest.sh --run-all \\
        ttnn/ttnn/bringup/flat_routed_expert_ttnn/tests/unit/test_hot_experts.py
"""

import importlib.util
import os
from pathlib import Path

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc
from ttnn.bringup.combine_rf.sparse_combine_rf import L1_MAX_WRITERS
from ttnn.bringup.flat_routed_expert_ttnn.flat_expert import FlatRoutedExpert

_TESTS = Path(__file__).resolve().parents[1]


def _load(name):
    spec = importlib.util.spec_from_file_location(
        f"bringup_flat_routed_expert_ttnn_tests_{name}", _TESTS / f"{name}.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_MESH = tuple(int(v) for v in os.environ.get("BRINGUP_TEST_MESH", "2x2").split("x"))
H, I, K = 5120, 2304, 6


def _device_params():
    rc = ttnn._ttnn.fabric.FabricRouterConfig()
    rc.max_packet_payload_size_bytes = 10880  # (the weighted combine's fabric packets, as the model's mesh)
    return {
        "fabric_config": ttnn.FabricConfig.FABRIC_2D,
        "l1_small_size": 24576,
        "fabric_router_config": rc,
    }


pytestmark = pytest.mark.skipif(not ttnn.device.is_blackhole(), reason="the flat routed expert is Blackhole-only")


class _Setup:
    """The op (FlatRoutedExpert) and the model's wiring (WeightedCase) on this mesh, built per test (the mesh fixture is per test)."""

    def __init__(self, mesh_device):
        wcase = _load("weighted_case")
        self.ref = _load("reference")
        R, C = tuple(mesh_device.shape)
        self.R, self.C = R, C
        self.E = 384 if R * C == 32 else 96  # (the model's experts on 8x4; the 2x2 seed's scaled 96)
        self.EPC = self.E // (R * C)
        self.S = min(5120 // R, 1024)  # (the 5120-token chunk's home tokens a chip, one pass)
        self.Eg = R * self.EPC
        g = torch.Generator().manual_seed(5)
        w = lambda *shape: self.ref.block_round(  # noqa: E731
            torch.randn(*shape, generator=g) * 0.02, 3, ties="even", clamp=True
        )
        self.w = (w(I, H), w(I, H), w(H, I))  # (one expert in every slot: [N, K] as nn.Linear stores them)
        self.fe = FlatRoutedExpert(
            mesh_device,
            [[self.w] * self.EPC for _ in range(R * C)],
            m=max(256, R * self.S),
            H=H,
            I=I,
            gids=[[r * self.EPC + le for le in range(self.EPC)] for r in range(R) for cc in range(C)],
            n_global=self.Eg,
            wdtype="bf4",
            act="clamped_silu",
            pin=1,
            weights_nk=True,
            check_exact=True,
            max_y_writers=L1_MAX_WRITERS,  # (the model's cap)
        )
        self.wc = wcase.WeightedCase(
            mesh_device,
            dict(self.fe.plan),
            H=H,
            I=I,
            epc=self.EPC,
            K=K,
            S=self.S,
            out_dtype=ttnn.bfloat16,
            x_bf16=False,
            h_bf16=False,
            acc_dtype=ttnn.bfloat16,
            planner="multi",
            patch_addresses=True,  # (the model's EXPERTS_PROGRAM_KEY=patch default)
        )
        assert self.wc.P == 1
        self.mesh = mesh_device

    def poison(self, g):
        """L1 as a model leaves it: random bits where this call's arena lands, and 16 KB a core below it (the y
        writers' and relays' words)."""
        shape, dt, lay, mesh, mc = self.wc._arena_spec
        ncores = shape[0] // self.wc.a_rows
        held = []
        for rows in (self.wc.a_rows, 256):
            spec = ttnn.MemoryConfig(
                mc.memory_layout,
                mc.buffer_type,
                ttnn.ShardSpec(mc.shard_spec.grid, [rows, shape[1]], mc.shard_spec.orientation),
            )
            bits = torch.randint(-(2**31), 2**31 - 1, (ncores * rows, shape[1]), generator=g, dtype=torch.int64)
            held.append(
                ttnn.from_torch(bits.to(torch.int32), dtype=ttnn.uint32, layout=lay, device=mesh, memory_config=spec)
            )
        ttnn.synchronize_device(mesh)
        for t in held:
            ttnn.deallocate(t)

    def gid(self, r, c, le):
        """Global id of local expert le on chip (r, c) (ExpertMapping column-major)."""
        return c * self.Eg + r * self.EPC + le

    def run(self, ids, wts, label, seed=6, check_tokens=48):
        """ids / wts [R, S, K] (global ids, rows of mesh row r = home r's tokens) -> runs the fused call, checks a
        sample of every chip's home tokens against the CPU model."""
        R, C, S, mesh = self.R, self.C, self.S, self.mesh
        gi = torch.Generator().manual_seed(seed)
        x = (torch.randn(R, S, H, generator=gi) * 0.3).to(torch.bfloat16)
        rows = ttnn.ShardTensor2dMesh(mesh, mesh_shape=(R, C), dims=(2, None))
        put = lambda t, dt: ttnn.from_torch(  # noqa: E731
            t.reshape(1, 1, R * S, -1).contiguous(),
            dtype=dt,
            layout=ttnn.TILE_LAYOUT,
            device=mesh,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=rows,
        )
        cnt = torch.bincount(ids.reshape(-1).long(), minlength=self.E)
        per_chip = cnt.reshape(C, R, self.EPC).sum(-1)  # [c, r]
        logger.info(
            f"{label}: top experts {[(int(e), int(n)) for n, e in zip(*cnt.sort(descending=True))][:8]}, "
            f"rows per chip (max) {int(per_chip.max())}, flat rows_cap {self.wc.rows_cap}, cap {self.wc.cap}"
        )
        x_d = put(x, ttnn.bfloat16)
        self.poison(gi)
        xg = self.wc.gather_x(x_d)
        ids_d, wts_d = put(ids.to(torch.int32), ttnn.uint32), put(wts.to(torch.bfloat16), ttnn.bfloat16)
        ids_g, wts_g = self.wc.routing(ids_d, wts_d)
        out = self.wc(self.fe, xg, ids_g, wts_g, 0)  # LoFi (the model's EXPERTS_PRECISION glm)
        ttnn.synchronize_device(mesh)
        got = [ttnn.to_torch(t).float() for t in ttnn.get_device_tensors(out)]  # chip d = (d // C, d % C)
        for t in (xg, x_d, ids_d, wts_d, ids_g, wts_g):
            ttnn.deallocate(t)
        # CPU model on a sample of home tokens per mesh row (every chip of the row holds their column sums)
        W1, W3, W2 = self.w
        worst = 1.0
        for r in range(R):
            sel = sorted(set(torch.randperm(S, generator=gi)[:check_tokens].tolist()) | {0, S - 1})
            xs = x[r, sel].float()
            f = self.ref.ffn(xs, W1.float(), W3.float(), W2.float(), "clamped_silu")
            col = ids[r, sel].long() // self.Eg  # [n, K] the column of each pick
            wsum = torch.stack([(wts[r, sel].float() * (col == c)).sum(-1) for c in range(C)])  # [C, n]
            for c in range(C):
                want = wsum[c][:, None] * f
                have = got[r * C + c][sel]
                ok, pcc = comp_pcc(want, have, 0.99)
                worst = min(worst, float(pcc))
                assert ok, f"{label}: chip ({r}, {c}) PCC {pcc}"
        logger.info(f"{label}: done, min PCC {worst:.5f} over {R * C} chips")
        return got


def _pad_routing(st, n_real, hot, seed=7):
    """The model's PAD pattern: home 0's first n_real tokens random top-K, every other token on the experts ``hot``."""
    R, S, E = st.R, st.S, st.E
    g = torch.Generator().manual_seed(seed)
    ids = torch.tensor(hot, dtype=torch.int64).repeat(R, S, 1)
    for t in range(n_real):
        ids[0, t] = torch.randperm(E, generator=g)[:K]
    wts = torch.tensor([0.2793, 0.25, 0.2461, 0.2461, 0.2422, 0.2354]).repeat(R, S, 1)
    wts[0, :n_real] = torch.rand(n_real, K, generator=g) * 0.3 + 0.05
    return ids, wts


def _case(st, name):
    R, C, EPC = st.R, st.C, st.EPC
    if name == "pad6":  # (8x4: the model's six hot experts; elsewhere six experts on six chips)
        hot = [318, 379, 99, 54, 355, 153] if R * C == 32 else [st.gid(i % R, i % C, i) for i in range(K)]
        return _pad_routing(st, 128, hot)
    if name == "one_chip":  # (every token on six experts of chip (1, 0): its rows at rows_cap)
        return _pad_routing(st, 0, [st.gid(1 % R, 0, le) for le in range(K)])
    if name == "full_column":  # (every token of every column on that column's experts: K picks over the columns)
        hot = [st.gid((i % C + i // C) % R, i % C, i // C) for i in range(K)]
        return _pad_routing(st, 0, hot)
    raise ValueError(name)


@pytest.mark.timeout(3600)
@pytest.mark.parametrize(
    "mesh_device, device_params",
    [(_MESH, _device_params())],
    ids=["x".join(map(str, _MESH))],
    indirect=True,
)
@pytest.mark.parametrize("case", ["pad6", "one_chip", "full_column"])
def test_hot_experts(mesh_device, device_params, case):
    setup = _Setup(mesh_device)
    ids, wts = _case(setup, case)
    setup.run(ids, wts, case)


@pytest.mark.timeout(3600)
@pytest.mark.parametrize(
    "mesh_device, device_params",
    [(_MESH, _device_params())],
    ids=["x".join(map(str, _MESH))],
    indirect=True,
)
def test_hot_capture(mesh_device, device_params):
    path = os.environ.get("HOT_CAPTURE")
    if not path or not os.path.exists(path):
        pytest.skip("HOT_CAPTURE=<a MoE routing dump .pt> not given")
    setup = _Setup(mesh_device)
    rec = torch.load(path)
    R, C = setup.R, setup.C
    assert tuple(rec["mesh"]) == (R, C), (rec["mesh"], (R, C))
    # per chip d = (d // C, d % C): [1, 1, S, K]; replicated over the row's chips
    ids = torch.stack([rec["idx"][r * C][0, 0].to(torch.int64) for r in range(R)])
    wts = torch.stack([rec["wts"][r * C][0, 0].float() for r in range(R)])
    setup.run(ids, wts, f"capture {os.path.basename(path)}")
