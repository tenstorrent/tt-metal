# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""flat_routed_expert_with_program(patch_addresses=True) (DeepSeek-V4.1 FIX.5): no buffer address in the program-cache
key, every one patched on a hit. Built as the model wires it (tests/weighted_case.py: the routed experts with the
weighted combine appended), on the box's mesh (BRINGUP_TEST_MESH=RxC, default 2x2), at the model's per-chip shape
(DeepSeek-V4.1: 384 experts on 32 chips, 12 a chip, 640 home tokens a chip, top 6; the 2x2 seed's 96 experts, 24 a
chip, 1024 tokens), one random expert (bfp4-exact values) in every slot (cheap to build; only bit-identity between the
two modes is checked). On 8x4 this needed the model's column-barrier fix (DS41-118) in weighted_case.py: without it the
harness hung on its first, address-keyed call.

  1. the address-keyed op (patch_addresses=False, the default) on inputs A: the reference output;
  2. patch_addresses=True on A: bit-identical to 1;
  3. the gathered x again at a NEW address (the first copy kept alive): no program-cache entry added, bit-identical;
  4. an L1 tensor held across the call, so the arena and words land lower (the appended combine's arena CBs and its
     runtime args move: extra_addrs change, its descriptor's args are copied in): bit-identical.

    BRINGUP_TEST_MESH=8x4 scripts/run_safe_pytest.sh --run-all \\
        ttnn/ttnn/bringup/flat_routed_expert_ttnn/tests/unit/test_with_program_patch.py
"""

import importlib.util
import os
from pathlib import Path

import pytest
import torch
from loguru import logger

import ttnn
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


def _bits(t):
    return ttnn.to_torch(t).contiguous().view(torch.int16).clone()


@pytest.mark.timeout(3600)
@pytest.mark.parametrize(
    "mesh_device, device_params",
    [(_MESH, _device_params())],
    ids=["x".join(map(str, _MESH))],
    indirect=True,
)
def test_with_program_patch_addresses(mesh_device, device_params):
    wcase = _load("weighted_case")
    R, C = tuple(mesh_device.shape)
    E = 384 if R * C == 32 else 96  # (the model's experts on 8x4; the 2x2 seed's scaled 96)
    EPC = E // (R * C)
    S = min(5120 // R, 1024)  # (the 5120-token chunk's home tokens a chip, one pass)
    Eg = R * EPC
    ref = _load("reference")
    g = torch.Generator().manual_seed(5)
    w = lambda *shape: ref.block_round(torch.randn(*shape, generator=g) * 0.02, 3, ties="even", clamp=True)  # noqa
    one = (w(I, H), w(I, H), w(H, I))  # (one random expert in every slot: cheap to build, exact bfp4 values)
    fe = FlatRoutedExpert(
        mesh_device,
        [[one] * EPC for _ in range(R * C)],
        m=max(256, R * S),
        H=H,
        I=I,
        gids=[[r * EPC + le for le in range(EPC)] for r in range(R) for cc in range(C)],
        n_global=Eg,
        wdtype="bf4",
        act="clamped_silu",
        pin=1,
        weights_nk=True,
        check_exact=True,
    )
    wc = wcase.WeightedCase(
        mesh_device,
        dict(fe.plan),
        H=H,
        I=I,
        epc=EPC,
        K=K,
        S=S,
        out_dtype=ttnn.bfloat16,
        x_bf16=False,
        h_bf16=False,
        acc_dtype=ttnn.bfloat16,
        planner="multi",
    )
    rows = ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=(R, C), dims=(2, None))
    gi = torch.Generator().manual_seed(6)
    x = (torch.randn(R, S, H, generator=gi) * 0.3).to(torch.bfloat16)
    ids = torch.stack([torch.stack([torch.randperm(E, generator=gi)[:K] for _ in range(S)]) for _ in range(R)])
    wts = (torch.rand(R, S, K, generator=gi) + 0.5).to(torch.bfloat16)
    put = lambda t, dt: ttnn.from_torch(  # noqa: E731
        t.reshape(1, 1, R * S, -1).contiguous(),
        dtype=dt,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=rows,
    )
    x_d = put(x, ttnn.bfloat16)
    xg = wc.gather_x(x_d)
    ids_g, wts_g = wc.routing(put(ids.to(torch.int32), ttnn.uint32), put(wts, ttnn.bfloat16))
    fid = 0  # LoFi (the model's EXPERTS_PRECISION glm)

    def run(xg_):
        out = wc(fe, xg_, ids_g, wts_g, fid)
        ttnn.synchronize_device(mesh_device)
        return [_bits(t) for t in ttnn.get_device_tensors(out)]

    want = run(xg)  # 1. the address-keyed op
    assert wc.P == 1, f"{S} home tokens in {wc.P} passes"
    wc.patch_addresses = True
    got = run(xg)  # 2. patched: its first call builds its program
    assert all(torch.equal(a, b) for a, b in zip(got, want)), "patch_addresses: output differs (same buffers)"
    xg2 = wc.gather_x(x_d)  # 3. the same x at a new address (the first copy still allocated)
    assert int(xg2.buffer_address()) != int(xg.buffer_address())
    n0 = mesh_device.num_program_cache_entries()
    got = run(xg2)
    n1 = mesh_device.num_program_cache_entries()
    logger.info(f"moved x {int(xg.buffer_address())} -> {int(xg2.buffer_address())}: +{n1 - n0} program-cache entries")
    assert n1 == n0, f"patch_addresses: a moved x built {n1 - n0} programs"
    assert all(torch.equal(a, b) for a, b in zip(got, want)), "patch_addresses: output differs (moved x)"
    hold = ttnn.allocate_tensor_on_device(  # 4. the arena and words land lower
        ttnn.Shape([32, 32 * 64]), ttnn.bfloat16, ttnn.TILE_LAYOUT, mesh_device, ttnn.L1_MEMORY_CONFIG
    )
    n0 = mesh_device.num_program_cache_entries()
    got = run(xg2)
    n1 = mesh_device.num_program_cache_entries()
    logger.info(f"moved arena (L1 held): +{n1 - n0} program-cache entries (the zeroing program may add its own)")
    assert all(torch.equal(a, b) for a, b in zip(got, want)), "patch_addresses: output differs (moved arena)"
    moved = [k for k in wc.combines]
    assert len(moved) >= 2, f"the arena / words did not move: {moved}"
    got = run(xg)  # back to the first addresses (x and arena): the combine's args copied in again
    ttnn.deallocate(hold)
    got = run(xg)
    assert all(torch.equal(a, b) for a, b in zip(got, want)), "patch_addresses: output differs (arena back)"
    for t in (xg, xg2, x_d):
        ttnn.deallocate(t)
