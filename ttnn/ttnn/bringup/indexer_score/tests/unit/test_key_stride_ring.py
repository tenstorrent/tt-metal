# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""ttnn.bringup.ring_indexer_score_dsa key_compression_ratio (pooled keys) on a full-mesh snake ring (cluster_axis None).

Cases: CASES below (GLM-5.3 Flash on the 2x4 LoudBox: R 4, 640 query tokens per chip per 5120-token chunk,
block_cyclic_chunk_local 640 tokens -> a 160-key stripe per chip per chunk; T shortened to three chunks, each listed
chunk start run with kv_len = (start + 5120) / 4 keys). The model's captured call is a model case in tests/cases.py.

Layout (key units, reference.full_mesh_key_local_positions): chip d (row-major) holds, for every chunk c, keys
c * 8 * 160 + d * 160 + [0, 160) at k_local rows c * 160 + [0, 160); the ring's full-T k buffer starts as random
garbage. Queries: chip d holds tokens chunk_start + d * 640 + [0, 640). kv_len = (chunk_start + 5120) / 4 keys.
Per chip, against a float32 reference on the same bf16 inputs (reference.dsa_score with key_stride):
- columns [0, kv_len): every pool-future column -inf and every visible column finite (exact);
- visible columns: PCC and relative L2 error. Columns >= kv_len are don't-care.
Also: key_compression_ratio > 1 with chunk_start_idx_tensor (the trace / metadata path) is refused.
"""

import importlib.util
from pathlib import Path

import pytest
import torch
from loguru import logger

import ttnn

_TESTS = Path(__file__).resolve().parents[1]


def _load(name):
    spec = importlib.util.spec_from_file_location(f"bringup_indexer_score_tests_ks_{name}", _TESTS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ref = _load("reference")
CASES = [
    {
        "id": "2x4-ring-fullmesh-keystride4-q640-t3840-h32-d128-fp32dest",
        "mesh": [2, 4],
        "device_params": {"fabric_config": "FABRIC_2D"},
        "heads": 32,
        "head_dim": 128,
        "q_rows": 640,
        "t": 3840,
        "k_local_rows": 480,
        "num_semaphores": 2,
        "topology": "Ring",
        "num_links": 1,
        "key_compression_ratio": 4,
        "chunk_starts": [5120, 10240],
        "block_cyclic_chunk_local": 640,
        "program_config": {"q_chunk_size": 64, "k_chunk_size": 32, "head_group_size": 0},
        "compute_kernel_config": {
            "math_fidelity": "HiFi4",
            "math_approx_mode": False,
            "fp32_dest_acc_en": True,
            "packer_l1_acc": False,
            "dst_full_sync_en": False,
        },
        "gate_scale": 1.0 / 64,
        "seed": 0,
        "pcc": 0.9999,
        "rel": 0.004,
    },
]

pytestmark = pytest.mark.skipif(not ttnn.device.is_blackhole(), reason="indexer_score is Blackhole-only")


def _device_params(c):
    p = dict(c["device_params"])
    p["fabric_config"] = getattr(ttnn.FabricConfig, p["fabric_config"])
    return p


def _pcc(a, b):
    a, b = a.double().flatten(), b.double().flatten()
    a, b = a - a.mean(), b - b.mean()
    return float((a @ b) / (a.norm() * b.norm()))


def _put(mesh_device, t, mapper):
    return ttnn.from_torch(
        t,
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=mapper,
    )


def _setup(mesh_device, c):
    rows, cols = c["mesh"]
    n_dev = rows * cols
    H, D, Sq, T, R, cl = (
        c["heads"],
        c["head_dim"],
        c["q_rows"],
        c["t"],
        c["key_compression_ratio"],
        c["block_cyclic_chunk_local"],
    )
    assert cl == Sq and c["k_local_rows"] * n_dev == T
    g = torch.Generator().manual_seed(c["seed"])
    q = torch.randn(1, H, n_dev * Sq, D, generator=g).to(torch.bfloat16)
    w = (torch.randn(1, 1, n_dev * Sq, H, generator=g) * c["gate_scale"]).to(torch.bfloat16)
    k_nat = torch.randn(T, D, generator=g).to(torch.bfloat16)
    garbage = torch.randn(1, 1, T, D, generator=g).to(torch.bfloat16)
    pos = ref.full_mesh_key_local_positions(T, n_dev, cl // R)  # [n_dev, T / n_dev]
    k_local_host = k_nat[pos.reshape(-1)].reshape(1, 1, T, D)  # chip d = rows [d * T / n_dev, (d + 1) * T / n_dev)
    per_dev = ttnn.ShardTensorToMesh(mesh_device, dim=2)  # row-major over the whole mesh
    dev = (_put(mesh_device, q, per_dev), _put(mesh_device, w, per_dev), _put(mesh_device, k_local_host, per_dev))
    assert list(dev[0].shape) == [1, H, Sq, D] and list(dev[2].shape) == [1, 1, T // n_dev, D]
    grid = mesh_device.compute_with_storage_grid_size()
    crs = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))})
    sems = [ttnn.create_global_semaphore(mesh_device, crs, 0) for _ in range(c["num_semaphores"])]
    return (q, w, k_nat, garbage), dev, sems


def _call(mesh_device, c, dev, sems, garbage, chunk_start, **extra):
    tt_q, tt_w, tt_k_local = dev
    tt_k = _put(mesh_device, garbage, ttnn.ReplicateTensorToMesh(mesh_device))
    pc, k = c["program_config"], c["compute_kernel_config"]
    chunk_tokens = c["mesh"][0] * c["mesh"][1] * c["q_rows"]
    kv_len = None if "chunk_start_idx_tensor" in extra else (chunk_start + chunk_tokens) // c["key_compression_ratio"]
    return ttnn.bringup.ring_indexer_score_dsa(
        tt_q,
        tt_k,
        tt_w,
        tt_k_local,
        sems,
        cluster_axis=None,
        topology=getattr(ttnn.Topology, c["topology"]),
        num_links=c["num_links"],
        chunk_start_idx=None if "chunk_start_idx_tensor" in extra else chunk_start,
        program_config=ttnn.bringup.IndexerScoreProgramConfig(
            q_chunk_size=pc["q_chunk_size"], k_chunk_size=pc["k_chunk_size"], head_group_size=pc["head_group_size"]
        ),
        compute_kernel_config=ttnn.WormholeComputeKernelConfig(
            math_fidelity=getattr(ttnn.MathFidelity, k["math_fidelity"]),
            math_approx_mode=k["math_approx_mode"],
            fp32_dest_acc_en=k["fp32_dest_acc_en"],
            packer_l1_acc=k["packer_l1_acc"],
            dst_full_sync_en=k["dst_full_sync_en"],
        ),
        kv_len=kv_len,
        block_cyclic_chunk_local=c["block_cyclic_chunk_local"],
        key_compression_ratio=c["key_compression_ratio"],
        **extra,
    )


@pytest.mark.timeout(1800)
@pytest.mark.parametrize(
    "mesh_device, device_params, case",
    [(tuple(c["mesh"]), _device_params(c), c) for c in CASES],
    ids=[c["id"] for c in CASES],
    indirect=["mesh_device", "device_params"],
)
def test_ring_key_stride_full_mesh(mesh_device, device_params, case, expect_error):
    c = case
    rows, cols = c["mesh"]
    n_dev = rows * cols
    Sq, T, R = c["q_rows"], c["t"], c["key_compression_ratio"]
    (q, w, k_nat, garbage), dev, sems = _setup(mesh_device, c)

    for chunk_start in c["chunk_starts"]:
        kv_len = (chunk_start + n_dev * Sq) // R
        out = _call(mesh_device, c, dev, sems, garbage, chunk_start)
        outs = [ttnn.to_torch(t).float().reshape(Sq, T) for t in ttnn.get_device_tensors(out)]
        assert len(outs) == n_dev
        for d in range(n_dev):
            q0 = chunk_start + d * Sq
            blk = slice(d * Sq, (d + 1) * Sq)
            want = ref.dsa_score(q[0, :, blk], k_nat, w[0, 0, blk], q0, key_stride=R)[:, :kv_len]
            got = outs[d][:, :kv_len]
            future = torch.isneginf(want)
            n_mask = int((~torch.isneginf(got[future])).sum())
            n_nonfinite = int((~torch.isfinite(got[~future])).sum())
            assert n_mask == 0, f"start {chunk_start} dev {d}: {n_mask} pool-future columns not -inf"
            assert n_nonfinite == 0, f"start {chunk_start} dev {d}: {n_nonfinite} visible columns not finite"
            a, b = got[~future], want[~future]
            pcc, rel = _pcc(a, b), float((a - b).norm() / b.norm())
            logger.info(
                f"start {chunk_start} dev {d}: tokens from {q0}, kv_len {kv_len}, {int((~future).sum())} visible, "
                f"{int(future.sum())} masked, pcc {pcc:.7f}, rel {rel:.5f}"
            )
            assert pcc >= c["pcc"] and rel <= c["rel"], f"start {chunk_start} dev {d}: pcc {pcc:.7f}, rel {rel:.5f}"
        ttnn.deallocate(out)

    # The trace-safe metadata path is not wired for key_compression_ratio > 1: refused before any dispatch.
    meta = ttnn.from_torch(
        torch.tensor([c["chunk_starts"][0]], dtype=torch.int32),
        device=mesh_device,
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    with expect_error(RuntimeError, "key_compression_ratio 1 only"):
        _call(mesh_device, c, dev, sems, garbage, 0, chunk_start_idx_tensor=meta)
