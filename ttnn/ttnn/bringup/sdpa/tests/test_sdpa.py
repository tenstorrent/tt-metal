# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Model cases of the sdpa fork (cases.py): each captured ttnn.bringup call on random inputs, checked against
reference.py and for bit-determinism (A, B, A x5).

Run through the bring-up runner (each case's mesh on the cards that form it):
    python -m bringup_fw.testing.fork_tests --fork sdpa
"""

import importlib.util
from pathlib import Path

import pytest

import ttnn
from bringup_fw.testing import determinism

_HERE = Path(__file__).parent


def _load(name):
    # by path: every fork has a package named `tests`, and pytest runs with --import-mode=importlib
    spec = importlib.util.spec_from_file_location(f"bringup_sdpa_tests_{name}", _HERE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


CASES = _load("cases").CASES
ref = _load("reference")

DTYPES = {"bfloat16": ttnn.bfloat16}


def _pcc(a, b):
    a, b = a.double().flatten(), b.double().flatten()
    a, b = a - a.mean(), b - b.mean()
    return float((a @ b) / (a.norm() * b.norm()))


def _ring_mla_inputs(case, mesh, seed):
    """Q (this chunk's rows, heads over TP, rows over SP) and the KV cache slot (block-cyclic over SP, replicated over
    TP), both random. Returns the device tensors and the host Q / KV (KV in position order) at the device's dtype."""
    import torch  # not at module level: ttnn's pre-commit forbids a global torch import under ttnn/ttnn

    g = torch.Generator().manual_seed(seed)
    dt = DTYPES[case["dtype"]]
    sp, tp = case["mesh"][case["sp_axis"]], case["mesh"][case["tp_axis"]]
    _, h, sq, d = case["q_shape"]
    q = torch.randn(1, h * tp, sq * sp, d, generator=g).to(torch.bfloat16)
    n = case["kv_shape"][2] * sp
    kv = torch.randn(1, 1, n, d, generator=g).to(torch.bfloat16)  # position order

    # device layout: position p sits at row rows[p] of the SP-concatenated cache
    rows = ref.block_cyclic_rows(n, sp, case["chunk_local"])
    kv_dev = torch.empty_like(kv)
    kv_dev[0, 0, rows] = kv[0, 0]

    q_dims, kv_dims = [None, None], [None, None]
    q_dims[case["sp_axis"]], q_dims[case["tp_axis"]] = 2, 1
    kv_dims[case["sp_axis"]] = 2
    shape = tuple(case["mesh"])
    common = dict(dtype=dt, layout=ttnn.TILE_LAYOUT, device=mesh, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    tt_q = ttnn.from_torch(q, mesh_mapper=ttnn.ShardTensor2dMesh(mesh, mesh_shape=shape, dims=q_dims), **common)
    tt_kv = ttnn.from_torch(kv_dev, mesh_mapper=ttnn.ShardTensor2dMesh(mesh, mesh_shape=shape, dims=kv_dims), **common)
    return tt_q, tt_kv, q, kv


def _ring_mla_q_start(case):
    """Absolute position of the chunk's first query. With the chunk a whole period (sp x chunk_local rows) and starting
    on one, SP device d holds the chunk's rows d * chunk_local .. (d + 1) * chunk_local - 1, in order."""
    start = case["kv_actual_isl"]
    sp = case["mesh"][case["sp_axis"]]
    assert start % (sp * case["chunk_local"]) == 0 and case["q_shape"][2] == case["chunk_local"], case["id"]
    return start


def run_ring_mla(case, mesh):
    import torch

    grid = mesh.compute_with_storage_grid_size()
    assert [grid.x, grid.y] == case["semaphore_grid"], (grid, case["semaphore_grid"])
    full = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))})
    semaphores = [ttnn.create_global_semaphore(mesh, full, 0) for _ in range(2)]
    pc = case["program_config"]
    program_config = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(*pc["grid"]),
        q_chunk_size=pc["q_chunk_size"],
        k_chunk_size=pc["k_chunk_size"],
        exp_approx_mode=pc["exp_approx_mode"],
        segmented_accumulation=pc["segmented_accumulation"],
    )
    ck = case["compute_kernel_config"]
    compute_kernel_config = ttnn.types.BlackholeComputeKernelConfig(
        math_fidelity=getattr(ttnn.MathFidelity, ck["math_fidelity"]),
        math_approx_mode=ck["math_approx_mode"],
        fp32_dest_acc_en=ck["fp32_dest_acc_en"],
        packer_l1_acc=ck["packer_l1_acc"],
    )
    gather = ttnn.from_torch(
        torch.zeros(case["gather_shape"]),
        dtype=DTYPES[case["dtype"]],
        layout=ttnn.TILE_LAYOUT,
        device=mesh,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    )

    def call(tt_q, tt_kv):
        out, _ = ttnn.bringup.ring_mla(
            tt_q,
            tt_kv,
            persistent_output_buffer_kv=gather,
            head_dim_v=case["head_dim_v"],
            logical_n=case["logical_n"],
            program_config=program_config,
            scale=case["scale"],
            compute_kernel_config=compute_kernel_config,
            dim=case["dim"],
            multi_device_global_semaphore=semaphores,
            num_links=case["num_links"],
            cluster_axis=case["sp_axis"],
            mesh_device=mesh,
            topology=getattr(ttnn.Topology, case["topology"]),
            ccl_core_grid_offset=ttnn.CoreCoord(*case["ccl_core_grid_offset"]),
            use_column_major_ccl=case["use_column_major_ccl"],
            is_balanced=case["is_balanced"],
            kv_cache_batch_idx=case["kv_cache_batch_idx"],
            kv_actual_isl=case["kv_actual_isl"],
            exact_score_exp=case["exact_score_exp"],
        )
        return out

    q_a, kv_a, q_host, kv_host = _ring_mla_inputs(case, mesh, case["seed"])
    out = call(q_a, kv_a)
    dims = [None, None]
    dims[case["sp_axis"]], dims[case["tp_axis"]] = 2, 1
    got = ttnn.to_torch(out, mesh_composer=ttnn.ConcatMesh2dToTensor(mesh, mesh_shape=tuple(case["mesh"]), dims=dims))

    # every output row is a real query (the chunk is full): no don't-care elements
    want = ref.ring_mla(
        q_host,
        kv_host,
        head_dim_v=case["head_dim_v"],
        scale=case["scale"],
        q_start=_ring_mla_q_start(case),
        logical_n=case["logical_n"],
    )
    got = got.float()
    pcc = _pcc(got, want)
    rel = float((got - want).norm() / want.norm())
    print(f"{case['id']}: pcc {pcc:.6f}, rel {rel:.5f}")
    assert pcc >= case["pcc"] and rel <= case["rel"], f"{case['id']}: pcc {pcc:.6f} (min {case['pcc']}), rel {rel:.5f}"

    # A, B, A x5: the gather buffer is rewritten whole by every call
    q_b, kv_b, _, _ = _ring_mla_inputs(case, mesh, case["seed"] + 1)
    determinism.assert_deterministic(lambda: call(q_a, kv_a), lambda: call(q_b, kv_b), first=out, label=case["id"])


def _sparse_sdpa_inputs(case, seed):
    """Random q, kv, sink; valid ids: per row a random count of distinct keys of [0, T) (a quarter of the rows use
    every slot, as at the model's full context), the rest of the row MASKED_INDEX (the op wants a contiguous tail)."""
    import torch  # local: tt-metal's check-torch-imports-in-ttnn forbids a global torch import under ttnn/

    g = torch.Generator().manual_seed(seed)
    _, H, S, _ = case["q_shape"]
    T, TOPK = case["kv_shape"][2], case["indices_shape"][3]
    q = torch.randn(case["q_shape"], generator=g).to(torch.bfloat16)
    kv = torch.randn(case["kv_shape"], generator=g).to(torch.bfloat16)
    idx = torch.full(case["indices_shape"], ref.MASKED_INDEX, dtype=torch.int64)
    nvalid = torch.randint(1, TOPK + 1, (S,), generator=g)
    nvalid[torch.randperm(S, generator=g)[: S // 4]] = TOPK
    for s in range(S):
        idx[0, 0, s, : nvalid[s]] = torch.randperm(T, generator=g)[: nvalid[s]]
    # sinks from negligible to dominant softmax shares (scaled-logit domain; the op multiplies the sink by scale)
    sink = ((torch.randn(case["sink_shape"], generator=g) * 3 + 2) / case["scale"]).to(torch.bfloat16)
    return q, kv, idx, sink


def _sparse_sdpa_dev(host, mesh):
    import torch  # local: tt-metal's check-torch-imports-in-ttnn forbids a global torch import under ttnn/

    def put(t, dtype):
        return ttnn.from_torch(
            t,
            dtype=dtype,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
        )

    q, kv, idx, sink = host
    # MASKED_INDEX wraps to -1 in int32: the same 32 bits
    return (
        put(q, ttnn.bfloat16),
        put(kv, ttnn.bfloat16),
        put(idx.to(torch.int32), ttnn.uint32),
        put(sink, ttnn.bfloat16),
    )


def run_sparse_sdpa(case, mesh):
    """The inputs are replicated, so every device's output is checked against the same torch reference."""
    import torch  # local: tt-metal's check-torch-imports-in-ttnn forbids a global torch import under ttnn/

    built = {"dtype": "bfloat16", "layout": "ROW_MAJOR", "memory": "DRAM_INTERLEAVED", "kv_format": "BF16"}
    for name, value in built.items():
        assert case[name] == value, f"{name}={case[name]}: not built by this test"
    k = case["compute_kernel_config"]
    kwargs = dict(
        kv_format=getattr(ttnn.bringup.SparseKVFormat, case["kv_format"]),
        scale=case["scale"],
        k_chunk_size=case["k_chunk_size"],
        compute_kernel_config=ttnn.WormholeComputeKernelConfig(
            math_fidelity=getattr(ttnn.MathFidelity, k["math_fidelity"]),
            math_approx_mode=k["math_approx_mode"],
            fp32_dest_acc_en=k["fp32_dest_acc_en"],
            packer_l1_acc=k["packer_l1_acc"],
            dst_full_sync_en=k["dst_full_sync_en"],
        ),
        exact_exp=case["exact_exp"],
    )

    def call(inp):
        q, kv, idx, sink = inp
        return ttnn.bringup.sparse_sdpa(q, kv, idx, case["v_dim"], attention_sink=sink, **kwargs)

    host_a = _sparse_sdpa_inputs(case, case["seed"])
    inp_a = _sparse_sdpa_dev(host_a, mesh)
    out = call(inp_a)
    shards = [ttnn.to_torch(t).float() for t in ttnn.get_device_tensors(out)]
    assert len(shards) == case["mesh"][0] * case["mesh"][1]
    q, kv, idx, sink = host_a
    want = ref.sparse_sdpa(q.float(), kv.float(), idx, case["scale"], case["v_dim"], sink.float())
    for d, got in enumerate(shards):
        assert tuple(got.shape) == tuple(want.shape), f"device {d}: shape {tuple(got.shape)}"
        assert torch.isfinite(got).all(), f"device {d}: non-finite output"
        pcc = _pcc(got, want)
        rel = float((got - want).norm() / want.norm())
        if d == 0:
            print(f"{case['id']}: pcc {pcc:.6f}, rel {rel:.5f}")
        assert pcc >= case["pcc"], f"device {d}: pcc {pcc:.6f} < {case['pcc']}"
        assert rel <= case["max_rel"], f"device {d}: rel {rel:.5f} > {case['max_rel']}"

    inp_b = _sparse_sdpa_dev(_sparse_sdpa_inputs(case, case["seed"] + 1), mesh)
    determinism.assert_deterministic(lambda: call(inp_a), lambda: call(inp_b), first=out, label=case["id"])


RUNNERS = {"ring_mla": run_ring_mla, "sparse_sdpa": run_sparse_sdpa}
FABRIC = {"FABRIC_1D_RING": ttnn.FabricConfig.FABRIC_1D_RING}


@pytest.mark.timeout(1800)
@pytest.mark.parametrize(
    "case, mesh_device, device_params",
    [pytest.param(c, tuple(c["mesh"]), {"fabric_config": FABRIC[c["fabric_config"]]}, id=c["id"]) for c in CASES],
    indirect=["mesh_device", "device_params"],
)
def test_sdpa_case(case, mesh_device, device_params):
    RUNNERS[case["op"]](case, mesh_device)
