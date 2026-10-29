# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""The indexer_score fork's DSA scorers against their torch semantics (reference.py), one random-input case per
captured call (cases.py).

ring_indexer_score_dsa: random natural-order keys are laid out as the model's block-cyclic, 4-chip-striped cache would
leave them after its TP-inner gather (k_local per SP row, reference.k_local_positions), the ring's full-T k buffer
starts as random garbage, and each chip's scores [Sq, T] are checked against a float32 reference on the same bf16
q / k / gates:
- every future column (t > the row's position) is -inf and every causal column finite (exact);
- causal columns: PCC and the relative L2 error, per chip.
Don't-care: nothing (kv_len = T, so every column of every row is written).

ring_indexer_score_dsa with key_compression_ratio R (pooled keys) on a full-mesh snake ring (cluster_axis None): chip d
(row-major) holds, for every chunk c, keys c * n * stripe + d * stripe + [0, stripe) at k_local rows c * stripe + ..
(stripe = block_cyclic_chunk_local / R, reference.full_mesh_key_local_positions) and queries chunk_start + d * Sq + ..;
per chip, columns [0, kv_len): every pool-future column -inf and every visible column finite (exact), then PCC and
relative L2 error on the visible ones. Don't-care: columns >= kv_len.

Cases with `noise` (deepseek_v41_flash_d_p O.1: natural query blocks, ratio 2 / 1, the block epilogue) run in
test_ring_indexer_score_dsa_ds41 (its docstring)."""

import importlib.util
from pathlib import Path

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.common.bringup.testing import determinism
from models.demos.common.bringup.testing.fork_cases import noise_limit

_HERE = Path(__file__).resolve().parent


def _load(name):
    spec = importlib.util.spec_from_file_location(f"bringup_indexer_score_tests_{name}", _HERE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ref = _load("reference")
CASES = _load("cases").CASES

pytestmark = pytest.mark.skipif(not ttnn.device.is_blackhole(), reason="indexer_score is Blackhole-only")


def _device_params(c):
    p = dict(c["device_params"])
    p["fabric_config"] = getattr(ttnn.FabricConfig, p["fabric_config"])
    if "fabric_payload_bytes" in p:  # the model's fabric packet payload (harness.device_params)
        rc = ttnn._ttnn.fabric.FabricRouterConfig()
        rc.max_packet_payload_size_bytes = int(p.pop("fabric_payload_bytes"))
        p["fabric_router_config"] = rc
    return p


def _kernel_config(k):
    return ttnn.WormholeComputeKernelConfig(
        math_fidelity=getattr(ttnn.MathFidelity, k["math_fidelity"]),
        math_approx_mode=k["math_approx_mode"],
        fp32_dest_acc_en=k["fp32_dest_acc_en"],
        packer_l1_acc=k["packer_l1_acc"],
        dst_full_sync_en=k["dst_full_sync_en"],
    )


def _pcc(a, b):
    a, b = a.double().flatten(), b.double().flatten()
    a, b = a - a.mean(), b - b.mean()
    return float((a @ b) / (a.norm() * b.norm()))


@pytest.mark.timeout(1800)
@pytest.mark.parametrize(
    "mesh_device, device_params, case",
    [
        (tuple(c["mesh"]), _device_params(c), c)
        for c in CASES
        if c["op"] == "ring_indexer_score_dsa" and "key_compression_ratio" not in c
    ],
    ids=[c["id"] for c in CASES if c["op"] == "ring_indexer_score_dsa" and "key_compression_ratio" not in c],
    indirect=["mesh_device", "device_params"],
)
def test_ring_indexer_score_dsa(mesh_device, device_params, case):
    c = case
    rows, cols = c["mesh"]
    n_dev = rows * cols
    sp_axis, tp_axis = c["block_cyclic_sp_axis"], c["seq_subshard_axis"]
    assert c["cluster_axis"] == sp_axis and (sp_axis, tp_axis) == (0, 1), "the test lays out SP on rows, TP on cols"
    assert c["block_cyclic_cache_tp_sharded"]
    sp, tp = rows, cols
    H, D, Sq, T, cl = c["heads"], c["head_dim"], c["q_rows"], c["t"], c["block_cyclic_chunk_local"]
    assert cl == tp * Sq and c["k_local_rows"] == T // sp and c["kv_len"] == T
    grid = mesh_device.compute_with_storage_grid_size()
    crs = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))})
    sems = [ttnn.create_global_semaphore(mesh_device, crs, 0) for _ in range(c["num_semaphores"])]
    host, dev = _inputs(mesh_device, c, c["seed"])
    q, w, k_nat = host
    out, tt_k = _call(mesh_device, c, dev, sems)
    outs = [ttnn.to_torch(t).float().reshape(Sq, T) for t in ttnn.get_device_tensors(out)]
    assert len(outs) == n_dev

    for d in range(n_dev):
        r, cc = divmod(d, cols)
        q0 = ref.query_start(c["chunk_start_idx"], r, cc, Sq, cl)
        blk = slice(d * Sq, (d + 1) * Sq)
        want = ref.dsa_score(q[0, :, blk], k_nat, w[0, 0, blk], q0)
        got = outs[d]
        future = torch.isneginf(want)
        n_mask = int((~torch.isneginf(got[future])).sum())
        n_nonfinite = int((~torch.isfinite(got[~future])).sum())
        assert n_mask == 0, f"dev {d}: {n_mask} future columns not -inf"
        assert n_nonfinite == 0, f"dev {d}: {n_nonfinite} causal columns not finite"
        a, b = got[~future], want[~future]
        pcc = _pcc(a, b)
        rel = float((a - b).norm() / b.norm())
        logger.info(f"dev {d}: rows from {q0}, {int((~future).sum())} causal scores, pcc {pcc:.7f}, rel {rel:.5f}")
        assert (
            pcc >= c["pcc"] and rel <= c["rel"]
        ), f"dev {d}: pcc {pcc:.7f} (min {c['pcc']}), rel {rel:.5f} (max {c['rel']})"

    # The op fills the ring's k buffer (an input): each call gets a fresh copy of its starting garbage (_call), and
    # the filled buffer is compared too.
    dev_b = _inputs(mesh_device, c, c["seed"] + 1)[1]
    determinism.assert_deterministic(
        lambda: _call(mesh_device, c, dev, sems),
        lambda: _call(mesh_device, c, dev_b, sems),
        first=(out, tt_k),
        label=c["id"],
    )
    ttnn.deallocate(out)


def _put(mesh_device, c, t, mapper):
    return ttnn.from_torch(
        t,
        device=mesh_device,
        dtype=getattr(ttnn.DataType, c["dtype"]),
        layout=getattr(ttnn.Layout, c["layout"]),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=mapper,
    )


def _inputs(mesh_device, c, seed):
    """Host (q, w, k_nat) for `seed` and the device inputs (q, w, k_local) plus the host garbage of the k buffer."""
    rows, cols = c["mesh"]
    n_dev = rows * cols
    sp, tp = rows, cols
    H, D, Sq, T, cl = c["heads"], c["head_dim"], c["q_rows"], c["t"], c["block_cyclic_chunk_local"]
    g = torch.Generator().manual_seed(seed)

    # Host inputs, bf16. Device d = r * cols + c (row-major) gets query block d; its rows start at
    # chunk_start + r * chunk_local + c * Sq = chunk_start + d * Sq.
    q = torch.randn(1, H, n_dev * Sq, D, generator=g).to(torch.bfloat16)
    w = (torch.randn(1, 1, n_dev * Sq, H, generator=g) * c["gate_scale"]).to(torch.bfloat16)
    k_nat = torch.randn(T, D, generator=g).to(torch.bfloat16)
    garbage = torch.randn(1, 1, T, D, generator=g).to(torch.bfloat16)
    pos = ref.k_local_positions(T, sp, tp, cl)  # [sp, T / sp]
    k_local_host = k_nat[pos.reshape(-1)].reshape(1, 1, T, D)  # SP row r = rows [r * T / sp, (r + 1) * T / sp)

    per_dev = ttnn.ShardTensorToMesh(mesh_device, dim=2)  # dim 2 split over all devices, row-major
    tt_q, tt_w = _put(mesh_device, c, q, per_dev), _put(mesh_device, c, w, per_dev)
    k_map = ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=(rows, cols), dims=(2, None))
    tt_k_local = _put(mesh_device, c, k_local_host, k_map)
    assert list(tt_q.shape) == [1, H, Sq, D] and list(tt_w.shape) == [1, 1, Sq, H]
    assert list(tt_k_local.shape) == [1, 1, T // sp, D]
    return (q, w, k_nat), (tt_q, tt_w, tt_k_local, garbage)


def _call(mesh_device, c, dev, sems):
    """One call on a fresh copy of the k buffer's garbage; returns (scores, the filled k buffer)."""
    tt_q, tt_w, tt_k_local, garbage = dev
    tt_k = _put(mesh_device, c, garbage, ttnn.ReplicateTensorToMesh(mesh_device))
    assert list(tt_k.shape) == [1, 1, c["t"], c["head_dim"]]
    pc = c["program_config"]
    out = ttnn.bringup.ring_indexer_score_dsa(
        tt_q,
        tt_k,
        tt_w,
        tt_k_local,
        sems,
        cluster_axis=c["cluster_axis"],
        topology=getattr(ttnn.Topology, c["topology"]),
        num_links=c["num_links"],
        chunk_start_idx=c["chunk_start_idx"],
        program_config=ttnn.bringup.IndexerScoreProgramConfig(
            q_chunk_size=pc["q_chunk_size"], k_chunk_size=pc["k_chunk_size"], head_group_size=pc["head_group_size"]
        ),
        compute_kernel_config=_kernel_config(c["compute_kernel_config"]),
        kv_len=c["kv_len"],
        seq_subshard_axis=c["seq_subshard_axis"],
        block_cyclic_sp_axis=c["block_cyclic_sp_axis"],
        block_cyclic_chunk_local=c["block_cyclic_chunk_local"],
        block_cyclic_cache_tp_sharded=c["block_cyclic_cache_tp_sharded"],
    )
    return out, tt_k


_KS = [c for c in CASES if c["op"] == "ring_indexer_score_dsa" and "key_compression_ratio" in c and "noise" not in c]


@pytest.mark.timeout(1800)
@pytest.mark.parametrize(
    "mesh_device, device_params, case",
    [(tuple(c["mesh"]), _device_params(c), c) for c in _KS],
    ids=[c["id"] for c in _KS],
    indirect=["mesh_device", "device_params"],
)
def test_ring_indexer_score_dsa_key_stride(mesh_device, device_params, case):
    c = case
    rows, cols = c["mesh"]
    n_dev = rows * cols
    assert c["cluster_axis"] is None, "the test lays out the full-mesh block-cyclic ring"
    Sq, T, R, kv = c["q_rows"], c["t"], c["key_compression_ratio"], c["kv_len"]
    assert c["block_cyclic_chunk_local"] == Sq and c["k_local_rows"] * n_dev == T
    assert kv == (c["chunk_start_idx"] + n_dev * Sq) // R
    grid = mesh_device.compute_with_storage_grid_size()
    crs = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))})
    sems = [ttnn.create_global_semaphore(mesh_device, crs, 0) for _ in range(c["num_semaphores"])]
    (q, w, k_nat), dev = _inputs_key_stride(mesh_device, c, c["seed"])
    out, tt_k = _call_key_stride(mesh_device, c, dev, sems)
    outs = [ttnn.to_torch(t).float().reshape(Sq, T) for t in ttnn.get_device_tensors(out)]
    assert len(outs) == n_dev

    for d in range(n_dev):
        q0 = c["chunk_start_idx"] + d * Sq
        blk = slice(d * Sq, (d + 1) * Sq)
        want = ref.dsa_score(q[0, :, blk], k_nat, w[0, 0, blk], q0, key_stride=R)[:, :kv]
        got = outs[d][:, :kv]
        future = torch.isneginf(want)
        n_mask = int((~torch.isneginf(got[future])).sum())
        n_nonfinite = int((~torch.isfinite(got[~future])).sum())
        assert n_mask == 0, f"dev {d}: {n_mask} pool-future columns not -inf"
        assert n_nonfinite == 0, f"dev {d}: {n_nonfinite} visible columns not finite"
        a, b = got[~future], want[~future]
        pcc = _pcc(a, b)
        rel = float((a - b).norm() / b.norm())
        logger.info(
            f"dev {d}: tokens from {q0}, kv_len {kv}, {int((~future).sum())} visible, {int(future.sum())} masked, "
            f"pcc {pcc:.7f}, rel {rel:.5f}"
        )
        assert (
            pcc >= c["pcc"] and rel <= c["rel"]
        ), f"dev {d}: pcc {pcc:.7f} (min {c['pcc']}), rel {rel:.5f} (max {c['rel']})"

    # The op fills the ring's k buffer (an input): each call gets a fresh copy of its starting garbage, and the filled
    # buffer is compared too. Score columns >= kv_len are unspecified (the op leaves them as the fresh allocation
    # held), so only [0, kv_len) is compared.
    def defined(res):
        o, k = res
        cut = [ttnn.to_torch(t).reshape(Sq, T)[:, :kv] for t in ttnn.get_device_tensors(o)]
        ttnn.deallocate(o)
        return cut, k

    dev_b = _inputs_key_stride(mesh_device, c, c["seed"] + 1)[1]
    determinism.assert_deterministic(
        lambda: defined(_call_key_stride(mesh_device, c, dev, sems)),
        lambda: defined(_call_key_stride(mesh_device, c, dev_b, sems)),
        first=defined((out, tt_k)),
        label=c["id"],
    )


def _inputs_key_stride(mesh_device, c, seed):
    """Host (q, w, k_nat) and the device inputs (q, w, k_local) plus the host garbage of the k buffer."""
    rows, cols = c["mesh"]
    n_dev = rows * cols
    H, D, Sq, T = c["heads"], c["head_dim"], c["q_rows"], c["t"]
    g = torch.Generator().manual_seed(seed)
    q = torch.randn(1, H, n_dev * Sq, D, generator=g).to(torch.bfloat16)
    w = (torch.randn(1, 1, n_dev * Sq, H, generator=g) * c["gate_scale"]).to(torch.bfloat16)
    k_nat = torch.randn(T, D, generator=g).to(torch.bfloat16)
    garbage = torch.randn(1, 1, T, D, generator=g).to(torch.bfloat16)
    pos = ref.full_mesh_key_local_positions(T, n_dev, c["block_cyclic_chunk_local"] // c["key_compression_ratio"])
    k_local_host = k_nat[pos.reshape(-1)].reshape(1, 1, T, D)  # chip d = rows [d * T / n_dev, (d + 1) * T / n_dev)
    per_dev = ttnn.ShardTensorToMesh(mesh_device, dim=2)  # row-major over the whole mesh
    tt_q, tt_w = _put(mesh_device, c, q, per_dev), _put(mesh_device, c, w, per_dev)
    tt_k_local = _put(mesh_device, c, k_local_host, per_dev)
    assert list(tt_q.shape) == [1, H, Sq, D] and list(tt_w.shape) == [1, 1, Sq, H]
    assert list(tt_k_local.shape) == [1, 1, c["k_local_rows"], D]
    return (q, w, k_nat), (tt_q, tt_w, tt_k_local, garbage)


def _call_key_stride(mesh_device, c, dev, sems):
    """One call on a fresh copy of the k buffer's garbage; returns (scores, the filled k buffer)."""
    tt_q, tt_w, tt_k_local, garbage = dev
    tt_k = _put(mesh_device, c, garbage, ttnn.ReplicateTensorToMesh(mesh_device))
    assert list(tt_k.shape) == [1, 1, c["t"], c["head_dim"]]
    pc = c["program_config"]
    out = ttnn.bringup.ring_indexer_score_dsa(
        tt_q,
        tt_k,
        tt_w,
        tt_k_local,
        sems,
        cluster_axis=None,
        topology=getattr(ttnn.Topology, c["topology"]),
        num_links=c["num_links"],
        chunk_start_idx=c["chunk_start_idx"],
        program_config=ttnn.bringup.IndexerScoreProgramConfig(
            q_chunk_size=pc["q_chunk_size"], k_chunk_size=pc["k_chunk_size"], head_group_size=pc["head_group_size"]
        ),
        compute_kernel_config=_kernel_config(c["compute_kernel_config"]),
        kv_len=c["kv_len"],
        block_cyclic_chunk_local=c["block_cyclic_chunk_local"],
        key_compression_ratio=c["key_compression_ratio"],
    )
    return out, tt_k


# ---------------------------------------------------------------------------------------------------------------- DS41
_DS41 = [c for c in CASES if c["op"] == "ring_indexer_score_dsa" and "noise" in c]
SENTINEL = 12345.0


@pytest.mark.timeout(1800)
@pytest.mark.parametrize(
    "mesh_device, device_params, case",
    [(tuple(c["mesh"]), _device_params(c), c) for c in _DS41],
    ids=[c["id"] for c in _DS41],
    indirect=["mesh_device", "device_params"],
)
def test_ring_indexer_score_dsa_ds41(mesh_device, device_params, case):
    """Natural query blocks over the block-cyclic, SP-sharded / TP-replicated key cache (the layout of
    tests/unit/test_natural_query_blocks.py), the ring's k buffer starting as random garbage, per chip over columns
    [0, kv_len):
    - every future column -inf (and, with block_bias, every column of a masked block), every other column finite:
      exact;
    - the visible scores vs float64 (reference.dsa_score_at): rel L2 within noise_limit at the declared formats (bf16
      q / k / gates, bf16 out) with the FPU's operand bits at the call's fidelity; PCC printed (a diagnostic);
    - block_max_out: bit-exact vs the block max of the op's own bf16 scores (as fp32), +inf at each query's own block
      with block_max_pin; columns past kv_len / block_keys keep the buffer's prior value.
    Determinism: the op writes its k buffer and block_max_out (inputs): each call gets fresh copies, compared too.
    Every call's k buffer is written (garbage) before the first call is enqueued, as the model's load-time buffer
    (tt/indexer.py _kbuf): a peer that finished call N starts call N+1 and writes its shard into this chip's buffer
    while this chip still runs call N, so a host write queued after call N would overwrite that shard (DS41-111)."""
    c = case
    sp, tp = c["mesh"]
    H, D, Sq, S, T, r = c["heads"], c["head_dim"], c["q_rows"], c["chunk"], c["t"], c["key_compression_ratio"]
    W, start, kv = c["block_cyclic_chunk_local"], c["chunk_start_idx"], c["kv_len"]
    assert W == S // sp and Sq == W // tp and c["k_local_rows"] == T // sp
    assert (sp, tp) == (2, 2) or c["cluster_axis"] == 0, "the test lays out SP on rows, TP on cols"
    bk = c.get("block_keys", 0)
    grid = mesh_device.compute_with_storage_grid_size()
    crs = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))})
    sems = [ttnn.create_global_semaphore(mesh_device, crs, 0) for _ in range(c["num_semaphores"])]

    def put(t, mapper, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT):
        return ttnn.from_torch(
            t.contiguous(),
            device=mesh_device,
            dtype=dtype,
            layout=layout,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=mapper,
        )

    chips = ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=(sp, tp), dims=(0, 1))

    qk_dt = getattr(ttnn, c.get("qk_dtype", "bfloat16"))  # lowbit: bfloat4_b q / k (inputs on its grid)
    on_grid = (lambda t: ref.bfp_grid(t).to(torch.bfloat16)) if qk_dt == ttnn.bfloat4_b else (lambda t: t)

    def inputs(seed):
        g = torch.Generator().manual_seed(seed)
        q = on_grid(torch.randn(1, H, S, D, generator=g).to(torch.bfloat16))
        wt = (torch.randn(1, 1, S, H, generator=g) * c["gate_scale"]).to(torch.bfloat16)
        k_nat = on_grid(torch.randn(T, D, generator=g).to(torch.bfloat16))
        garbage = torch.randn(1, 1, T, D, generator=g).to(torch.bfloat16)
        pos = ref.k_local_positions(T, sp, 1, W // r)
        k_local = k_nat[pos.reshape(-1)].reshape(sp, 1, T // sp, D)
        qd = q.reshape(1, H, sp, tp, Sq, D).permute(2, 3, 1, 4, 5, 0).reshape(sp, tp * H, Sq, D)
        wd = wt.reshape(1, 1, sp, tp, Sq, H).permute(2, 3, 0, 1, 4, 5).reshape(sp, tp, Sq, H)
        dev = {
            "q": put(qd, chips, qk_dt),
            "w": put(wd, chips),
            "k_local": put(k_local, ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=(sp, tp), dims=(0, None)), qk_dt),
        }
        host = {"q": q, "w": wt, "k": k_nat, "garbage": garbage}
        if "block_bias" in c:
            nb = c["block_bias"]["shape"][-1]
            keep = torch.rand(sp, tp, Sq, nb, generator=g) < 0.5
            host["keep"] = keep
            dev["block_bias"] = put(torch.where(keep, 0.0, float("-inf")).to(torch.bfloat16), chips)
        return host, dev

    def kbufs(host):
        """Fresh k buffers for every call on these inputs (the first, B or the A repeats), all written now."""
        return [
            put(host["garbage"], ttnn.ReplicateTensorToMesh(mesh_device), qk_dt)
            for _ in range(determinism.repeats() + 1)
        ]

    def call(host, dev):
        """One call on a fresh, pre-written copy of the k buffer's garbage (and the block-max buffer's sentinel, which
        only this chip writes)."""
        tt_k = host["kbufs"].pop()
        kw = {}
        bmax = None
        if bk:
            kw["block_keys"] = bk
        if "block_bias" in c:
            kw["block_bias"] = dev["block_bias"]
        if "block_max_out" in c:
            nbm = c["block_max_out"]["shape"][-1]
            bmax = put(torch.full((sp, tp, Sq, nbm), SENTINEL), chips, ttnn.float32, ttnn.ROW_MAJOR_LAYOUT)
            kw["block_max_out"] = bmax
            kw["block_max_pin"] = c["block_max_pin"]
        pc, ck = c["program_config"], c["compute_kernel_config"]
        out = ttnn.bringup.ring_indexer_score_dsa(
            dev["q"],
            tt_k,
            dev["w"],
            dev["k_local"],
            sems,
            cluster_axis=c["cluster_axis"],
            topology=getattr(ttnn.Topology, c["topology"]),
            num_links=c["num_links"],
            chunk_start_idx=start,
            key_compression_ratio=r,
            program_config=ttnn.bringup.IndexerScoreProgramConfig(
                q_chunk_size=pc["q_chunk_size"], k_chunk_size=pc["k_chunk_size"], head_group_size=pc["head_group_size"]
            ),
            compute_kernel_config=_kernel_config(ck),
            kv_len=kv,
            seq_subshard_axis=c["seq_subshard_axis"],
            block_cyclic_sp_axis=c["block_cyclic_sp_axis"],
            block_cyclic_chunk_local=W,
            natural_query_blocks=c["natural_query_blocks"],
            **kw,
        )
        return (out, tt_k) if bmax is None else (out, tt_k, bmax)

    host, dev = inputs(c["seed"])
    host_b, dev_b = inputs(c["seed"] + 1)  # built before the first call: its k buffers too
    host["kbufs"], host_b["kbufs"] = kbufs(host), kbufs(host_b)
    res = call(host, dev)
    outs = [ttnn.to_torch(t).float().reshape(Sq, T) for t in ttnn.get_device_tensors(res[0])]
    maxes = (
        [ttnn.to_torch(t).float().reshape(Sq, -1) for t in ttnn.get_device_tensors(res[2])] if len(res) > 2 else None
    )
    fid = c["compute_kernel_config"]["math_fidelity"]
    for i in range(sp * tp):
        d, cc = divmod(i, tp)
        blk = slice(d * W + cc * Sq, d * W + (cc + 1) * Sq)
        q0 = start + d * W + cc * Sq
        ins = [host["q"][0, :, blk], host["k"], host["w"][0, 0, blk]]
        want = ref.dsa_score_at(*[t.double() for t in ins], q0, key_stride=r)[:, :kv]
        got = outs[i][:, :kv]
        masked = torch.isneginf(want)
        if "block_bias" in c:
            masked = masked | ~host["keep"][d, cc][:, : kv // bk].repeat_interleave(bk, dim=-1)
        n_mask = int((~torch.isneginf(got[masked])).sum())
        n_bad = int((~torch.isfinite(got[~masked])).sum())
        assert n_mask == 0, f"chip {i}: {n_mask} future / masked columns not -inf"
        assert n_bad == 0, f"chip {i}: {n_bad} visible columns not finite"
        vis = ~masked

        def fn(qq, kk, ww):
            s = ref.dsa_score_at(qq, kk, ww, q0, key_stride=r, fidelity=fid)[:, :kv]
            return s[vis]

        lim = noise_limit(fn, ins, c["noise"]["inputs"], c["noise"]["out"])
        a, b = got[vis].double(), want[vis]
        rel = float((a - b).norm() / b.norm())
        pa, pb = a - a.mean(), b - b.mean()
        pcc = float(pa @ pb / (pa.norm() * pb.norm()))
        logger.info(
            f"{c['id']} chip {i}: tokens from {q0}, {int(vis.sum())} visible, rel {rel:.3e} (limit {lim['rel']:.3e}, "
            f"noise {lim['noise_rel']:.3e}) pcc {pcc:.7f} (a diagnostic)"
        )
        assert rel <= lim["rel"], f"chip {i}: rel L2 {rel:.3e} > limit {lim['rel']:.3e}"
        if maxes is not None:
            nbv = kv // bk
            wm = got.reshape(Sq, nbv, bk).amax(-1)  # the op's own bf16 scores (as fp32)
            if c["block_max_pin"]:
                pos = q0 + torch.arange(Sq)
                wm[torch.arange(Sq), ((pos + 1) // r - 1) // bk] = float("inf")
            assert torch.equal(maxes[i][:, :nbv], wm), f"chip {i}: block max differs from the op's own scores"
            assert bool((maxes[i][:, nbv:] == SENTINEL).all()), f"chip {i}: block max past kv_len written"
    determinism.assert_deterministic(lambda: call(host, dev), lambda: call(host_b, dev_b), first=res, label=c["id"])
