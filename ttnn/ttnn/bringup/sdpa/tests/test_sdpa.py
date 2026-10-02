# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""ttnn.bringup.scaled_dot_product_attention / chunked_scaled_dot_product_attention / sparse_sdpa against their torch semantics
(reference.py), one random-input case per captured call (cases.py). Math: PCC plus a bound on the relative L2 error,
per device. Every input differs per device (sharded on dim 0 over the mesh) so each chip is checked on its own data.
The chunked case uses a random permutation as the page table (the model's is the identity; any valid table must work).
"""

import importlib.util
from pathlib import Path

import pytest
import torch

import ttnn
from models.demos.common.bringup.testing import determinism

_HERE = Path(__file__).resolve().parent


def _load(name):
    spec = importlib.util.spec_from_file_location(f"bringup_sdpa_tests_{name}", _HERE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ref = _load("reference")
CASES = _load("cases").CASES


def _device_params(c):
    p = dict(c["device_params"])
    p["fabric_config"] = getattr(ttnn.FabricConfig, p["fabric_config"])
    # A case runs on a box of its own mesh size only (conftest skips it elsewhere): a smaller mesh opened on a bigger
    # box fails the FABRIC_2D router handshake (e.g. a 2x2 case on a 4x2 box), and the case's math depends on its mesh.
    p["require_exact_physical_num_devices"] = True
    return p


def _pcc(a, b):
    a, b = a.double().flatten(), b.double().flatten()
    a, b = a - a.mean(), b - b.mean()
    return float((a @ b) / (a.norm() * b.norm()))


def _shard(mesh, t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT):
    return ttnn.from_torch(
        t,
        dtype=dtype,
        layout=layout,
        device=mesh,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=0),
    )


def _randn(g, n_dev, shape):
    """n_dev per-device tensors of `shape`, bf16-rounded, concatenated on dim 0."""
    return torch.randn([n_dev * shape[0], *shape[1:]], generator=g).to(torch.bfloat16)


def _sparse_sdpa(mesh_device, c):
    """ttnn.bringup.sparse_sdpa: q [1, H, S, K_DIM] and the latent cache kv [1, 1, T, K_DIM] bf16 ROW_MAJOR, idx
    [1, 1, S, W] uint32 ROW_MAJOR, per device. Query row s sits at position q_pos + d * S + s (chip d holds rows d S ..
    of the chunk, as the model splits it); its ids are distinct, causal (<= its position) and random, the first
    n_valid slots, then the 0xFFFFFFFF sentinel. Every n_short_every-th row has fewer valid ids (partly and wholly
    masked k_chunks). Checked per device vs the float32 reference on the same bf16 inputs: PCC, rel L2 and the per
    (head, row) output norm ratio."""
    rows, cols = c["mesh"]
    n_dev = rows * cols
    _, nh, sq, kd = c["q"]
    (q, kv, idx), dev = _sparse_inputs(mesh_device, c, c["seed"])
    out = _sparse_call(c, dev)
    outs = [ttnn.to_torch(t).float() for t in ttnn.get_device_tensors(out)]
    assert len(outs) == n_dev
    for d in range(n_dev):
        got = outs[d]
        assert list(got.shape) == [1, nh, sq, c["v_dim"]], f"dev {d}: output shape {list(got.shape)}"
        want = ref.sparse_sdpa(q[d : d + 1], kv[d : d + 1], idx[d : d + 1], scale=c["scale"], v_dim=c["v_dim"])
        pcc = _pcc(got, want)
        rel = float((got - want).norm() / want.norm())
        ratio = got.norm(dim=-1) / want.norm(dim=-1)
        lo, hi = float(ratio.min()), float(ratio.max())
        print(f"dev {d}: pcc {pcc:.7f} rel L2 err {rel:.5f} per-row norm ratio [{lo:.4f}, {hi:.4f}]")
        assert pcc >= c["pcc"], f"dev {d}: pcc {pcc} < {c['pcc']}"
        assert rel <= c["rel"], f"dev {d}: rel L2 err {rel} > {c['rel']}"
        assert c["ratio"][0] <= lo and hi <= c["ratio"][1], f"dev {d}: norm ratio [{lo}, {hi}] outside {c['ratio']}"
    dev_b = _sparse_inputs(mesh_device, c, c["seed"] + 1)[1]
    determinism.assert_deterministic(
        lambda: _sparse_call(c, dev), lambda: _sparse_call(c, dev_b), first=out, label=c["id"]
    )


def _sparse_inputs(mesh_device, c, seed):
    """Host (q, kv, idx) for `seed` and the device (q, kv, idx, compute kernel config)."""
    rows, cols = c["mesh"]
    n_dev = rows * cols
    g = torch.Generator().manual_seed(seed)
    _, nh, sq, kd = c["q"]
    t_kv = c["kv"][2]
    w = c["indices"][-1]
    q = (torch.randn(n_dev, nh, sq, kd, generator=g) * c.get("q_scale", 1.0)).to(torch.bfloat16)
    kv = torch.randn(n_dev, 1, t_kv, kd, generator=g).to(torch.bfloat16)
    idx = torch.full((n_dev, 1, sq, w), -1, dtype=torch.int64)
    for d in range(n_dev):
        pos = c["q_pos"] + d * sq + torch.arange(sq)
        assert int(pos.max()) < t_kv
        r = torch.rand(sq, t_kv, generator=g)
        r[torch.arange(t_kv)[None, :] > pos[:, None]] = -1.0  # never pick a future row
        nv = torch.full((sq,), c["n_valid"], dtype=torch.int64)
        short = torch.arange(sq) % c["n_short_every"] == 0
        nv[short] = torch.randint(1, c["n_valid"] + 1, (int(short.sum()),), generator=g)
        top = r.topk(c["n_valid"], dim=-1).indices
        keep = torch.arange(c["n_valid"])[None, :] < nv[:, None]
        idx[d, 0, :, : c["n_valid"]] = torch.where(keep, top, -1)

    ck = c["compute_kernel_config"]
    ckc = ttnn.WormholeComputeKernelConfig(
        math_fidelity=getattr(ttnn.MathFidelity, ck["math_fidelity"]),
        math_approx_mode=ck["math_approx_mode"],
        fp32_dest_acc_en=ck["fp32_dest_acc_en"],
        packer_l1_acc=ck["packer_l1_acc"],
        dst_full_sync_en=ck["dst_full_sync_en"],
    )
    rm = ttnn.ROW_MAJOR_LAYOUT
    tq, tkv = _shard(mesh_device, q, layout=rm), _shard(mesh_device, kv, layout=rm)
    ti = _shard(mesh_device, idx.to(torch.int32), dtype=ttnn.uint32, layout=rm)
    assert list(tq.shape) == c["q"] and list(tkv.shape) == c["kv"] and list(ti.shape) == c["indices"]
    return (q, kv, idx), (tq, tkv, ti, ckc)


def _sparse_call(c, dev):
    tq, tkv, ti, ckc = dev
    return ttnn.bringup.sparse_sdpa(
        tq,
        tkv,
        ti,
        c["v_dim"],
        kv_format=getattr(ttnn.bringup.SparseKVFormat, c["kv_format"]),
        scale=c["scale"],
        k_chunk_size=c["k_chunk_size"],
        compute_kernel_config=ckc,
        high_precision=c["high_precision"],
    )


def _ring_mla(mesh_device, c):
    """ttnn.bringup.ring_mla (latent-V ring attention over cluster_axis 0, the mesh rows). The chunk's queries
    [isl, isl + chunk) are split over the rows in order (row r holds isl + r * chunk / rows ..), the heads over the
    mesh columns (column col holds heads col * H ..); the latent cache holds keys [0, logical_n) in the block-cyclic
    order (period = chunk, row r's shard holds positions slab * chunk + r * chunk / rows + i), replicated over the
    columns, ND-sharded over the DRAM banks as captured. K = the 576 columns, V = the first head_dim_v. Every output
    row of every head of both columns is checked vs a float32 causal reference on the same bf16 inputs: PCC, rel L2
    and the worst row's relative error."""
    rows, cols = c["mesh"]
    _, nh, sq, kd = c["q"]
    dv, chunk, isl, n = c["head_dim_v"], c["chunk"], c["kv_actual_isl"], c["logical_n"]
    local = c["kv"][2]
    max_seq = local * rows
    assert sq * rows == chunk and n == isl + chunk and c["persistent_output_buffer_kv"][2] == max_seq
    gr = mesh_device.compute_with_storage_grid_size()
    cores = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(gr.x - 1, gr.y - 1))})
    sems = [ttnn.create_global_semaphore(mesh_device, cores, 0) for _ in range(2)]
    (q, kv), dev = _mla_inputs(mesh_device, c, c["seed"])
    out, stats, buf = _mla_call(mesh_device, c, dev, sems)
    devs = [ttnn.to_torch(t).float() for t in ttnn.get_device_tensors(out)]
    assert len(devs) == rows * cols
    for d in devs:
        assert list(d.shape) == [1, nh, sq, dv], d.shape
    keys = kv[:n].float()
    causal = torch.arange(n)[None, :] > torch.arange(isl, n)[:, None]  # [chunk, n]
    for col in range(cols):
        got = torch.cat([devs[r * cols + col] for r in range(rows)], dim=2)[0]  # [nh, chunk, dv]
        want = torch.empty_like(got)
        for h in range(nh):
            s_ = (q[0, col * nh + h].float() @ keys.T) * c["scale"]
            want[h] = s_.masked_fill_(causal, float("-inf")).softmax(-1) @ keys[:, :dv]
        pcc = _pcc(got, want)
        rel = float((got - want).norm() / want.norm())
        row = float(((got - want).norm(dim=-1) / want.norm(dim=-1)).max())
        print(f"column {col}: pcc {pcc:.7f} rel L2 err {rel:.5f} worst row {row:.5f}")
        assert pcc >= c["pcc"], f"column {col}: pcc {pcc} < {c['pcc']}"
        assert rel <= c["rel"], f"column {col}: rel L2 err {rel} > {c['rel']}"
        assert row <= c["row"], f"column {col}: worst row rel err {row} > {c['row']}"
    # The op gathers the ring's KV into persistent_output_buffer_kv (an input): each call gets a fresh zeroed buffer
    # (_mla_call), and the gathered buffer is compared too.
    dev_b = _mla_inputs(mesh_device, c, c["seed"] + 1)[1]
    determinism.assert_deterministic(
        lambda: _mla_call(mesh_device, c, dev, sems),
        lambda: _mla_call(mesh_device, c, dev_b, sems),
        first=(out, stats, buf),
        label=c["id"],
    )
    for t in (*dev, buf, out, stats):
        ttnn.deallocate(t)


def _mla_inputs(mesh_device, c, seed):
    """Host (q, natural-order kv) for `seed` and the device (q, block-cyclic ND-sharded latent cache)."""
    rows, cols = c["mesh"]
    _, nh, sq, kd = c["q"]
    chunk, n = c["chunk"], c["logical_n"]
    local = c["kv"][2]
    max_seq = local * rows
    g = torch.Generator().manual_seed(seed)
    q = (torch.randn(1, nh * cols, chunk, kd, generator=g) * c["q_scale"]).to(torch.bfloat16)
    kv = torch.zeros(max_seq, kd)
    kv[:n] = torch.randn(n, kd, generator=g)
    kv = kv.to(torch.bfloat16)

    # block-cyclic shard order: shard row -> natural position (deepseek_v3_d_p tt/mla/utils.blockcyclic_positions)
    cl = chunk // rows
    r_of = torch.arange(rows).repeat_interleave(local)
    lr = torch.arange(local).repeat(rows)
    pos = (lr // cl) * chunk + r_of * cl + lr % cl
    cache = kv[pos].reshape(1, 1, max_seq, kd)
    nd = ttnn.MemoryConfig(
        buffer_type=ttnn.BufferType.DRAM,
        nd_shard_spec=ttnn.NdShardSpec(
            shard_shape=[1, 1, 32, kd],
            grid=ttnn.CoreRangeSet(
                [
                    ttnn.CoreRange(ttnn.CoreCoord(b, 0), ttnn.CoreCoord(b, 0))
                    for b in range(mesh_device.dram_grid_size().x)
                ]
            ),
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            shard_distribution_strategy=ttnn.ShardDistributionStrategy.ROUND_ROBIN_1D,
        ),
    )
    shard = lambda dims: ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=tuple(mesh_device.shape), dims=dims)
    tq = ttnn.from_torch(
        q,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=shard((2, 1)),
    )
    tkv = ttnn.from_torch(
        cache,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=nd,
        mesh_mapper=shard((2, None)),
    )
    assert list(tq.shape) == c["q"] and list(tkv.shape) == c["kv"], (tq.shape, tkv.shape)
    assert "ND_SHARDED" in str(tkv.memory_config().memory_layout), tkv.memory_config()
    return (q, kv), (tq, tkv)


def _mla_call(mesh_device, c, dev, sems):
    """One call on a fresh zeroed persistent KV buffer; returns (out, stats, the gathered buffer)."""
    tq, tkv = dev
    buf = ttnn.from_torch(
        torch.zeros(c["persistent_output_buffer_kv"]),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    ck, pc = c["compute_kernel_config"], c["program_config"]
    out, stats = ttnn.bringup.ring_mla(
        tq,
        tkv,
        persistent_output_buffer_kv=buf,
        head_dim_v=c["head_dim_v"],
        logical_n=c["logical_n"],
        program_config=ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(*pc["grid"]),
            q_chunk_size=pc["q_chunk_size"],
            k_chunk_size=pc["k_chunk_size"],
            exp_approx_mode=pc["exp_approx_mode"],
            max_cores_per_head_batch=16,
        ),
        scale=c["scale"],
        compute_kernel_config=ttnn.WormholeComputeKernelConfig(
            math_fidelity=getattr(ttnn.MathFidelity, ck["math_fidelity"]),
            math_approx_mode=ck["math_approx_mode"],
            fp32_dest_acc_en=ck["fp32_dest_acc_en"],
            packer_l1_acc=ck["packer_l1_acc"],
            dst_full_sync_en=ck["dst_full_sync_en"],
        ),
        dim=c["dim"],
        multi_device_global_semaphore=sems,
        num_links=c["num_links"],
        cluster_axis=c["cluster_axis"],
        mesh_device=mesh_device,
        topology=getattr(ttnn.Topology, c["topology"]),
        ccl_core_grid_offset=tuple(c["ccl_core_grid_offset"]),
        use_column_major_ccl=c["use_column_major_ccl"],
        is_balanced=c["is_balanced"],
        kv_cache_batch_idx=c["kv_cache_batch_idx"],
        kv_actual_isl=c["kv_actual_isl"],
    )
    return out, stats, buf


@pytest.mark.timeout(1200)
@pytest.mark.parametrize(
    "mesh_device, device_params, case",
    [(tuple(c["mesh"]), _device_params(c), c) for c in CASES],
    ids=[c["id"] for c in CASES],
    indirect=["mesh_device", "device_params"],
)
def test_sdpa(mesh_device, device_params, case):
    c = case
    if c["op"] == "sparse_sdpa":
        return _sparse_sdpa(mesh_device, c)
    if c["op"] == "ring_mla":
        return _ring_mla(mesh_device, c)
    rows, cols = c["mesh"]
    n_dev = rows * cols
    (q, k, v, pts, sink), dev = _inputs(mesh_device, c, c["seed"])
    fork, source = _ops(c)
    # The source op (ttnn.transformer) takes V only as wide as K: it runs on V zero-padded to K's width, and the fork's
    # output must equal its first V columns bit for bit (same QK / softmax / PV arithmetic; tests/unit does the same).
    outs_ = [_call(c, dev, fork)] + ([_call(c, dev, source, dev["v_pad"])] if dev["v_pad"] is not None else [])
    out = outs_[0]
    src = [ttnn.to_torch(t).float() for t in ttnn.get_device_tensors(outs_[1])] if len(outs_) > 1 else None

    outs = [ttnn.to_torch(t).float() for t in ttnn.get_device_tensors(out)]
    assert len(outs) == n_dev
    sq = c["q"][2]
    want_shape = [1, c["q"][1], sq, c["v"][-1]]
    kb = c["k"][0]
    for d in range(n_dev):
        got = outs[d]
        assert list(got.shape) == want_shape, f"dev {d}: output shape {list(got.shape)} != {want_shape}"
        qd, kd, vd = q[d : d + 1], k[d * kb : (d + 1) * kb], v[d * kb : (d + 1) * kb]
        if pts is not None:
            kd, vd = ref.unpage(kd, pts[d]), ref.unpage(vd, pts[d])
        want = ref.sdpa(
            qd,
            kd,
            vd,
            scale=c["scale"],
            q_start=c["chunk_start_idx"] or 0,
            window=c["sliding_window_size"] or 0,
            sink=sink[d].flatten() if sink is not None else None,
        )
        if src is not None:
            s_d = src[d][..., : got.shape[-1]]
            same = torch.equal(got, s_d)
            assert same, f"dev {d}: differs from the padded-V source op, max {(got - s_d).abs().max()}"
        pcc = _pcc(got, want)
        rel = float((got - want).norm() / want.norm())
        print(f"dev {d}: pcc {pcc:.7f} rel L2 err {rel:.5f} max abs err {float((got - want).abs().max()):.4g}")
        assert pcc >= c["pcc"], f"dev {d}: pcc {pcc} < {c['pcc']}"
        if c["rel"] is not None:
            assert rel <= c["rel"], f"dev {d}: rel L2 err {rel} > {c['rel']}"
    dev_b = _inputs(mesh_device, c, c["seed"] + 1)[1]
    determinism.assert_deterministic(
        lambda: _call(c, dev, fork), lambda: _call(c, dev_b, fork), first=out, label=c["id"]
    )


def _ops(c):
    """(the fork's op, the source op) of a scaled_dot_product_attention / chunked_... case."""
    return getattr(ttnn.bringup, c["op"]), getattr(ttnn.transformer, c["op"])


def _inputs(mesh_device, c, seed):
    """Host (q, k, v, page tables, sink) for `seed` and the device inputs (with V zero-padded to K's width for the
    source op, None when V is as wide as K) and configs."""
    rows, cols = c["mesh"]
    n_dev = rows * cols
    g = torch.Generator().manual_seed(seed)
    q, k, v = _randn(g, n_dev, c["q"]), _randn(g, n_dev, c["k"]), _randn(g, n_dev, c["v"])

    ck = c["compute_kernel_config"]
    ckc = ttnn.WormholeComputeKernelConfig(
        math_fidelity=getattr(ttnn.MathFidelity, ck["math_fidelity"]),
        math_approx_mode=ck["math_approx_mode"],
        fp32_dest_acc_en=ck["fp32_dest_acc_en"],
        packer_l1_acc=ck["packer_l1_acc"],
        dst_full_sync_en=ck["dst_full_sync_en"],
    )
    pc = c["program_config"]
    prog = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(*pc["grid"]),
        q_chunk_size=pc["q_chunk_size"],
        k_chunk_size=pc["k_chunk_size"],
        exp_approx_mode=pc["exp_approx_mode"],
        max_cores_per_head_batch=16,
    )
    dev = {"q": _shard(mesh_device, q), "k": _shard(mesh_device, k), "v": _shard(mesh_device, v)}
    dev.update(prog=prog, ckc=ckc, page_table=None, extra={})
    pts, sink = None, None
    pad = c["k"][-1] - c["v"][-1]
    dev["v_pad"] = _shard(mesh_device, torch.nn.functional.pad(v, (0, pad))) if pad else None

    if c["op"] == "chunked_scaled_dot_product_attention":
        nb = c["page_table"][1]
        pts = torch.stack([torch.randperm(nb, generator=g) for _ in range(n_dev)]).to(torch.int32)  # [n_dev, nb]
        dev["page_table"] = _shard(mesh_device, pts, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT)
    else:
        if c["attention_sink"] is not None:
            logit = torch.rand([n_dev, *c["attention_sink"][1:]], generator=g) * 3.0
            sink = (logit / c["scale"]).to(torch.bfloat16)  # stored pre-divided, like the model's
            dev["extra"]["attention_sink"] = _shard(mesh_device, sink)
        if c["sliding_window_size"]:
            dev["extra"]["sliding_window_size"] = c["sliding_window_size"]
    return (q, k, v, pts, sink), dev


def _call(c, dev, op, v=None):
    """op on the case's device inputs; v replaces V (the padded V of the source op)."""
    v = dev["v"] if v is None else v
    if c["op"] == "chunked_scaled_dot_product_attention":
        return op(
            input_tensor_q=dev["q"],
            input_tensor_k=dev["k"],
            input_tensor_v=v,
            page_table_tensor=dev["page_table"],
            chunk_start_idx=c["chunk_start_idx"],
            scale=c["scale"],
            program_config=dev["prog"],
            compute_kernel_config=dev["ckc"],
        )
    return op(
        dev["q"],
        dev["k"],
        v,
        is_causal=c["is_causal"],
        scale=c["scale"],
        program_config=dev["prog"],
        compute_kernel_config=dev["ckc"],
        **dev["extra"],
    )
