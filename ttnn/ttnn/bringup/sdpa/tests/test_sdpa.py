# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""ttnn.bringup.scaled_dot_product_attention / chunked_scaled_dot_product_attention against their torch semantics
(reference.py), one random-input case per captured call (cases.py). Math: PCC plus a bound on the relative L2 error,
per device. Every input differs per device (sharded on dim 0 over the mesh) so each chip is checked on its own data.
The chunked case uses a random permutation as the page table (the model's is the identity; any valid table must work)."""

import importlib.util
from pathlib import Path

import pytest
import torch

import ttnn

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


@pytest.mark.timeout(1200)
@pytest.mark.parametrize(
    "mesh_device, device_params, case",
    [(tuple(c["mesh"]), _device_params(c), c) for c in CASES],
    ids=[c["id"] for c in CASES],
    indirect=["mesh_device", "device_params"],
)
def test_sdpa(mesh_device, device_params, case):
    c = case
    rows, cols = c["mesh"]
    n_dev = rows * cols
    g = torch.Generator().manual_seed(c["seed"])
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
    tq, tk, tv = _shard(mesh_device, q), _shard(mesh_device, k), _shard(mesh_device, v)
    pts, sink = None, None

    # The source op (ttnn.transformer) takes V only as wide as K: it runs on V zero-padded to K's width, and the fork's
    # output must equal its first V columns bit for bit (same QK / softmax / PV arithmetic; tests/unit does the same).
    pad = c["k"][-1] - c["v"][-1]
    tv_pad = _shard(mesh_device, torch.nn.functional.pad(v, (0, pad))) if pad else None

    if c["op"] == "chunked_scaled_dot_product_attention":
        nb = c["page_table"][1]
        pts = torch.stack([torch.randperm(nb, generator=g) for _ in range(n_dev)]).to(torch.int32)  # [n_dev, nb]
        tpt = _shard(mesh_device, pts, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT)
        outs_ = []
        for op, vv in (
            (ttnn.bringup.chunked_scaled_dot_product_attention, tv),
            (ttnn.transformer.chunked_scaled_dot_product_attention, tv_pad),
        ):
            if vv is None:
                continue
            outs_.append(
                op(
                    input_tensor_q=tq,
                    input_tensor_k=tk,
                    input_tensor_v=vv,
                    page_table_tensor=tpt,
                    chunk_start_idx=c["chunk_start_idx"],
                    scale=c["scale"],
                    program_config=prog,
                    compute_kernel_config=ckc,
                )
            )
    else:
        extra = {}
        if c["attention_sink"] is not None:
            logit = torch.rand([n_dev, *c["attention_sink"][1:]], generator=g) * 3.0
            sink = (logit / c["scale"]).to(torch.bfloat16)  # stored pre-divided, like the model's
            extra["attention_sink"] = _shard(mesh_device, sink)
        if c["sliding_window_size"]:
            extra["sliding_window_size"] = c["sliding_window_size"]
        outs_ = [
            op(
                tq,
                tk,
                vv,
                is_causal=c["is_causal"],
                scale=c["scale"],
                program_config=prog,
                compute_kernel_config=ckc,
                **extra,
            )
            for op, vv in (
                (ttnn.bringup.scaled_dot_product_attention, tv),
                (ttnn.transformer.scaled_dot_product_attention, tv_pad),
            )
            if vv is not None
        ]
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
