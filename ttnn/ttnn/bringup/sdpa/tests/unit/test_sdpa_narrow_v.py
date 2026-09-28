# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""ttnn.bringup SDPA with a V head dim narrower than K's (non-MLA GQA), at MiMo-V2 shapes.

MiMo's attention has Q/K head dim 192 and V head dim 128; the source op requires V as wide as K, so the model pads V
to 192 with zeros. The fork takes V at its own width: a [.., 128] V gives a [.., 128] output. Covered here, each
against a float32 torch reference (PCC) and against the source op run the way MiMo runs it today, V zero-padded to
K's width with the output sliced back: the fork's output must be bit-identical to that (same QK, softmax and PV
arithmetic, only the padded columns are no longer computed):
- the plain causal path (chunk 0), the sliding-window path with an attention sink (window 128), and the chunked path
  on a shuffled paged cache with a KV prefix (chunk_start_idx as an int and as a device tensor);
- both compute paths: non-streaming (fp32 dest acc on, MiMo's "base") and streaming (fp32 dest acc off, MiMo's "A"
  for full layers and "S" for sliding ones);
- V as wide as K: the fork's output is bit-identical to the source op's (the program is unchanged);
- the program cache: V 192 and V 128 with the same Q/K are two programs;
- the refusals: V not a multiple of 32, V wider than K.
"""

import pytest
import torch

import ttnn

from tests.ttnn.utils_for_testing import comp_pcc

NQH, DK, DV = 16, 192, 128
# float32-exact: the chunked binding takes scale as a no-convert float, which refuses a double that float32 rounds.
SCALE = torch.tensor(DK**-0.5, dtype=torch.float32).item()
BLOCK = 64  # paged-cache page size (MiMo's KV_BLOCK)

# name -> (fidelity, fp32 dest acc, approx exp, q chunk, k chunk). fp32 dest acc off = the streaming kernel.
CONFIGS = {
    "nonstreaming": (ttnn.MathFidelity.HiFi4, True, False, 128, 128),  # MiMo "base"
    "streaming": (ttnn.MathFidelity.HiFi2, False, True, 512, 128),  # MiMo "A" (full layers)
    "streaming_S": (ttnn.MathFidelity.HiFi4, False, False, 128, 128),  # MiMo "S" (sliding layers)
}
PCC = 0.999


def _cfg(device, name):
    fid, fp32, approx, q, k = CONFIGS[name]
    prog = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=device.compute_with_storage_grid_size(),
        q_chunk_size=q,
        k_chunk_size=k,
        exp_approx_mode=approx,
    )
    ckc = ttnn.WormholeComputeKernelConfig(
        math_fidelity=fid, math_approx_mode=False, fp32_dest_acc_en=fp32, packer_l1_acc=False
    )
    return prog, ckc


def _dev(t, device):
    return ttnn.from_torch(
        t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )


def _bf16(*shape):
    return torch.randn(*shape).to(torch.bfloat16).float()


def reference(q, k, v, q_start=0, window=0, sink=None):
    """softmax(Q K^T * SCALE + mask [+ sink column]) V in float32, per head. q [1, NQH, Sq, DK] holds positions
    [q_start, q_start + Sq); k [1, nkv, Sk, DK], v [1, nkv, Sk, dv]; causal; window W: key j visible to query i iff
    i - W < j <= i. sink [NQH] (unscaled, like the op's attention_sink; its logit is sink * SCALE)."""
    nkv, sq, sk = k.shape[1], q.shape[2], k.shape[2]
    i = torch.arange(q_start, q_start + sq)[:, None]
    j = torch.arange(sk)[None, :]
    visible = j <= i
    if window:
        visible &= j > i - window
    out = []
    for h in range(NQH):
        g = h // (NQH // nkv)
        s = (q[0, h] @ k[0, g].T) * SCALE
        s = s.masked_fill(~visible, float("-inf"))
        if sink is not None:
            s = torch.cat([s, torch.full((sq, 1), float(sink[h]) * SCALE)], dim=-1)
        p = torch.softmax(s, dim=-1)[:, :sk]
        out.append(p @ v[0, g])
    return torch.stack(out)[None]


def _check(tt_out, ref, padded=None):
    """PCC against the torch reference; with `padded` (the source op's output for V zero-padded to K's width),
    bit-identical to its first V columns. (The relative error is not bounded here: at a long prefix the attention of
    random inputs is nearly uniform and the output small, so the bf16 output's relative error grows with the prefix,
    the same for the padded source.)"""
    out = ttnn.to_torch(tt_out).float()
    assert list(out.shape) == list(ref.shape), f"output shape {list(out.shape)} != {list(ref.shape)}"
    ok, pcc = comp_pcc(ref, out, PCC)
    rel = ((out - ref).norm() / ref.norm()).item()
    assert ok, f"PCC {pcc} < {PCC} (rel err {rel:.4f})"
    if padded is not None:
        src = ttnn.to_torch(padded).float()[..., : out.shape[-1]]
        assert torch.equal(out, src), f"differs from the padded-V source op: max diff {(out - src).abs().max()}"
    return pcc


def _pad_v(v):
    return torch.nn.functional.pad(v, (0, DK - v.shape[-1]))


def _paged(x, perm):
    """[1, nkv, S, D] contiguous -> [nb, nkv, BLOCK, D] paged, logical block b stored at physical perm[b]."""
    _, nkv, s, d = x.shape
    nb = s // BLOCK
    blocks = x[0].reshape(nkv, nb, BLOCK, d).transpose(0, 1)  # [nb(logical), nkv, BLOCK, D]
    out = torch.empty_like(blocks)
    out[perm] = blocks
    return out


# ---- plain causal (chunk 0)
@pytest.mark.parametrize("cfg", ["nonstreaming", "streaming"])
@pytest.mark.parametrize("nkv, s", [(1, 5120), (2, 2048)], ids=["nkv1-s5120", "nkv2-s2048"])
def test_causal_narrow_v(device, cfg, nkv, s):
    torch.manual_seed(0)
    q, k, v = _bf16(1, NQH, s, DK), _bf16(1, nkv, s, DK), _bf16(1, nkv, s, DV)
    prog, ckc = _cfg(device, cfg)
    out, padded = (
        op(
            _dev(q, device),
            _dev(k, device),
            _dev(vv, device),
            is_causal=True,
            scale=SCALE,
            program_config=prog,
            compute_kernel_config=ckc,
        )
        for op, vv in (
            (ttnn.bringup.scaled_dot_product_attention, v),
            (ttnn.transformer.scaled_dot_product_attention, _pad_v(v)),
        )
    )
    _check(out, reference(q, k, v), padded)


# ---- sliding window + attention sink (MiMo's sliding layers: window 128, 2 KV heads)
@pytest.mark.parametrize("cfg", ["nonstreaming", "streaming_S"])
def test_sliding_sink_narrow_v(device, cfg):
    torch.manual_seed(1)
    s, nkv, w = 1024, 2, 128
    q, k, v = _bf16(1, NQH, s, DK), _bf16(1, nkv, s, DK), _bf16(1, nkv, s, DV)
    sink = (torch.rand(NQH) * 4.0).to(torch.bfloat16).float()
    prog, ckc = _cfg(device, cfg)
    out, padded = (
        op(
            _dev(q, device),
            _dev(k, device),
            _dev(vv, device),
            is_causal=True,
            scale=SCALE,
            sliding_window_size=w,
            attention_sink=_dev(sink.reshape(1, NQH, 1, 1), device),
            program_config=prog,
            compute_kernel_config=ckc,
        )
        for op, vv in (
            (ttnn.bringup.scaled_dot_product_attention, v),
            (ttnn.transformer.scaled_dot_product_attention, _pad_v(v)),
        )
    )
    _check(out, reference(q, k, v, window=w, sink=sink), padded)


# ---- chunked prefill on a paged cache with a KV prefix
def _chunked_data(prefix, sq, nkv, dv, seed=2):
    torch.manual_seed(seed)
    total = prefix + sq
    q = _bf16(1, NQH, sq, DK)
    k, v = _bf16(1, nkv, total, DK), _bf16(1, nkv, total, dv)
    return (q, k, v), torch.randperm(total // BLOCK)


def _run_chunked(device, cfg, prefix, sq, nkv, dv, start_as_tensor, op=None, seed=2, data=None):
    (q, k, v), perm = data or _chunked_data(prefix, sq, nkv, dv, seed)
    page_table = perm[None].to(torch.int32)  # logical block b -> physical perm[b]
    tt_pt = ttnn.from_torch(page_table, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    prog, ckc = _cfg(device, cfg)
    op = op or ttnn.bringup.chunked_scaled_dot_product_attention
    args = (_dev(q, device), _dev(_paged(k, perm), device), _dev(_paged(v, perm), device), tt_pt)
    kw = dict(scale=SCALE, program_config=prog, compute_kernel_config=ckc)
    if start_as_tensor:
        idx = ttnn.from_torch(torch.tensor([prefix], dtype=torch.int32), dtype=ttnn.int32, device=device)
        out = op(*args, chunk_start_idx_tensor=idx, **kw)
    else:
        out = op(*args, prefix, **kw)
    return out, (q, k, v)


@pytest.mark.parametrize("cfg", ["nonstreaming", "streaming"])
@pytest.mark.parametrize("start_as_tensor", [False, True], ids=["start_int", "start_tensor"])
def test_chunked_paged_narrow_v(device, cfg, start_as_tensor):
    _chunked_case(device, cfg, 4096, 1024, start_as_tensor)


def test_chunked_paged_narrow_v_50k_prefix(device):
    """MiMo's full-attention call at a 49k prefix: 1 KV head, q512/k128 streaming, chunk_start_idx an int."""
    _chunked_case(device, "streaming", 49152, 1024, False)


def _chunked_case(device, cfg, prefix, sq, start_as_tensor):
    (q, k, v), perm = _chunked_data(prefix, sq, 1, DV)
    out, _ = _run_chunked(device, cfg, prefix, sq, 1, DV, start_as_tensor, data=((q, k, v), perm))
    padded, _ = _run_chunked(
        device,
        cfg,
        prefix,
        sq,
        1,
        DK,
        start_as_tensor,
        op=ttnn.transformer.chunked_scaled_dot_product_attention,
        data=((q, k, _pad_v(v)), perm),
    )
    _check(out, reference(q, k, v, q_start=prefix), padded)


# ---- V as wide as K: the fork's program is the source op's (bit-identical output)
@pytest.mark.parametrize("cfg", ["nonstreaming", "streaming"])
def test_v_equal_k_matches_source_causal(device, cfg):
    torch.manual_seed(3)
    s, nkv = 1024, 1
    q, k, v = _bf16(1, NQH, s, DK), _bf16(1, nkv, s, DK), _bf16(1, nkv, s, DK)
    prog, ckc = _cfg(device, cfg)
    outs = []
    for op in (ttnn.transformer.scaled_dot_product_attention, ttnn.bringup.scaled_dot_product_attention):
        o = op(
            _dev(q, device),
            _dev(k, device),
            _dev(v, device),
            is_causal=True,
            scale=SCALE,
            program_config=prog,
            compute_kernel_config=ckc,
        )
        outs.append(ttnn.to_torch(o))
    assert torch.equal(outs[0], outs[1]), "fork differs from the source op with V as wide as K"
    _check(ttnn.from_torch(outs[1]), reference(q, k, v))


@pytest.mark.parametrize("cfg", ["nonstreaming", "streaming"])
def test_v_equal_k_matches_source_chunked(device, cfg):
    outs = []
    for op in (
        ttnn.transformer.chunked_scaled_dot_product_attention,
        ttnn.bringup.chunked_scaled_dot_product_attention,
    ):
        o, (q, k, v) = _run_chunked(device, cfg, 2048, 1024, 1, DK, False, op=op, seed=4)
        outs.append(ttnn.to_torch(o))
    assert torch.equal(outs[0], outs[1]), "fork differs from the source op with V as wide as K"


# ---- program cache: V 192 and V 128 with the same Q/K are different programs, both right
def test_program_cache_v_width(device):
    torch.manual_seed(5)
    s, nkv = 1024, 1
    q, k = _bf16(1, NQH, s, DK), _bf16(1, nkv, s, DK)
    prog, ckc = _cfg(device, "streaming")
    tq, tk = _dev(q, device), _dev(k, device)
    device.clear_program_cache()
    for dv in (DK, DV, DK, DV):
        v = _bf16(1, nkv, s, dv)
        out = ttnn.bringup.scaled_dot_product_attention(
            tq, tk, _dev(v, device), is_causal=True, scale=SCALE, program_config=prog, compute_kernel_config=ckc
        )
        _check(out, reference(q, k, v))
    assert device.num_program_cache_entries() == 2, device.num_program_cache_entries()


# ---- refusals
@pytest.mark.parametrize("dv", [112, 224], ids=["v112-not-tile-multiple", "v224-wider-than-k"])
@pytest.mark.parametrize("path", ["causal", "chunked"])
def test_refuses_bad_v_width(device, expect_error, dv, path):
    s, nkv = 256, 1
    prog, ckc = _cfg(device, "nonstreaming")
    q, k, v = _bf16(1, NQH, s, DK), _bf16(1, nkv, s, DK), _bf16(1, nkv, s, dv)
    msg = "V head dim must be a multiple of 32 and at most K's head dim"
    if path == "causal":
        with expect_error(RuntimeError, msg):
            ttnn.bringup.scaled_dot_product_attention(
                _dev(q, device),
                _dev(k, device),
                _dev(v, device),
                is_causal=True,
                scale=SCALE,
                program_config=prog,
                compute_kernel_config=ckc,
            )
    else:
        perm = torch.arange(s // BLOCK)
        tt_pt = ttnn.from_torch(
            perm[None].to(torch.int32), dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device
        )
        with expect_error(RuntimeError, msg):
            ttnn.bringup.chunked_scaled_dot_product_attention(
                _dev(_bf16(1, NQH, 128, DK), device),
                _dev(_paged(k, perm), device),
                _dev(_paged(v, perm), device),
                tt_pt,
                128,
                scale=SCALE,
                program_config=prog,
                compute_kernel_config=ckc,
            )
