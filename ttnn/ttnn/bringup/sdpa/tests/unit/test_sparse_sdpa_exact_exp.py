# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""ttnn.bringup.sparse_sdpa(exact_exp=True): the softmax score exp in the accurate form.

The source op's fast (Schraudolph) score exp over-estimates exp by ~1 % on average. With a sink that takes a large
share of the softmax, every key's probability is then too large relative to the sink, and vice versa; the
per-head-row norm of the output carries a bias that grows with the sink share. exact_exp removes that part; the op's
bf16 running output / row-sum (unchanged) still leaves a smaller one.
"""

import pytest

import ttnn
from tests.ttnn.unit_tests.operations.sdpa.sparse_sdpa_test_utils import golden, make_inputs, pcc, to_dev

pytestmark = pytest.mark.use_module_device

K_DIM, V_DIM = 576, 512


def _hifi4():
    return ttnn.types.BlackholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=False
    )


def _run(device, q, kv, idx, sink, scale, kc, exact_exp):
    import torch  # local: tt-metal's check-torch-imports-in-ttnn forbids a global torch import under ttnn/

    out = ttnn.bringup.sparse_sdpa(
        to_dev(q, device, ttnn.bfloat16),
        to_dev(kv, device, ttnn.bfloat16),
        to_dev(idx.to(torch.int32), device, ttnn.uint32),
        V_DIM,
        kv_format=ttnn.bringup.SparseKVFormat.BF16,
        scale=scale,
        k_chunk_size=kc,
        compute_kernel_config=_hifi4(),
        attention_sink=None if sink is None else to_dev(sink, device, ttnn.bfloat16),
        exact_exp=exact_exp,
    )
    return ttnn.to_torch(out).float()


def _ratio(out, ref):
    """Per head-row projection of out on ref: 1 = no scale error."""
    return (out * ref).sum(-1) / (ref * ref).sum(-1)


@pytest.mark.parametrize(
    "H,S,T,TOPK,kc,nvalid",
    [
        pytest.param(64, 160, 4096, 2048, 128, "full", id="hy4-geometry-full"),
        pytest.param(64, 64, 2048, 2048, 128, "causal", id="hy4-geometry-sentinel-tail"),
        pytest.param(32, 32, 256, 128, 32, "causal", id="small"),
    ],
)
def test_exact_exp_removes_sink_bias(device, H, S, T, TOPK, kc, nvalid):
    import torch  # local: tt-metal's check-torch-imports-in-ttnn forbids a global torch import under ttnn/

    n_fn = (lambda s: TOPK) if nvalid == "full" else (lambda s: 1 + (s * 37) % TOPK)
    q, kv, idx = make_inputs(H, S, T, TOPK, K_DIM, n_fn, seed=7)
    scale = K_DIM**-0.5
    q, kv = (q * 2).to(torch.bfloat16), kv.to(torch.bfloat16)
    # sinks from negligible to dominant (scaled-logit domain, passed as sink / scale)
    sink_logit = torch.linspace(-2, 9, H)
    sink = (sink_logit / scale).reshape(1, 1, 1, H).to(torch.bfloat16)
    ref = golden(q.float(), kv.float(), idx, scale, V_DIM, attention_sink=sink.float().reshape(1, H, 1, 1))
    fast = _run(device, q, kv, idx, sink, scale, kc, False)
    exact = _run(device, q, kv, idx, sink, scale, kc, True)
    r_fast, r_exact = _ratio(fast, ref), _ratio(exact, ref)
    rel = lambda o: ((o - ref).norm() / ref.norm()).item()  # noqa: E731
    print(
        f"fast: rel {rel(fast):.5f} ratio mean {r_fast.mean():.5f}; exact: rel {rel(exact):.5f} "
        f"ratio mean {r_exact.mean():.5f} min {r_exact.min():.5f} max {r_exact.max():.5f}"
    )
    assert pcc(exact, ref) >= 0.999
    # what remains is the op's bf16 running output / row-sum (measured here: -0.57 % / -0.36 % / +0.18 %)
    assert abs(r_exact.mean().item() - 1) <= 0.0075, "exact_exp leaves a large systematic scale"
    assert abs(r_exact.mean().item() - 1) < abs(r_fast.mean().item() - 1), "exact_exp is not closer than the fast exp"
    assert rel(exact) <= rel(fast) + 1e-4


def test_exact_exp_no_sink_and_masked_keys(device):
    import torch  # local: tt-metal's check-torch-imports-in-ttnn forbids a global torch import under ttnn/

    """No sink, rows with a single valid key (output = that key's latent) and sentinel tails: -inf -> exp 0."""
    H, S, T, TOPK, kc = 32, 32, 256, 128, 32
    q, kv, idx = make_inputs(H, S, T, TOPK, K_DIM, lambda s: 1 if s < 4 else s * 3, seed=3)
    q, kv = q.to(torch.bfloat16), kv.to(torch.bfloat16)
    scale = K_DIM**-0.5
    ref = golden(q.float(), kv.float(), idx, scale, V_DIM)
    exact = _run(device, q, kv, idx, None, scale, kc, True)
    assert pcc(exact, ref) >= 0.999
    torch.testing.assert_close(exact[:, :, :4], ref[:, :, :4], atol=0.02, rtol=0.02)


def test_exact_exp_is_its_own_program(device):
    import torch  # local: tt-metal's check-torch-imports-in-ttnn forbids a global torch import under ttnn/

    """exact_exp is in the program hash: the two settings never share a cached program, and off is unchanged."""
    device.clear_program_cache()
    H, S, T, TOPK, kc = 32, 32, 256, 128, 32
    q, kv, idx = make_inputs(H, S, T, TOPK, K_DIM, lambda s: TOPK, seed=5)
    q, kv = q.to(torch.bfloat16), kv.to(torch.bfloat16)
    scale = K_DIM**-0.5
    off1 = _run(device, q, kv, idx, None, scale, kc, False)
    n = device.num_program_cache_entries()
    on = _run(device, q, kv, idx, None, scale, kc, True)
    assert device.num_program_cache_entries() > n
    off2 = _run(device, q, kv, idx, None, scale, kc, False)
    assert torch.equal(off1, off2)
    assert not torch.equal(off1, on)
