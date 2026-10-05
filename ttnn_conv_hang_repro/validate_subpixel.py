#!/usr/bin/env python3
"""Validate the sub-pixel decomposition of ConvTranspose1d in pure torch.

The decoder currently implements each transposed conv as zero-insertion followed by
one wide conv1d, so the conv sees (L-1)*stride + 2(k-1) input positions. That is the
op that deadlocks in ttnn's block-sharded mcast reader, and it also does `stride`
times more arithmetic than necessary (most multiplies are against inserted zeros).

Sub-pixel (phase) form: output position t = q*stride + r depends only on the kernel
taps j = r (mod stride), so for each phase r there is a narrow causal correlation

    y[q*stride + r] = sum_m x[q - m] * W[:, :, m*stride + r]

over the ORIGINAL length L. Interleaving the `stride` phase outputs reconstructs the
transposed conv, and it yields exactly the first L*stride samples -- which is what
conv_decoder_block keeps after trimming (k - stride) anyway.

This script checks the identity exactly, for the real (kernel, stride) pairs used by
the decoder, before any of it is ported to ttnn.
"""

import torch
import torch.nn.functional as F


def transposed_conv_reference(x, weight, bias, stride):
    """What the decoder effectively wants: conv_transpose1d then trim to L*stride."""
    y = F.conv_transpose1d(x, weight, bias=bias, stride=stride, padding=0)
    return y[..., : x.shape[-1] * stride]


def transposed_conv_subpixel(x, weight, bias, stride):
    """Phase decomposition: `stride` narrow causal convs over L, then interleave."""
    in_c, out_c, k = weight.shape
    b, _, L = x.shape

    phase_outputs = []
    for r in range(stride):
        sub = weight[:, :, r::stride]  # [in_c, out_c, M_r]
        m = sub.shape[-1]
        # Causal correlation: out[q] = sum_m x[q-m] * sub[m]  ->  left-pad by m-1
        # and flip the taps, since F.conv1d correlates forwards.
        v = sub.permute(1, 0, 2).flip(-1).contiguous()  # [out_c, in_c, M_r]
        x_pad = F.pad(x, (m - 1, 0))
        y_r = F.conv1d(x_pad, v, bias=bias, padding=0)  # [b, out_c, L]
        assert y_r.shape[-1] == L, (y_r.shape, L)
        phase_outputs.append(y_r)

    # Interleave phases along time: y[q*stride + r] = phase_outputs[r][q].
    # Stack on a new trailing axis then flatten, which is the transpose-free form of
    # the concat-on-channels + reshape that ttnn will do in NLC layout.
    stacked = torch.stack(phase_outputs, dim=-1)  # [b, out_c, L, stride]
    return stacked.reshape(b, out_c, L * stride)


def main():
    torch.manual_seed(0)
    # (in_c, out_c, kernel, stride) covering the decoder's upsample stages
    cases = [
        (64, 64, 4, 2),
        (128, 64, 8, 4),
        (256, 128, 10, 5),
        (512, 256, 16, 8),
        (256, 128, 6, 3),
        (1024, 512, 4, 2),
        (32, 32, 3, 1),
    ]
    all_ok = True
    for in_c, out_c, k, s in cases:
        for L in (16, 64, 128):
            x = torch.randn(1, in_c, L, dtype=torch.float64)
            w = torch.randn(in_c, out_c, k, dtype=torch.float64)
            for use_bias in (False, True):
                bias = torch.randn(out_c, dtype=torch.float64) if use_bias else None
                ref = transposed_conv_reference(x, w, bias, s)
                got = transposed_conv_subpixel(x, w, bias, s)
                ok = ref.shape == got.shape and torch.allclose(ref, got, atol=1e-10, rtol=0)
                all_ok &= ok
                if not ok:
                    err = (ref - got).abs().max().item() if ref.shape == got.shape else float("nan")
                    print(f"FAIL in={in_c} out={out_c} k={k} s={s} L={L} bias={use_bias} "
                          f"ref{tuple(ref.shape)} got{tuple(got.shape)} maxerr={err}")
                else:
                    taps = (k + s - 1) // s
                    old_in = (L - 1) * s + 2 * (k - 1)
                    new_in = L + taps - 1
                    print(f"ok   k={k:2d} s={s} L={L:3d} bias={int(use_bias)}  "
                          f"conv input {old_in:4d} -> {new_in:3d} ({old_in/new_in:.1f}x smaller), "
                          f"taps {k:2d} -> {taps}")

    print("\nALL EXACT" if all_ok else "\nMISMATCHES FOUND")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
