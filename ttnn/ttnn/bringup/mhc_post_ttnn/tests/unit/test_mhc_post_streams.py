# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Stream-count coverage for the compute kernel's n-dependent DEST windows (Refinement 2).

The mix packs P = ceil((n+1)/2) coefficient tiles + (n+1) data tiles per output column into the 8
SyncFull fp32 DEST slots: n <= 2 runs K > 1 columns per window, n = 3 / 4 one column, n = 5 the
grouped (accumulating) WeightedSum path. The golden suite pins n = 4 only.
"""

import pytest
import torch
import ttnn

mhc_post = ttnn.bringup.mhc_post  # the C++ op (the Python builder is the parity reference)

from .test_mhc_post import reference_mhc_post, to_device

TORCH_DTYPE = {ttnn.float32: torch.float32, ttnn.bfloat16: torch.bfloat16}


@pytest.mark.parametrize("dtype", [ttnn.float32, ttnn.bfloat16], ids=["fp32", "bf16"])
@pytest.mark.parametrize("n", [1, 2, 3, 5])
@pytest.mark.parametrize("T, C", [(64, 32 * 7), (45, 32 * 3)], ids=["T64_C224", "T45_C96_non_aligned"])
def test_mhc_post_streams(device, n, T, C, dtype):
    torch.manual_seed(n)
    f = torch.randn(1, T, C).to(TORCH_DTYPE[dtype])
    x = torch.randn(1, T, n * C).to(TORCH_DTYPE[dtype])
    post = 2.0 * torch.rand(1, T, n)
    comb = torch.rand(1, T, n * n)
    out = mhc_post(
        to_device(f, device, dtype),
        to_device(x, device, dtype),
        to_device(post, device, ttnn.float32),
        to_device(comb, device, ttnn.float32),
    )
    got = ttnn.to_torch(out).float()
    ref = reference_mhc_post(f, x, post, comb)
    if dtype == ttnn.float32:
        torch.testing.assert_close(got, ref, rtol=1e-5, atol=1e-5)
    else:
        torch.testing.assert_close(got, ref.bfloat16().float(), rtol=1.6e-2, atol=1e-2)
