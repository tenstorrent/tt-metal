# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn


@pytest.mark.parametrize("H,W", [(32, 64)])
def test_stage6_kernels_compile_and_run(device, H, W):
    """
    Stage 6: Verify kernels compile at runtime and operation completes without hanging.
    Output values will be garbage (stub kernels) - correctness is Stage 7's job.
    """
    torch.manual_seed(0)

    # Create input tensor
    input_torch = torch.randn(H, W, dtype=torch.bfloat16)

    # Convert to TTNN (row-major)
    input_tt = ttnn.from_torch(input_torch, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)

    # Run operation - should complete without hang
    output_tt = ttnn.centralize_w_rm(input_tt)

    # Convert back
    output_torch = ttnn.to_torch(output_tt)

    # Verify output shape matches input (no dimension reduction in centralize)
    assert (
        output_torch.shape == input_torch.shape
    ), f"Output shape {output_torch.shape} != input shape {input_torch.shape}"

    print(f"✓ Stage 6 PASS: Kernels compiled and ran without hang. Output shape: {output_torch.shape}")
    print(f"  Note: Output values are garbage (stub kernels). Correctness testing is Stage 7.")
