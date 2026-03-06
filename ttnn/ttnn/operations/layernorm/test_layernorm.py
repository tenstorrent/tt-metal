# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

import pytest
import ttnn

from .layernorm_op import layernorm


@pytest.mark.parametrize(
    "shape",
    [
        pytest.param((32, 32), id="single_tile"),
        pytest.param((64, 64), id="2x2_tiles"),
        pytest.param((32, 128), id="1x4_tiles"),
        pytest.param((1, 32, 64), id="batch1"),
        pytest.param((2, 64, 64), id="batch2"),
    ],
)
def test_layernorm(device, shape):
    import torch

    torch.manual_seed(42)
    W = shape[-1]

    torch_input = torch.randn(shape, dtype=torch.bfloat16)
    torch_gamma = torch.ones(W, dtype=torch.bfloat16)
    torch_beta = torch.zeros(W, dtype=torch.bfloat16)
    eps = 1e-5

    ttnn_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    ttnn_gamma = ttnn.from_torch(
        torch_gamma,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    ttnn_beta = ttnn.from_torch(
        torch_beta,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    ttnn_output = layernorm(ttnn_input, ttnn_gamma, ttnn_beta, eps=eps)

    assert list(ttnn_output.shape) == list(shape), f"Shape mismatch: {ttnn_output.shape} vs {shape}"

    torch_output = ttnn.to_torch(ttnn_output)
    torch_expected = torch.nn.functional.layer_norm(
        torch_input.float(), [W], weight=torch_gamma.float(), bias=torch_beta.float(), eps=eps
    ).to(torch.bfloat16)

    assert torch.allclose(
        torch_output.float(),
        torch_expected.float(),
        rtol=1e-1,
        atol=1e-1,
    ), f"Output mismatch. Max diff: {(torch_output.float() - torch_expected.float()).abs().max()}"


@pytest.mark.parametrize(
    "shape",
    [
        pytest.param((32, 64), id="with_random_weights"),
        pytest.param((64, 64), id="with_random_weights_2x2"),
    ],
)
def test_layernorm_random_weights(device, shape):
    import torch

    torch.manual_seed(123)
    W = shape[-1]

    torch_input = torch.randn(shape, dtype=torch.bfloat16)
    torch_gamma = torch.randn(W, dtype=torch.bfloat16)
    torch_beta = torch.randn(W, dtype=torch.bfloat16)
    eps = 1e-5

    ttnn_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    ttnn_gamma = ttnn.from_torch(
        torch_gamma,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    ttnn_beta = ttnn.from_torch(
        torch_beta,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    ttnn_output = layernorm(ttnn_input, ttnn_gamma, ttnn_beta, eps=eps)

    assert list(ttnn_output.shape) == list(shape)

    torch_output = ttnn.to_torch(ttnn_output)
    torch_expected = torch.nn.functional.layer_norm(
        torch_input.float(), [W], weight=torch_gamma.float(), bias=torch_beta.float(), eps=eps
    ).to(torch.bfloat16)

    assert torch.allclose(
        torch_output.float(),
        torch_expected.float(),
        rtol=1e-1,
        atol=1e-1,
    ), f"Output mismatch. Max diff: {(torch_output.float() - torch_expected.float()).abs().max()}"
