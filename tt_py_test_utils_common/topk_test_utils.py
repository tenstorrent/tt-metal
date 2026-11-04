# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn
from tt_py_test_utils_common.utils_for_testing import assert_equal, assert_numeric_metrics

UINT16_MAX = 65535

TEST_PADDING_VALUE = -42


def run_topk_test(N, C, H, W, k, dtype, dim, sorted, largest, device, sub_core_grids=None, pass_indices_tensor=False):
    torch.manual_seed(2005)

    if dtype == ttnn.bfloat8_b:
        pytest.xfail("BFLOAT8_B not supported by pad operation in topk")

    # float32 sorts at full 32-bit precision (no bf16 downcast), so its single-core compute
    # buffers are ~2x the bf16 size. The result-prep (2*Kt) + output (Kt) tiles dominate L1, so
    # the ceiling is a tile count, not a width: measured max is 54 output tiles (k <= 1728) in
    # Wormhole's ~1.5MB per-core L1; beyond that the buffers overflow. Multi-core can't help
    # here (it requires k <= 64). Kt = ceil(k/32).
    if dtype == ttnn.float32 and (k + 31) // 32 > 54:
        pytest.skip("exact float32 top-K exceeds single-core L1 beyond k=1728 (54 output tiles)")

    # Input tensor
    shape = [N, C, H, W]
    # Index width follows the tile-padded width; 16 bits hold every width up to 65536 (index 65535).
    padded_W = -(-W // 32) * 32
    ttnn_indices_dtype = ttnn.uint16 if padded_W <= UINT16_MAX + 1 else ttnn.uint32
    torch_indices_dtype = torch.uint16 if padded_W <= UINT16_MAX + 1 else torch.uint32
    torch_dtype = torch.float32 if dtype == ttnn.float32 else torch.bfloat16
    input = torch.randn(shape, dtype=torch_dtype) * 0.9
    ttnn_input = ttnn.from_torch(input, dtype, layout=ttnn.Layout.TILE, device=device)
    ttnn_input = ttnn.fill_implicit_tile_padding(ttnn_input, TEST_PADDING_VALUE)

    pyt_topk_values, pyt_topk_indices = torch.topk(input, k, dim=dim, largest=largest, sorted=True)

    if pass_indices_tensor:
        # The payload must differ from the iota topk generates on its own, or the gather below
        # cannot tell a read from a regeneration. Column i is labelled W - 1 - i.
        indices_tensor_torch = (W - 1 - torch.arange(W)).expand(shape).to(torch_indices_dtype)
        indices_tensor = ttnn.from_torch(
            indices_tensor_torch, ttnn_indices_dtype, layout=ttnn.Layout.TILE, device=device
        )
        indices_tensor = ttnn.fill_implicit_tile_padding(indices_tensor, TEST_PADDING_VALUE)
    else:
        indices_tensor = None

    ttnn_topk_values, ttnn_topk_indices = ttnn.topk(
        ttnn_input,
        k,
        dim=dim,
        largest=largest,
        sorted=sorted,
        sub_core_grids=sub_core_grids,
        indices_tensor=indices_tensor,
    )

    # Convert TTNN outputs to Torch for comparison
    ttnn_torch_values = ttnn.to_torch(ttnn_topk_values)
    ttnn_torch_indices = ttnn.to_torch(ttnn_topk_indices, dtype=torch_indices_dtype)

    # Assert output shapes
    desired_shape = [N, C, H, W]
    desired_shape[dim] = k
    assert list(ttnn_topk_values.shape) == desired_shape
    assert list(ttnn_topk_indices.shape) == desired_shape

    # test for equivalance
    assert_numeric_metrics(
        pyt_topk_values,
        ttnn_torch_values,
        pcc_threshold=0.9999,
        rtol=1e-06,
        atol=1e-06,
        frobenius_threshold=1e-09,
    )
    assert_equal(ttnn_torch_values, pyt_topk_values)

    # Assert indices correctness using gather
    # pcc is not a good measure for the raw indices
    # if index 49 and index 8 are tied, the order of the indices can be different
    # but the values associated with the indices should be the same
    # if index 7 and 8 are tied, but swapped, the pcc will be better than if index 49 and 8 are tied but swapped
    # rounding may also cause more ties than expected
    # the bigger we get, the tighter the distribution of the top K elements, so the pcc will be worse as stability/rounding will cause more ties
    # use cosine similarity on the gathered indices as this will show the top elements are all about the same
    # topk returns the labels it is given, so turn each label back into the column it names.
    # The labelling above is its own inverse: label L names column W - 1 - L.
    # Without a payload the op generates plain column indices, so no conversion is needed.
    ttnn_torch_columns = ttnn_torch_indices.to(torch.int64)
    if pass_indices_tensor:
        ttnn_torch_columns = W - 1 - ttnn_torch_columns
    ttnn_torch_gather_from_indices = torch.gather(input, dim, ttnn_torch_columns)
    cosine = torch.nn.CosineSimilarity(dim=dim)
    ttnn_torch_cosine = torch.mean(cosine(pyt_topk_values, ttnn_torch_gather_from_indices))

    assert ttnn_torch_cosine > 0.99, "Cosine similarity between topk values and gather from indices is less than 0.99"
