// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "rotary_embedding_hf_nanobind.hpp"

#include <optional>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>

#include "ttnn-nanobind/bind_function.hpp"
#include "rotary_embedding_hf.hpp"

namespace ttnn::operations::experimental::transformer {

void bind_rotary_embedding_hf(nb::module_& mod) {
    ttnn::bind_function<"rotary_embedding_hf", "ttnn.experimental.">(
        mod,
        R"doc(
        Applies HuggingFace-style rotary position embedding to input tensor.

        This operation supports both prefill and decode modes:
        - Prefill: input [1, num_heads, seq_len, head_dim], cos/sin [1, 1, seq_len, head_dim]
        - Decode: input [1, batch, num_heads, head_dim], cos/sin [1, batch, 1, head_dim]

        In decode mode, each batch element can have a different position (different cos/sin values).

        Args:
            input_tensor (ttnn.Tensor): Input tensor to apply rotation to
            cos_cache (ttnn.Tensor): Precomputed cosine values
            sin_cache (ttnn.Tensor): Precomputed sine values

        Keyword Args:
            is_decode (bool): Whether to use decode mode (default: False)
            memory_config (Optional[ttnn.MemoryConfig]): Memory configuration for output tensor
            compute_kernel_config (Optional[ttnn.DeviceComputeKernelConfig]): Compute kernel configuration

        Returns:
            ttnn.Tensor: Output tensor with rotary embedding applied

        Example:
            >>> # Prefill mode
            >>> input = ttnn.from_torch(torch.randn(1, 32, 128, 64), device=device)
            >>> cos = ttnn.from_torch(torch.randn(1, 1, 128, 64), device=device)
            >>> sin = ttnn.from_torch(torch.randn(1, 1, 128, 64), device=device)
            >>> output = ttnn.experimental.rotary_embedding_hf(input, cos, sin, is_decode=False)

            >>> # Decode mode
            >>> input = ttnn.from_torch(torch.randn(1, 32, 8, 64), device=device)  # batch=32
            >>> cos = ttnn.from_torch(torch.randn(1, 32, 1, 64), device=device)
            >>> sin = ttnn.from_torch(torch.randn(1, 32, 1, 64), device=device)
            >>> output = ttnn.experimental.rotary_embedding_hf(input, cos, sin, is_decode=True)
    )doc",
        &ttnn::experimental::rotary_embedding_hf,
        nb::arg("input_tensor"),
        nb::arg("cos_cache"),
        nb::arg("sin_cache"),
        nb::kw_only(),
        nb::arg("is_decode") = false,
        nb::arg("memory_config") = nb::none(),
        nb::arg("compute_kernel_config") = nb::none());
}

}  // namespace ttnn::operations::experimental::transformer
