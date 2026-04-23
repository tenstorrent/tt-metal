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
        - Prefill: input [batch_size=1, num_heads, seq_len, head_dim], cos/sin [1, 1, seq_len, head_dim]
        - Decode: input [seq_len=1, batch, num_heads, head_dim], cos/sin [1, batch, 1, head_dim]

        In decode mode, each batch element can have a different position (different cos/sin values).

        Tensors must use TILE layout. The padded ``head_dim`` (last dimension of ``input_tensor``,
        ``cos_cache``, and ``sin_cache``) must be divisible by ``2 * ttnn.TILE_SIZE`` (typically
        ``TILE_SIZE`` is 32, so ``head_dim`` is a multiple of 64).

        Args:
            input_tensor (ttnn.Tensor): Input tensor to apply rotation to
            cos_cache (ttnn.Tensor): Precomputed cosine values
            sin_cache (ttnn.Tensor): Precomputed sine values

        Keyword Args:
            is_decode_mode (bool): When ``True``, use decode mode (height-sharded input and caches).
                Default: ``False``.
            memory_config (Optional[ttnn.MemoryConfig]): Output memory configuration. If ``None``
                (default), device-resident ``input_tensor`` uses ``input_tensor.memory_config()``;
                host tensors use the TT-Metal default output memory configuration
                (``tt::tt_metal::operation::DEFAULT_OUTPUT_MEMORY_CONFIG`` in C++).
            compute_kernel_config (Optional[ttnn.DeviceComputeKernelConfig]): Compute kernel settings.
                If ``None`` (default), the op uses ``init_device_compute_kernel_config`` with
                ``math_fidelity=HiFi4``, ``math_approx_mode=True``, ``fp32_dest_acc_en=False``,
                ``packer_l1_acc=False``, and ``dst_full_sync_en=False``.

        Returns:
            ttnn.Tensor: Output tensor with rotary embedding applied

        Example:
            >>> # Prefill mode
            >>> input = ttnn.from_torch(torch.randn(1, 32, 128, 64), device=device)
            >>> cos = ttnn.from_torch(torch.randn(1, 1, 128, 64), device=device)
            >>> sin = ttnn.from_torch(torch.randn(1, 1, 128, 64), device=device)
            >>> output = ttnn.experimental.rotary_embedding_hf(input, cos, sin, is_decode_mode=False)

            >>> # Decode mode
            >>> input = ttnn.from_torch(torch.randn(1, 32, 8, 64), device=device)  # batch=32
            >>> cos = ttnn.from_torch(torch.randn(1, 32, 1, 64), device=device)
            >>> sin = ttnn.from_torch(torch.randn(1, 32, 1, 64), device=device)
            >>> output = ttnn.experimental.rotary_embedding_hf(input, cos, sin, is_decode_mode=True)
    )doc",
        &ttnn::experimental::rotary_embedding_hf,
        nb::arg("input_tensor"),
        nb::arg("cos_cache"),
        nb::arg("sin_cache"),
        nb::kw_only(),
        nb::arg("is_decode_mode") = false,
        nb::arg("memory_config") = nb::none(),
        nb::arg("compute_kernel_config") = nb::none());
}

}  // namespace ttnn::operations::experimental::transformer
