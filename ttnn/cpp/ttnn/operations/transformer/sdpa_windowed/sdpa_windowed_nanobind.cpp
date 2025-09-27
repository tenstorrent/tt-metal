// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "sdpa_windowed_nanobind.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>

#include "sdpa_windowed.hpp"
#include "ttnn-nanobind/decorators.hpp"

namespace ttnn::operations::transformer {

void bind_sdpa_windowed(nb::module_& mod) {
    auto doc =
        R"doc(
        Windowed scaled dot product attention. This is similar to the standard SDPA but instead of
        accepting an explicit attention mask, it accepts cumulative window sequence lengths and builds
        the attention mask internally to create block-diagonal attention patterns.

        This is particularly useful for vision transformers with windowed attention mechanisms like
        Qwen2.5-VL where attention is restricted to specific windows in the sequence.

        Args:
            input_tensor_q (ttnn.Tensor): the query tensor.     [b x nqh x s x dh]
            input_tensor_k (ttnn.Tensor): the key tensor.       [b x nkv x s x dh]
            input_tensor_v (ttnn.Tensor): the value tensor.     [b x nkv x s x dh]
            cu_window_seqlens (ttnn.Tensor): cumulative window sequence lengths that define attention boundaries. [window_count + 1]

        Keyword args:
            scale (float, optional): Defaults to `None`. Scale factor for QK^T.
            memory_config (ttnn.MemoryConfig, optional): Memory configuration for the operation. Defaults to `None`.
            program_config (SDPAProgramConfig, optional): Defaults to `None`.
            compute_kernel_config (ttnn.DeviceComputeKernelConfig, optional): Defaults to `None`.

        Returns:
            ttnn.Tensor: the output tensor [b x nqh x s x dh].

        Example:
            # For a sequence with 3 windows of sizes 10, 15, and 20 tokens:
            cu_window_seqlens = [0, 10, 25, 45]
            output = ttnn.transformer.windowed_scaled_dot_product_attention(
                q, k, v, cu_window_seqlens
            )
        )doc";

    using OperationType = decltype(ttnn::transformer::windowed_scaled_dot_product_attention);
    ttnn::bind_registered_operation(
        mod,
        ttnn::transformer::windowed_scaled_dot_product_attention,
        doc,
        ttnn::nanobind_overload_t{
            [](const OperationType& self,
               const ttnn::Tensor& input_tensor_q,
               const ttnn::Tensor& input_tensor_k,
               const ttnn::Tensor& input_tensor_v,
               const ttnn::Tensor& cu_window_seqlens,
               std::optional<float> scale,
               const std::optional<MemoryConfig>& memory_config,
               std::optional<SDPAProgramConfig> program_config,
               std::optional<DeviceComputeKernelConfig> compute_kernel_config) {
                return self(
                    input_tensor_q,
                    input_tensor_k,
                    input_tensor_v,
                    cu_window_seqlens,
                    scale,
                    memory_config,
                    program_config,
                    compute_kernel_config);
            },
            nb::arg("input_tensor_q").noconvert(),
            nb::arg("input_tensor_k").noconvert(),
            nb::arg("input_tensor_v").noconvert(),
            nb::arg("cu_window_seqlens").noconvert(),
            nb::kw_only(),
            nb::arg("scale").noconvert() = nb::none(),
            nb::arg("memory_config").noconvert() = nb::none(),
            nb::arg("program_config").noconvert() = nb::none(),
            nb::arg("compute_kernel_config").noconvert() = nb::none(),
        });
}

}  // namespace ttnn::operations::transformer
