// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "sdpa_pybind.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "sdpa.hpp"
#include "ttnn/cpp/pybind11/decorators.hpp"

namespace ttnn::operations::transformer {

void py_bind_sdpa(py::module &module) {
    auto doc =
        R"doc(
        Causal scaled dot product attention. This API mimicks the PyTorch API of the same name.
        The implementation is FlashAttention-2."

        Accepts a `SDPAProgramConfig` which specifies the grid size and chunk tiles in the Q and K sequence lengths. The op parallelizes over `b`, `nqh`, and Q's `s` dimension.

        Args:
            input_tensor_q (ttnn.Tensor): the input tensor.          [b x nqh x s x dh]
            input_tensor_k (ttnn.Tensor): the input tensor.          [b x nkv x s x dh]
            input_tensor_v (ttnn.Tensor): the input tensor.          [b x nkv x s x dh]

        Keyword args:
            attn_mask (ttnn.Tensor, optional): Defaults to `None`. [b x 1 x s x s]. Head broadcasting is implied.
            is_casual (bool): Defaults to `true`.
            memory_config (ttnn.MemoryConfig, optional): Memory configuration for the operation. Defaults to `None`.
            queue_id (int, optional): command queue id. Defaults to `0`.
            scale (float, optional): Defaults to `None`.
            program_config (SDPAProgramConfig, optional): Defaults to `None`.
            compute_kernel_config (ttnn.DeviceComputeKernelConfig, optional): Defaults to `None`.


        Returns:
            ttnn.Tensor: the output tensor [b x nqh x s x dh].

        )doc";

    using OperationType = decltype(ttnn::transformer::scaled_dot_product_attention);
    ttnn::bind_registered_operation(
        module,
        ttnn::transformer::scaled_dot_product_attention,
        doc,
        ttnn::pybind_overload_t{
            [](const OperationType &self,
               const ttnn::Tensor &input_tensor_q,
               const ttnn::Tensor &input_tensor_k,
               const ttnn::Tensor &input_tensor_v,
               std::optional<ttnn::Tensor> attn_mask,
               bool is_causal,
               std::optional<float> scale,
               const std::optional<MemoryConfig> &memory_config,
               std::optional<SDPAProgramConfig> program_config,
               std::optional<DeviceComputeKernelConfig> compute_kernel_config,
               uint8_t queue_id) {
                return self(
                    queue_id,
                    input_tensor_q,
                    input_tensor_k,
                    input_tensor_v,
                    attn_mask,
                    is_causal,
                    scale,
                    memory_config,
                    program_config,
                    compute_kernel_config);
            },
            py::arg("input_tensor_q").noconvert(),
            py::arg("input_tensor_k").noconvert(),
            py::arg("input_tensor_v").noconvert(),
            py::kw_only(),
            py::arg("attn_mask").noconvert() = std::nullopt,
            py::arg("is_causal").noconvert() = true,
            py::arg("scale").noconvert() = std::nullopt,
            py::arg("memory_config").noconvert() = std::nullopt,
            py::arg("program_config").noconvert() = std::nullopt,
            py::arg("compute_kernel_config").noconvert() = std::nullopt,
            py::arg("queue_id") = 0,
        });
}
}  // namespace ttnn::operations::transformer
