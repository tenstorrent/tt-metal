// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "speculative_sdpa_decode_pybind.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "speculative_sdpa_decode.hpp"
#include "ttnn/cpp/pybind11/decorators.hpp"

namespace ttnn::operations::experimental::transformer {

using SDPAProgramConfig = ttnn::operations::transformer::SDPAProgramConfig;

void py_bind_speculative_sdpa_decode(py::module& module) {
    namespace py = pybind11;

    auto doc =
        R"doc(
        Speculative version of scaled dot product attention decode.


        Accepts a `SDPAMultiCoreProgramConfig` which specifies the grid size and chunk tiles in the K/V/Mask sequence lengths (Q chunk tiles is not used). The op parallelizes over `b` and K/V/Mask's `s` dimension.


        Args:
            input_tensor_q (ttnn.Tensor): the input tensor [1 x b x nh x dh]
            input_tensor_k (ttnn.Tensor): the input tensor [b x nkv x   s x dh]
            input_tensor_v (ttnn.Tensor): the input tensor [b x nkv x   s x dh]

        Keyword args:
            lambda_ (float): the lambda value for the speculation error tolerance threshold. Defaults to `0.2`.
            speculative_chunk_size (int): the speculative chunk size. Defaults to `128`.
            is_causal (bool): whether the attention is is_causal. Defaults to `True`.
            attn_mask (ttnn.Tensor, optional): the input tensor [b x 1 x s x s]. Defaults to `None`.
            cur_pos (List of int, optional): list of integers of length b. Defaults to `None`.
            memory_config (ttnn.MemoryConfig, optional): Memory configuration for the operation. Defaults to `None`.
            queue_id (int, optional): command queue id. Defaults to `0`.
            cur_pos_tensor (ttnn.Tensor, optional): [b] tensor of integers of length b. Defaults to `None`.
            scale (float, optional): Defaults to `None`.
            program_config (SDPAProgramConfig, optional): Defaults to `None`.
            compute_kernel_config (ttnn.DeviceComputeKernelConfig, optional): Defaults to `None`.


        Returns:
            ttnn.Tensor: the full output tensor [1 x b x pnh x dh].
            ttnn.Tensor: the speculative output tensor [1 x b x pnh x dh].
            ttnn.Tensor: the row major l2 distance tensor [b].
            ttnn.Tensor: the row major l2 norm x tensor [b].


        "Accepts a `SDPAMultiCoreProgramConfig` which specifies the grid size and chunk tiles in the K/V/Mask sequence lengths (Q chunk tiles is not used). The op parallelizes over `b` and K/V/Mask's `s` dimension."
        "If a position is given as (-1), compute for the corresponding index in the batch is skipped."
        )doc";

    using OperationType = decltype(ttnn::experimental::speculative_scaled_dot_product_attention_decode);
    ttnn::bind_registered_operation(
        module,
        ttnn::experimental::speculative_scaled_dot_product_attention_decode,
        doc,
        ttnn::pybind_overload_t{
            [](const OperationType& self,
               const ttnn::Tensor& input_tensor_q,
               const ttnn::Tensor& input_tensor_k,
               const ttnn::Tensor& input_tensor_v,
               std::optional<float> lambda_,
               std::optional<uint32_t> speculative_chunk_size,
               const bool is_causal,
               const std::optional<const Tensor>& attn_mask,
               const std::vector<uint32_t>& cur_pos,
               const std::optional<const Tensor>& cur_pos_tensor,
               std::optional<float> scale,
               const std::optional<MemoryConfig>& memory_config,
               std::optional<SDPAProgramConfig> program_config,
               std::optional<DeviceComputeKernelConfig> compute_kernel_config,
               uint8_t queue_id) {
                return self(
                    queue_id,
                    input_tensor_q,
                    input_tensor_k,
                    input_tensor_v,
                    lambda_,
                    speculative_chunk_size,
                    is_causal,
                    attn_mask,
                    cur_pos,
                    cur_pos_tensor,
                    scale,
                    memory_config,
                    program_config,
                    compute_kernel_config);
            },
            py::arg("input_tensor_q").noconvert(),
            py::arg("input_tensor_k").noconvert(),
            py::arg("input_tensor_v").noconvert(),
            py::kw_only(),
            py::arg("lambda_") = std::nullopt,
            py::arg("speculative_chunk_size") = std::nullopt,
            py::arg("is_causal").noconvert() = true,
            py::arg("attn_mask").noconvert() = std::nullopt,
            py::arg("cur_pos").noconvert() = std::vector<uint32_t>(),
            py::arg("cur_pos_tensor").noconvert() = std::nullopt,
            py::arg("scale").noconvert() = std::nullopt,
            py::arg("memory_config").noconvert() = std::nullopt,
            py::arg("program_config").noconvert() = std::nullopt,
            py::arg("compute_kernel_config").noconvert() = std::nullopt,
            py::arg("queue_id") = 0,
        });
}
}  // namespace ttnn::operations::experimental::transformer
