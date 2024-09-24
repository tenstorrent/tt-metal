// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "sdpa_decode_gqa_pybind.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "sdpa_decode_gqa.hpp"
#include "ttnn/cpp/pybind11/decorators.hpp"

namespace ttnn::operations::transformer {

void py_bind_sdpa_gqa_decode(py::module &module) {
    auto doc =
        R"doc(
        A version of scaled dot product attention specifically for GQA decode.


        Accepts a `SDPAMultiCoreProgramConfig` which specifies the grid size and chunk tiles in the K/V/Mask sequence lengths (Q chunk tiles is not used). The op parallelizes over `b` and K/V/Mask's `s` dimension.


        Args:
            input_tensor_q (ttnn.Tensor): the input tensor [1 x qh x b x dh]
            input_tensor_k (ttnn.Tensor): the input tensor [b x kh x s x dh]
            input_tensor_v (ttnn.Tensor): the input tensor [b x kh x s x dh]
            cur_pos (List of int): list of integers of length b.



        Keyword args:
            memory_config (ttnn.MemoryConfig, optional): Memory configuration for the operation. Defaults to `None`.
            queue_id (int, optional): command queue id. Defaults to `0`.
            scale (float, optional): Defaults to `None`.
            program_config (SDPAProgramConfig, optional): Defaults to `None`.
            compute_kernel_config (ttnn.DeviceComputeKernelConfig, optional): Defaults to `None`.


        Returns:
            ttnn.Tensor: the output tensor [1 x b x qh x dh].

        "Q:      [1 x qh x b x dh] or [1 x b x qh x dh]"
        "K:      [b x kh x s x dh] or [1 x kh x s x dh]"
        "V:      [b x kh x s x dh] or [1 x kh x s x dh]"
        "cur_pos: list of integers of length b"
        "transpose_q: bool default true. If true, expects Q in [1 x qh x b x dh] format"
        "share_cache: bool default false. If true, shares cache across all batch, so K and V are [1 x kh x s x dh]"
        "output: [1 x b x qh x dh]"

        )doc";

    using OperationType = decltype(ttnn::transformer::scaled_dot_product_attention_decode_gqa);
    ttnn::bind_registered_operation(
        module,
        ttnn::transformer::scaled_dot_product_attention_decode_gqa,
        doc,
        ttnn::pybind_overload_t{
            [](const OperationType &self,
               const ttnn::Tensor &input_tensor_q,
               const ttnn::Tensor &input_tensor_k,
               const ttnn::Tensor &input_tensor_v,
               const std::vector<uint32_t> cur_pos,
               const std::optional<const Tensor> cur_pos_tensor,
               std::optional<bool> transpose_q,
               std::optional<bool> share_cache,
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
                    cur_pos,
                    cur_pos_tensor,
                    transpose_q,
                    share_cache,
                    scale,
                    memory_config,
                    program_config,
                    compute_kernel_config);
            },
            py::arg("input_tensor_q").noconvert(),
            py::arg("input_tensor_k").noconvert(),
            py::arg("input_tensor_v").noconvert(),
            py::arg("cur_pos").noconvert() = std::vector<uint32_t>(),
            py::kw_only(),
            py::arg("cur_pos_tensor").noconvert() = std::nullopt,
            py::arg("transpose_q") = true,
            py::arg("share_cache") = false,
            py::arg("scale").noconvert() = std::nullopt,
            py::arg("memory_config").noconvert() = std::nullopt,
            py::arg("program_config").noconvert() = std::nullopt,
            py::arg("compute_kernel_config").noconvert() = std::nullopt,
            py::arg("queue_id") = 0,
        });
}
}  // namespace ttnn::operations::transformer
