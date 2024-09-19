// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "sdpa_decode_pybind.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "sdpa_decode.hpp"
#include "ttnn/cpp/pybind11/decorators.hpp"

namespace ttnn::operations::transformer {

void py_bind_sdpa_decode(py::module &module) {
    auto doc =
        R"doc(
        "A version of scaled dot product attention specifically for decode."
        "The implementation is Flash-Decode and it currently only supports MQA on decoding single token.\n"

        "Q:      [1 x b x pnh x dh]"
        "K:      [1 x b x   s x dh]"
        "V:      [1 x b x   s x dh]"
        "cur_pos: list of integers of length b"
        "cur_pos_tensor: [b] tensor of integers of length b"
        "output: [1 x b x pnh x dh]"

        "Accepts a `SDPAMultiCoreProgramConfig` which specifies the grid size and chunk tiles in the K/V/Mask sequence lengths (Q chunk tiles is not used). The op parallelizes over `b` and K/V/Mask's `s` dimension."
        "If a position is given as (-1), compute for the corresponding index in the batch is skipped."
        )doc";

    using OperationType = decltype(ttnn::transformer::scaled_dot_product_attention_decode);
    ttnn::bind_registered_operation(
        module,
        ttnn::transformer::scaled_dot_product_attention_decode,
        doc,
        ttnn::pybind_overload_t{
            [](const OperationType &self,
               const ttnn::Tensor &input_tensor_q,
               const ttnn::Tensor &input_tensor_k,
               const ttnn::Tensor &input_tensor_v,
               const std::vector<uint32_t> cur_pos,
               const std::optional<const Tensor> cur_pos_tensor,
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
            py::arg("scale").noconvert() = std::nullopt,
            py::arg("memory_config").noconvert() = std::nullopt,
            py::arg("program_config").noconvert() = std::nullopt,
            py::arg("compute_kernel_config").noconvert() = std::nullopt,
            py::arg("queue_id") = 0,
        });

    using PagedOperationType = decltype(ttnn::transformer::paged_scaled_dot_product_attention_decode);
    ttnn::bind_registered_operation(
        module,
        ttnn::transformer::paged_scaled_dot_product_attention_decode,
        doc,
        ttnn::pybind_overload_t{
            [](
                const PagedOperationType &self,
                const ttnn::Tensor &input_tensor_q,
                const ttnn::Tensor &input_tensor_k,
                const ttnn::Tensor &input_tensor_v,
                const ttnn::Tensor &cur_pos_tensor,
                const ttnn::Tensor &page_table_tensor,
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
                    cur_pos_tensor,
                    page_table_tensor,
                    scale,
                    memory_config,
                    program_config,
                    compute_kernel_config);
            },
            py::arg("input_tensor_q").noconvert(),
            py::arg("input_tensor_k").noconvert(),
            py::arg("input_tensor_v").noconvert(),
            py::arg("cur_pos_tensor").noconvert(),
            py::arg("page_table_tensor").noconvert(),
            py::kw_only(),
            py::arg("scale").noconvert() = std::nullopt,
            py::arg("memory_config").noconvert() = std::nullopt,
            py::arg("program_config").noconvert() = std::nullopt,
            py::arg("compute_kernel_config").noconvert() = std::nullopt,
            py::arg("queue_id") = 0,
        });
}
}  // namespace ttnn::operations::transformer
