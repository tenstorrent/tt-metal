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
        "Causal scaled dot product attention. This API mimicks the PyTorch API of the same name."
        "The implementation is FlashAttention-2 and it currently only supports MQA with causal masking.\n"

        "Q:      [b x nqh x s x dh]"
        "K:      [b x 1   x s x dh]"
        "V:      [b x 1   x s x dh]"
        "mask:   [b x 1   x s x s ]"
        "output: [b x nqh x s x dh]"

        "Mask must be a causal mask with 0s in the lower triangle and -inf in the upper triangle."

        "Accepts a `SDPAProgramConfig` which specifies the grid size and chunk tiles in the Q and K sequence lengths. The op parallelizes over `b`, `nqh`, and Q's `s` dimension."
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
               std::optional<ttnn::Tensor> causal_mask,
               bool is_causal,
               std::optional<float> scale,
               const std::optional<MemoryConfig> &memory_config,
               std::optional<SDPAProgramConfig> program_config,
               std::optional<DeviceComputeKernelConfig> compute_kernel_config,
               std::optional<uint32_t> valid_seq_len,
               uint8_t queue_id) {
                return self(
                    queue_id,
                    input_tensor_q,
                    input_tensor_k,
                    input_tensor_v,
                    causal_mask,
                    is_causal,
                    scale,
                    memory_config,
                    program_config,
                    compute_kernel_config,
                    valid_seq_len);
            },
            py::arg("input_tensor_q").noconvert(),
            py::arg("input_tensor_k").noconvert(),
            py::arg("input_tensor_v").noconvert(),
            py::arg("causal_mask").noconvert(),
            py::kw_only(),
            py::arg("is_causal").noconvert() = true,
            py::arg("scale").noconvert() = std::nullopt,
            py::arg("memory_config").noconvert() = std::nullopt,
            py::arg("program_config").noconvert() = std::nullopt,
            py::arg("compute_kernel_config").noconvert() = std::nullopt,
            py::arg("valid_seq_len").noconvert() = std::nullopt,
            py::arg("queue_id") = 0,
        });
}
}  // namespace ttnn::operations::transformer
