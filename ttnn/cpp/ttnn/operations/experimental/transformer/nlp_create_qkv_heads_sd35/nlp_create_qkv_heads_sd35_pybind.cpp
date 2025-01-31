// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/cpp/pybind11/decorators.hpp"

#include "ttnn/operations/experimental/transformer/nlp_create_qkv_heads_sd35/nlp_create_qkv_heads_sd35.hpp"
#include "ttnn/operations/experimental/transformer/nlp_create_qkv_heads_sd35/nlp_create_qkv_heads_sd35_pybind.hpp"

namespace ttnn::operations::experimental::transformer::detail {

void bind_nlp_create_qkv_heads_sd35(pybind11::module& module) {
    ttnn::bind_registered_operation(
        module,
        ttnn::experimental::nlp_create_qkv_heads_sd35,
        R"doc(
            Shuffles [B, 1, S, hidden_dim] one of q,k,v tensors into heads with shapes [B, head_count, S, head_dim].
        )doc",
        ttnn::pybind_overload_t{
            [](const decltype(ttnn::experimental::nlp_create_qkv_heads_sd35)& self,
               const ttnn::Tensor& input_tensor_q,
               const std::optional<ttnn::MemoryConfig>& memory_config,
               std::optional<std::vector<std::optional<ttnn::Tensor>>>& optional_output_tensors,
               uint8_t queue_id) { return self(queue_id, input_tensor_q, memory_config, optional_output_tensors); },
            pybind11::arg("input").noconvert(),
            pybind11::kw_only(),
            pybind11::arg("memory_config").noconvert() = std::nullopt,
            pybind11::arg("output_tensors").noconvert() = std::nullopt,
            pybind11::arg("queue_id") = 0});
};
}  // namespace ttnn::operations::experimental::transformer::detail
