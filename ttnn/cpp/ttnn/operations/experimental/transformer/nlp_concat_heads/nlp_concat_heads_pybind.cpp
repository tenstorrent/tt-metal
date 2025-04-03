// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <pybind11/cast.h>
#include <pybind11/pybind11.h>
#include <memory>
#include <optional>

#include "cpp/pybind11/decorators.hpp"
#include "ttnn/common/queue_id.hpp"
#include "ttnn/operations/experimental/transformer/nlp_concat_heads/nlp_concat_heads.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"

namespace tt {
namespace tt_metal {
struct MemoryConfig;
}  // namespace tt_metal
}  // namespace tt

namespace ttnn::operations::experimental::transformer::detail {

namespace py = pybind11;

void bind_nlp_concat_heads(py::module& module) {
    using OperationType = decltype(ttnn::experimental::nlp_concat_heads);
    ttnn::bind_registered_operation(
        module,
        ttnn::experimental::nlp_concat_heads,
        R"doc(
            Shuffles [B, num_heads, S, head_dim] tensor into tensor with shape [B, 1, S, num_heads * head_dim].
        )doc",
        ttnn::pybind_overload_t{
            [](const OperationType& self,
               const ttnn::Tensor& input_tensor,
               const std::optional<ttnn::MemoryConfig>& memory_config,
               std::optional<ttnn::Tensor> optional_output_tensor,
               QueueId queue_id) { return self(queue_id, input_tensor, memory_config, optional_output_tensor); },
            py::arg("input_tensor").noconvert(),
            py::kw_only(),
            py::arg("memory_config") = std::nullopt,
            py::arg("output_tensor") = std::nullopt,
            py::arg("queue_id") = DefaultQueueId});
}

}  // namespace ttnn::operations::experimental::transformer::detail
