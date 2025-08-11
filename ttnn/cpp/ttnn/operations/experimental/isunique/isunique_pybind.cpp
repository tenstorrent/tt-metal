// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "isunique_pybind.hpp"

#include "isunique.hpp"

#include <optional>

namespace ttnn::operations::experimental::isunique::detail {

using namespace ttnn;

void bind_isunique_operation(py::module& module) {
    auto doc = "";

    using OperationType = decltype(ttnn::experimental::isunique);
    bind_registered_operation(
        module,
        ttnn::experimental::isunique,
        doc,
        ttnn::pybind_overload_t{
            [](const OperationType& self,
               const Tensor& input_tensor,
               const std::optional<int32_t>& dim,
               const bool& invert,
               const std::optional<Tensor>& opt_out,
               const std::optional<MemoryConfig>& memory_config,
               const QueueId& queue_id = DefaultQueueId) -> Tensor {
                return self(queue_id, input_tensor, dim, invert, opt_out, memory_config);
            },
            py::arg("input").noconvert(),
            py::kw_only(),
            py::arg("dim") = std::nullopt,
            py::arg("invert") = false,
            py::arg("out") = std::nullopt,
            py::arg("memory_config") = std::nullopt,
            py::arg("queue_id") = DefaultQueueId});
}

}  // namespace ttnn::operations::experimental::isunique::detail
