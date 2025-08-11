// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "isin_pybind.hpp"

#include "isin.hpp"

#include <optional>

namespace ttnn::operations::experimental::isin::detail {

using namespace ttnn;

void bind_isin_operation(py::module& module) {
    auto doc = "";

    using OperationType = decltype(ttnn::experimental::isin);
    bind_registered_operation(
        module,
        ttnn::experimental::isin,
        doc,
        ttnn::pybind_overload_t{
            [](const OperationType& self,
               const Tensor& elements,
               const Tensor& test_elements,
               const bool& assume_unique,
               const bool& invert,
               const std::optional<MemoryConfig>& memory_config,
               const std::optional<Tensor>& opt_out,
               const QueueId& queue_id = DefaultQueueId) -> Tensor {
                return self(queue_id, elements, test_elements, assume_unique, invert, memory_config, opt_out);
            },
            py::arg("elements").noconvert(),
            py::arg("test_elements").noconvert(),
            py::kw_only(),
            py::arg("assume_unique") = false,
            py::arg("invert") = false,
            py::arg("memory_config") = std::nullopt,
            py::arg("out") = std::nullopt,
            py::arg("queue_id") = DefaultQueueId});
}

}  // namespace ttnn::operations::experimental::isin::detail
