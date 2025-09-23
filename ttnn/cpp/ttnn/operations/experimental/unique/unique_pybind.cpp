// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "unique_pybind.hpp"

#include "unique.hpp"

#include <optional>

namespace ttnn::operations::experimental::unique::detail {

using namespace ttnn;

void bind_unique_operation(py::module& module) {
    auto doc = "";

    using OperationType = decltype(ttnn::experimental::unique);
    bind_registered_operation(
        module,
        ttnn::experimental::unique,
        doc,
        ttnn::pybind_overload_t{
            [](const OperationType& self,
               const Tensor& input,
               const bool& sorted,
               const bool& return_inverse,
               const bool& return_counts,
               const std::optional<int32_t>& dim,
               const std::optional<MemoryConfig>& memory_config) -> Tensor {
                return self(input, sorted, return_inverse, return_counts, dim, memory_config);
            },
            py::arg("input").noconvert(),
            py::kw_only(),
            py::arg("sorted") = false,
            py::arg("return_inverse") = false,
            py::arg("return_counts") = false,
            py::arg("dim") = std::nullopt,
            py::arg("memory_config") = std::nullopt});
}

}  // namespace ttnn::operations::experimental::unique::detail
