// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/data_movement/non_zero_indices/non_zero_indices_pybind.hpp"

#include <optional>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ttnn-pybind/decorators.hpp"
#include "ttnn/operations/data_movement/non_zero_indices/non_zero_indices.hpp"

namespace ttnn::operations::data_movement::detail {
namespace py = pybind11;

void bind_non_zero_indices(py::module& module) {
    auto doc = fmt::format(
        R"doc(
        Returns the number of elements (N) that are non-zero as well as a tensor of the same shape as input where the first N elements are the indices of non-zero elements.

        Args:
        input_tensor (ttnn.Tensor): Input Tensor should be 1D and in row major layout.

        Keyword Args:
        memory_config (ttnn.MemoryConfig, optional): Memory configuration for the operation. Defaults to `None`.

        Returns:
        List of ttnn.Tensor: the output tensors.
    )doc",
        ttnn::nonzero.base_name());

    using OperationType = decltype(ttnn::nonzero);

    ttnn::bind_registered_operation(
        module,
        ttnn::nonzero,
        doc,
        ttnn::pybind_overload_t{
            [](const OperationType& self,
               const Tensor& input_tensor,
               const std::optional<tt::tt_metal::MemoryConfig>& memory_config) {
                return self(input_tensor, memory_config);
            },
            py::arg("input_tensor").noconvert(),
            py::kw_only(),
            py::arg("memory_config") = std::nullopt});
}

}  // namespace ttnn::operations::data_movement::detail
