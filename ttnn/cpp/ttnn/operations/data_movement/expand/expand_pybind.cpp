// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "expand_pybind.hpp"

#include "pybind11/decorators.hpp"
#include "ttnn/operations/data_movement/expand/expand.hpp"

namespace ttnn::operations::data_movement::detail {
void py_bind_expand(py::module &module) {
    const auto *doc =
        R"doc(expand(input: ttnn.Tensor, sizes: List[int], output: Optional[ttnn.Tensor] = None, memory_config: Optional[ttnn.MemoryConfig] = None) -> ttnn.Tensor
        Returns a new tensor where singleton dimensions are expanded to a larger side.
        Unlike :func:`torch.expand`, this function is not zero-cost and perform a memory copy to create the expanded tensor. This is due to `ttnn.Tensor`'s lack of strided tensor support.

        Args:
            * :attr:`input`: The tensor to be expanded.
            * :attr:`sizes`: The desired expanded size.
            * :attr:`output`: An optional tensor to store the expanded result.
            * :attr:`memory_config`: The memory configuration for the expanded tensor.
        )doc";
    bind_registered_operation(
        module,
        ttnn::expand,
        doc,
        ttnn::pybind_arguments_t{
            py::arg("input"),
            py::arg("sizes"),
            py::kw_only(),
            py::arg("output") = std::nullopt,
            py::arg("memory_config") = std::nullopt});
}
}  // namespace ttnn::operations::data_movement::detail
