// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "rotate_half_pybind.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ttnn/cpp/pybind11/decorators.hpp"
#include "rotate_half.hpp"

namespace ttnn::operations::experimental::transformer {

void py_bind_rotate_half(pybind11::module& module) {
    namespace py = pybind11;
    ttnn::bind_registered_operation(
        module,
        ttnn::experimental::rotate_half,
        R"doc(rotate_half(input: ttnn.Tensor, *, memory_config: Optional[MemoryConfig] = None) -> ttnn.Tensor

            Performs a rotate half operation used by RotaryEmbedding

            Args:
                * :attr:`input`: Input Tensor

            Keyword Args:
                * :attr:`memory_config`: Memory Config of the output tensor, if None then it gets set to input.memory_config()
        )doc",
        ttnn::pybind_arguments_t{
            py::arg("input"),
            py::kw_only(),
            py::arg("memory_config") = std::nullopt});
}

}  // namespace ttnn::operations::experimental::transformer
