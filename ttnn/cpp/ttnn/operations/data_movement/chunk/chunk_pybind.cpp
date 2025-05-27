// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "chunk_pybind.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ttnn-pybind/decorators.hpp"
#include "ttnn/operations/data_movement/chunk/chunk.hpp"
#include "ttnn/types.hpp"

namespace ttnn::operations::data_movement {

namespace detail {

template <typename data_movement_operation_t>
void bind_chunk(pybind11::module& module, const data_movement_operation_t& operation, const char* doc) {
    bind_registered_operation(
        module,
        operation,
        doc,
        ttnn::pybind_overload_t{
            [](const data_movement_operation_t& self,
               const ttnn::Tensor& input_tensor,
               const uint32_t chunks,
               const int dim) -> std::vector<ttnn::Tensor> { return self(input_tensor, chunks, dim); },
            py::arg("input_tensor"),
            py::arg("chunks"),
            py::arg("dim")});
}

}  // namespace detail

void py_bind_chunk(pybind11::module& module) {
    detail::bind_chunk(
        module,
        ttnn::chunk,
        R"doc(chunk(input_tensor: ttnn.Tensor, chunks: int, dim: int) -> List[ttnn.Tensor]

        Splits a tensor into multiple chunks along a specified dimension.

        Args:
            * :attr:`input_tensor`: The tensor to split.
            * :attr:`chunks`: Number of chunks to divide the tensor into.
            * :attr:`dim`: Dimension along which to split.

        )doc");
}

}  // namespace ttnn::operations::data_movement
