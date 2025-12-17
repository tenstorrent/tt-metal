// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "repeat_interleave_pybind.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ttnn-pybind/decorators.hpp"

#include "repeat_interleave.hpp"

namespace ttnn::operations::data_movement::detail {
namespace py = pybind11;

void bind_repeat_interleave(py::module& module) {
    const auto* doc = R"doc(
        Repeats elements of a :attr:`tensor` in the given :attr:`dim`.

        Args:
            input_tensor (ttnn.Tensor): the input tensor.
            repeats (number): he number of repetitions for each element. repeats is broadcasted to fit the shape of the given axis.
            dim (number): the dimension to expand with the repetitions.

        Keyword args:
            memory_config (ttnn.MemoryConfig, optional): Memory configuration for the operation. Defaults to `None`.

        Returns:
            ttnn.Tensor: the output tensor.
    )doc";

    ttnn::bind_registered_operation(
        module,
        ttnn::repeat_interleave,
        doc,
        ttnn::pybind_arguments_t{
            py::arg("input_tensor"),
            py::arg("repeats"),
            py::arg("dim"),
            py::kw_only(),
            py::arg("memory_config") = std::nullopt});
}

}  // namespace ttnn::operations::data_movement::detail
