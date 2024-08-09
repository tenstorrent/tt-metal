// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "concatenate_heads_pybind.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ttnn/cpp/pybind11/decorators.hpp"
#include "concatenate_heads.hpp"

namespace ttnn::operations::transformer {

void py_bind_concatenate_heads(pybind11::module& module) {
    namespace py = pybind11;
    ttnn::bind_registered_operation(
        module,
        ttnn::transformer::concatenate_heads,
        R"doc(concatenate_heads(input_tensor: ttnn.Tensor, *, memory_config: Optional[MemoryConfig] = None) -> ttnn.Tensor

            Takes in a tensor of shape ``[batch_size, num_heads, sequence_size, head_size]``, concatenates heads back along the width dimension and returns the tensor of shape ``[batch_size, sequence_size, num_heads * head_size]``

            Args:
                * :attr:`input_tensor`: Input Tensor

            Keyword Args:
                * :attr:`memory_config`: Memory Config of the output tensor, if None then it gets set to input_tensor.memory_config()
        )doc",
        ttnn::pybind_arguments_t{
            py::arg("input_tensor"),
            py::kw_only(),
            py::arg("memory_config") = std::nullopt});
}

}  // namespace ttnn::operations::ccl
