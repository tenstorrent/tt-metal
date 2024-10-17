// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "nonzero_pybind.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "nonzero.hpp"
#include "ttnn/cpp/pybind11/decorators.hpp"

namespace ttnn::operations::nonzero {

void bind_nonzero(py::module& module) {
    auto doc = fmt::format(
        R"doc(

            Returns the number of elements (N) that are non-zero as well as a tensor of the same shape as input where the first N elements are the indices of non-zero elements.

            Args:
                input_tensor (ttnn.Tensor): Input Tensor should be 1D and in row major layout.

            Keyword Args:
                memory_config (ttnn.MemoryConfig, optional): Memory configuration for the operation. Defaults to `None`.

            Returns:
                List of ttnn.Tensor: the output tensor.

            Example:

                >>> tensor = ttnn.to_device(ttnn.from_torch(torch.zeros((1, 1, 1, 32), dtype=torch.bfloat16)), device)
                >>> output = ttnn.nonzero(tensor)
        )doc",
        ttnn::nonzero.base_name());
    bind_registered_operation(
        module,
        ttnn::nonzero,
        doc,
        ttnn::pybind_arguments_t{
            py::arg("input"),
            py::kw_only(),
            py::arg("memory_config") = std::nullopt,
        });
}

}  // namespace ttnn::operations::nonzero
