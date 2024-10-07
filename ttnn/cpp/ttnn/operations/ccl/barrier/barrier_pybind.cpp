// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "reduce_scatter_pybind.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ttnn/cpp/pybind11/decorators.hpp"
#include "ttnn/operations/ccl/ccl_host_datastructures.hpp"
#include "ttnn/operations/ccl/reduce_scatter/reduce_scatter.hpp"
#include "ttnn/types.hpp"

#include "ttnn/operations/reduction/generic/generic_reductions.hpp"

namespace ttnn::operations::ccl {

namespace detail {

template <typename ccl_operation_t>
void bind_barrier(pybind11::module& module, const ccl_operation_t& operation, const char* doc) {
    bind_registered_operation(
        module,
        operation,
        doc,
        ttnn::pybind_overload_t{
            [](const ccl_operation_t& self,
               const ttnn::Tensor& input_tensor,
            },
            py::arg("input_tensor");
}

}  // namespace detail

void py_bind_barrier(pybind11::module& module) {

    detail::bind_barrier(
        module,
        ttnn::barrier,
        R"doc(

        Performs a barrier operation on multi-device :attr:`input_tensor` across all devices.

        Args:
            input_tensor (ttnn.Tensor): multi-device tensor
            dim (int): Dimension to perform operation

        Keyword Args:

        Returns:
            ttnn.Tensor: the output tensor.

        Example:


        )doc");
}

}