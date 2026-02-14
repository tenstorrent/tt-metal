// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "gather_nanobind.hpp"

#include <cstdint>
#include <optional>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>

#include "gather.hpp"
#include "ttnn-nanobind/bind_function.hpp"

namespace ttnn::operations::data_movement::detail {

void bind_gather_operation(nb::module_& mod) {
    const auto* doc = R"doc(
        The `gather` operation extracts values from the input tensor based on indices provided in the index tensor along a specified dimension.

        The input tensor and the index tensor must have the same number of dimensions.
        For all dimensions except the specified one (`dim`), the size of the index tensor must not exceed the size of the input tensor.
        The output tensor will have the same shape as the index tensor. Note that the input and index tensors do not broadcast against each other.

        Args:
            input (ttnn.Tensor): The source tensor from which values are gathered.
            dim (int): The dimension along which values are gathered.
            index (ttnn.Tensor): A tensor containing the indices of elements to gather, with the same number of dimensions as the input tensor.

        Keyword Arguments:
            sparse_grad (bool, optional): If `True`, the gradient computation will be sparse. Defaults to `False`.
            memory_config (ttnn.MemoryConfig, optional): Specifies the memory configuration for the output tensor. Defaults to `None`.
            out (ttnn.Tensor, optional): A preallocated tensor to store the gathered values. Defaults to `None`.
            sub_core_grids (ttnn.CoreRangeSet, optional): Custom core range set for operation execution. Allows specification of which cores should be used for the operation. Defaults to `None`.

        Additional Information:
            * Currently, the `sparse_grad` argument is not supported.

        Note:

            Supported dtypes and layout for input tensor values:

            .. list-table::
                :header-rows: 1

                * - Dtypes
                  - Layouts
                * - BFLOAT16, FLOAT32
                  - TILE
                * - UINT16, UINT32
                  - TILE
                * - INT32
                  - TILE

            Supported dtypes and layout for index tensor values:

            .. list-table::
                :header-rows: 1

                * - Dtypes
                  - Layouts
                * - UINT16, UINT32
                  - TILE

        Memory Support:
            - Interleaved: DRAM and L1
    )doc";

    ttnn::bind_function<"gather">(
        mod,
        doc,
        ttnn::overload_t(
            &ttnn::gather,
            nb::arg("input").noconvert(),
            nb::arg("dim"),
            nb::arg("index"),
            nb::kw_only(),
            nb::arg("sparse_grad") = false,
            nb::arg("out") = nb::none(),
            nb::arg("memory_config") = nb::none(),
            nb::arg("sub_core_grids") = nb::none()));
}

}  // namespace ttnn::operations::data_movement::detail
