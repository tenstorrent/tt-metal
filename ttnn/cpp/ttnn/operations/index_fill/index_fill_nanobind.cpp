// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "index_fill_nanobind.hpp"

#include <optional>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>

#include "index_fill.hpp"
#include "ttnn-nanobind/bind_function.hpp"

namespace ttnn::operations::index_fill {

void bind_index_fill_operation(nb::module_& mod) {
    const auto* doc =
        R"doc(

        Fills the input tensor with the given value at the specified indices along the specified dimension.

        Args:
            input (ttnn.Tensor): The input tensor.
            dim (int): The dimension along which to fill the value.
            index (ttnn.Tensor): A tensor containing the indices along `dim` to fill with the given `value`.
            value (int or float): The value which will be used to fill the output tensor.

        Keyword Args:
            memory_config (ttnn.MemoryConfig, optional): The memory configuration for the output tensor. Defaults to `None`.

        Returns:
            ttnn.Tensor: The output tensor.

        Note:
            This operation supports tensors according to the following data types and layouts:

            .. list-table:: input tensor
                :header-rows: 1

                * - dtype
                    - layout
                * - BFLOAT16, FLOAT32, INT32
                    - ROW_MAJOR

            .. list-table:: index tensor
                :header-rows: 1

                * - dtype
                    - layout
                * - UINT32
                    - ROW_MAJOR
                * - UINT32
                    - TILE

            Memory Support:
                - Interleaved: DRAM and L1

            Limitations:
                -  The input tensor must be on the device.
                -  The index tensor must be on the device and must be a 1D tensor.
                -  The `dim` must be less than the number of dimensions of the input tensor and >= 0.
                -  The value must be a float or int and must match the dtype of the input tensor.
    )doc";

    ttnn::bind_function<"index_fill">(
        mod,
        doc,
        ttnn::overload_t(
            &ttnn::index_fill,
            nb::arg("input"),
            nb::arg("dim"),
            nb::arg("index"),
            nb::arg("value"),
            nb::kw_only(),
            nb::arg("memory_config") = nb::none()));
}

}  // namespace ttnn::operations::index_fill
