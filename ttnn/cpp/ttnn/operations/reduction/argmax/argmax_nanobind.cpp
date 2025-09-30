// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "argmax_nanobind.hpp"

#include <optional>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>

#include "ttnn-nanobind/decorators.hpp"
#include "ttnn/operations/reduction/argmax/argmax.hpp"

namespace ttnn::operations::reduction::detail {
void bind_reduction_argmax_operation(nb::module_& mod) {
    auto doc =
        R"doc(
            Returns the indices of the maximum value of elements in the :attr:`input_tensor`.
            If no :attr:`dim` is provided, it will return the indices of maximum value of all elements in given :attr:`input_tensor`.

            Args:
                input_tensor (ttnn.Tensor): the input tensor.

            Keyword args:
                dim (int, optional): dimension to reduce. Defaults to `None`.
                keepdim (bool, optional): whether to keep the reduced dimension. Defaults to `False`.
                memory_config (ttnn.MemoryConfig, optional): Memory configuration for the operation. Defaults to `None`.
                output_tensor (ttnn.Tensor, optional): Preallocated output tensor. Defaults to `None`.

            Returns:
                ttnn.Tensor: Output tensor containing the indices of the maximum value.

            Note:
                The input tensor supports the following data types and layouts:

                .. list-table:: Input Tensor
                    :header-rows: 1

                    * - dtype
                        - layout
                    * - FLOAT32
                        - ROW_MAJOR
                    * - BFLOAT16
                        - ROW_MAJOR
                    * - UINT32
                        - ROW_MAJOR
                    * - INT32
                        - ROW_MAJOR
                    * - UINT16
                        - ROW_MAJOR

                The output tensor will be of the following data type and layout:

                .. list-table:: Output Tensor
                    :header-rows: 1

                    * - dtype
                        - layout
                    * - UINT32
                        - ROW_MAJOR

            Limitations:
                Currently this op only supports dimension-specific reduction on the last dimension (i.e. :attr:`dim` = -1).

            Example:
                input_tensor = ttnn.rand([1, 1, 32, 64], device=device, layout=ttnn.ROW_MAJOR_LAYOUT)

                # Last dim reduction yields shape of [1, 1, 32, 1]
                output_onedim = ttnn.argmax(input_tensor, dim=-1, keepdim=True)

                # All dim reduction yields shape of []
                output_alldim = ttnn.argmax(input_tensor)

        )doc";

    using OperationType = decltype(ttnn::argmax);
    bind_registered_operation(
        mod,
        ttnn::argmax,
        doc,
        ttnn::nanobind_overload_t{
            [](const OperationType& self,
               const ttnn::Tensor& input_tensor,
               const std::optional<int> dim,
               const bool keepdim,
               const std::optional<CoreRangeSet>& sub_core_grids,
               const bool use_multicore,
               const std::optional<ttnn::MemoryConfig>& memory_config,
               std::optional<ttnn::Tensor> optional_output_tensor) {
                return self(
                    input_tensor, dim, keepdim, sub_core_grids, use_multicore, memory_config, optional_output_tensor);
            },
            nb::arg("input_tensor").noconvert(),
            nb::arg("dim") = nb::none(),
            nb::arg("keepdim") = false,
            nb::kw_only(),
            nb::arg("sub_core_grids") = nb::none(),
            nb::arg("use_multicore") = false,
            nb::arg("memory_config") = nb::none(),
            nb::arg("output_tensor") = nb::none()});
}

}  // namespace ttnn::operations::reduction::detail
