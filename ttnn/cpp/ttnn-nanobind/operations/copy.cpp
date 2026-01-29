// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "copy.hpp"

#include <optional>

#include <fmt/format.h>
#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>

#include "ttnn-nanobind/decorators.hpp"
#include "ttnn/operations/copy/typecast/typecast.hpp"
#include "ttnn/types.hpp"

namespace ttnn::operations::copy {

namespace {

void bind_global_typecast(nb::module_& mod) {
    auto doc = fmt::format(
        R"doc(
        Performs {0} on elements of a tensor on the host or device to the desired dtype.

        Args:
            input_tensor (ttnn.Tensor): input tensor to be typecast (can be on the host or device).
            dtype (ttnn.DataType): data type to cast the tensor elements to.

        Keyword Args:
            memory_config (Optional[ttnn.MemoryConfig]): Memory configuration for the operation.
            output_tensor (Optional[ttnn.Tensor]): Preallocated tensor to store the output.

        Returns:
            ttnn.Tensor: The tensor with the updated data type. The output tensor will be in the same layout as the input tensor and have the given data type.

        Note:
            This operations supports tensors according to the following data types and layout:

            .. list-table:: input_tensor
                :header-rows: 1

                * - dtype
                    - layout
                * - BFLOAT16, BFLOAT8_B, BFLOAT4_B, FLOAT32, UINT32, INT32, UINT16, UINT8
                    - TILE
                * - BFLOAT16, FLOAT32, UINT32, INT32, UINT16, UINT8
                    - ROW_MAJOR

            Memory Support:
                - Interleaved: DRAM and L1
                - Height, Width, and Block Sharded: DRAM and L1

            Limitations:
                -  ND Sharded tensors are not supported.
                -  If preallocated output tensor is used, it must match the input tensor's shape and layout.
        )doc",
        ttnn::typecast.base_name());

    using TypecastType = decltype(ttnn::typecast);
    bind_registered_operation(
        mod,
        ttnn::typecast,
        doc,
        ttnn::nanobind_overload_t{
            [](const TypecastType& self,
               const ttnn::Tensor& input_tensor,
               const DataType dtype,
               const std::optional<ttnn::MemoryConfig>& memory_config,
               const std::optional<ttnn::Tensor>& output_tensor,
               const std::optional<CoreRangeSet>& sub_core_grids) -> ttnn::Tensor {
                return self(input_tensor, dtype, memory_config, output_tensor, sub_core_grids);
            },
            nb::arg("input_tensor"),
            nb::arg("dtype"),
            nb::kw_only(),
            nb::arg("memory_config") = nb::none(),
            nb::arg("output_tensor") = nb::none(),
            nb::arg("sub_core_grids") = nb::none()},

        ttnn::nanobind_overload_t{
            [](const TypecastType& self,
               const ttnn::Tensor& input_tensor,
               const DataType input_dtype,
               const DataType output_dtype,
               const std::optional<MemoryConfig>& memory_config,
               const std::optional<Tensor>& output_tensor,
               const std::optional<CoreRangeSet>& sub_core_grids) -> ttnn::Tensor {
                return self(input_tensor, input_dtype, output_dtype, memory_config, output_tensor, sub_core_grids);
            },
            nb::arg("input_tensor"),
            nb::arg("input_dtype"),
            nb::arg("output_dtype"),
            nb::kw_only(),
            nb::arg("memory_config") = nb::none(),
            nb::arg("output_tensor") = nb::none(),
            nb::arg("sub_core_grids") = nb::none()});
}

}  // namespace

void py_module(nb::module_& mod) { bind_global_typecast(mod); }

}  // namespace ttnn::operations::copy
