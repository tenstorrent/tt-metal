// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "pad_nanobind.hpp"

#include <array>
#include <cstdint>
#include <optional>

#include <nanobind/nanobind.h>
#include <nanobind/stl/array.h>
#include <nanobind/stl/optional.h>

#include "ttnn-nanobind/small_vector_caster.hpp"
#include "ttnn-nanobind/decorators.hpp"

#include "pad.hpp"

namespace ttnn::operations::data_movement::detail {

void bind_pad(nb::module_& mod) {
    const auto* doc = R"doc(
        Returns a padded tensor, with a specified value at the specified location. If the input tensor is on host, the pad will be performed on host, and if its on device it will be performed on device.
        Any rank of tensor is supported, however tensors with rank > 4 can only apply padding to the lower 3 dimensions.

        Args:
            * :attr:`input_tensor`: (ttnn.Tensor): the input tensor.
            * :attr:`padding`: (list[Tuple[int,int]]): padding to apply. Each element of padding should be a tuple of 2 integers, with the first integer specifying the number of values to add before the tensor and the second integer specifying the number of values to add after the tensor. Mutually exclusive to output_tensor_shape and input_tensor_start.
            * :attr:`value`: (Union[float,int]): value to pad with.

        Keyword Args:
            * :attr:`use_multicore`: (Optional[bool]) switch to use multicore implementation
            * :attr:`memory_config`: (Optional[ttnn.MemoryConfig]): Memory configuration for the operation. Defaults to `None`.
            * :attr:`sub_core_grids`: (Optional[ttnn.CoreRangeSet]): Sub core grids to run the operation on. Defaults to `None`.

        Returns:
            List of ttnn.Tensor: the output tensor.
    )doc";

    using OperationType = decltype(ttnn::pad);
    ttnn::bind_registered_operation(
        mod,
        ttnn::pad,
        doc,
        ttnn::nanobind_overload_t{
            [](const OperationType& self,
               const ttnn::Tensor& input_tensor,
               const ttnn::SmallVector<std::array<uint32_t, 2>>& padding,
               const float value,
               const bool use_multicore,
               const std::optional<ttnn::MemoryConfig>& memory_config,
               const std::optional<CoreRangeSet>& sub_core_grids) -> ttnn::Tensor {
                return self(input_tensor, padding, value, use_multicore, memory_config, sub_core_grids);
            },
            nb::arg("input_tensor"),
            nb::arg("padding"),
            nb::arg("value"),
            nb::kw_only(),
            nb::arg("use_multicore") = true,
            nb::arg("memory_config") = nb::none(),
            nb::arg("sub_core_grids") = nb::none()},
        ttnn::nanobind_overload_t{
            [](const OperationType& self,
               const ttnn::Tensor& input_tensor,
               const tt::tt_metal::Array4D& output_padded_shape,
               const tt::tt_metal::Array4D& input_tensor_start,
               const float value,
               const bool use_multicore,
               const std::optional<ttnn::MemoryConfig>& memory_config,
               const std::optional<CoreRangeSet>& sub_core_grids) -> ttnn::Tensor {
                return self(
                    input_tensor,
                    output_padded_shape,
                    input_tensor_start,
                    value,
                    use_multicore,
                    memory_config,
                    sub_core_grids);
            },
            nb::arg("input_tensor"),
            nb::arg("output_padded_shape"),
            nb::arg("input_tensor_start"),
            nb::arg("value"),
            nb::kw_only(),
            nb::arg("use_multicore") = false,
            nb::arg("memory_config") = nb::none(),
            nb::arg("sub_core_grids") = nb::none()});
}
}  // namespace ttnn::operations::data_movement::detail
