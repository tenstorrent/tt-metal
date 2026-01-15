// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "slice_nanobind.hpp"

#include <array>
#include <cstdint>
#include <optional>

#include <nanobind/nanobind.h>
#include <nanobind/stl/array.h>
#include <nanobind/stl/optional.h>

#include "ttnn-nanobind/small_vector_caster.hpp"
#include "ttnn-nanobind/decorators.hpp"

#include "slice.hpp"

namespace ttnn::operations::data_movement::detail {

void bind_slice(nb::module_& mod) {
    const auto* doc = R"doc(
        Returns a sliced tensor. If the input tensor is on host, the slice will be performed on host, and if its on device it will be performed on device.

        Args:
            input_tensor: Input Tensor.
            slice_start: Start indices of input tensor. Values along each dim must be < input_tensor_shape[i].
            slice_end: End indices of input tensor. Values along each dim must be < input_tensor_shape[i].
            slice_step: (Optional[List[int[tensor rank]]) Step size for each dim. Default is None, which works out be 1 for each dimension.

        Keyword Args:
            memory_config: Memory Config of the output tensor
            pad_value: Optional value to fill padding for tiled tensors. Padding values are unmodified (and undefined) by default
            sub_core_grids: (ttnn.CoreRangeSet, optional): Sub core grids. Defaults to `None`.

        Returns:
            ttnn.Tensor: the output tensor.
    )doc";

    // TODO: implementing the array version and overloading the nanobind with all the possible array sizes is better
    // than a vector with a fixed size default value
    using OperationType = decltype(ttnn::slice);
    ttnn::bind_registered_operation(
        mod,
        ttnn::slice,
        doc,
        ttnn::nanobind_overload_t{
            [](const OperationType& self,
               const ttnn::Tensor& input_tensor,
               const ttnn::Tensor& slice_start,
               const ttnn::Tensor& slice_end,
               const std::optional<ttnn::SmallVector<uint32_t>>& step,
               const std::optional<ttnn::MemoryConfig>& memory_config,
               const std::optional<Tensor>& optional_output_tensor,
               const std::optional<float>& pad_value,
               // the following two args are used to calculate the output shape for slice op with tensor args
               // running on device to avoid host-device data transfer for mesh device and trace cases
               const std::optional<uint32_t>& slice_dim,
               const std::optional<uint32_t>& num_devices,
               const std::optional<CoreRangeSet>&& sub_core_grids) {
                return self(
                    input_tensor,
                    slice_start,
                    slice_end,
                    step,
                    memory_config,
                    optional_output_tensor,
                    pad_value,
                    slice_dim,
                    num_devices,
                    sub_core_grids);
            },
            nb::arg("input_tensor"),
            nb::arg("starts"),
            nb::arg("ends"),
            nb::arg("slice_step") = nb::none(),  // should consider a better default value
            nb::kw_only(),
            nb::arg("memory_config") = nb::none(),
            nb::arg("output_tensor") = nb::none(),
            nb::arg("pad_value") = nb::none(),
            nb::arg("slice_dim") = nb::none(),
            nb::arg("num_devices") = nb::none(),
            nb::arg("sub_core_grids") = nb::none(),
        },
        ttnn::nanobind_overload_t{
            [](const OperationType& self,
               const ttnn::Tensor& input_tensor,
               const std::array<uint32_t, 4>& begins,
               const std::array<uint32_t, 4>& ends,
               const std::array<uint32_t, 4>& step,
               const std::optional<ttnn::MemoryConfig>& memory_config,
               const std::optional<Tensor>& optional_output_tensor,
               const std::optional<float>& pad_value,
               const std::optional<CoreRangeSet>&& sub_core_grids) {
                return self(
                    input_tensor, begins, ends, step, memory_config, optional_output_tensor, pad_value, sub_core_grids);
            },
            nb::arg("input_tensor"),
            nb::arg("starts"),
            nb::arg("ends"),
            nb::arg("steps"),
            nb::kw_only(),
            nb::arg("memory_config") = nb::none(),
            nb::arg("output_tensor") = nb::none(),
            nb::arg("pad_value") = nb::none(),
            nb::arg("sub_core_grids") = nb::none()},
        ttnn::nanobind_overload_t{
            [](const OperationType& self,
               const ttnn::Tensor& input_tensor,
               const ttnn::SmallVector<int>& slice_start,
               const ttnn::SmallVector<int>& slice_end,
               const std::optional<ttnn::SmallVector<int>>& step,
               const std::optional<ttnn::MemoryConfig>& memory_config,
               const std::optional<Tensor>& optional_output_tensor,
               const std::optional<float>& pad_value,
               const std::optional<CoreRangeSet>&& sub_core_grids) {
                const auto step_value = step.value_or(ttnn::SmallVector<int>(slice_end.size(), 1));
                return self(
                    input_tensor,
                    slice_start,
                    slice_end,
                    step_value,
                    memory_config,
                    optional_output_tensor,
                    pad_value,
                    sub_core_grids);
            },
            nb::arg("input_tensor"),
            nb::arg("slice_start"),
            nb::arg("slice_end"),
            nb::arg("slice_step") = nb::none(),  // should consider a better default value
            nb::kw_only(),
            nb::arg("memory_config") = nb::none(),
            nb::arg("output_tensor") = nb::none(),
            nb::arg("pad_value") = nb::none(),
            nb::arg("sub_core_grids") = std::nullopt}

    );
}
}  // namespace ttnn::operations::data_movement::detail
