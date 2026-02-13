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
#include "ttnn-nanobind/bind_function.hpp"

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
    ttnn::bind_function<"slice">(
        mod,
        doc,
        // Overload 1: Tensor args version (uint32_t template parameter)
        ttnn::overload_t(
            static_cast<ttnn::Tensor (*)(
                const ttnn::Tensor&,
                const ttnn::Tensor&,
                const ttnn::Tensor&,
                const std::optional<ttnn::SmallVector<uint32_t>>&,
                const std::optional<MemoryConfig>&,
                const std::optional<Tensor>&,
                const std::optional<float>&,
                const std::optional<uint32_t>&,
                const std::optional<uint32_t>&,
                const std::optional<CoreRangeSet>&)>(&ttnn::slice<uint32_t>),
            nb::arg("input_tensor"),
            nb::arg("starts"),
            nb::arg("ends"),
            nb::arg("slice_step") = nb::none(),
            nb::kw_only(),
            nb::arg("memory_config") = nb::none(),
            nb::arg("output_tensor") = nb::none(),
            nb::arg("pad_value") = nb::none(),
            nb::arg("slice_dim") = nb::none(),
            nb::arg("num_devices") = nb::none(),
            nb::arg("sub_core_grids") = nb::none()),
        // Overload 2: std::array version (uint32_t template parameter, size 4)
        ttnn::overload_t(
            static_cast<ttnn::Tensor (*)(
                const ttnn::Tensor&,
                const std::array<uint32_t, 4>&,
                const std::array<uint32_t, 4>&,
                const std::array<uint32_t, 4>&,
                const std::optional<MemoryConfig>&,
                const std::optional<Tensor>&,
                const std::optional<float>&,
                const std::optional<CoreRangeSet>&)>(&ttnn::slice<uint32_t, 4>),
            nb::arg("input_tensor"),
            nb::arg("starts"),
            nb::arg("ends"),
            nb::arg("steps"),
            nb::kw_only(),
            nb::arg("memory_config") = nb::none(),
            nb::arg("output_tensor") = nb::none(),
            nb::arg("pad_value") = nb::none(),
            nb::arg("sub_core_grids") = nb::none()),
        // Overload 3: SmallVector<int> version (int32_t template parameter)
        ttnn::overload_t(
            static_cast<ttnn::Tensor (*)(
                const ttnn::Tensor&,
                const ttnn::SmallVector<int32_t>&,
                const ttnn::SmallVector<int32_t>&,
                const ttnn::SmallVector<int32_t>&,
                const std::optional<MemoryConfig>&,
                const std::optional<Tensor>&,
                const std::optional<float>&,
                const std::optional<CoreRangeSet>&)>(&ttnn::slice<int32_t>),
            nb::arg("input_tensor"),
            nb::arg("slice_start"),
            nb::arg("slice_end"),
            nb::arg("slice_step"),
            nb::kw_only(),
            nb::arg("memory_config") = nb::none(),
            nb::arg("output_tensor") = nb::none(),
            nb::arg("pad_value") = nb::none(),
            nb::arg("sub_core_grids") = nb::none()));
}
}  // namespace ttnn::operations::data_movement::detail
