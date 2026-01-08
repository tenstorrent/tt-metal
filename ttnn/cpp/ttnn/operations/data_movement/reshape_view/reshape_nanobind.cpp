// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "reshape_nanobind.hpp"

#include <optional>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>

#include "ttnn-nanobind/decorators.hpp"
#include "ttnn-nanobind/export_enum.hpp"
#include "ttnn/operations/data_movement/reshape_view/reshape.hpp"
#include "ttnn/types.hpp"
#include "ttnn/operations/data_movement/reshape_view/reshape_common.hpp"

namespace ttnn::operations::data_movement {

namespace detail {

template <typename data_movement_operation_t>
void bind_reshape_view(nb::module_& mod, const data_movement_operation_t& operation, const char* doc) {
    bind_registered_operation(
        mod,
        operation,
        doc,
        ttnn::nanobind_overload_t{
            [](const data_movement_operation_t& self,
               const ttnn::Tensor& input_tensor,
               const ttnn::Shape& shape,
               const std::optional<MemoryConfig>& memory_config,
               const std::optional<PadValue>& pad_value,
               const ttnn::TileReshapeMapMode reshape_tile_mode,
               const std::optional<CoreRangeSet>& sub_core_grids) -> ttnn::Tensor {
                return self(input_tensor, shape, memory_config, pad_value, reshape_tile_mode, sub_core_grids);
            },
            nb::arg("input_tensor"),
            nb::arg("shape"),
            nb::kw_only(),
            nb::arg("memory_config") = nb::none(),
            nb::arg("pad_value") = nb::none(),
            nb::arg("reshape_tile_mode") = nb::cast(ttnn::TileReshapeMapMode::CACHE),
            nb::arg("sub_core_grids") = nb::none()},
        ttnn::nanobind_overload_t{
            [](const data_movement_operation_t& self,
               const ttnn::Tensor& input_tensor,
               const ttnn::Shape& logical_shape,
               const ttnn::Shape& padded_shape,
               const std::optional<MemoryConfig>& memory_config,
               const std::optional<PadValue>& pad_value,
               const ttnn::TileReshapeMapMode reshape_tile_mode,
               const std::optional<CoreRangeSet>& sub_core_grids) -> ttnn::Tensor {
                return self(
                    input_tensor,
                    logical_shape,
                    padded_shape,
                    memory_config,
                    pad_value,
                    reshape_tile_mode,
                    sub_core_grids);
            },
            nb::arg("input_tensor"),
            nb::arg("logical_shape"),
            nb::arg("padded_shape"),
            nb::kw_only(),
            nb::arg("memory_config") = nb::none(),
            nb::arg("pad_value") = nb::none(),
            nb::arg("reshape_tile_mode") = nb::cast(ttnn::TileReshapeMapMode::CACHE),
            nb::arg("sub_core_grids") = nb::none()},
        ttnn::nanobind_overload_t{
            [](const data_movement_operation_t& self,
               const ttnn::Tensor& input_tensor,
               const ttnn::SmallVector<int32_t>& shape,
               const std::optional<MemoryConfig>& memory_config,
               const std::optional<PadValue>& pad_value,
               const ttnn::TileReshapeMapMode reshape_tile_mode,
               const std::optional<CoreRangeSet>& sub_core_grids) -> ttnn::Tensor {
                return self(input_tensor, shape, memory_config, pad_value, reshape_tile_mode, sub_core_grids);
            },
            nb::arg("input_tensor"),
            nb::arg("shape"),
            nb::kw_only(),
            nb::arg("memory_config") = nb::none(),
            nb::arg("pad_value") = nb::none(),
            nb::arg("recreate_mapping_tensor") = nb::cast(ttnn::TileReshapeMapMode::CACHE),
            nb::arg("sub_core_grids") = nb::none()});
}
}  // namespace detail

void bind_reshape_enum(nb::module_& mod) { export_enum<ttnn::TileReshapeMapMode>(mod, "TileReshapeMapMode"); }

void bind_reshape_view(nb::module_& mod) {
    detail::bind_reshape_view(
        mod,
        ttnn::reshape,
        R"doc(
            Note: for a 0 cost view, the following conditions must be met:
                * the last dimension must not change
                * In Tiled the second last two dimensions must not change OR there is no padding on the second last dimension

            Args:
                * input_tensor: Input Tensor.
                * shape: Shape of tensor.

            Keyword Args:
                * :attr:`memory_config`: Memory Config of the output tensor. Default is to match input tensor memory config
                * :attr:`pad_value` (number): Value to pad the output tensor. Default is 0
                * :attr:`recreate_mapping_tensor` (bool): Advanced option. Set to true to recompute and realloc mapping tensor. This may alleviate DRAM fragmentation but is slow.
                * :attr:`sub_core_grids` (CoreRangeSet, optional): Specifies sub-core grid ranges for advanced core selection control. Default uses all the cores in the device.


            Returns:
                ttnn.Tensor: the output tensor with the new shape.
        )doc");
}

}  // namespace ttnn::operations::data_movement
