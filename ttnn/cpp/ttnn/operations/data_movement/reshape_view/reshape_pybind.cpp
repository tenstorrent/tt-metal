// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "reshape_pybind.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ttnn-pybind/decorators.hpp"
#include "ttnn/operations/data_movement/reshape_view/reshape.hpp"
#include "ttnn/types.hpp"
#include "ttnn/operations/data_movement/reshape_view/reshape_common.hpp"

namespace ttnn::operations::data_movement {

namespace detail {

template <typename data_movement_operation_t>
void bind_reshape_view(pybind11::module& module, const data_movement_operation_t& operation, const char* doc) {
    bind_registered_operation(
        module,
        operation,
        doc,
        ttnn::pybind_overload_t{
            [](const data_movement_operation_t& self,
               const ttnn::Tensor& input_tensor,
               const ttnn::Shape& shape,
               const std::optional<MemoryConfig>& memory_config,
               const std::optional<PadValue>& pad_value,
               const ttnn::TileReshapeMapMode reshape_tile_mode,
               const std::optional<CoreRangeSet>& sub_core_grids) -> ttnn::Tensor {
                return self(input_tensor, shape, memory_config, pad_value, reshape_tile_mode, sub_core_grids);
            },
            py::arg("input_tensor"),
            py::arg("shape"),
            py::kw_only(),
            py::arg("memory_config") = std::nullopt,
            py::arg("pad_value") = std::nullopt,
            py::arg("reshape_tile_mode") = ttnn::TileReshapeMapMode::CACHE,
            py::arg("sub_core_grids") = std::nullopt},
        ttnn::pybind_overload_t{
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
            py::arg("input_tensor"),
            py::arg("logical_shape"),
            py::arg("padded_shape"),
            py::kw_only(),
            py::arg("memory_config") = std::nullopt,
            py::arg("pad_value") = std::nullopt,
            py::arg("reshape_tile_mode") = ttnn::TileReshapeMapMode::CACHE,
            py::arg("sub_core_grids") = std::nullopt},
        ttnn::pybind_overload_t{
            [](const data_movement_operation_t& self,
               const ttnn::Tensor& input_tensor,
               const ttnn::SmallVector<int32_t>& shape,
               const std::optional<MemoryConfig>& memory_config,
               const std::optional<PadValue>& pad_value,
               const ttnn::TileReshapeMapMode reshape_tile_mode,
               const std::optional<CoreRangeSet>& sub_core_grids) -> ttnn::Tensor {
                return self(input_tensor, shape, memory_config, pad_value, reshape_tile_mode, sub_core_grids);
            },
            py::arg("input_tensor"),
            py::arg("shape"),
            py::kw_only(),
            py::arg("memory_config") = std::nullopt,
            py::arg("pad_value") = std::nullopt,
            py::arg("recreate_mapping_tensor") = ttnn::TileReshapeMapMode::CACHE,
            py::arg("sub_core_grids") = std::nullopt});
}
}  // namespace detail

void py_bind_reshape_enum(pybind11::module& module) {
    py::enum_<ttnn::TileReshapeMapMode>(module, "TileReshapeMapMode")
        .value("CACHE", ttnn::TileReshapeMapMode::CACHE)
        .value("RECREATE", ttnn::TileReshapeMapMode::RECREATE);
}

void py_bind_reshape_view(pybind11::module& module) {
    detail::bind_reshape_view(
        module,
        ttnn::reshape,
        R"doc(
            Note: for a 0 cost view, the following conditions must be met:
                * the last dimension must not change
                * In Tiled the second last two dimensions must not change OR there is no padding on the second last dimension

            Args:
                * input_tensor: Input Tensor.
                * new_shape: New shape of tensor.

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
