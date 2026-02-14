// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "reshape_nanobind.hpp"

#include <optional>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>

#include "ttnn-nanobind/bind_function.hpp"
#include "ttnn-nanobind/export_enum.hpp"
#include "ttnn-nanobind/small_vector_caster.hpp"
#include "ttnn/operations/data_movement/reshape_view/reshape.hpp"
#include "ttnn/types.hpp"
#include "ttnn/operations/data_movement/reshape_view/reshape_common.hpp"

namespace ttnn::operations::data_movement {

namespace detail {

void bind_reshape_view_operation(nb::module_& mod) {
    const auto* doc = R"doc(
            Note: for a 0 cost view, the following conditions must be met:
                * the last dimension must not change
                * In Tiled the second last two dimensions must not change OR there is no padding on the second last dimension

            Args:
                * input_tensor: Input Tensor.
                * shape: Shape of tensor.

            Keyword Args:
                * :attr:`memory_config`: Memory Config of the output tensor. Default is to match input tensor memory config
                * :attr:`pad_value` (number): Value to pad the output tensor. Default is 0
                * :attr:`reshape_map_mode` (TileReshapeMapMode): Advanced option. Set to RECREATE to recompute and realloc mapping tensor. This may alleviate DRAM fragmentation but is slow. Default is CACHE.
                * :attr:`sub_core_grid` (CoreRangeSet, optional): Specifies sub-core grid ranges for advanced core selection control. Default uses all the cores in the device.


            Returns:
                ttnn.Tensor: the output tensor with the new shape.
        )doc";

    ttnn::bind_function<"reshape">(
        mod,
        doc,

        // Overload 1: single shape (ttnn::Shape)
        ttnn::overload_t(
            nb::overload_cast<
                const ttnn::Tensor&,
                const ttnn::Shape&,
                const std::optional<MemoryConfig>&,
                const std::optional<PadValue>&,
                TileReshapeMapMode,
                const std::optional<CoreRangeSet>&>(&ttnn::reshape),
            nb::arg("input_tensor"),
            nb::arg("shape"),
            nb::kw_only(),
            nb::arg("memory_config") = nb::none(),
            nb::arg("pad_value") = nb::none(),
            nb::arg("reshape_map_mode") = ttnn::TileReshapeMapMode::CACHE,
            nb::arg("sub_core_grid") = nb::none()),

        // Overload 2: logical_shape and padded_shape (ttnn::Shape, ttnn::Shape)
        ttnn::overload_t(
            nb::overload_cast<
                const ttnn::Tensor&,
                const ttnn::Shape&,
                const ttnn::Shape&,
                const std::optional<MemoryConfig>&,
                const std::optional<PadValue>&,
                TileReshapeMapMode,
                const std::optional<CoreRangeSet>&>(&ttnn::reshape),
            nb::arg("input_tensor"),
            nb::arg("logical_shape"),
            nb::arg("padded_shape"),
            nb::kw_only(),
            nb::arg("memory_config") = nb::none(),
            nb::arg("pad_value") = nb::none(),
            nb::arg("reshape_map_mode") = ttnn::TileReshapeMapMode::CACHE,
            nb::arg("sub_core_grid") = nb::none()),

        // Overload 3: shape vector (SmallVector<int32_t>)
        ttnn::overload_t(
            nb::overload_cast<
                const ttnn::Tensor&,
                tt::stl::Span<const int32_t>,
                const std::optional<MemoryConfig>&,
                const std::optional<PadValue>&,
                TileReshapeMapMode,
                const std::optional<CoreRangeSet>&>(&ttnn::reshape),
            nb::arg("input_tensor"),
            nb::arg("shape"),
            nb::kw_only(),
            nb::arg("memory_config") = nb::none(),
            nb::arg("pad_value") = nb::none(),
            nb::arg("reshape_map_mode") = ttnn::TileReshapeMapMode::CACHE,
            nb::arg("sub_core_grid") = nb::none()));
}

}  // namespace detail

void bind_reshape_enum(nb::module_& mod) { export_enum<ttnn::TileReshapeMapMode>(mod, "TileReshapeMapMode"); }

void bind_reshape_view(nb::module_& mod) { detail::bind_reshape_view_operation(mod); }

}  // namespace ttnn::operations::data_movement
