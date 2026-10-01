// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
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

ttnn::Tensor reshape_shape_vector_wrapper(
    const ttnn::Tensor& input_tensor,
    const ttsl::SmallVector<int32_t>& shape,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<PadValue>& pad_value,
    const ttnn::TileReshapeMapMode reshape_tile_mode,
    const std::optional<CoreRangeSet>& sub_core_grids,
    const bool skip_padding_fill) {
    return ttnn::reshape(
        input_tensor, shape, memory_config, pad_value, reshape_tile_mode, sub_core_grids, skip_padding_fill);
}

void bind_reshape_view_operation(nb::module_& mod) {
    const auto* doc = R"doc(
            Note: for a 0 cost view, the following conditions must be met:
                * the last dimension must not change
                * In Tiled the second last two dimensions must not change OR there is no padding on the second last dimension

            Args:
                * input_tensor: Input Tensor.
                * shape: Shape of tensor.

            Keyword Args:
                * :attr:`memory_config`: Memory Config of the output tensor. Default is to match input tensor memory config. If ``memory_config`` specifies a sharded layout without ``shard_spec``, the input tensor's ``shard_spec`` is used as the seed grid (layout should match the input).
                * :attr:`pad_value` (number): Value to pad the output tensor. Default is 0
                * :attr:`reshape_tile_mode` (TileReshapeMapMode): Advanced option. Set to RECREATE to recompute and reallocate the mapping tensor. This may alleviate DRAM fragmentation but is slow. Default is CACHE. This keyword is named :attr:`reshape_tile_mode` on all overloads; the small-vector (tuple/list) shape overload previously used the name ``recreate_mapping_tensor`` for the same option—update callers to ``reshape_tile_mode``.
                * :attr:`sub_core_grids` (CoreRangeSet, optional): Specifies sub-core grid ranges for advanced core selection control. Default uses all the cores in the device.
                * :attr:`skip_padding_fill` (bool): If False, ``pad_value`` is applied to tile padding lanes. If True, ``pad_value`` is ignored and tile padding is left as-is. Default is False. Note: this option is silently ignored for ``BFLOAT8_B`` outputs because the BF8 typecast computes a shared exponent across each 16-element sub-block, and unfilled padding would corrupt logical values in straddling sub-blocks; the fill always runs in that case.


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
                const std::optional<CoreRangeSet>&,
                bool>(&ttnn::reshape),
            nb::arg("input_tensor"),
            nb::arg("shape"),
            nb::kw_only(),
            nb::arg("memory_config") = nb::none(),
            nb::arg("pad_value") = nb::none(),
            nb::arg("reshape_tile_mode") = nb::cast(ttnn::TileReshapeMapMode::CACHE),
            nb::arg("sub_core_grids") = nb::none(),
            nb::arg("skip_padding_fill") = false),

        // Overload 2: logical_shape and padded_shape (ttnn::Shape, ttnn::Shape)
        ttnn::overload_t(
            nb::overload_cast<
                const ttnn::Tensor&,
                const ttnn::Shape&,
                const ttnn::Shape&,
                const std::optional<MemoryConfig>&,
                const std::optional<PadValue>&,
                TileReshapeMapMode,
                const std::optional<CoreRangeSet>&,
                bool>(&ttnn::reshape),
            nb::arg("input_tensor"),
            nb::arg("logical_shape"),
            nb::arg("padded_shape"),
            nb::kw_only(),
            nb::arg("memory_config") = nb::none(),
            nb::arg("pad_value") = nb::none(),
            nb::arg("reshape_tile_mode") = nb::cast(ttnn::TileReshapeMapMode::CACHE),
            nb::arg("sub_core_grids") = nb::none(),
            nb::arg("skip_padding_fill") = false),

        // Overload 3: shape vector (ttsl::SmallVector<int32_t>)
        ttnn::overload_t(
            &reshape_shape_vector_wrapper,
            nb::arg("input_tensor"),
            nb::arg("shape"),
            nb::kw_only(),
            nb::arg("memory_config") = nb::none(),
            nb::arg("pad_value") = nb::none(),
            nb::arg("reshape_tile_mode") = nb::cast(ttnn::TileReshapeMapMode::CACHE),
            nb::arg("sub_core_grids") = nb::none(),
            nb::arg("skip_padding_fill") = false));
}

}  // namespace detail

void bind_reshape_enum(nb::module_& mod) { export_enum<ttnn::TileReshapeMapMode>(mod, "TileReshapeMapMode"); }

void bind_reshape_view(nb::module_& mod) { detail::bind_reshape_view_operation(mod); }

}  // namespace ttnn::operations::data_movement
