// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/data_movement/fold/fold_nanobind.hpp"

#include <array>
#include <cstdint>
#include <optional>
#include <variant>

#include <nanobind/nanobind.h>
#include <nanobind/stl/array.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/variant.h>

#include "ttnn/operations/data_movement/fold/fold.hpp"
#include "ttnn/operations/data_movement/fold/device/fold_device_op.hpp"
#include "ttnn-nanobind/bind_function.hpp"
#include "ttnn/types.hpp"

namespace ttnn::operations::data_movement {

void bind_fold_operation(nb::module_& mod) {
    const auto* doc = R"doc(
        NHWC space-to-depth. Packs every stride_h * stride_w neighbourhood at position
        (h * stride_h, w * stride_w) into the channel dim at position (h, w). Shapes below
        use the padded H/W/C (any HW / channel padding is applied before folding).

        Output shape: (N, Hp/stride_h, Wp/stride_w, Cp * stride_h * stride_w)
        where Hp, Wp, Cp are the padded input dims.

        Args:
            input (ttnn.Tensor): Input tensor [N, H, W, C].
            stride_h (int): Stride along H.
            stride_w (int): Stride along W.
            collapse_output (bool, optional): Default False. When True, returns
                (1, 1, N * Hp/stride_h * Wp/stride_w, Cp * stride_h * stride_w) instead.
    )doc";

    ttnn::bind_function<"fold">(
        mod,
        doc,
        &ttnn::fold,
        nb::arg("input"),
        nb::arg("stride_h"),
        nb::arg("stride_w"),
        nb::arg("use_transpose_as_fold") = false,
        nb::arg("output_shape") = nb::none(),
        nb::arg("padding") = std::array<uint32_t, 2>{0, 0},
        nb::arg("grid_size") = nb::none(),
        nb::arg("override_memory_config") = nb::none(),
        nb::arg("collapse_output") = false);

    // Test-only hooks used by test_fold_tile_gate_pins_routing / test_fold_tile_gate_pins_alignment
    // to pin the tile-native routing. _prim_fold bypasses the composite gate so validate_fold's
    // FATAL fires on unsupported inputs; _is_tile_native_fold_supported lets tests compute their
    // fits/over shapes from the same predicate both C++ callers consult.
    mod.def(
        "_prim_fold",
        [](const ttnn::Tensor& input, uint32_t stride_h, uint32_t stride_w, bool collapse_output) {
            return ttnn::prim::fold(input, stride_h, stride_w, collapse_output);
        },
        nb::arg("input"),
        nb::arg("stride_h"),
        nb::arg("stride_w"),
        nb::arg("collapse_output") = false);
    mod.def(
        "_is_tile_native_fold_supported",
        &ttnn::operations::data_movement::is_tile_native_fold_supported,
        nb::arg("input"),
        nb::arg("stride_h"),
        nb::arg("stride_w"));
}

}  // namespace ttnn::operations::data_movement
