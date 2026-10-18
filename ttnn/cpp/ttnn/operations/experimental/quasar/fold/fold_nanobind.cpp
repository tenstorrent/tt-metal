// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/experimental/quasar/fold/fold_nanobind.hpp"

#include <array>
#include <cstdint>
#include <optional>
#include <variant>

#include <nanobind/nanobind.h>
#include <nanobind/stl/array.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/variant.h>

#include "ttnn/operations/experimental/quasar/fold/fold.hpp"
#include "ttnn/operations/experimental/quasar/fold/device/fold_device_op.hpp"
#include "ttnn-nanobind/bind_function.hpp"
#include "ttnn/types.hpp"

namespace ttnn::operations::experimental::quasar::detail {

void bind_fold_operation(nb::module_& mod) {
    const auto* doc = R"doc(
        Fold TT Tensor.
        Input tensor must be on TT accelerator device, in ROW_MAJOR.
        Output tensor will be on TT accelerator device, in ROW_MAJOR.

        Args:
            input (ttnn.Tensor): Input tensor to be folded. Tensor of shape [N, H, W, C].
            stride_h (int): Stride along the H-dimension.
            stride_w (int): Stride along the W-dimension.
    )doc";

    ttnn::bind_function<"fold", "ttnn.experimental.quasar.">(
        mod,
        doc,
        &ttnn::operations::experimental::quasar::fold,
        nb::arg("input"),
        nb::arg("stride_h"),
        nb::arg("stride_w"),
        nb::arg("use_transpose_as_fold") = false,
        nb::arg("output_shape") = nb::none(),
        nb::arg("padding") = std::array<uint32_t, 2>{0, 0},
        nb::arg("grid_size") = nb::none(),
        nb::arg("override_memory_config") = nb::none(),
        nb::arg("input_is_nhwc") = false);

    // Test-only hook: _prim_fold bypasses the composite gate so validate_fold's FATAL surfaces.
    mod.def(
        "_prim_fold",
        [](const ttnn::Tensor& input, uint32_t stride_h, uint32_t stride_w) {
            return ttnn::prim::qsr::fold(input, stride_h, stride_w);
        },
        nb::arg("input"),
        nb::arg("stride_h"),
        nb::arg("stride_w"));
    mod.def(
        "_is_tile_native_fold_supported",
        &ttnn::operations::experimental::quasar::is_tile_native_fold_supported,
        nb::arg("input"),
        nb::arg("stride_h"),
        nb::arg("stride_w"));
}

}  // namespace ttnn::operations::experimental::quasar::detail
