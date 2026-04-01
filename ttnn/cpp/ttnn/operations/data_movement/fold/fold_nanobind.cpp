// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
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
#include "ttnn-nanobind/bind_function.hpp"
#include "ttnn/types.hpp"

namespace ttnn::operations::data_movement {

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

    ttnn::bind_function<"fold">(
        mod,
        doc,
        ttnn::overload_t(
            &ttnn::fold,
            nb::arg("input"),
            nb::arg("stride_h"),
            nb::arg("stride_w"),
            nb::arg("use_transpose_as_fold") = false,
            nb::arg("output_shape") = nb::none(),
            nb::arg("padding") = std::array<uint32_t, 2>{0, 0},
            nb::arg("grid_size") = nb::none(),
            nb::arg("override_memory_config") = nb::none()));
}

}  // namespace ttnn::operations::data_movement
