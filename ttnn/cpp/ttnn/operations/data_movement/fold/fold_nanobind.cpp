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
#include "ttnn-nanobind/bind_function.hpp"
#include "ttnn/types.hpp"

namespace ttnn::operations::data_movement {

void bind_fold_operation(nb::module_& mod) {
    const auto* doc = R"doc(
        Fold (NHWC space-to-depth): packs a stride_h x stride_w neighbourhood along H,W into C.

        Args:
            input (ttnn.Tensor): Input tensor of shape ``[N, H, W, C]``.
            stride_h (int): Stride along H.
            stride_w (int): Stride along W.
            use_transpose_as_fold (bool, optional): Transpose-based fold path. Defaults to False.
            output_shape (ttnn.Shape, optional): Explicit output shape (transpose path only).
            padding (list[int], optional): Pre-fold padding, length 2/4/6. Defaults to ``[0, 0]``.
            grid_size (ttnn.CoreRangeSet, optional): Grid for the transpose path.
            override_memory_config (ttnn.MemoryConfig, optional): Requested output memory config.

        Returns:
            ttnn.Tensor: Folded tensor.

        Notes:
            - Output shape is 4D ``(N, H/stride_h, W/stride_w, C*stride_h*stride_w)`` for TILE,
              sharded, DRAM RM, and override paths. L1 RM interleaved returns the legacy collapsed
              shape ``(1, 1, N*H*W/(stride_h*stride_w), C*stride_h*stride_w)``.
            - Requires ``H % stride_h == 0`` and ``W % stride_w == 0``; sharded RM additionally
              requires ``shard_shape[0] % (stride_h*stride_w) == 0``.
            - ``override_memory_config`` without a shard_spec synthesises one over the device grid,
              inheriting orientation from the input.
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
        nb::arg("override_memory_config") = nb::none());
}

}  // namespace ttnn::operations::data_movement
