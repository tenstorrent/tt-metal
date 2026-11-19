// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "move_nanobind.hpp"

#include <optional>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>

#include "ttnn-nanobind/bind_function.hpp"

#include "move.hpp"

namespace ttnn::operations::data_movement::detail {

void bind_move(nb::module_& mod) {
    const auto* doc = R"doc(
            Moves `input_tensor` into a newly allocated tensor and deallocates `input_tensor`.

            `input_tensor` is deallocated before the output is allocated, so the output can take its place, or memory
            that tensors deallocated earlier left free. This can compact device memory.

            Args:
                input_tensor (ttnn.Tensor): the tensor to move, on device. No other tensor, such as a view from ``ttnn.view``, may share its device buffer. It is deallocated by the call and must not be used afterwards.

            Keyword Args:
                memory_config (ttnn.MemoryConfig, optional): Memory configuration of the output. Defaults to `None`, which uses the input tensor's memory configuration. For an interleaved input it must be interleaved, and for a sharded input `None` or the input's memory configuration.

            Returns:
                ttnn.Tensor: the output tensor, with the shape, dtype, layout and values of the input.

            Note:
                Supported dtypes and layouts:

                .. list-table::
                   :header-rows: 1

                   * - Dtypes
                     - Layouts
                   * - BFLOAT16, FLOAT32, INT32, UINT32, UINT16, UINT8
                     - TILE, ROW_MAJOR
                   * - BFLOAT8_B, BFLOAT4_B
                     - TILE

                On Blackhole, the input can also be an ``FP8_E4M3`` tensor in ROW_MAJOR layout.

                When an interleaved output is in the input's buffer type, each core's share of the tensor must fit in
                that core's free L1: anywhere above the allocator's base for a DRAM tensor, and below the output for an
                L1 tensor. A core's share is the tensor's number of pages divided by the number of cores in the
                device's compute grid, rounded up, times the page size rounded up to the tensor's buffer alignment. A
                page is a tile for a TILE tensor and a row for a ROW_MAJOR one.

            Memory Support:
                - Interleaved: DRAM and L1; the input and the output can be in different buffer types
                - Sharded: HEIGHT_SHARDED, WIDTH_SHARDED and BLOCK_SHARDED in L1, with the output in the input's memory configuration
        )doc";

    ttnn::bind_function<"move">(
        mod,
        doc,
        &ttnn::move,
        nb::arg("input_tensor").noconvert(),
        nb::kw_only(),
        nb::arg("memory_config") = nb::none());
}

}  // namespace ttnn::operations::data_movement::detail
