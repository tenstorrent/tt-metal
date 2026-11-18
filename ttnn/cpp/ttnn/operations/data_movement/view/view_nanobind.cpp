// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "view_nanobind.hpp"

#include <cstdint>

#include <nanobind/nanobind.h>

#include "ttnn-nanobind/small_vector_caster.hpp"
#include "ttnn-nanobind/bind_function.hpp"
#include "ttnn/operations/data_movement/view/view.hpp"
#include "ttnn/types.hpp"

namespace ttnn::operations::data_movement {

namespace {

ttnn::Tensor view_shape_vector_wrapper(const ttnn::Tensor& input_tensor, const ttsl::SmallVector<int32_t>& shape) {
    return ttnn::view(input_tensor, shape);
}

}  // namespace

void bind_view(nb::module_& mod) {
    const auto* doc =
        R"doc(
        Returns a tensor with a new shape that shares the input tensor's device buffer, without copying or moving data.

        Args:
            input_tensor (ttnn.Tensor): the input tensor, on device, in ROW_MAJOR layout or in TILE layout with 32x32 tiles. A TILE input's padded shape must be its shape with the last two dimensions rounded up to multiples of 32.
            shape (ttnn.Shape or List[int]): the new shape. It must have the input's number of elements and the input's last dimension. For a TILE input, its second-to-last dimension must be the input's, or a multiple of 32 in both the input and `shape`. As a list or tuple, one dimension may be `-1`; its size is inferred from the other dimensions.

        Returns:
            ttnn.Tensor: the output tensor, with the new shape and the input's dtype, layout and memory configuration. It holds the input's values in the same row-major order, in the input's device buffer.

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

            On Blackhole, the input can also be a ``FP8_E4M3`` tensor in ROW_MAJOR layout.

        Memory Support:
            - Interleaved: DRAM and L1
            - Sharded: HEIGHT_SHARDED, WIDTH_SHARDED and BLOCK_SHARDED in L1; a sharded input needs a `shape` of rank 2 or more
        )doc";

    ttnn::bind_function<"view">(
        mod,
        doc,
        // Bind first overload: view(Tensor, ttsl::SmallVector<int32_t>)
        ttnn::overload_t(&view_shape_vector_wrapper, nb::arg("input_tensor"), nb::arg("shape")),
        // Bind second overload: view(Tensor, ttsl::SmallVector<int32_t>)
        ttnn::overload_t(
            nb::overload_cast<const ttnn::Tensor&, const ttnn::Shape&>(&ttnn::view),
            nb::arg("input_tensor"),
            nb::arg("shape")));
}
}  // namespace ttnn::operations::data_movement
