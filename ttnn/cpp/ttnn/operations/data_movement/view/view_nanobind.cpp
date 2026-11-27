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
        Returns a zero-cost view of the input tensor with a new shape. No data is copied or moved: the returned tensor
        shares the input's buffer, so writes through either tensor are visible in the other. Use :func:`ttnn.reshape`
        for shape changes that need data movement.

        Args:
            input_tensor (ttnn.Tensor): the input tensor, on device.
            shape (ttnn.Shape or List[int]): the new shape, with the same number of elements as the input. As a list, one dimension may be ``-1``; its size is inferred from the other dimensions.

        Returns:
            ttnn.Tensor: a view of the input with the new shape, and the input's dtype, layout and memory configuration. If the shape does not change, the input tensor itself is returned.

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

            The last dimension must not change.

            For TILE layout, the second-to-last dimension must not change, or must be a multiple of the tile height
            (32 for the standard tile) both before and after the view. A rank 1 shape counts as a second-to-last
            dimension of 1.

        Memory Support:
            - Interleaved: DRAM and L1
            - Sharded: HEIGHT_SHARDED, WIDTH_SHARDED and BLOCK_SHARDED, in DRAM or L1
            - ND sharded: only a rank 0 or 1 input viewed as a rank 2 shape ``[1, N]``, or a view that leaves the padded shape unchanged

        Example:

            >>> tensor = ttnn.from_torch(torch.rand(2, 1, 1, 1, 4, dtype=torch.bfloat16), device=device)
            >>> output = ttnn.view(tensor, (2, 1, 4))

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
