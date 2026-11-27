// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "squeeze_nanobind.hpp"

#include <optional>
#include <variant>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/variant.h>

#include "ttnn-nanobind/bind_function.hpp"
#include "ttnn-nanobind/small_vector_caster.hpp"
#include "ttnn/operations/data_movement/squeeze/squeeze.hpp"

namespace ttnn::operations::data_movement {

namespace {
// nanobind variant/optional casters perform the Python -> C++ conversion at
// the wrapper boundary (GIL held), so the body runs with the GIL released
// (call_guard applied by bind_function).
ttnn::Tensor squeeze_wrapper(
    const ttnn::Tensor& input_tensor, std::optional<std::variant<int, ttsl::SmallVector<int>>> dim) {
    if (!dim.has_value()) {  // None
        return ttnn::squeeze(input_tensor);
    }
    return std::visit([&](const auto& d) { return ttnn::squeeze(input_tensor, d); }, *dim);
}
}  // namespace

void bind_squeeze(nb::module_& mod) {
    ttnn::bind_function<"squeeze">(
        mod,
        R"doc(
        Returns a tensor with the specified dimensions squeezed. If `dim` is not provided, all dimensions of size 1 will be squeezed. If `dim` is an integer, only the specified dimension will be squeezed. If `dim` is a list of integers, all specified dimensions will be squeezed.

        If a specified dimension in `dim` does not have size 1, it will be ignored.

        Example:

        .. code-block:: python

            input_tensor = ttnn.from_torch(torch.rand((1, 1, 1, 256), dtype=torch.bfloat16), device=device)
            output_tensor = ttnn.squeeze(input_tensor, 2)  # shape [1, 1, 256]

        Args:
            input_tensor (ttnn.Tensor): the input tensor.
            dim (int or List[int], optional): the dimension or dimensions to squeeze, each in ``[-r, r)`` for an input of rank ``r``. Negative values count from the last dimension. A list must not repeat a dimension. Defaults to `None`, which squeezes every dimension of size 1.

        Returns:
            ttnn.Tensor: the input with the size 1 dimensions removed, with the same dtype and layout. A view keeps the input's memory layout, buffer type and shard grid, with shard metadata adapted to the new shape as in :func:`ttnn.view`. When data is copied, the output keeps the input's buffer type and normally its memory layout: a sharded output gets a shard spec derived for the new shape, as in :func:`ttnn.reshape`, and falls back to interleaved when the input's shard grid is not rectangular. If no dimension is removed, the input tensor itself is returned.

        Note:
            The squeeze is done with :func:`ttnn.reshape`. For a host tensor it is always a metadata-only view that
            shares the input's data. For a device tensor it is a zero-cost view that shares the input's buffer
            when the size of the last dimension does not change and, for TILE layout, the size of the second-to-last
            dimension does not change or is a multiple of 32 both before and after the squeeze. Otherwise the data
            is copied into a new tensor: for example, squeezing ``[32, 4, 1]`` at ``-1`` copies, while squeezing
            ``[32, 1, 1]`` at ``-1`` is a view in ROW_MAJOR layout but copies in TILE layout, where the
            second-to-last dimension goes from 1 to 32. An ND sharded input is copied when the squeeze changes its padded
            shape, as removing a dimension normally does. Squeezing a non-empty TILE ND sharded input from ``[1, N]``
            to ``[N]`` also copies, although the padded shape is unchanged.

            With a tile other than 32x32, a view also has the padding and buffer-size limits of :func:`ttnn.view`: a
            view whose padded output does not fit in the input's buffer raises an error instead of copying, as for
            ``[1, 16, 32]`` with 16x32 tiles squeezed at ``0``.

            Supported dtypes and layouts:

            .. list-table::
               :header-rows: 1

               * - Dtypes
                 - Layouts
               * - BFLOAT16, FLOAT32, INT32, UINT32, UINT16, UINT8
                 - TILE, ROW_MAJOR
               * - BFLOAT8_B, BFLOAT4_B
                 - TILE

            A squeeze that copies data has the dtype restrictions of :func:`ttnn.reshape`. In addition, a ``UINT16``
            TILE squeeze that copies needs every dimension it keeps from the input's last two to be a multiple of 32:
            squeezing ``[64, 1, 64]`` at ``1`` works, but squeezing ``[32, 4, 1]`` at ``-1`` or ``[64, 1, 30]`` at
            ``1`` raises an error.

            For a rank 0 input, ``dim`` must be `None`, 0 or -1, and the input is returned unchanged.

        Memory Support:
            - Interleaved: DRAM and L1
            - Sharded: HEIGHT_SHARDED, WIDTH_SHARDED and BLOCK_SHARDED, in DRAM or L1. A DRAM-sharded ROW_MAJOR, BFLOAT8_B or BFLOAT4_B input is supported only when the squeeze is a view. An L1-sharded ROW_MAJOR copy needs input and output shard widths, in bytes, that are multiples of the L1 alignment, unless the output falls back to interleaved. This applies to the physical shard width, which can be padded beyond a logical last dimension of size 1: an input with one-element-wide shards fails, and a padded input works when the output falls back to interleaved, as for a non-rectangular shard grid
            - ND sharded: supported as in :func:`ttnn.reshape`, through a DRAM-interleaved intermediate when data is copied
        )doc",
        &squeeze_wrapper,
        nb::arg("input_tensor"),
        nb::arg("dim") = nb::none()  // Default value is None
    );
}

}  // namespace ttnn::operations::data_movement
