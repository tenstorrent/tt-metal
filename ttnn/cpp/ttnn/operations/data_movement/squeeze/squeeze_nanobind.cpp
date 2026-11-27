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

        Equivalent pytorch code:

        .. code-block:: python

            input_tensor = torch.rand((1,1,1,256), dtype=torch.bfloat16)
            output_tensor = torch.squeeze(input_tensor, 2) # tensor of shape (1,1,256), where at dimension 2 we removed it

        Args:
            input_tensor (ttnn.Tensor): the input tensor.
            dim (int or List[int], optional): the dimension or dimensions to squeeze, each in ``[-r, r)`` for an input of rank ``r``. Negative values count from the last dimension. A list must not repeat a dimension. Defaults to `None`, which squeezes every dimension of size 1.

        Returns:
            ttnn.Tensor: the input with the size 1 dimensions removed, with the same dtype, layout and memory configuration. If no dimension is removed, the input tensor itself is returned.

        Note:
            The squeeze is done with :func:`ttnn.reshape`. It is a zero-cost view that shares the input's buffer
            when the last dimension is kept and, for TILE layout, the second-to-last dimension does not change or is a
            multiple of 32 both before and after the squeeze. Otherwise, as when squeezing the last dimension, the
            data is copied into a new tensor. An ND sharded input is always copied when a dimension is removed, since
            that changes its padded shape.

            Supported dtypes and layouts:

            .. list-table::
               :header-rows: 1

               * - Dtypes
                 - Layouts
               * - BFLOAT16, FLOAT32, INT32, UINT32, UINT16, UINT8
                 - TILE, ROW_MAJOR
               * - BFLOAT8_B, BFLOAT4_B
                 - TILE

            A squeeze that copies data has the dtype restrictions of :func:`ttnn.reshape`.

            For a rank 0 input, ``dim`` must be `None`, 0 or -1, and the input is returned unchanged.

        Memory Support:
            - Interleaved: DRAM and L1
            - Sharded: HEIGHT_SHARDED, WIDTH_SHARDED and BLOCK_SHARDED
            - ND sharded: supported as in :func:`ttnn.reshape`, through a DRAM-interleaved intermediate when data is copied
        )doc",
        &squeeze_wrapper,
        nb::arg("input_tensor"),
        nb::arg("dim") = nb::none()  // Default value is None
    );
}

}  // namespace ttnn::operations::data_movement
