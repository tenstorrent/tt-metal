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
        Returns `input_tensor` with dimensions of size 1 removed, like ``torch.squeeze``: every dimension of size 1 when `dim` is `None`, or only the dimensions of size 1 that `dim` lists.

        When every removed dimension comes before the last two, the output is a view of `input_tensor`: it shares the input's buffer and no data moves.

        Args:
            input_tensor (ttnn.Tensor): the input tensor.
            dim (int or List[int], optional): the dimension or dimensions to remove, each in ``[-rank, rank - 1]`` and listed once, where ``rank`` is the rank of `input_tensor`; for a rank-0 input, ``0`` or ``-1``, which return the input unchanged. Negative values count from the last dimension. A listed dimension whose size is not 1 is kept. Defaults to `None`, which removes every dimension of size 1.

        Returns:
            ttnn.Tensor: the output tensor, with the input's dtype, layout and memory configuration.

        Note:
            Supported dtypes and layouts:

            .. list-table::
               :header-rows: 1

               * - Dtypes
                 - Layouts
               * - BFLOAT16, FLOAT32, INT32, UINT32
                 - TILE, ROW_MAJOR
               * - BFLOAT8_B, BFLOAT4_B
                 - TILE

            Also supported when every removed dimension comes before the last two:

            .. list-table::
               :header-rows: 1

               * - Dtypes
                 - Layouts
               * - UINT16, UINT8
                 - TILE, ROW_MAJOR

            Removing the last dimension of a ``BFLOAT8_B`` or ``BFLOAT4_B`` tensor regroups its values into new blocks of 16 that share one exponent.

        Memory Support:
            - Interleaved: DRAM and L1
            - Sharded: HEIGHT_SHARDED, WIDTH_SHARDED and BLOCK_SHARDED in L1, when every removed dimension comes before the last two
        )doc",
        &squeeze_wrapper,
        nb::arg("input_tensor"),
        nb::arg("dim") = nb::none()  // Default value is None
    );
}

}  // namespace ttnn::operations::data_movement
