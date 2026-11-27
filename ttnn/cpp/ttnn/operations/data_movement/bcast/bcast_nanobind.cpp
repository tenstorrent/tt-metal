// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "bcast_nanobind.hpp"

#include <optional>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>

#include "ttnn-nanobind/bind_function.hpp"
#include "ttnn/operations/data_movement/bcast/bcast.hpp"

namespace ttnn::operations::data_movement::detail {
namespace nb = nanobind;

void bind_bcast(nb::module_& mod) {
    const char* doc =
        R"doc(
            Applies `math_op` elementwise to `input_a` and `input_b`, broadcasting `input_b` over the last two
            dimensions of `input_a` as `dim` sets. For an `input_a` of shape ``[..., H, W]``:

            - ``ttnn.BcastOpDim.H``: `input_b` has shape ``[..., 1, W]``. Its row is applied to every row of `input_a`.
            - ``ttnn.BcastOpDim.W``: `input_b` has shape ``[..., H, 1]``. Its column is applied to every column of `input_a`.
            - ``ttnn.BcastOpDim.HW``: `input_b` has shape ``[..., 1, 1]``. Its value is applied to every element of `input_a`.

            The dimensions of `input_b` before the last two are either all 1, which applies the same `input_b` to every
            batch of `input_a`, or equal to those of `input_a`, which applies each batch of `input_b` to the same batch
            of `input_a`.

            Args:
                input_a (ttnn.Tensor): the first operand, of rank 2, 3 or 4.
                input_b (ttnn.Tensor): the operand to broadcast, with the shape that `dim` sets and the same dtype and layout as `input_a`.
                math_op (ttnn.BcastOpMath): the operation. ``ttnn.BcastOpMath.ADD`` computes `input_a` + `input_b`, ``ttnn.BcastOpMath.SUB`` computes `input_a` - `input_b`, and ``ttnn.BcastOpMath.MUL`` computes `input_a` * `input_b`.
                dim (ttnn.BcastOpDim): how `input_b` is broadcast: ``ttnn.BcastOpDim.H``, ``ttnn.BcastOpDim.W`` or ``ttnn.BcastOpDim.HW``, as described above.

            Keyword Args:
                memory_config (ttnn.MemoryConfig, optional): Memory configuration of the output. Defaults to `None`, which uses the memory configuration of `input_a`. For an interleaved `input_a` it must be interleaved, and for a sharded `input_a` it must be `None` or the memory configuration of `input_a`. When `output_tensor` is given, it must be `None` or the memory configuration of `output_tensor`.
                output_tensor (ttnn.Tensor, optional): Preallocated output tensor for an interleaved `input_a`. It must be interleaved and in TILE layout, with the shape and dtype of `input_a`. Defaults to `None`.

            Returns:
                ttnn.Tensor: the output tensor, in TILE layout with the shape and dtype of `input_a`. When `output_tensor` is given, it is returned with the output written to it.

            Note:
                Supported dtypes and layouts:

                .. list-table::
                   :header-rows: 1

                   * - Dtypes
                     - Layouts
                   * - BFLOAT16
                     - TILE, ROW_MAJOR
                   * - BFLOAT8_B
                     - TILE

                `input_a` and `input_b` must have the same dtype and layout, and their values must be finite.

            Memory Support:
                - Interleaved: DRAM and L1
                - Sharded, with ``ttnn.BcastOpDim.H``: a TILE `input_a` that is WIDTH_SHARDED in L1 on one row of cores starting at core (0, 0), or BLOCK_SHARDED in L1 on a rectangular core grid starting at core (0, 0)
                - Sharded, with ``ttnn.BcastOpDim.HW``: a TILE `input_a` that is HEIGHT_SHARDED in L1
                - A sharded `input_a` needs an interleaved `input_b` whose dimensions before the last two are all 1, and gives an output with the same sharding
        )doc";

    ttnn::bind_function<"bcast">(
        mod,
        doc,
        &ttnn::bcast,
        nb::arg("input_a").noconvert(),
        nb::arg("input_b").noconvert(),
        nb::arg("math_op"),
        nb::arg("dim"),
        nb::kw_only(),
        nb::arg("memory_config") = nb::none(),
        nb::arg("output_tensor") = nb::none());
}

}  // namespace ttnn::operations::data_movement::detail
