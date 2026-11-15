// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "step_context_nanobind.hpp"

#include <nanobind/stl/vector.h>

#include "ttnn-nanobind/bind_function.hpp"
#include "ttnn/operations/experimental/deepseek/step_context/step_context.hpp"

namespace ttnn::operations::experimental::deepseek::detail {

void bind_step_context(::nanobind::module_& mod) {
    namespace nb = nanobind;
    ttnn::bind_function<"decode_step_context", "ttnn.experimental.deepseek.">(
        mod,
        R"doc(
        One decode step's integer index tensors and RoPE rows from its INT32 packet, in one program.

        Args:
            packet (ttnn.Tensor): ``[1, 1, 1, W]`` INT32 ROW_MAJOR DRAM interleaved.
            rope_tables (List[ttnn.Tensor]): ``[1, 1, H + 1025, 2 * Rd]`` INT32 ROW_MAJOR DRAM Q28 tables: rows
                ``[cos | sin]`` of ``h * 1024 * w`` (``h < H``), then ``scale * [cos | sin]`` of ``l * w``,
                ``l = -1 .. 1023``.
            int_shapes (List[List[int]]): each integer output's shape (at most 16 elements).
            int_dtypes (List[ttnn.DataType]): each integer output's dtype, INT32 or UINT32.
            int_terms (List[List[int]]): 7 words per element ``{w0, pre_add, div, mul, post_add, clamp_max,
                add_mul}``, ``w0 = (src + 1) | mode << 8 | (add_field + 1) << 16``; ``v = pkt[src] + pre_add``,
                floored ``v // div`` (mode 1) or ``v % div`` (mode 2), ``min(v * mul + post_add, clamp_max)
                + pkt[add_field] * add_mul``.
            rope_table_ids, rope_fields, rope_offsets (List[int]): per RoPE output, its table and the position
                ``pkt[field] + offset`` (``>= -1``).

        Returns:
            List[ttnn.Tensor]: the integer outputs (ROW_MAJOR, DRAM), then ``cos, sin, -sin`` ``[1, 1, 1, Rd]``
            BFLOAT16 TILE (DRAM) per RoPE output.
        )doc",
        &ttnn::experimental::deepseek::decode_step_context,
        nb::arg("packet"),
        nb::arg("rope_tables"),
        nb::arg("int_shapes"),
        nb::arg("int_dtypes"),
        nb::arg("int_terms"),
        nb::arg("rope_table_ids"),
        nb::arg("rope_fields"),
        nb::arg("rope_offsets"));
}

}  // namespace ttnn::operations::experimental::deepseek::detail
