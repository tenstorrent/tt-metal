// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dispatch_tilize_nanobind.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>

#include "ttnn-nanobind/bind_function.hpp"
#include "dispatch_tilize.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::dispatch_tilize::detail {

void bind_dispatch_tilize(nb::module_& mod) {
    ttnn::bind_function<"dispatch_tilize", "ttnn.experimental.deepseek_prefill.">(
        mod,
        R"doc(
            Region-aware tilize for the MoE dispatch buffer.

            Converts the ROW_MAJOR dispatched-token buffer to TILE layout (with an
            optional dtype change, e.g. bf16/fp8_e4m3 -> bfloat8_b) for the routed-expert
            matmul, replacing ttnn.to_layout(dispatched_buffer, TILE, dtype=...).

            When total_counts_per_expert (from routing_setup) is supplied, the kernel tilizes
            only the filled prefix of the worst-case-padded dispatch buffer (the same rows the
            expert matmul reads) and leaves the padded tail untouched, so device time scales
            with real tokens instead of capacity. The filled prefix is the fullest chip's fill,
            valid_rows = max_chip sum_{e in chip} align32(count[e]), grouping the [1,E] counts
            into consecutive experts_per_chip chips. Omitting the counts tilizes the whole buffer
            (byte-identical to to_layout).

            Args:
                input_tensor (ttnn.Tensor): dispatched buffer, ROW_MAJOR.
                total_counts_per_expert (ttnn.Tensor, optional): uint32 [1, num_routed_experts],
                    per-expert token counts (from routing_setup). Supply to enable the skip.
                output_dtype (ttnn.DataType, optional): output dtype. Defaults to input dtype.
                experts_per_chip (int): experts hosted per device; groups the counts into per-chip
                    fills. Required (>0) when total_counts_per_expert is given; 0 => full tilize.
                output_memory_config (ttnn.MemoryConfig, optional): defaults to input's.

            Returns:
                ttnn.Tensor: TILE-layout output (padded tail undefined when skipping).
        )doc",
        &ttnn::operations::experimental::deepseek_prefill::dispatch_tilize::dispatch_tilize,
        nb::arg("input_tensor").noconvert(),
        nb::arg("total_counts_per_expert") = nb::none(),
        nb::kw_only(),
        nb::arg("output_dtype") = nb::none(),
        nb::arg("experts_per_chip") = 0,
        nb::arg("output_memory_config") = nb::none());
}

}  // namespace ttnn::operations::experimental::deepseek_prefill::dispatch_tilize::detail

namespace ttnn::operations::experimental::deepseek_prefill::detail {

void bind_dispatch_tilize(::nanobind::module_& mod) { dispatch_tilize::detail::bind_dispatch_tilize(mod); }

}  // namespace ttnn::operations::experimental::deepseek_prefill::detail
