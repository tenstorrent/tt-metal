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

            When expert_region_offsets and total_counts_per_expert (from routing_setup)
            are supplied, the kernel tilizes only the filled prefix of the worst-case-padded
            dispatch buffer (the same rows the expert matmul reads) and leaves the padded
            tail untouched, so device time scales with real tokens instead of capacity.
            Omitting them tilizes the whole buffer (byte-identical to to_layout).

            Args:
                input_tensor (ttnn.Tensor): dispatched buffer, ROW_MAJOR.
                expert_region_offsets (ttnn.Tensor, optional): uint32 [1, num_routed_experts],
                    exclusive tile-aligned prefix sum of per-expert counts (from routing_setup).
                total_counts_per_expert (ttnn.Tensor, optional): uint32 [1, num_routed_experts],
                    per-expert token counts (from routing_setup).
                output_dtype (ttnn.DataType, optional): output dtype. Defaults to input dtype.
                experts_per_chip (int): experts hosted per device (drives the per-chip valid_rows).
                    0 => plain full tilize (no skip).
                output_memory_config (ttnn.MemoryConfig, optional): defaults to input's.

            Returns:
                ttnn.Tensor: TILE-layout output (padded tail undefined when skipping).
        )doc",
        &ttnn::operations::experimental::deepseek_prefill::dispatch_tilize::dispatch_tilize,
        nb::arg("input_tensor").noconvert(),
        nb::arg("expert_region_offsets") = nb::none(),
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
