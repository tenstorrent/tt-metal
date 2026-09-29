// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "select_final_carry_nanobind.hpp"

#include "select_final_carry.hpp"
#include "ttnn-nanobind/bind_function.hpp"

#include <nanobind/stl/optional.h>

namespace ttnn::operations::experimental::kda::select_final_carry::detail {

void bind_select_final_carry(nb::module_& mod) {
    ttnn::bind_function<"select_final_carry", "ttnn.experimental.kda.">(
        mod,
        R"doc(
        Select the recurrent state after the chronologically last valid token.

        When the valid interval ends in a separated tail, the answer is the final state of the rank
        that owns that tail, taken from ``rank_finals``; otherwise it is the completed distributed
        prefix ``prefix_final``. The chronology is derived on device from ``actual_start`` and the
        optional ``actual_end``, exactly as ``chronological_selections`` derives it.

        Args:
            rank_finals (ttnn.Tensor): FLOAT32 TILE-layout ``[P * B*H, K, V]`` final states gathered
                along the sequence-parallel axis in physical rank order.
            prefix_final (ttnn.Tensor): FLOAT32 TILE-layout ``[B*H, K, V]`` completed prefix carry.

        Keyword Args:
            actual_start (ttnn.Tensor): Replicated UINT32 row-major scalar with the chunk's first
                absolute position.
            local_rows (int): Positive, 32-aligned token rows per SP device.
            actual_end (ttnn.Tensor, optional): Replicated UINT32 row-major exclusive valid end.
            memory_config (ttnn.MemoryConfig, optional): Interleaved output memory. Defaults to DRAM.
            sequence_parallel_axis (int, optional): Mesh axis partitioning the sequence.

        Returns:
            ttnn.Tensor: FLOAT32 TILE-layout ``[B*H, K, V]`` final state.
        )doc",
        &ttnn::experimental::kda::select_final_carry,
        nb::arg("rank_finals").noconvert(),
        nb::arg("prefix_final").noconvert(),
        nb::kw_only(),
        nb::arg("actual_start").noconvert(),
        nb::arg("local_rows"),
        nb::arg("actual_end") = nb::none(),
        nb::arg("memory_config") = nb::none(),
        nb::arg("sequence_parallel_axis") = 0);
}

}  // namespace ttnn::operations::experimental::kda::select_final_carry::detail
