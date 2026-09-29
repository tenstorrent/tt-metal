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
        Distribute the recurrent state after the chronologically last valid token.

        When the valid interval ends in a separated tail, the answer is the final state of the rank
        that owns that tail: that rank multicasts its ``rank_final`` along the sequence-parallel line
        over the fabric. Otherwise every rank copies its replicated ``prefix_final``, with no
        collective traffic. The chronology is derived on device from ``actual_start`` and the
        optional ``actual_end``, exactly as ``chronological_selections`` derives it, so the choice
        follows the bounds on every trace replay.

        Args:
            rank_final (ttnn.Tensor): FLOAT32 TILE-layout ``[B*H, K, V]`` final state of this rank's
                local scan, or ``[B*H, groups, K, V]`` group states whose last group is read in place.
            prefix_final (ttnn.Tensor): FLOAT32 TILE-layout ``[B*H, K, V]`` completed prefix carry.

        Keyword Args:
            actual_start (ttnn.Tensor): Replicated UINT32 row-major scalar with the chunk's first
                absolute position.
            local_rows (int): Positive, 32-aligned token rows per SP device.
            actual_end (ttnn.Tensor, optional): Replicated UINT32 row-major exclusive valid end.
            memory_config (ttnn.MemoryConfig, optional): Interleaved output memory. Defaults to DRAM.
            sequence_parallel_axis (int, optional): Mesh axis partitioning the sequence.
            num_links (int, optional): Fabric links used by the owner's multicast. Defaults to the
                links available on the sequence-parallel axis.

        Returns:
            ttnn.Tensor: FLOAT32 TILE-layout ``[B*H, K, V]`` final state, replicated along the line.
        )doc",
        &ttnn::experimental::kda::select_final_carry,
        nb::arg("rank_final").noconvert(),
        nb::arg("prefix_final").noconvert(),
        nb::kw_only(),
        nb::arg("actual_start").noconvert(),
        nb::arg("local_rows"),
        nb::arg("actual_end") = nb::none(),
        nb::arg("memory_config") = nb::none(),
        nb::arg("sequence_parallel_axis") = 0,
        nb::arg("num_links") = nb::none());
}

}  // namespace ttnn::operations::experimental::kda::select_final_carry::detail
