// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "select_final_carry_nanobind.hpp"

#include "select_final_carry.hpp"
#include "ttnn-nanobind/bind_function.hpp"

#include <nanobind/stl/optional.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>

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
        optional ``actual_end``, exactly as the other KDA operations derive it, so the choice
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
            cores (list[ttnn.CoreCoord], optional): The worker cores: the first ``num_links`` send over
                the fabric and all of them copy. Defaults to enough cores from the grid's start.
            prefix_on_last_rank (bool, optional): ``prefix_final`` is valid only on the last
                chronological rank (as the SP relay leaves it): without a split tail that rank
                multicasts it instead of every rank copying its own.
            semaphores (list[ttnn.GlobalSemaphore], optional): The zeroed barrier and arrival global
                semaphores, kept alive across calls. Without them each cached program allocates its
                own when it is built.

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
        nb::arg("num_links") = nb::none(),
        nb::arg("cores") = nb::none(),
        nb::arg("prefix_on_last_rank") = false,
        nb::arg("semaphores") = nb::none());

    ttnn::bind_function<"select_final_carry_with_projection", "ttnn.experimental.kda.">(
        mod,
        R"doc(
        ``select_final_carry`` with a projection ``projection_input @ projection_weight`` in the same
        dispatch, on worker cores the final carry's ``cores`` leave (``ttnn.linear``'s 2D multicast matmul with
        ``projection_program_config``, on its cores). Nothing reads the carry until the next call, and its program mostly waits on the
        sequence-parallel fabric, so that wait hides behind the projection's compute.

        Args:
            rank_final, prefix_final (ttnn.Tensor): As ``select_final_carry``'s.
            projection_input (ttnn.Tensor): Interleaved TILE ``[..., M, K]``.
            projection_weight (ttnn.Tensor): Interleaved TILE ``[K, N]``.

        Keyword Args:
            actual_start, local_rows, actual_end, memory_config, sequence_parallel_axis, num_links,
                prefix_on_last_rank: As ``select_final_carry``'s.
            cores (list[ttnn.CoreCoord]): The final carry's worker cores.
            projection_program_config (ttnn.MatmulMultiCoreReuseMultiCastProgramConfig): The
                projection's schedule and cores (``allowed_worker_cores``, or the grid from (0, 0)).
            projection_memory_config (ttnn.MemoryConfig, optional): Interleaved memory of the
                projection's BFLOAT16 output. Defaults to DRAM.
            projection_compute_kernel_config (ttnn.DeviceComputeKernelConfig, optional): The
                projection's fidelity; defaults to HiFi2.

        Returns:
            tuple[ttnn.Tensor, ttnn.Tensor]: The final carry and the projection's output.
        )doc",
        &ttnn::experimental::kda::select_final_carry_with_projection,
        nb::arg("rank_final").noconvert(),
        nb::arg("prefix_final").noconvert(),
        nb::arg("projection_input").noconvert(),
        nb::arg("projection_weight").noconvert(),
        nb::kw_only(),
        nb::arg("actual_start").noconvert(),
        nb::arg("local_rows"),
        nb::arg("cores"),
        nb::arg("projection_program_config"),
        nb::arg("actual_end") = nb::none(),
        nb::arg("memory_config") = nb::none(),
        nb::arg("sequence_parallel_axis") = 0,
        nb::arg("num_links") = nb::none(),
        nb::arg("prefix_on_last_rank") = false,
        nb::arg("projection_memory_config") = nb::none(),
        nb::arg("projection_compute_kernel_config") = nb::none(),
        nb::arg("semaphores") = nb::none());
}

}  // namespace ttnn::operations::experimental::kda::select_final_carry::detail
