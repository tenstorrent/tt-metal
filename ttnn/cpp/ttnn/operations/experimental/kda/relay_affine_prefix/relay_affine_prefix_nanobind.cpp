// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "relay_affine_prefix_nanobind.hpp"

#include "relay_affine_prefix.hpp"
#include "ttnn-nanobind/bind_function.hpp"

#include <nanobind/stl/optional.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>

namespace ttnn::operations::experimental::kda::relay_affine_prefix::detail {

void bind_relay_affine_prefix(nb::module_& mod) {
    ttnn::bind_function<"relay_affine_prefix", "ttnn.experimental.kda.">(
        mod,
        R"doc(
        The sequence-parallel prefix as a relay around the SP ring: rank ``p`` (chronological position, from
        ``actual_start``) takes its entry state ``E_p`` from its predecessor, computes
        ``E_{p+1} = A_p E_p + B_p`` from its packed ``[A | B]`` summary and forwards it, one head per core and
        one tile column at a time; the last rank sends the final state on to the first. ``cores`` lists a core
        per head, then a sender core per link. The landing buffers and the two global ``semaphores`` outlive
        the call: a peer writes them before this rank's call starts. Consecutive calls must be separated by a
        barrier across the SP line, since a rank's landing buffers are copied out at the end of its call and
        its predecessor's next call overwrites them as soon as it starts.

        With ``output_gate_program_config``, ``output_gate_input @ output_gate_weight`` runs in the same
        dispatch as ``ttnn.linear`` runs it with that config, on its cores (``allowed_worker_cores``, or the
        grid from (0, 0)); its first ``output_gate_copied_columns`` tile columns are copied from
        ``output_gate_copy_source`` by ``output_gate_copy_cores``.

        Args:
            packed (ttnn.Tensor): BFLOAT16 TILE-layout interleaved ``[1, B*H, K, 2K]`` rows of this rank's
                packed transition ``[A | B]``.
            initial_state (ttnn.Tensor): FLOAT32 TILE-layout interleaved ``[B*H, K, K]`` state the
                chronologically first rank enters at.
            actual_start (ttnn.Tensor): Replicated UINT32 row-major scalar with the absolute position of the
                chunk's first token. Keep its address stable and update its contents before replay of a
                captured trace.
            entry_landing (ttnn.Tensor): FLOAT32 TILE-layout ``[B*H*K, K]`` height-sharded in L1, one ``K x K``
                state per head core: the predecessor's sender writes this rank's entry states here.
            state_staging (ttnn.Tensor): Sharded like ``entry_landing``: this rank's next states, read out by
                its senders.
            final_landing (ttnn.Tensor): Sharded like ``entry_landing``: the last rank's final states land here
                on the first rank.
            semaphores (list[ttnn.GlobalSemaphore]): The zeroed entry-state and final-state arrival counters.

        Keyword Args:
            local_rows (int): Positive, 32-aligned token rows per SP device.
            num_links (int): Fabric links to the successor, one sender core each.
            cores (list[ttnn.CoreCoord]): One core per head, then one sender per link, all distinct.
            sequence_parallel_axis (int, optional): Mesh axis partitioning the sequence; the fabric must wrap
                it into a ring of more than two ranks.
            entry_memory_config (ttnn.MemoryConfig, optional): Interleaved memory of the entry states.
                Defaults to DRAM.
            compute_kernel_config (ttnn.DeviceComputeKernelConfig, optional): The hop's matmul fidelity;
                its products accumulate in DST. Defaults to HiFi2 with FP32 destination accumulation.
            output_gate_input (ttnn.Tensor, optional): Interleaved TILE ``[..., M, K']`` BFLOAT16 or BFLOAT8_B.
            output_gate_weight (ttnn.Tensor, optional): Interleaved TILE ``[K', N]`` BFLOAT16 or BFLOAT8_B.
            output_gate_program_config (ttnn.MatmulMultiCoreReuseMultiCastProgramConfig, optional): The
                projection's schedule and cores; without it no output gate runs.
            output_gate_compute_kernel_config (ttnn.DeviceComputeKernelConfig, optional): The projection's
                fidelity; defaults to HiFi2.
            output_gate_memory_config (ttnn.MemoryConfig, optional): Interleaved memory of the projection's
                output. Defaults to DRAM.
            output_gate_copy_source (ttnn.Tensor, optional): TILE tensor of the projection's dtype with ``M``
                rows whose tile columns from ``output_gate_copy_source_column`` fill the output's first
                ``output_gate_copied_columns`` tile columns.
            output_gate_copy_source_column (int, optional): The first copied tile column of the source.
            output_gate_copied_columns (int, optional): Copied tile columns; 0 copies nothing.
            output_gate_copy_cores (list[ttnn.CoreCoord], optional): The cores that copy, splitting the rows.

        Returns:
            tuple[ttnn.Tensor, ttnn.Tensor, ttnn.Tensor | None]: FLOAT32 TILE-layout ``[B*H, K, K]`` entry
                states, the final state in DRAM (valid on the first and last chronological ranks) and the
                output gate's BFLOAT16 ``[..., M, 32 * output_gate_copied_columns + N]`` output, or None
                without an output gate.

        Note:
            ``K`` must be positive, tile aligned and at most the DST's tiles wide; the fabric payload must hold
            an FP32 tile. Inputs are not modified.
        )doc",
        &ttnn::experimental::kda::relay_affine_prefix,
        nb::arg("packed").noconvert(),
        nb::arg("initial_state").noconvert(),
        nb::arg("actual_start").noconvert(),
        nb::arg("entry_landing").noconvert(),
        nb::arg("state_staging").noconvert(),
        nb::arg("final_landing").noconvert(),
        nb::arg("semaphores"),
        nb::kw_only(),
        nb::arg("local_rows"),
        nb::arg("num_links"),
        nb::arg("cores"),
        nb::arg("sequence_parallel_axis") = 0,
        nb::arg("entry_memory_config") = nb::none(),
        nb::arg("compute_kernel_config") = nb::none(),
        nb::arg("output_gate_input") = nb::none(),
        nb::arg("output_gate_weight") = nb::none(),
        nb::arg("output_gate_program_config") = nb::none(),
        nb::arg("output_gate_compute_kernel_config") = nb::none(),
        nb::arg("output_gate_memory_config") = nb::none(),
        nb::arg("output_gate_copy_source") = nb::none(),
        nb::arg("output_gate_copy_source_column") = 0,
        nb::arg("output_gate_copied_columns") = 0,
        nb::arg("output_gate_copy_cores") = std::vector<tt::tt_metal::CoreCoord>{});
}

}  // namespace ttnn::operations::experimental::kda::relay_affine_prefix::detail
