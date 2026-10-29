// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "entry_correction_nanobind.hpp"

#include "entry_correction.hpp"
#include "ttnn-nanobind/bind_function.hpp"

#include <nanobind/stl/optional.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>

namespace ttnn::operations::experimental::kda::entry_correction::detail {

void bind_entry_correction(nb::module_& mod) {
    ttnn::bind_function<"entry_correction", "ttnn.experimental.kda.">(
        mod,
        R"doc(
        Every chunk's output and this rank's final state from the summary's emitted terms
        (``summarize_chunk_recurrence(emit_outputs=True)``) once the entry states are known. The recurrence is
        affine in its seed, so ``output_c = F0_c + (FI_c - F0_c) @ seed`` and ``final = A @ seed + B``, all
        chunks in parallel. The seed is ``entry`` for the chunks before the chronological reset chunk and
        ``tail_entry`` from it on; the final state takes ``tail_entry`` when the rank's valid rows reach its
        split tail. Each head's chunks are split over ``len(cores) / (B*H)`` cores, and
        ``zero_output``/``identity_output`` are height-sharded over ``cores`` so each core's chunk rows are in
        its L1. Chunks past ``actual_end`` are left unspecified.

        Args:
            zero_output (ttnn.Tensor): FLOAT32 TILE-layout ``[B*H, chunks, 32, K]`` chunk outputs for a zero
                seed, height-sharded in L1 over ``cores`` in their order, one core's chunk rows per shard.
            identity_output (ttnn.Tensor): The chunk outputs for an identity seed, sharded the same way.
            final_a (ttnn.Tensor): FLOAT32 TILE-layout interleaved ``[B*H, K, K]`` multiplier of the rank's
                final transition.
            final_b (ttnn.Tensor): FLOAT32 TILE-layout interleaved ``[B*H, K, K]`` offset of the rank's final
                transition.
            entry (ttnn.Tensor): FLOAT32 TILE-layout interleaved ``[B*H, K, K]`` entry states.
            tail_entry (ttnn.Tensor): FLOAT32 TILE-layout interleaved ``[B*H, K, K]`` entry states of the
                split tail.

        Keyword Args:
            actual_start (ttnn.Tensor): Replicated UINT32 row-major scalar with the absolute position of the
                chunk's first token. Keep its address stable and update its contents before replay of a
                captured trace.
            local_rows (int): Positive, 32-aligned token rows per SP device; ``chunks = local_rows / 32``.
            cores (list[ttnn.CoreCoord]): The same number of cores per head, head by head: core ``j`` of a
                head takes its ``j``-th share of the chunks and forms every ``cores_per_head``-th tile row of
                the final state from row ``j``. ``cores_per_head`` must divide ``chunks``.
            actual_end (ttnn.Tensor, optional): Replicated UINT32 row-major exclusive valid end.
            sequence_parallel_axis (int, optional): Mesh axis partitioning the sequence.
            memory_config (ttnn.MemoryConfig, optional): Interleaved memory of the chunk outputs. Defaults to
                DRAM.
            compute_kernel_config (ttnn.DeviceComputeKernelConfig, optional): Compute-kernel configuration;
                the products accumulate in DST. Defaults to HiFi2 with FP32 destination accumulation.

        Returns:
            tuple[ttnn.Tensor, ttnn.Tensor]: BFLOAT16 TILE-layout ``[B*H, chunks, 32, K]`` chunk outputs and
                the FLOAT32 TILE-layout ``[B*H, K, K]`` final state in DRAM.

        Note:
            Every state is ``K x K`` (``K == V``), with ``K`` positive, tile-aligned and at most the DST's
            tiles wide. Inputs are not modified.
        )doc",
        &ttnn::experimental::kda::entry_correction,
        nb::arg("zero_output").noconvert(),
        nb::arg("identity_output").noconvert(),
        nb::arg("final_a").noconvert(),
        nb::arg("final_b").noconvert(),
        nb::arg("entry").noconvert(),
        nb::arg("tail_entry").noconvert(),
        nb::kw_only(),
        nb::arg("actual_start").noconvert(),
        nb::arg("local_rows"),
        nb::arg("cores"),
        nb::arg("actual_end") = nb::none(),
        nb::arg("sequence_parallel_axis") = 0,
        nb::arg("memory_config") = nb::none(),
        nb::arg("compute_kernel_config") = nb::none());
}

}  // namespace ttnn::operations::experimental::kda::entry_correction::detail
