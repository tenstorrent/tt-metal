// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "affine_exclusive_scan_nanobind.hpp"

#include "affine_exclusive_scan.hpp"
#include "ttnn-nanobind/bind_function.hpp"

#include <nanobind/stl/optional.h>

namespace ttnn::operations::experimental::kda::affine_exclusive_scan::detail {

void bind_affine_exclusive_scan(nb::module_& mod) {
    ttnn::bind_function<"affine_exclusive_scan", "ttnn.experimental.kda.">(
        mod,
        R"doc(
        Compute the recurrent-state entry for every ordered group of chunks.

        Each transform represents one group-level affine transition:

            F_g(S) = A_g @ S + B_g

        Transforms are read from packed summaries, as produced by
        ``summarize_chunk_recurrence``, whose transform index is flattened
        batch-head-group order:

            index = batch_head * groups_per_head + group

        The scan is exclusive: the first group receives ``initial_state``. On an
        unsplit rank, each later group receives the state produced by all preceding
        groups:

            entry[0] = initial_state
            entry[g] = A_{g-1} @ entry[g-1] + B_{g-1}

        On the rank containing both the chronological beginning and end of the
        sequence, the tail restarts from ``tail_entry_states`` after the intervening
        ranks. For a group-aligned split, the first tail group receives this seed
        directly. For a split inside a group, that group's entry still belongs to
        the head; its tail summary is applied to the tail seed to produce the next
        group's entry. The chunk scan handles the reset inside the split group.
        Subsequent groups use the tail transforms (slot 1) instead of the head
        transforms (slot 0).

        Optional ``actual_end`` is a replicated UINT32 row-major scalar, with
        the same lifetime as ``actual_start``. It defines a nonempty interval within
        physical capacity and need not be 32-aligned: a partial last chunk is
        valid, and its rows past the end must be identity steps, as produced by
        ``prepare_chunk_recurrence``. Omission uses the full capacity.
        Bounds may change during trace replay. Padded group outputs are unspecified.
        Bounds are caller preconditions and are not read back on the host.

        Args:
            summaries (ttnn.Tensor): Packed group transforms ``[S, B*H*G, K, K+V]``:
                ``K`` rows of ``[A | B]`` per transform, head transforms in slot 0
                and, with ``S = 2``, tail transforms in slot 1. ``S = 2`` is
                required when the sequence-parallel axis has more than one device.
                Must be a TILE-layout FLOAT32 or BFLOAT16 device tensor.
            initial_state (ttnn.Tensor): Entry state ``[B*H, K, V]`` for the first
                group of each batch-head. Must be a TILE-layout FLOAT32 device tensor.
            groups_per_head (int): Number of consecutive groups ``G`` belonging to
                each batch-head. Must be positive and divide ``B*H*G``.

        Keyword Args:
            actual_start (ttnn.Tensor): Replicated UINT32 row-major scalar
                containing the absolute position of the chunk's first token; pass [0]
                for zero-offset execution. Its
                value must be nonnegative and 32-aligned. Keep its address stable
                and update its contents before replay of a captured trace.
            sequence_parallel_axis (int, optional): Mesh axis partitioning the
                sequence. Native mesh coordinates supply each device's rank.
            local_rows (int): Positive, 32-aligned token rows per SP device. Summary tensor
                shapes do not encode this sequence length.
            tail_entry_states (ttnn.Tensor, optional): FLOAT32 tail seed ``[B*H,K,V]``,
                required exactly when ``summaries`` hold a tail slot. The kernel
                derives the local split and reads the seed only on a split rank.
            memory_config (ttnn.MemoryConfig, optional): Interleaved output memory
                configuration. Defaults to DRAM.
            compute_kernel_config (ttnn.DeviceComputeKernelConfig, optional):
                Compute-kernel configuration. Defaults to exact HiFi2 math with
                FP32 destination accumulation.

        Returns:
            ttnn.Tensor: New FLOAT32 TILE-layout group-entry states
                ``[B*H*G, K, V]`` in flattened batch-head-group order.

        Note:
            ``K`` and ``V`` must be positive and tile-aligned. ``summaries`` may be
            interleaved, height-sharded or ND-sharded. All inputs must be allocated
            on the same device and are not modified. Output memory must be
            interleaved.
        )doc",
        &ttnn::experimental::kda::affine_exclusive_scan,
        nb::arg("summaries").noconvert(),
        nb::arg("initial_state").noconvert(),
        nb::arg("groups_per_head"),
        nb::kw_only(),
        nb::arg("actual_start").noconvert(),
        nb::arg("local_rows"),
        nb::arg("tail_entry_states") = nb::none(),

        nb::arg("memory_config") = nb::none(),
        nb::arg("compute_kernel_config") = nb::none(),
        nb::arg("sequence_parallel_axis") = 0,
        nb::arg("actual_end") = nb::none());
}

}  // namespace ttnn::operations::experimental::kda::affine_exclusive_scan::detail
