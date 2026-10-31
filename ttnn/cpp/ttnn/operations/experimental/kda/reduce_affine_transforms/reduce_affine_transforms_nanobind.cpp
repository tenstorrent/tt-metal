// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "reduce_affine_transforms_nanobind.hpp"

#include "reduce_affine_transforms.hpp"
#include "ttnn-nanobind/bind_function.hpp"

#include <nanobind/stl/optional.h>

namespace ttnn::operations::experimental::kda::reduce_affine_transforms::detail {

void bind_reduce_affine_transforms(nb::module_& mod) {
    ttnn::bind_function<"reduce_affine_transforms", "ttnn.experimental.kda.">(
        mod,
        R"doc(
        Compose ordered group-level affine state transitions into one transition
        per batch-head.

        Each transform represents one group of chunks for one batch-head:

            F_g(S) = A_g @ S + B_g

        It is read in complement form, ``E_g = A_g - I``, from slot 0 (head
        transforms) of packed summaries, as produced by
        ``summarize_chunk_recurrence``; ``A_g = E_g + I`` is formed in FP32. The
        transform index is flattened batch-head-group order:

            index = batch_head * groups_per_head + group

        Groups are composed in sequence order:

            A_total = A_g @ A_total
            B_total = A_g @ B_total + B_g

        On the rank containing both the chronological beginning and end of the
        sequence, only the head groups are composed. Other ranks compose all
        groups. If the head ends inside a group, that group's slot-0 transform
        must be its head-only summary. A tail slot, when present, is not read;
        the separated tail is handled later in the recurrence pipeline.

        The composed transition is returned in complement form, packed as the
        distributed prefix gathers it (see ``chain_affine_transforms``):

            S_after = S_before + (E @ S_before + B),   E = A_total - I

        E is formed in FP32 from the FP32 composition and rounded once to BF16, so
        a transition close to the identity keeps the precision of its small change.
        A rank with no valid group writes the identity map, ``(0, 0)``.

        Optional ``actual_end`` is a replicated UINT32 row-major scalar, with
        the same lifetime as ``actual_start``. It defines a nonempty interval within
        physical capacity and need not be 32-aligned: a partial last chunk is
        valid, and its rows past the end must be identity steps, as produced by
        ``prepare_chunk_recurrence``. Omission uses the full capacity.
        Bounds may change during trace replay. Padded group outputs are unspecified.
        Bounds are caller preconditions and are not read back on the host.

        Args:
            summaries (ttnn.Tensor): Packed group transforms ``[S, B*H*G, K, K+V]``
                with ``S`` 1 or 2: ``K`` rows of ``[E | B]`` per transform. Must be
                a TILE-layout FLOAT32 or BFLOAT16 device tensor.
            groups_per_head (int): Number of consecutive groups ``G`` belonging
                to each batch-head. Must be positive and divide the transform
                count ``B*H*G`` (dimension 1).

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
            memory_config (ttnn.MemoryConfig, optional): Interleaved output memory
                configuration. Defaults to DRAM.
            compute_kernel_config (ttnn.DeviceComputeKernelConfig, optional):
                Compute-kernel configuration. Requires ``fp32_dest_acc_en``.

        Returns:
            ttnn.Tensor: New BFLOAT16 TILE-layout tensor ``[1,B*H,K,K+V]`` whose
                rows are ``[E | B]``, one composed transition per batch-head.

        Note:
            ``K`` and ``V`` must be positive and tile-aligned. ``summaries`` may be
            interleaved, height-sharded or ND-sharded and is not modified. Output
            memory must be interleaved.
        )doc",
        &ttnn::experimental::kda::reduce_affine_transforms,
        nb::arg("summaries").noconvert(),
        nb::arg("groups_per_head"),
        nb::kw_only(),
        nb::arg("actual_start").noconvert(),
        nb::arg("local_rows"),
        nb::arg("memory_config") = nb::none(),
        nb::arg("compute_kernel_config") = nb::none(),
        nb::arg("sequence_parallel_axis") = 0,
        nb::arg("actual_end") = nb::none());
}

}  // namespace ttnn::operations::experimental::kda::reduce_affine_transforms::detail
