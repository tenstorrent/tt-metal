// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "reduce_affine_transforms_nanobind.hpp"

#include "reduce_affine_transforms.hpp"
#include "ttnn-nanobind/bind_function.hpp"

#include <nanobind/stl/optional.h>
#include <nanobind/stl/pair.h>

namespace ttnn::operations::experimental::kda::reduce_affine_transforms::detail {

void bind_reduce_affine_transforms(nb::module_& mod) {
    ttnn::bind_function<"reduce_affine_transforms", "ttnn.experimental.kda.">(
        mod,
        R"doc(
        Compose ordered group-level affine state transitions into one transition
        per batch-head.

        Each input pair represents one group of chunks for one batch-head:

            F_g(S) = A_g @ S + B_g

        The leading dimension is flattened batch-head-group order:

            index = batch_head * groups_per_head + group

        Groups are composed in sequence order:

            A_total = A_g @ A_total
            B_total = A_g @ B_total + B_g

        On the rank containing both the chronological beginning and end of the
        sequence, only the head groups are composed. Other ranks compose all
        groups. If the head ends inside a group, that group's input must contain
        its head-only summary, as produced by ``summarize_chunk_recurrence``.
        The separated tail is handled later in the recurrence pipeline.

        Optional ``actual_end`` is a replicated UINT32 row-major scalar, with
        the same lifetime as ``actual_start``. It defines a nonempty 32-aligned
        interval within physical capacity; omission uses the full capacity.
        Bounds may change during trace replay. Padded group outputs are unspecified.
        Bounds are caller preconditions and are not read back on the host.

        Args:
            a (ttnn.Tensor): Group multipliers ``[B*H*G, K, K]``. Each leading
                entry represents one batch-head-group. Must be a TILE-layout
                FLOAT32 or BFLOAT16 device tensor.
            b (ttnn.Tensor): Group offsets ``[B*H*G, K, V]``. Must have the same
                dtype, device, and leading dimension as ``a``.
            groups_per_head (int): Number of consecutive groups ``G`` belonging
                to each batch-head. Must be positive and divide the leading
                dimension.

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
                Compute-kernel configuration.

        Returns:
            tuple[ttnn.Tensor, ttnn.Tensor]: New FLOAT32 TILE-layout tensors
                ``A[B*H,K,K]`` and ``B[B*H,K,V]``, containing one composed
                transition per batch-head:

                    S_after = A @ S_before + B

        Note:
            ``K`` and ``V`` must be positive and tile-aligned, and each ``A_g``
            must be square. Inputs may be interleaved or height-sharded and are
            not modified. Output memory must be interleaved.
        )doc",
        &ttnn::experimental::kda::reduce_affine_transforms,
        nb::arg("a").noconvert(),
        nb::arg("b").noconvert(),
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
