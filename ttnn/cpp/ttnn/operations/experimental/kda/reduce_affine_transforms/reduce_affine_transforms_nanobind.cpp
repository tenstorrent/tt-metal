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
        nb::arg("memory_config") = nb::none(),
        nb::arg("compute_kernel_config") = nb::none());
}

}  // namespace ttnn::operations::experimental::kda::reduce_affine_transforms::detail
