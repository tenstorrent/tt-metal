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

        Each input pair represents one group-level affine transition:

            F_g(S) = A_g @ S + B_g

        The leading dimension is flattened batch-head-group order:

            index = batch_head * groups_per_head + group

        The scan is exclusive: the first group receives ``initial_state``, and each
        later group receives the state produced by all preceding groups:

            entry[0] = initial_state
            entry[g] = A_{g-1} @ entry[g-1] + B_{g-1}

        Args:
            a (ttnn.Tensor): Group multipliers ``[B*H*G, K, K]``. Must be a
                TILE-layout FLOAT32 or BFLOAT16 device tensor.
            b (ttnn.Tensor): Group offsets ``[B*H*G, K, V]``. Must have the same
                dtype, device, and leading dimension as ``a``.
            initial_state (ttnn.Tensor): Entry state ``[B*H, K, V]`` for the first
                group of each batch-head. Must be a TILE-layout FLOAT32 device tensor.
            groups_per_head (int): Number of consecutive groups ``G`` belonging to
                each batch-head. Must be positive and divide the leading dimension.

        Keyword Args:
            memory_config (ttnn.MemoryConfig, optional): Interleaved output memory
                configuration. Defaults to DRAM.
            compute_kernel_config (ttnn.DeviceComputeKernelConfig, optional):
                Compute-kernel configuration. Defaults to exact HiFi2 math with
                FP32 destination accumulation.

        Returns:
            ttnn.Tensor: New FLOAT32 TILE-layout group-entry states
                ``[B*H*G, K, V]`` in flattened batch-head-group order.

        Note:
            ``K`` and ``V`` must be positive and tile-aligned, and each ``A_g`` must
            be square. All inputs must be allocated on the same device and are not
            modified. Output memory must be interleaved.
        )doc",
        &ttnn::experimental::kda::affine_exclusive_scan,
        nb::arg("a").noconvert(),
        nb::arg("b").noconvert(),
        nb::arg("initial_state").noconvert(),
        nb::arg("groups_per_head"),
        nb::kw_only(),
        nb::arg("memory_config") = nb::none(),
        nb::arg("compute_kernel_config") = nb::none());
}

}  // namespace ttnn::operations::experimental::kda::affine_exclusive_scan::detail
