// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "chain_affine_transforms_nanobind.hpp"

#include "chain_affine_transforms.hpp"
#include "ttnn-nanobind/bind_function.hpp"

#include <nanobind/stl/optional.h>
#include <nanobind/stl/pair.h>

namespace ttnn::operations::experimental::kda::chain_affine_transforms::detail {

void bind_chain_affine_transforms(nb::module_& mod) {
    ttnn::bind_function<"chain_affine_transforms", "ttnn.experimental.kda.">(
        mod,
        R"doc(
        Apply one affine state transition per sequence-parallel rank in chronological order.

        ``transforms`` holds every rank's packed transition ``[A | B]`` gathered along the
        sequence-parallel axis in physical rank order. Starting from ``initial_state``, the
        transitions are applied in chronological order, the order of ranks starting at the
        rank that holds the first token of the chunk:

            S_{s+1} = A_{rank(s)} @ S_s + B_{rank(s)},   rank(s) = (first_rank + s) mod P

        Each step is a matmul that accumulates the whole key dimension in FP32, followed by an
        FP32 add. Returns this rank's entry state ``S_c``, where ``c`` is the
        rank's chronological index, and the completed carry ``S_P``.

        Args:
            transforms (ttnn.Tensor): Packed transitions ``[P, B*H, K, K + V]`` with ``P`` the
                sequence-parallel mesh size. TILE-layout BFLOAT16 interleaved device tensor.
            initial_state (ttnn.Tensor): FLOAT32 TILE-layout ``[B*H, K, V]`` interleaved state.

        Keyword Args:
            actual_start (ttnn.Tensor): Replicated UINT32 row-major scalar with the absolute
                position of the chunk's first token; pass [0] for zero-offset execution. Its value must
                be nonnegative and 32-aligned. Keep its address stable and update its contents before
                replay of a captured trace.
            local_rows (int): Positive, 32-aligned token rows per SP device.
            memory_config (ttnn.MemoryConfig, optional): Interleaved output memory. Defaults to DRAM.
            compute_kernel_config (ttnn.DeviceComputeKernelConfig, optional): Compute-kernel configuration.
                Defaults to HiFi2 with FP32 destination accumulation, which is required.
            sequence_parallel_axis (int, optional): Mesh axis partitioning the sequence. Native mesh
                coordinates supply each device's rank.

        Returns:
            tuple[ttnn.Tensor, ttnn.Tensor]: FLOAT32 TILE-layout ``[B*H, K, V]`` entry state
                and final carry.

        Note:
            K and V must be positive and tile-aligned, with one core per ``B*H`` head. Inputs are
            not modified.
        )doc",
        &ttnn::experimental::kda::chain_affine_transforms,
        nb::arg("transforms").noconvert(),
        nb::arg("initial_state").noconvert(),
        nb::kw_only(),
        nb::arg("actual_start").noconvert(),
        nb::arg("local_rows"),
        nb::arg("memory_config") = nb::none(),
        nb::arg("compute_kernel_config") = nb::none(),
        nb::arg("sequence_parallel_axis") = 0);
}

}  // namespace ttnn::operations::experimental::kda::chain_affine_transforms::detail
