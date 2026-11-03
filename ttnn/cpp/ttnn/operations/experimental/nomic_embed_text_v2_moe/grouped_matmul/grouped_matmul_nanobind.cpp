// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "grouped_matmul_nanobind.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/variant.h>

#include "ttnn-nanobind/bind_function.hpp"
#include "grouped_matmul.hpp"

namespace ttnn::experimental::nomic_embed_text_v2_moe {

void bind_grouped_matmul(nb::module_& mod) {
    ttnn::bind_function<"grouped_matmul", "ttnn.experimental.nomic_embed_text_v2_moe.">(
        mod,
        R"doc(
        Multiplies each 32-row block of an expert-sorted matrix by its own expert's weights.

        The rows are routing_metadata's gathered layout: expert e's rows fill whole 32-row blocks
        from block first_block[e], the regions in expert order, each expert_counts[e] rounded up to
        a multiple of 32. Block b is multiplied by the weights of the expert whose region holds it.
        The kernels read expert_counts on the device, so nothing is read back to the host and the
        call is trace-safe. Blocks past the last region are neither computed nor written: their
        output rows hold whatever the buffer held. Regions stop at the input's last row, so counts
        that do not match input_tensor never read or write past it or the output.

        One program runs every expert. Each core column multiplies by its share of the output
        columns. By default the blocks are dealt round-robin to the core rows and each expert's
        weights are read once and multicast down every column. With per_expert_lanes, core row e
        multiplies all of expert e's blocks and each core reads its own share of expert e's weights,
        so the experts' weights load at once: faster when the experts have few blocks each.

        Args:
            input_tensor (ttnn.Tensor): (1, 1, rows, K) BFLOAT16 or BFLOAT8_B, TILE, interleaved,
                rows a multiple of 32.
            weight (ttnn.Tensor): (1, 1, K, num_experts * N) BFLOAT16 or BFLOAT8_B, TILE,
                interleaved, expert e's (K, N) matrix in columns e * N to (e + 1) * N. With
                transpose_weight, (1, 1, N, num_experts * K), expert e's (N, K) matrix in columns
                e * K to (e + 1) * K, read transposed.
            expert_counts (ttnn.Tensor): (1, num_experts) UINT32, ROW_MAJOR, interleaved, as
                routing_metadata writes it. num_experts is 1 to 64.

        Keyword Args:
            dtype (ttnn.DataType): output dtype, BFLOAT16 or BFLOAT8_B.
            transpose_weight (bool): read each expert's weight transposed. Defaults to False.
            activation (ttnn.UnaryWithParam, optional): GELU_TANH with no parameter or 0 (accurate)
                or 1 (fast), applied on the pack thread. Defaults to none.
            output_layout (ttnn.Layout): TILE, or ROW_MAJOR for a BFLOAT16 output whose N splits
                evenly over the core columns. Defaults to TILE.
            memory_config (ttnn.MemoryConfig, optional): interleaved DRAM or L1. Defaults to the
                input's.
            compute_kernel_config (ttnn.DeviceComputeKernelConfig, optional): defaults to HiFi4
                with fp32 accumulation.
            per_expert_lanes (bool): give each expert a core row of its own; num_experts must be
                at most the grid's rows. The output is the same bit for bit. Defaults to False.

        Returns:
            ttnn.Tensor: (1, 1, rows, N).
        )doc",
        &ttnn::experimental::nomic_embed_text_v2_moe::grouped_matmul,
        nb::arg("input_tensor"),
        nb::arg("weight"),
        nb::arg("expert_counts"),
        nb::kw_only(),
        nb::arg("dtype"),
        nb::arg("transpose_weight") = false,
        nb::arg("activation") = nb::none(),
        nb::arg("output_layout") = nb::cast(tt::tt_metal::Layout::TILE),
        nb::arg("memory_config") = nb::none(),
        nb::arg("compute_kernel_config") = nb::none(),
        nb::arg("per_expert_lanes") = false);
}

}  // namespace ttnn::experimental::nomic_embed_text_v2_moe
