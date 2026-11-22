// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "grouped_matmul_w1_nanobind.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/variant.h>

#include "ttnn-nanobind/bind_function.hpp"
#include "grouped_matmul_w1.hpp"

namespace ttnn::experimental::nomic_embed_text_v2_moe {

void bind_grouped_matmul_w1(nb::module_& mod) {
    ttnn::bind_function<"grouped_matmul_w1", "ttnn.experimental.nomic_embed_text_v2_moe.">(
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
        columns. By default the blocks are dealt round-robin to the core rows; each column's top core
        streams each expert's weights and multicasts them down the column, and each row's input
        blocks are multicast along it, from its east core or, for large inputs where three columns
        are free, from dedicated sender cores east of the grid. With per_expert_lanes, core row e
        multiplies all of expert e's blocks and each core reads its own share of expert e's weights.

        The weight is fastest as a chunked weight from prepare_grouped_matmul_w1_weight, read in large
        contiguous chunks; a plain TILE weight is read one tile at a time.

        Args:
            input_tensor (ttnn.Tensor): (1, 1, rows, K) BFLOAT16 or BFLOAT8_B, TILE, interleaved or
                sharded, rows a multiple of 32; or (B, 1, S, K), whose B x ceil32(S) tile rows, each
                sequence's padding included, are the rows, and the output is then (B, 1, S, N).
            weight (ttnn.Tensor): (1, 1, K, num_experts * N) BFLOAT16 or BFLOAT8_B, TILE,
                interleaved, expert e's (K, N) matrix in columns e * N to (e + 1) * N. With
                transpose_weight, (1, 1, N, num_experts * K), expert e's (N, K) matrix in columns
                e * K to (e + 1) * K, read transposed. Or such a weight (and bias) as
                prepare_grouped_matmul_w1_weight returns it, with weight_chunk_tiles and weight_n.
            expert_counts (ttnn.Tensor, optional): (1, num_experts) UINT32, ROW_MAJOR, interleaved, as
                routing_metadata writes it. num_experts is 1 to 64. None: one expert taking every row,
                with no counts to read.

        Keyword Args:
            dtype (ttnn.DataType): output dtype, BFLOAT16 or BFLOAT8_B.
            transpose_weight (bool): read each expert's weight transposed. Defaults to False.
            activation (ttnn.UnaryWithParam, optional): GELU_TANH with no parameter or 0 (accurate),
                1 (fast) or 2 (a degree-4 polynomial of the normal CDF, |error| <= 1.3e-3, for a
                BFLOAT8_B output), applied on the pack thread. Defaults to none.
            output_layout (ttnn.Layout): TILE, or ROW_MAJOR for a BFLOAT16 output whose N splits
                evenly over the core columns. Defaults to TILE.
            memory_config (ttnn.MemoryConfig, optional): interleaved DRAM or L1. Defaults to
                interleaved in the input's buffer type.
            compute_kernel_config (ttnn.DeviceComputeKernelConfig, optional): defaults to HiFi4
                with fp32 accumulation.
            per_expert_lanes (bool): give each expert a core row of its own; num_experts must be
                at most the grid's rows. The output is the same bit for bit. Defaults to False.
            bias (ttnn.Tensor, optional): (1, 1, 1, num_experts * N) TILE, the weight's dtype, each
                expert's N columns side by side, added before the activation. Needs a BFLOAT16 input
                and an untransposed weight. With a chunked weight the bias is in the weight and this
                is None. Defaults to none.
            weight_chunk_tiles (int): the chunk_tiles weight was prepared with; 0 for a plain TILE
                weight. Defaults to 0.
            weight_n (int): an expert's N, for a chunked weight. Defaults to 0.
            weight_has_bias (bool): the chunked weight holds a bias. Defaults to False.

        Returns:
            ttnn.Tensor: (1, 1, rows, N), or (B, 1, S, N) for a (B, 1, S, K) input.
        )doc",
        &ttnn::experimental::nomic_embed_text_v2_moe::grouped_matmul_w1,
        nb::arg("input_tensor"),
        nb::arg("weight"),
        nb::arg("expert_counts").none(),
        nb::kw_only(),
        nb::arg("dtype"),
        nb::arg("transpose_weight") = false,
        nb::arg("activation") = nb::none(),
        nb::arg("output_layout") = nb::cast(tt::tt_metal::Layout::TILE),
        nb::arg("memory_config") = nb::none(),
        nb::arg("compute_kernel_config") = nb::none(),
        nb::arg("per_expert_lanes") = false,
        nb::arg("bias") = nb::none(),
        nb::arg("weight_chunk_tiles") = 0,
        nb::arg("weight_n") = 0,
        nb::arg("weight_has_bias") = false);

    ttnn::bind_function<"prepare_grouped_matmul_w1_weight", "ttnn.experimental.nomic_embed_text_v2_moe.">(
        mod,
        R"doc(
        Lays out grouped_matmul_w1's weight (and bias) for this device's grid as a chunked weight.

        Each core column's share of each expert's weights is stored in the order grouped_matmul_w1 reads
        it, in contiguous chunks of chunk_tiles tiles spread over the DRAM banks, so a column reads a
        chunk at a time. The tiles are copied whole, so grouped_matmul_w1's results are the same bit for
        bit as with the TILE weight. Meant to run once, at load.

        Args:
            weight (ttnn.Tensor): grouped_matmul_w1's TILE weight, on the host or a device.
            num_experts (int): the experts the weight holds.

        Keyword Args:
            transpose_weight (bool): the weight is read transposed. Defaults to False.
            bias (ttnn.Tensor, optional): grouped_matmul_w1's bias, stored with the weight. Defaults to none.
            chunk_tiles (int): tiles a chunk. Defaults to 60.
            device (ttnn.MeshDevice): the device whose grid the layout is for, and where it is placed.

        Returns:
            ttnn.Tensor: (1, 1, 32, 8 * shard tiles * 32) TILE, WIDTH_SHARDED over the DRAM banks. Pass
            it to grouped_matmul_w1 with weight_chunk_tiles = chunk_tiles, weight_n = an expert's N and
            weight_has_bias = bias is not None.
        )doc",
        &ttnn::experimental::nomic_embed_text_v2_moe::prepare_grouped_matmul_w1_weight,
        nb::arg("weight"),
        nb::arg("num_experts"),
        nb::kw_only(),
        nb::arg("transpose_weight") = false,
        nb::arg("bias") = nb::none(),
        nb::arg("chunk_tiles") = 60,
        nb::arg("device"));
}

}  // namespace ttnn::experimental::nomic_embed_text_v2_moe
