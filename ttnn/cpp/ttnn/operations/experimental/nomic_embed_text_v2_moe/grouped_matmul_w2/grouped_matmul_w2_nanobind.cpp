// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "grouped_matmul_w2_nanobind.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/variant.h>

#include "ttnn-nanobind/bind_function.hpp"
#include "grouped_matmul_w2.hpp"

namespace ttnn::experimental::nomic_embed_text_v2_moe {

void bind_grouped_matmul_w2(nb::module_& mod) {
    ttnn::bind_function<"grouped_matmul_w2", "ttnn.experimental.nomic_embed_text_v2_moe.">(
        mod,
        R"doc(
        Multiplies each 32-row block of an expert-sorted matrix by its own expert's w2, into row-major rows.

        The rows are routing_metadata's gathered layout: expert e's rows fill whole 32-row blocks from
        block first_block[e], the regions in expert order, each expert_counts[e] rounded up to a
        multiple of 32. Block b is multiplied by the weights of the expert whose region holds it. The
        kernels read expert_counts on the device, so nothing is read back to the host and the call is
        trace-safe. Blocks past the last region are neither computed nor written. Each output tile
        accumulates all of K in fp32 in k order and the weight tiles are read transposed, as
        grouped_matmul_w1 reads w2, so the two agree bit for bit.

        The cores split N into 8 shares of 3 tiles and the blocks over 10 core-row lanes, or 13 with
        three core-column lanes once there are 156 blocks. Each lane's sender reads a block from the
        lane's own cores, where grouped_matmul_w2_input_memory_config puts it, and multicasts it along
        the lane; each share's top core streams the experts' weights from DRAM down its column.

        Args:
            input_tensor (ttnn.Tensor): (1, 1, rows, K) BFLOAT8_B TILE in
                grouped_matmul_w2_input_memory_config(rows, K)'s layout, rows a multiple of 32.
            weight (ttnn.Tensor): the weight from prepare_grouped_matmul_w2_weight.
            expert_counts (ttnn.Tensor): (1, num_experts) UINT32, ROW_MAJOR, interleaved, as
                routing_metadata writes it; num_experts is 1 to 64.

        Keyword Args:
            weight_chunk_tiles (int): the chunk_tiles the weight was prepared with.
            weight_n (int): an expert's N, 768 for 8 shares of 3 tiles.
            memory_config (ttnn.MemoryConfig, optional): interleaved DRAM or L1. Defaults to L1.
            compute_kernel_config (ttnn.DeviceComputeKernelConfig, optional): defaults to HiFi4
                with fp32 accumulation.
            first_slots (int): half slices of the first K half a core's weight ring holds. Defaults to 2.
            second_slots (int): the same for the second K half. Defaults to 2.

        Returns:
            ttnn.Tensor: (1, 1, rows, N) BFLOAT16, ROW_MAJOR.
        )doc",
        &ttnn::experimental::nomic_embed_text_v2_moe::grouped_matmul_w2,
        nb::arg("input_tensor"),
        nb::arg("weight"),
        nb::arg("expert_counts"),
        nb::kw_only(),
        nb::arg("weight_chunk_tiles"),
        nb::arg("weight_n"),
        nb::arg("memory_config") = nb::none(),
        nb::arg("compute_kernel_config") = nb::none(),
        nb::arg("first_slots") = 2,
        nb::arg("second_slots") = 2);

    ttnn::bind_function<"prepare_grouped_matmul_w2_weight", "ttnn.experimental.nomic_embed_text_v2_moe.">(
        mod,
        R"doc(
        Lays out grouped_matmul_w2's weight for this device as contiguous chunks in DRAM. Meant to run once, at load.

        Args:
            weight (ttnn.Tensor): (1, 1, N, num_experts * K) BFLOAT8_B TILE, expert e's w2 transposed in
                columns e * K to (e + 1) * K, on the host or a device.
            num_experts (int): the experts the weight holds.

        Keyword Args:
            chunk_tiles (int): tiles a chunk; must split each K half of a share's slice evenly.
            device (ttnn.MeshDevice): the device whose grid the layout is for, and where it is placed.

        Returns:
            ttnn.Tensor: WIDTH_SHARDED over the DRAM banks; pass it with weight_chunk_tiles = chunk_tiles.
        )doc",
        &ttnn::experimental::nomic_embed_text_v2_moe::prepare_grouped_matmul_w2_weight,
        nb::arg("weight"),
        nb::arg("num_experts"),
        nb::kw_only(),
        nb::arg("chunk_tiles"),
        nb::arg("device"));

    ttnn::bind_function<"grouped_matmul_w2_input_memory_config", "ttnn.experimental.nomic_embed_text_v2_moe.">(
        mod,
        R"doc(
        The memory config grouped_matmul_w2 reads a (1, 1, rows, k) input in: L1, ND-sharded in quarter-block
        pieces, each block's pieces on its own lane's cores.

        Args:
            rows (int): the input's rows, a multiple of 32.
            k (int): the input's K, a multiple of 128.
            device (ttnn.MeshDevice): the device.

        Returns:
            ttnn.MemoryConfig.
        )doc",
        &ttnn::experimental::nomic_embed_text_v2_moe::grouped_matmul_w2_input_memory_config,
        nb::arg("rows"),
        nb::arg("k"),
        nb::arg("device"));

    ttnn::bind_function<"grouped_matmul_w2_circular_buffer_bytes", "ttnn.experimental.nomic_embed_text_v2_moe.">(
        mod,
        R"doc(
        Circular-buffer bytes a core of a grouped_matmul_w2 call takes, for L1 planning.

        Args:
            k (int): the input's K.
            n (int): an expert's N.
            first_slots (int): the call's first_slots.
            second_slots (int): the call's second_slots.
            num_experts (int): the experts expert_counts holds.
            device (ttnn.MeshDevice): the device.

        Returns:
            int.
        )doc",
        &ttnn::experimental::nomic_embed_text_v2_moe::grouped_matmul_w2_circular_buffer_bytes,
        nb::arg("k"),
        nb::arg("n"),
        nb::arg("first_slots"),
        nb::arg("second_slots"),
        nb::arg("num_experts"),
        nb::arg("device"));
}

}  // namespace ttnn::experimental::nomic_embed_text_v2_moe
