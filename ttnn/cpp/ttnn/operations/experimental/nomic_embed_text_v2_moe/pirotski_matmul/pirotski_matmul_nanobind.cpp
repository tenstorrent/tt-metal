// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "pirotski_matmul_nanobind.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>

#include "ttnn-nanobind/bind_function.hpp"
#include "pirotski_matmul.hpp"

namespace ttnn::experimental::nomic_embed_text_v2_moe {

void bind_pirotski_matmul(nb::module_& mod) {
    ttnn::bind_function<"pirotski_matmul", "ttnn.experimental.nomic_embed_text_v2_moe.">(
        mod,
        R"doc(
        input_tensor @ weight + bias for M > N, bit for bit as ttnn.experimental.minimal_matmul with the same
        config, compute kernel config, dtype and bias.

        minimal_matmul's compute kernel on minimal_matmul's split: M over the config grid's columns and N over its
        rows, each padded to a multiple of its lanes. The activations are multicast down each core column from its
        top core and the weights west along each core row from its right core, instead of relayed core to core.
        An interleaved output's last block is written on both data-movement RISCs.

        Two layouts are taken directly when they match the grid:
        - an input block-sharded column-major in L1 with shard (x, s) on core (x, s), M_tiles_per_core rows high
          and dividing K (the split layer norm's layout): each K block is read from the shard holding it;
        - an output memory_config block-sharded column-major in L1 on the config's grid with shard (x, y) on core
          (x, y), M_tiles_per_core x N_tiles_per_core, M_block_size dividing M_tiles_per_core and N_block_size a
          core's whole N share: compute packs into the shard and nothing writes it.
        Any other layout is read and written tile by tile.

        With strip_m_tiles, the device grid's rows below the config grid (the strip) take the last strip_m_tiles
        rows of each column's M share: strip row j computes them for N tiles [j * S, (j + 1) * S), where S is the N
        share of (config rows / strip rows) config rows, its weights read by its right core and multicast along it.
        Every output tile keeps its K blocks and their order, so the output does not change. A sharded output's
        shards get the strip's rows written into their last strip_m_tiles rows; the main grid then packs only the
        first M_tiles_per_core - strip_m_tiles rows, which M_block_size must divide.

        Args:
            input_tensor (ttnn.Tensor): (..., M, K) BFLOAT16 or BFLOAT8_B, TILE; upper dims fold into M, each at its
                tile-padded length.
            weight (ttnn.Tensor): (K, N) BFLOAT16 or BFLOAT8_B, TILE, interleaved; leading dims of 1 are allowed.

        Keyword Args:
            config (ttnn.MinimalMatmulConfig): blocks, subblock and grid, as minimal_matmul takes them. N must split
                evenly over the grid's rows, N_block_size must divide a core's N share, K_block_size must divide K's
                tiles, and the subblock must divide the output block and fit the destination. With several N blocks
                a core, every second one runs K backwards from the last input block, as minimal_matmul's do.
            dtype (ttnn.DataType): output dtype, BFLOAT16 or BFLOAT8_B.
            bias (ttnn.Tensor, optional): (1, N) BFLOAT16, TILE, interleaved. Defaults to none.
            memory_config (ttnn.MemoryConfig, optional): the output's. Defaults to interleaved in the input's
                buffer type.
            compute_kernel_config (ttnn.DeviceComputeKernelConfig, optional): defaults to minimal_matmul's, HiFi2
                with fp32 and packer L1 accumulation.
            resident_weights (bool): keep a core's whole weight share in L1, K x N_tiles_per_core tiles, so only
                the first M block streams it. Defaults to False.
            strip_m_tiles (int): tile rows of each column's M share that the strip computes; 0 for none. It must be
                below the M share, the strip's row count must divide the config grid's rows, and the call must have
                one N block a core and streamed weights. The split layer norm's input layout is then read tile by
                tile. Defaults to 0.
            strip_read_ahead (bool): a strip row's weight reader issues block b + 1's reads before it multicasts
                block b. Defaults to False.
            read_ahead (bool): the same for the config grid's rows; needs one M block and one N block a core and
                streamed weights. Defaults to False.

        Returns:
            ttnn.Tensor: (..., M, N) TILE.
        )doc",
        &ttnn::experimental::nomic_embed_text_v2_moe::pirotski_matmul,
        nb::arg("input_tensor"),
        nb::arg("weight"),
        nb::kw_only(),
        nb::arg("config"),
        nb::arg("dtype"),
        nb::arg("bias") = nb::none(),
        nb::arg("memory_config") = nb::none(),
        nb::arg("compute_kernel_config") = nb::none(),
        nb::arg("resident_weights") = false,
        nb::arg("strip_m_tiles") = 0,
        nb::arg("strip_read_ahead") = false,
        nb::arg("read_ahead") = false);
}

}  // namespace ttnn::experimental::nomic_embed_text_v2_moe
