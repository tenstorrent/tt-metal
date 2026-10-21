// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include <array>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"
#include "ttnn/operations/experimental/transformer/nlp_concat_heads/device/kernels/nlp_concat_heads_split.hpp"

void kernel_main() {
    Noc noc;

    // Static core index plus the common row count in tiles; the (row, head) block range comes from the two.
    const uint32_t core_idx = get_arg(args::core_idx);
    const uint32_t in0_h_tiles = get_arg(args::h_tiles);

    // COMPILE TIME ARGS
    constexpr uint32_t in0_w_tiles = get_arg(args::in0_w_tiles);
    constexpr uint32_t in0_c = get_arg(args::in0_c);
    constexpr uint32_t batch = get_arg(args::batch);
    constexpr uint32_t num_cores = get_arg(args::num_cores);
    constexpr bool head_parallel_ok = get_arg(args::head_parallel_ok) != 0;

    const uint32_t in0_HtWt = in0_h_tiles * in0_w_tiles;
    const auto [start_block, end_block] = nlp_concat_heads_block_range(
        core_idx, in0_h_tiles, batch, in0_c, num_cores, head_parallel_ok);

    DataflowBuffer dfb_in0(dfb::in0);
    const uint32_t single_tile_size_bytes = dfb_in0.get_entry_size();
    const auto s0 = TensorAccessor(tensor::src);

    // Blocks are (row, head) in row-major order: one head's tile row each.
    for (uint32_t block = start_block; block < end_block; ++block) {
        const uint32_t row = block / in0_c;
        const uint32_t head = block % in0_c;
        uint32_t in0_tensor_current_tile_id =
            (row / in0_h_tiles * in0_c + head) * in0_HtWt + (row % in0_h_tiles) * in0_w_tiles;
#ifdef ARCH_QUASAR
        // On Quasar DM, get_write_ptr() returns the UNCACHED L1 alias and NOC APIs do not accept uncached
        // addresses, so read through the DFB endpoint one tile at a time.
        // TODO: batch per head once the Quasar DFB endpoint accepts an offset.
        for (uint32_t w_dim = 0; w_dim < in0_w_tiles; w_dim++) {
            dfb_in0.reserve_back(1);
            noc.async_read(s0, dfb_in0, single_tile_size_bytes, {.page_id = in0_tensor_current_tile_id}, {});
            noc.async_read_barrier();
            dfb_in0.push_back(1);
            in0_tensor_current_tile_id++;
        }
#else
        // One barrier per head: the buffer holds at least a row of heads.
        dfb_in0.reserve_back(in0_w_tiles);
        uint32_t l1_write_addr = dfb_in0.get_write_ptr();
        for (uint32_t w_dim = 0; w_dim < in0_w_tiles; w_dim++) {
            noc.async_read(
                s0,
                CoreLocalMem<uint32_t>(l1_write_addr),
                single_tile_size_bytes,
                {.page_id = in0_tensor_current_tile_id},
                {});
            l1_write_addr += single_tile_size_bytes;
            in0_tensor_current_tile_id++;
        }
        noc.async_read_barrier();
        dfb_in0.push_back(in0_w_tiles);
#endif
    }
}
