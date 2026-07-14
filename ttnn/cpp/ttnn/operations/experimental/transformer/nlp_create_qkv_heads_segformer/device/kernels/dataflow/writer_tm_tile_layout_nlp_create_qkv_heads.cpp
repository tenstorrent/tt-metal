// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include <array>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    Noc noc;

    // WRITER RUNTIME ARGS
    uint32_t q_tensor_addr = get_arg_val<uint32_t>(0);
    uint32_t num_blocks = get_arg_val<uint32_t>(1);
    uint32_t q_out_h_dim = get_arg_val<uint32_t>(2);
    uint32_t q_out_tensor_tile_id = get_arg_val<uint32_t>(3);

    // COMPILE TIME ARGS
    constexpr uint32_t q_out_h_tiles = get_compile_time_arg_val(0);
    constexpr uint32_t q_out_w_tiles = get_compile_time_arg_val(1);
    constexpr uint32_t q_out_HtWt = get_compile_time_arg_val(2);
    constexpr uint32_t q_out_c = get_compile_time_arg_val(3);
    constexpr auto q_args = TensorAccessorArgs<4>();

    constexpr uint32_t cb_id_qv = 1;  // cb for Q, V heads tiles

    const DataFormat data_format = get_dataformat(cb_id_qv);
    const auto sq = TensorAccessor(q_args, q_tensor_addr);

    CircularBuffer cb_qv(cb_id_qv);
    uint32_t tile_bytes = get_tile_size(cb_id_qv);

    constexpr uint32_t block_size = 1;  // micro-block size for read/write; nothing to do with num_blocks
    // TODO: This might negatively impact perf
    constexpr uint32_t out_num_tiles_read = block_size;  // always read and pop by micro-block size for generality
    uint32_t l1_read_addr;
    uint32_t q_out_tensor_current_tile_id;  // need this to update q_out_tensor_tile_id
    uint32_t out_tensor_current_tile_id_along_c;

    for (uint32_t block = 0; block < num_blocks; block++) {
        // q + create q head --> outputs: [B, num_q_heads, S, head_dim]
        out_tensor_current_tile_id_along_c = q_out_tensor_tile_id;
        for (uint32_t c_dim = 0; c_dim < q_out_c; c_dim++) {
            q_out_tensor_current_tile_id = out_tensor_current_tile_id_along_c;
            for (uint32_t w_dim = 0; w_dim < q_out_w_tiles; w_dim++) {
                cb_qv.wait_front(out_num_tiles_read);
                l1_read_addr = cb_qv.get_read_ptr();
                noc.async_write(
                    CoreLocalMem<uint32_t>(l1_read_addr),
                    sq,
                    tile_bytes,
                    {},
                    {.page_id = q_out_tensor_current_tile_id});

                noc.async_write_barrier();
                cb_qv.pop_front(out_num_tiles_read);

                q_out_tensor_current_tile_id++;
            }
            out_tensor_current_tile_id_along_c += q_out_HtWt;
        }

        // Update out_tensor_tile_id for next h_dim or batch if we finish one CHtWt
        q_out_h_dim++;
        if (q_out_h_dim < q_out_h_tiles) {
            q_out_tensor_tile_id += q_out_w_tiles;
        } else {
            // If we finish one batch, always roll over to next tile in memory
            // This is just the current_tile_id, except for K when we transpose heads
            // In this case, decrement k_out_tensor_current_tile_id by the stride (q_out_h_tiles) and add 1 to roll over
            q_out_tensor_tile_id = q_out_tensor_current_tile_id;
            q_out_h_dim = 0;
        }
    }
}
