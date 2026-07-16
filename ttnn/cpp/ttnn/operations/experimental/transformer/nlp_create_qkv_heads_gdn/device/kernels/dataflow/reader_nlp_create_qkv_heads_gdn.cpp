// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// GDN create-heads reader: reads the fused token-major input one seq-tile-row ("block") at a time.
// Per block the fused width is a contiguous run of q_num_tiles (Q) + k_num_tiles (K) + v_num_tiles
// (V) tiles; the Q|K|V column split is implicit in read order. The whole block's qkv_tiles are read
// into cb1 behind a SINGLE async_read_barrier so the NoC pipelines them (vs one barrier per tile).
// The writer pops them in the same q→k→v order and scatters them head-major. No transpose.

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    Noc noc;

    // RUNTIME ARGS
    uint32_t in0_tensor_addr = get_arg_val<uint32_t>(0);
    uint32_t num_blocks = get_arg_val<uint32_t>(1);
    uint32_t in0_tensor_tile_id = get_arg_val<uint32_t>(2);

    // COMPILE TIME ARGS
    constexpr uint32_t q_num_tiles = get_compile_time_arg_val(0);
    constexpr uint32_t k_num_tiles = get_compile_time_arg_val(1);
    constexpr uint32_t v_num_tiles = get_compile_time_arg_val(2);
    constexpr auto in0_args = TensorAccessorArgs<3>();

    constexpr uint32_t cb_id = 1;
    constexpr uint32_t qkv_tiles = q_num_tiles + k_num_tiles + v_num_tiles;

    const auto s0 = TensorAccessor(in0_args, in0_tensor_addr);
    CircularBuffer cb(cb_id);
    const uint32_t tile_bytes = get_tile_size(cb_id);

    for (uint32_t block = 0; block < num_blocks; block++) {
        // Reserve the whole block (Q run, then K run, then V run — contiguous in the fused input).
        // The CB is sized to an integer multiple of qkv_tiles (see the program factory), so the
        // reserved region never wraps the ring and the qkv_tiles tiles are contiguous in L1.
        cb.reserve_back(qkv_tiles);
        uint32_t l1_write_addr = cb.get_write_ptr();
        for (uint32_t i = 0; i < qkv_tiles; i++) {
            noc.async_read(
                s0,
                CoreLocalMem<uint32_t>(l1_write_addr + i * tile_bytes),
                tile_bytes,
                {.page_id = in0_tensor_tile_id},
                {});
            in0_tensor_tile_id++;
        }
        noc.async_read_barrier();  // one barrier for the whole block: the reads pipeline on the NoC
        cb.push_back(qkv_tiles);
    }
}
