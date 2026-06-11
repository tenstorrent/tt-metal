// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    Noc noc;

    // READER RUNTIME ARGS (Metal 2.0: named args via get_arg(args::), tensor addresses via TensorAccessor
    // bindings ta::, CB ids via DFB tokens dfb::)
    uint32_t num_blocks = get_arg(args::num_blocks);
    uint32_t in0_tensor_tile_id = get_arg(args::in0_tensor_tile_id);
    uint32_t in1_tensor_tile_id = get_arg(args::in1_tensor_tile_id);

    // COMPILE TIME ARGS
    // READER COMPILE TIME ARGS
    constexpr uint32_t q_num_tiles = get_arg(args::q_num_tiles);
    constexpr uint32_t kv_num_tiles = get_arg(args::kv_num_tiles);

    constexpr uint32_t cb_id_qv = dfb::cb_qv;  // cb for Q, V heads
#ifdef TRANSPOSE_K_HEADS
    constexpr uint32_t cb_id_k = dfb::cb_k;  // K goes to its own k_in DFB so compute can transpose it
#else
    constexpr uint32_t cb_id_k = dfb::cb_qv;  // not transposing: K shares the qv DFB (single binding per kernel)
#endif

    constexpr uint32_t onetile = 1;
    const auto s0 = TensorAccessor(ta::in0);

#ifdef READ_FROM_INPUT_TENSOR_KV
    const auto s1 = TensorAccessor(ta::in1);
#endif

    CircularBuffer cb_qv(cb_id_qv);
    CircularBuffer cb_k(cb_id_k);

    const uint32_t tile_bytes_qv = get_tile_size(cb_id_qv);
    const uint32_t tile_bytes_k = get_tile_size(cb_id_k);

    for (uint32_t block = 0; block < num_blocks; block++) {
        // Q
        for (uint32_t i = 0; i < q_num_tiles; i++) {
            cb_qv.reserve_back(onetile);
            uint32_t l1_write_addr = cb_qv.get_write_ptr();
            noc.async_read(
                s0, CoreLocalMem<uint32_t>(l1_write_addr), tile_bytes_qv, {.page_id = in0_tensor_tile_id}, {});
            noc.async_read_barrier();
            cb_qv.push_back(onetile);
            in0_tensor_tile_id++;
        }

        // K
        for (uint32_t i = 0; i < kv_num_tiles; i++) {
            cb_k.reserve_back(onetile);
            uint32_t l1_write_addr = cb_k.get_write_ptr();
#ifdef READ_FROM_INPUT_TENSOR_KV
            noc.async_read(
                s1, CoreLocalMem<uint32_t>(l1_write_addr), tile_bytes_k, {.page_id = in1_tensor_tile_id}, {});
            in1_tensor_tile_id++;
#else
            noc.async_read(
                s0, CoreLocalMem<uint32_t>(l1_write_addr), tile_bytes_k, {.page_id = in0_tensor_tile_id}, {});
            in0_tensor_tile_id++;
#endif
            noc.async_read_barrier();
            cb_k.push_back(onetile);
        }

        // V
        for (uint32_t i = 0; i < kv_num_tiles; i++) {
            cb_qv.reserve_back(onetile);
            uint32_t l1_write_addr = cb_qv.get_write_ptr();
#ifdef READ_FROM_INPUT_TENSOR_KV
            noc.async_read(
                s1, CoreLocalMem<uint32_t>(l1_write_addr), tile_bytes_qv, {.page_id = in1_tensor_tile_id}, {});
            in1_tensor_tile_id++;
#else
            noc.async_read(
                s0, CoreLocalMem<uint32_t>(l1_write_addr), tile_bytes_qv, {.page_id = in0_tensor_tile_id}, {});
            in0_tensor_tile_id++;
#endif
            noc.async_read_barrier();
            cb_qv.push_back(onetile);
        }
    }
}
