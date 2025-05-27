// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "dataflow_api.h"

template <bool DRAM, uint32_t tile_hw = 1024>
inline __attribute__((always_inline)) void read_tile(
    const uint32_t cb_id, const InterleavedAddrGenFast<DRAM, tile_hw>& s, uint32_t tensor_tile_id) {
    constexpr uint32_t onetile = 1;
    cb_reserve_back(cb_id, onetile);
    uint32_t l1_write_addr = get_write_ptr(cb_id);
    noc_async_read_tile(tensor_tile_id, s, l1_write_addr);
    noc_async_read_barrier();
    cb_push_back(cb_id, onetile);
}

void kernel_main() {
    // READER RUNTIME ARGS
    uint32_t in0_tensor_addr = get_arg_val<uint32_t>(0);
    uint32_t in1_tensor_addr = get_arg_val<uint32_t>(1);
    uint32_t num_blocks_q = get_arg_val<uint32_t>(2);
    uint32_t num_blocks_kv = get_arg_val<uint32_t>(3);
    uint32_t in0_tensor_tile_id = get_arg_val<uint32_t>(4);
    uint32_t in1_tensor_tile_id = get_arg_val<uint32_t>(5);

    // Read in interleaved fashion from q and kv tensors, up to min(num_blocks_q, num_blocks_kv)
    // and then read from other tensor to fill the rest of the blocks
    uint32_t num_blocks_common = num_blocks_q < num_blocks_kv ? num_blocks_q : num_blocks_kv;
    uint32_t num_blocks_q_remaining = num_blocks_q - num_blocks_common;
    uint32_t num_blocks_kv_remaining = num_blocks_kv - num_blocks_common;

    // COMPILE TIME ARGS
    // interleaved accessor args
    constexpr uint32_t in0_is_dram = get_compile_time_arg_val(0);
    constexpr uint32_t in1_is_dram = get_compile_time_arg_val(1);
    // READER COMPILE TIME ARGS
    constexpr uint32_t q_num_tiles = get_compile_time_arg_val(2);
    constexpr uint32_t kv_num_tiles = get_compile_time_arg_val(3);

    constexpr uint32_t cb_id_qv = 1;  // cb for Q, V heads
#ifdef TRANSPOSE_K_HEADS
    constexpr uint32_t cb_id_k = 0;  // cb for K heads (used by compute)
#else
    constexpr uint32_t cb_id_k = 1;  // cb for K heads (directly to writer)
#endif

    constexpr uint32_t onetile = 1;
    const uint32_t single_tile_size_bytes = get_tile_size(cb_id_qv);
    const DataFormat data_format = get_dataformat(cb_id_qv);

    constexpr bool in0_is_dram_bool = in0_is_dram == 1;
    const InterleavedAddrGenFast<in0_is_dram_bool> s0 = {
        .bank_base_address = in0_tensor_addr,
        .page_size = single_tile_size_bytes,
        .data_format = data_format,
    };

#ifdef READ_FROM_INPUT_TENSOR_KV
    constexpr bool in1_is_dram_bool = in1_is_dram == 1;
    const InterleavedAddrGenFast<in1_is_dram_bool> s1 = {
        .bank_base_address = in1_tensor_addr,
        .page_size = single_tile_size_bytes,
        .data_format = data_format,
    };
#endif

    for (uint32_t block = 0; block < num_blocks_common; block++) {
        // Q
        for (uint32_t i = 0; i < q_num_tiles; i++) {
            read_tile(cb_id_qv, s0, in0_tensor_tile_id);
            in0_tensor_tile_id++;
        }

        // K
        for (uint32_t i = 0; i < kv_num_tiles; i++) {
#ifdef READ_FROM_INPUT_TENSOR_KV
            read_tile(cb_id_k, s1, in1_tensor_tile_id);
            in1_tensor_tile_id++;
#else
            read_tile(cb_id_k, s0, in0_tensor_tile_id);
            in0_tensor_tile_id++;
#endif
        }

        // V
        for (uint32_t i = 0; i < kv_num_tiles; i++) {
#ifdef READ_FROM_INPUT_TENSOR_KV
            read_tile(cb_id_qv, s1, in1_tensor_tile_id);
            in1_tensor_tile_id++;
#else
            read_tile(cb_id_qv, s0, in0_tensor_tile_id);
            in0_tensor_tile_id++;
#endif
        }
    }
    // Read remaining blocks from q
    for (uint32_t block = 0; block < num_blocks_q_remaining; block++) {
        for (uint32_t i = 0; i < q_num_tiles; i++) {
            read_tile(cb_id_qv, s0, in0_tensor_tile_id);
            in0_tensor_tile_id++;
        }
    }
    // Read remaining blocks from kv
    for (uint32_t block = 0; block < num_blocks_kv_remaining; block++) {
        for (uint32_t i = 0; i < kv_num_tiles; i++) {
#ifdef READ_FROM_INPUT_TENSOR_KV
            read_tile(cb_id_k, s1, in1_tensor_tile_id);
            in1_tensor_tile_id++;
#else
            read_tile(cb_id_k, s0, in0_tensor_tile_id);
            in0_tensor_tile_id++;
#endif
        }
        for (uint32_t i = 0; i < kv_num_tiles; i++) {
#ifdef READ_FROM_INPUT_TENSOR_KV
            read_tile(cb_id_qv, s1, in1_tensor_tile_id);
            in1_tensor_tile_id++;
#else
            read_tile(cb_id_qv, s0, in0_tensor_tile_id);
            in0_tensor_tile_id++;
#endif
        }
    }
}
