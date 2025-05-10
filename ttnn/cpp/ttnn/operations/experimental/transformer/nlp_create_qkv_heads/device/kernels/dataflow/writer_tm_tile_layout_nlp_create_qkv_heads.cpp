// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include <array>
#include "dataflow_api.h"

template <bool DRAM, uint32_t tile_hw = 1024>
inline __attribute__((always_inline)) void write_tiles(
    const uint32_t cb_id, const InterleavedAddrGenFast<DRAM, tile_hw>& s, uint32_t tensor_tile_id, uint32_t num_tiles) {
    cb_wait_front(cb_id, num_tiles);
    uint32_t l1_read_addr = get_read_ptr(cb_id);
    noc_async_write_tile(tensor_tile_id, s, l1_read_addr);
    noc_async_write_barrier();
    cb_pop_front(cb_id, num_tiles);
}

template <bool DRAM, uint32_t tile_hw = 1024>
inline __attribute__((always_inline)) void process_q_block(
    const uint32_t cb_id,
    const InterleavedAddrGenFast<DRAM, tile_hw>& s,
    const uint32_t q_out_c,
    const uint32_t q_out_HtWt,
    const uint32_t q_out_w_tiles,
    const uint32_t q_out_h_tiles,
    const uint32_t num_tiles_read,
    uint32_t& q_out_h_dim,
    uint32_t& q_tensor_tile_id) {
    // q + create q head --> outputs: [B, num_q_heads, Sq, head_dim]
    uint32_t out_tensor_current_tile_id_along_c = q_tensor_tile_id;
    uint32_t q_out_tensor_current_tile_id = 0;
    for (uint32_t c_dim = 0; c_dim < q_out_c; c_dim++) {
        q_out_tensor_current_tile_id = out_tensor_current_tile_id_along_c;
        for (uint32_t w_dim = 0; w_dim < q_out_w_tiles; w_dim++) {
            write_tiles(cb_id, s, q_out_tensor_current_tile_id, num_tiles_read);
            q_out_tensor_current_tile_id++;
        }
        out_tensor_current_tile_id_along_c += q_out_HtWt;
    }

    // Update out_tensor_tile_id for next h_dim or batch if we finish one CHtWt
    q_out_h_dim++;
    if (q_out_h_dim < q_out_h_tiles) {
        q_tensor_tile_id += q_out_w_tiles;
    } else {
        // If we finish one batch, always roll over to next tile in memory
        // This is just the current_tile_id for Q
        q_tensor_tile_id = q_out_tensor_current_tile_id;
        q_out_h_dim = 0;
    }
}

template <bool DRAM, uint32_t tile_hw = 1024>
inline __attribute__((always_inline)) void process_kv_block(
    const uint32_t cb_id_k,
    const uint32_t cb_id_qv,
    const InterleavedAddrGenFast<DRAM, tile_hw>& s_k,
    const InterleavedAddrGenFast<DRAM, tile_hw>& s_v,
    const uint32_t kv_out_c,
    const uint32_t kv_out_HtWt,
    const uint32_t kv_out_w_tiles,
    const uint32_t kv_out_h_tiles,
    const uint32_t num_tiles_read,
    uint32_t& kv_out_h_dim,
    uint32_t& k_tensor_tile_id,
    uint32_t& v_tensor_tile_id) {
    // k + create k head --> outputs: [B, num_kv_heads, Skv, head_dim]
    uint32_t out_tensor_current_tile_id_along_c = 0;
    uint32_t k_out_tensor_current_tile_id = 0;
#ifndef TRANSPOSE_K_HEADS
    out_tensor_current_tile_id_along_c = k_tensor_tile_id;
#else
    k_out_tensor_current_tile_id = k_tensor_tile_id;
#endif
    for (uint32_t c_dim = 0; c_dim < kv_out_c; c_dim++) {
#ifndef TRANSPOSED_K_HEADS
        k_out_tensor_current_tile_id = out_tensor_current_tile_id_along_c;
#endif
        for (uint32_t w_dim = 0; w_dim < kv_out_w_tiles; w_dim++) {
            write_tiles(cb_id_k, s_k, k_out_tensor_current_tile_id, num_tiles_read);
#ifndef TRANSPOSED_K_HEADS
            k_out_tensor_current_tile_id++;
#else
            k_out_tensor_current_tile_id += kv_out_h_tiles;
#endif
        }
#ifndef TRANSPOSED_K_HEADS
        out_tensor_current_tile_id_along_c += kv_out_HtWt;
#endif
    }

    // v + create v head --> outputs: [B, num_kv_heads, S, head_dim]
    out_tensor_current_tile_id_along_c = v_tensor_tile_id;
    uint32_t v_out_tensor_current_tile_id = 0;
    for (uint32_t c_dim = 0; c_dim < kv_out_c; c_dim++) {
        v_out_tensor_current_tile_id = out_tensor_current_tile_id_along_c;
        for (uint32_t w_dim = 0; w_dim < kv_out_w_tiles; w_dim++) {
            write_tiles(cb_id_qv, s_v, v_out_tensor_current_tile_id, num_tiles_read);
            v_out_tensor_current_tile_id++;
        }
        out_tensor_current_tile_id_along_c += kv_out_HtWt;
    }

    kv_out_h_dim++;
    if (kv_out_h_dim < kv_out_h_tiles) {
#ifndef TRANSPOSE_K_HEADS
        k_tensor_tile_id += kv_out_w_tiles;
#else
        k_tensor_tile_id++;
#endif
        v_tensor_tile_id += kv_out_w_tiles;
    } else {
#ifndef TRANSPOSE_K_HEADS
        k_tensor_tile_id = k_out_tensor_current_tile_id;
#else
        k_tensor_tile_id = ++k_out_tensor_current_tile_id - kv_out_h_tiles;  // inc by 1 and decrement by stride
#endif
        v_tensor_tile_id = v_out_tensor_current_tile_id;
        kv_out_h_dim = 0;
    }
}

void kernel_main() {
    // WRITER RUNTIME ARGS
    uint32_t q_tensor_addr = get_arg_val<uint32_t>(0);
    uint32_t k_tensor_addr = get_arg_val<uint32_t>(1);
    uint32_t v_tensor_addr = get_arg_val<uint32_t>(2);
    uint32_t num_blocks_q = get_arg_val<uint32_t>(3);
    uint32_t num_blocks_kv = get_arg_val<uint32_t>(4);
    uint32_t q_out_h_dim = get_arg_val<uint32_t>(5);
    uint32_t kv_out_h_dim = get_arg_val<uint32_t>(6);
    uint32_t q_out_tensor_tile_id = get_arg_val<uint32_t>(7);
    uint32_t k_out_tensor_tile_id = get_arg_val<uint32_t>(8);
    uint32_t v_out_tensor_tile_id = get_arg_val<uint32_t>(9);

    uint32_t num_blocks_common = num_blocks_q < num_blocks_kv ? num_blocks_q : num_blocks_kv;
    uint32_t num_blocks_q_remaining = num_blocks_q - num_blocks_common;
    uint32_t num_blocks_kv_remaining = num_blocks_kv - num_blocks_common;

    // COMPILE TIME ARGS
    // interleaved accessor args
    constexpr uint32_t out_is_dram = get_compile_time_arg_val(0);
    constexpr uint32_t q_out_h_tiles = get_compile_time_arg_val(1);
    constexpr uint32_t q_out_w_tiles = get_compile_time_arg_val(2);
    constexpr uint32_t q_out_HtWt = get_compile_time_arg_val(3);
    constexpr uint32_t kv_out_h_tiles = get_compile_time_arg_val(4);
    constexpr uint32_t kv_out_w_tiles = get_compile_time_arg_val(5);
    constexpr uint32_t kv_out_HtWt = get_compile_time_arg_val(6);
    constexpr uint32_t q_out_c = get_compile_time_arg_val(7);
    constexpr uint32_t kv_out_c = get_compile_time_arg_val(8);

    constexpr uint32_t cb_id_qv = 1;  // cb for Q, V heads tiles
#ifdef TRANSPOSE_K_HEADS
    constexpr uint32_t cb_id_k = 16;  // cb for K heads (filled by compute)
#else
    constexpr uint32_t cb_id_k = 1;  // cb for K heads (directly from reader)
#endif
    const uint32_t single_tile_size_bytes = get_tile_size(cb_id_qv);
    const DataFormat data_format = get_dataformat(cb_id_qv);

    constexpr bool out_is_dram_bool = out_is_dram == 1;
    const InterleavedAddrGenFast<out_is_dram_bool> sq = {
        .bank_base_address = q_tensor_addr, .page_size = single_tile_size_bytes, .data_format = data_format};
    const InterleavedAddrGenFast<out_is_dram_bool> sk = {
        .bank_base_address = k_tensor_addr, .page_size = single_tile_size_bytes, .data_format = data_format};
    const InterleavedAddrGenFast<out_is_dram_bool> sv = {
        .bank_base_address = v_tensor_addr, .page_size = single_tile_size_bytes, .data_format = data_format};

    constexpr uint32_t block_size = 1;  // micro-block size for read/write; nothing to do with num_blocks
    // TODO: This might negatively impact perf
    constexpr uint32_t out_num_tiles_read = block_size;  // always read and pop by micro-block size for generality
    uint32_t l1_read_addr;
    uint32_t q_out_tensor_current_tile_id;  // need this to update q_out_tensor_tile_id
    uint32_t k_out_tensor_current_tile_id;  // need this to update k_out_tensor_tile_id
    uint32_t v_out_tensor_current_tile_id;  // need this to update v_out_tensor_tile_id
    uint32_t out_tensor_current_tile_id_along_c;

    for (uint32_t block = 0; block < num_blocks_common; block++) {
        // q + create q head --> outputs: [B, num_q_heads, Sq, head_dim]
        process_q_block(
            cb_id_qv,
            sq,
            q_out_c,
            q_out_HtWt,
            q_out_w_tiles,
            q_out_h_tiles,
            out_num_tiles_read,
            q_out_h_dim,
            q_out_tensor_tile_id);

        // k + create k head --> outputs: [B, num_kv_heads, Skv, head_dim]
        process_kv_block(
            cb_id_k,
            cb_id_qv,
            sk,
            sv,
            kv_out_c,
            kv_out_HtWt,
            kv_out_w_tiles,
            kv_out_h_tiles,
            out_num_tiles_read,
            kv_out_h_dim,
            k_out_tensor_tile_id,
            v_out_tensor_tile_id);
    }
    // Handle remaining blocks
    for (uint32_t block = 0; block < num_blocks_q_remaining; block++) {
        // q + create q head --> outputs: [B, num_q_heads, Sq, head_dim]
        process_q_block(
            cb_id_qv,
            sq,
            q_out_c,
            q_out_HtWt,
            q_out_w_tiles,
            q_out_h_tiles,
            out_num_tiles_read,
            q_out_h_dim,
            q_out_tensor_tile_id);
    }
    for (uint32_t block = 0; block < num_blocks_kv_remaining; block++) {
        // k + create k head --> outputs: [B, num_kv_heads, Skv, head_dim]
        process_kv_block(
            cb_id_k,
            cb_id_qv,
            sk,
            sv,
            kv_out_c,
            kv_out_HtWt,
            kv_out_w_tiles,
            kv_out_h_tiles,
            out_num_tiles_read,
            kv_out_h_dim,
            k_out_tensor_tile_id,
            v_out_tensor_tile_id);
    }
}
