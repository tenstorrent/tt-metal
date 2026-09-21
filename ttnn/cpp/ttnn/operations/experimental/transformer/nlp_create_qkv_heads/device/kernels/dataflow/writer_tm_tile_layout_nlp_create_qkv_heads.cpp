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
    uint32_t k_tensor_addr = get_arg_val<uint32_t>(1);
    uint32_t v_tensor_addr = get_arg_val<uint32_t>(2);
    uint32_t num_blocks = get_arg_val<uint32_t>(3);
    uint32_t q_out_h_dim = get_arg_val<uint32_t>(4);
    uint32_t q_out_tensor_tile_id = get_arg_val<uint32_t>(5);
    uint32_t k_out_tensor_tile_id = get_arg_val<uint32_t>(6);
    uint32_t v_out_tensor_tile_id = get_arg_val<uint32_t>(7);

    // COMPILE TIME ARGS
    constexpr uint32_t q_out_h_tiles = get_compile_time_arg_val(0);
    constexpr uint32_t q_out_w_tiles = get_compile_time_arg_val(1);
    constexpr uint32_t q_out_HtWt = get_compile_time_arg_val(2);
    constexpr uint32_t q_out_c = get_compile_time_arg_val(3);
    constexpr uint32_t kv_out_c = get_compile_time_arg_val(4);
    constexpr bool head_parallel = get_compile_time_arg_val(5) != 0;
    constexpr uint32_t split_width = get_compile_time_arg_val(6);  // zero keeps the full Q output
    constexpr auto q_args = TensorAccessorArgs<7>();
    constexpr auto k_args = TensorAccessorArgs<q_args.next_compile_time_args_offset()>();
    constexpr auto v_args = TensorAccessorArgs<k_args.next_compile_time_args_offset()>();

    constexpr uint32_t cb_id_qv = 1;  // cb for Q, V heads tiles
#ifdef TRANSPOSE_K_HEADS
    constexpr uint32_t cb_id_k = 16;  // cb for K heads (filled by compute)
#else
    constexpr uint32_t cb_id_k = 1;  // cb for K heads (directly from reader)
#endif
    const auto sq = TensorAccessor(q_args, q_tensor_addr);
    const auto sk = TensorAccessor(k_args, k_tensor_addr);
    const auto sv = TensorAccessor(v_args, v_tensor_addr);

    CircularBuffer cb_qv(cb_id_qv);
    CircularBuffer cb_k(cb_id_k);

    const uint32_t tile_bytes_qv = get_tile_size(cb_id_qv);
    const uint32_t tile_bytes_k = get_tile_size(cb_id_k);

    constexpr uint32_t block_size = 1;  // micro-block size for read/write; nothing to do with num_blocks
    // TODO: This might negatively impact perf
    constexpr uint32_t out_num_tiles_read = block_size;  // always read and pop by micro-block size for generality
    uint32_t l1_read_addr;
    uint32_t q_out_tensor_current_tile_id;  // need this to update q_out_tensor_tile_id
    uint32_t k_out_tensor_current_tile_id;  // need this to update k_out_tensor_tile_id
    uint32_t v_out_tensor_current_tile_id;  // need this to update v_out_tensor_tile_id
    uint32_t out_tensor_current_tile_id_along_c;

    // Keep reader/work partitioning in full-head coordinates; route each tile once at the writer.
    const auto write_q_tile = [&](uint32_t source, uint32_t full_tile_id) {
        if constexpr (split_width > 0) {
            constexpr uint32_t left_width = split_width;
            constexpr uint32_t right_width = q_out_w_tiles - left_width;
            const uint32_t row = full_tile_id / q_out_w_tiles;
            const uint32_t col = full_tile_id % q_out_w_tiles;
            if (col < left_width) {
                noc.async_write(
                    CoreLocalMem<uint32_t>(source), sq, tile_bytes_qv, {}, {.page_id = row * left_width + col});
            } else {
                noc.async_write(
                    CoreLocalMem<uint32_t>(source),
                    sk,
                    tile_bytes_qv,
                    {},
                    {.page_id = row * right_width + col - left_width});
            }
        } else {
            noc.async_write(CoreLocalMem<uint32_t>(source), sq, tile_bytes_qv, {}, {.page_id = full_tile_id});
        }
    };

    if constexpr (head_parallel) {
        constexpr uint32_t transfer_tiles = q_out_w_tiles % 4 == 0 ? 4 : (q_out_w_tiles % 2 == 0 ? 2 : 1);
        for (uint32_t tile = 0; tile < num_blocks * q_out_w_tiles; tile += transfer_tiles) {
            cb_qv.wait_front(transfer_tiles);
            uint32_t source = cb_qv.get_read_ptr();
            for (uint32_t j = 0; j < transfer_tiles; ++j) {
                write_q_tile(source, q_out_tensor_tile_id++);
                source += tile_bytes_qv;
            }
            noc.async_write_barrier();
            cb_qv.pop_front(transfer_tiles);
        }
    } else {
        for (uint32_t block = 0; block < num_blocks; block++) {
            // q + create q head --> outputs: [B, num_q_heads, S, head_dim]
            out_tensor_current_tile_id_along_c = q_out_tensor_tile_id;
            for (uint32_t c_dim = 0; c_dim < q_out_c; c_dim++) {
                q_out_tensor_current_tile_id = out_tensor_current_tile_id_along_c;
                for (uint32_t w_dim = 0; w_dim < q_out_w_tiles; w_dim++) {
                    cb_qv.wait_front(out_num_tiles_read);
                    l1_read_addr = cb_qv.get_read_ptr();
                    write_q_tile(l1_read_addr, q_out_tensor_current_tile_id);

                    noc.async_write_barrier();
                    cb_qv.pop_front(out_num_tiles_read);

                    q_out_tensor_current_tile_id++;
                }
                out_tensor_current_tile_id_along_c += q_out_HtWt;
            }

// k + create k head --> outputs: [B, num_kv_heads, S, head_dim]
#ifndef TRANSPOSE_K_HEADS
            out_tensor_current_tile_id_along_c = k_out_tensor_tile_id;
#else
            k_out_tensor_current_tile_id = k_out_tensor_tile_id;
#endif
            for (uint32_t c_dim = 0; c_dim < kv_out_c; c_dim++) {
#ifndef TRANSPOSE_K_HEADS
                k_out_tensor_current_tile_id = out_tensor_current_tile_id_along_c;
#endif
                for (uint32_t w_dim = 0; w_dim < q_out_w_tiles; w_dim++) {
                    cb_k.wait_front(out_num_tiles_read);
                    l1_read_addr = cb_k.get_read_ptr();
                    noc.async_write(
                        CoreLocalMem<uint32_t>(l1_read_addr),
                        sk,
                        tile_bytes_k,
                        {},
                        {.page_id = k_out_tensor_current_tile_id});

                    noc.async_write_barrier();
                    cb_k.pop_front(out_num_tiles_read);

#ifndef TRANSPOSE_K_HEADS
                    k_out_tensor_current_tile_id++;
#else
                    k_out_tensor_current_tile_id += q_out_h_tiles;
#endif
                }
#ifndef TRANSPOSE_K_HEADS
                out_tensor_current_tile_id_along_c += q_out_HtWt;
#endif
            }

            // v + create v head --> outputs: [B, num_kv_heads, S, head_dim]
            out_tensor_current_tile_id_along_c = v_out_tensor_tile_id;
            for (uint32_t c_dim = 0; c_dim < kv_out_c; c_dim++) {
                v_out_tensor_current_tile_id = out_tensor_current_tile_id_along_c;
                for (uint32_t w_dim = 0; w_dim < q_out_w_tiles; w_dim++) {
                    cb_qv.wait_front(out_num_tiles_read);
                    l1_read_addr = cb_qv.get_read_ptr();
                    noc.async_write(
                        CoreLocalMem<uint32_t>(l1_read_addr),
                        sv,
                        tile_bytes_qv,
                        {},
                        {.page_id = v_out_tensor_current_tile_id});

                    noc.async_write_barrier();
                    cb_qv.pop_front(out_num_tiles_read);

                    v_out_tensor_current_tile_id++;
                }
                out_tensor_current_tile_id_along_c += q_out_HtWt;
            }

            // Update out_tensor_tile_id for next h_dim or batch if we finish one CHtWt
            q_out_h_dim++;
            if (q_out_h_dim < q_out_h_tiles) {
                q_out_tensor_tile_id += q_out_w_tiles;
#ifndef TRANSPOSE_K_HEADS
                k_out_tensor_tile_id += q_out_w_tiles;
#else
                k_out_tensor_tile_id++;
#endif
                v_out_tensor_tile_id += q_out_w_tiles;
            } else {
                // If we finish one batch, always roll over to next tile in memory
                // This is just the current_tile_id, except for K when we transpose heads
                // In this case, decrement k_out_tensor_current_tile_id by the stride (q_out_h_tiles) and add 1 to roll
                // over
                q_out_tensor_tile_id = q_out_tensor_current_tile_id;
#ifndef TRANSPOSE_K_HEADS
                k_out_tensor_tile_id = k_out_tensor_current_tile_id;
#else
                k_out_tensor_tile_id =
                    ++k_out_tensor_current_tile_id - q_out_h_tiles;  // inc by 1 and decrement by stride
#endif
                v_out_tensor_tile_id = v_out_tensor_current_tile_id;
                q_out_h_dim = 0;
            }
        }
    }
}
