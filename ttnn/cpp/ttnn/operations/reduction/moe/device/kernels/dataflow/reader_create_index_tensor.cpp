// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "experimental/noc.h"
#include "experimental/circular_buffer.h"
#include "experimental/core_local_mem.h"
#include "experimental/tensor.h"

/**
 * add a cb full of indices for the tile
 * each row is identical in the index tensor, so we just need to add an offset based on which row tile it is
 * first 32 elements are {0,..31}, then next 32 are {32,..64}
 * wt is which tile it is along the row [0, Wt) so j + 32*wt is the value in the tile at each element
 */
FORCE_INLINE void generate_index_tile(const uint32_t cb_id, const uint32_t wt) {
    // TODO: investigate moving to compile time (binary size is at risk)
    experimental::CircularBuffer cb(cb_id);
    cb.reserve_back(1);
    experimental::CoreLocalMem<volatile uint32_t> ptr(cb.get_write_ptr());
    uint16_t wt_offset = wt << 5;

    uint32_t count = 0;
    for (uint32_t i = 0; i < 2; ++i) {
        for (uint32_t j = 0; j < 2; ++j) {
            for (uint32_t k = 0; k < 16; ++k) {
                for (uint32_t l = 0; l < 16; l += 2) {
                    uint16_t value = l + 16 * j + wt_offset;
                    ptr[count] = (value + 1) << 16 | value;
                    count++;
                }
            }
        }
    }
    cb.push_back(1);
}

void kernel_main() {
    uint32_t src_addr = get_arg_val<uint32_t>(0);
    uint32_t topk_addr = get_arg_val<uint32_t>(1);
    uint32_t expert_addr = get_arg_val<uint32_t>(2);

    constexpr uint32_t cb_id_in0 = get_compile_time_arg_val(0);
    constexpr uint32_t cb_intermed_index = get_compile_time_arg_val(1);
    constexpr uint32_t cb_topk_mask = get_compile_time_arg_val(2);
    constexpr uint32_t cb_expert_mask = get_compile_time_arg_val(3);

    constexpr uint32_t Ht = get_compile_time_arg_val(4);
    constexpr uint32_t Wt = get_compile_time_arg_val(5);
    constexpr uint32_t K = get_compile_time_arg_val(6);
    constexpr uint32_t Kt = K % 32 == 0 ? K / 32 : K / 32 + 1;

    constexpr auto s0_args = TensorAccessorArgs<7>();
    constexpr auto s1_args = TensorAccessorArgs<s0_args.next_compile_time_args_offset()>();
    constexpr auto s2_args = TensorAccessorArgs<s1_args.next_compile_time_args_offset()>();

    constexpr uint32_t onetile = 1;
    constexpr uint32_t tile_bytes_input = get_tile_size(cb_id_in0);

    const auto s0 = TensorAccessor(s0_args, src_addr, tile_bytes_input);

    constexpr uint32_t tile_bytes_topk = get_tile_size(cb_topk_mask);

    const auto s1 = TensorAccessor(s1_args, topk_addr, tile_bytes_topk);

    constexpr uint32_t tile_bytes_expert = get_tile_size(cb_expert_mask);

    const auto s2 = TensorAccessor(s2_args, expert_addr, tile_bytes_expert);

    experimental::Noc noc;
    experimental::CircularBuffer cb_in0(cb_id_in0);
    experimental::CircularBuffer cb_topk(cb_topk_mask);
    experimental::CircularBuffer cb_expert(cb_expert_mask);

    // Stream in input tensor, buffer has four tiles as we double-buffer to continue streaming while waiting for compute
    // and we need two tiles for the bitonic sort llk We could load in an entire row of tiles at a time but that would
    // require substantially more memory (we would be double buffering four Wt sized CBs)

    uint32_t tile_id = 0;
    uint32_t tile_id_topk = 0;
    uint32_t tile_id_expert = 0;
    for (uint32_t i = 0; i < Ht; ++i) {
        // input
        cb_in0.reserve_back(Wt);
        for (uint32_t j = 0; j < Wt; ++j) {
            noc.async_read(s0, cb_in0, tile_bytes_input, {.page_id = tile_id}, {.offset_bytes = j * tile_bytes_input});
            tile_id++;
            generate_index_tile(cb_intermed_index, j);
        }
        noc.async_read_barrier();
        cb_in0.push_back(Wt);

        // topk mask
        cb_topk.reserve_back(Kt);
        for (uint32_t j = 0; j < Kt; ++j) {
            noc.async_read(
                s1, cb_topk, tile_bytes_topk, {.page_id = tile_id_topk}, {.offset_bytes = j * tile_bytes_topk});
            tile_id_topk++;
        }
        noc.async_read_barrier();
        cb_topk.push_back(Kt);

        // expert mask
        cb_expert.reserve_back(Wt);
        for (uint32_t j = 0; j < Wt; ++j) {
            noc.async_read(
                s2, cb_expert, tile_bytes_expert, {.page_id = tile_id_expert}, {.offset_bytes = j * tile_bytes_expert});
            tile_id_expert++;
        }
        noc.async_read_barrier();
        cb_expert.push_back(Wt);
    }
}
