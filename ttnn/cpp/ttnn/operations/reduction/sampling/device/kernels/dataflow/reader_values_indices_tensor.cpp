// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"
/**
 * add a cb full of indices for the tile
 * each row is identical in the index tensor, so we just need to add an offset based on which row tile it is
 * first 32 elements are {0,..31}, then next 32 are {32,..64}
 * wt is which tile it is along the row [0, Wt) so j + 32*wt is the value in the tile at each element
 */
// USE_32BIT == false (WH/BH): pack two 16-bit indices per 32-bit word (UInt16 index tile, LO16
// dest mode). USE_32BIT == true (Quasar): write one 32-bit index per element (Int32 index tile,
// INT32 dest mode). The index width must match the top-k LLK's fp32_dest_acc_en setting.
template <bool USE_32BIT>
FORCE_INLINE void generate_index_tile(const uint32_t cb_id, const uint32_t wt) {
    // TODO: investigate moving to compile time (binary size is at risk)
    CircularBuffer cb(cb_id);
    cb.reserve_back(1);
    CoreLocalMem<volatile uint32_t> ptr(cb.get_write_ptr());
    uint32_t wt_offset = wt << 5;

    uint32_t count = 0;
    for (uint32_t i = 0; i < 2; ++i) {
        for (uint32_t j = 0; j < 2; ++j) {
            for (uint32_t k = 0; k < 16; ++k) {
                if constexpr (USE_32BIT) {
                    for (uint32_t l = 0; l < 16; ++l) {
                        uint32_t value = l + 16 * j + wt_offset;
                        ptr[count] = value;
                        count++;
                    }
                } else {
                    for (uint32_t l = 0; l < 16; l += 2) {
                        uint16_t value = l + 16 * j + wt_offset;
                        ptr[count] = (value + 1) << 16 | value;
                        count++;
                    }
                }
            }
        }
    }
    cb.push_back(1);
}

void kernel_main() {
    uint32_t values_addr = get_arg_val<uint32_t>(0);
    uint32_t indices_addr = get_arg_val<uint32_t>(1);

    constexpr uint32_t input_values_cb_index = get_compile_time_arg_val(0);
    constexpr uint32_t input_indices_cb_index = get_compile_time_arg_val(1);
    constexpr uint32_t cb_intermed_index = get_compile_time_arg_val(2);

    constexpr uint32_t Ht = get_compile_time_arg_val(3);
    constexpr uint32_t Wt = get_compile_time_arg_val(4);
    constexpr uint32_t input_indices_page_size = get_compile_time_arg_val(5);
    constexpr uint32_t tile_height = get_compile_time_arg_val(6);
    constexpr bool use_32bit_index = get_compile_time_arg_val(7) == 1;
    // Number of logical users (== number of running cores). Only this many input-index rows exist
    // and are streamed in, even though the values tile is padded to a full tile_height.
    constexpr uint32_t num_users = get_compile_time_arg_val(8);

    constexpr auto s0_args = TensorAccessorArgs<9>();
    constexpr auto s1_args = TensorAccessorArgs<s0_args.next_compile_time_args_offset()>();

    // ublocks size defined in tiles
    constexpr uint32_t onetile = 1;

    const auto s0 = TensorAccessor(s0_args, values_addr);

    const auto s1 = TensorAccessor(s1_args, indices_addr);

    Noc noc;
    CircularBuffer input_values_cb(input_values_cb_index);
    CircularBuffer input_indices_cb(input_indices_cb_index);
    const uint32_t tile_bytes_input_values = input_values_cb.get_tile_size();

    uint32_t tile_id_input_values = 0;
    uint32_t tile_id_input_indices = 0;
    for (uint32_t i = 0; i < Ht; ++i) {
        // input values TILE
        for (uint32_t j = 0; j < Wt; ++j) {
            input_values_cb.reserve_back(onetile);
            noc.async_read(
                s0, input_values_cb, tile_bytes_input_values, {.page_id = tile_id_input_values}, {.offset_bytes = 0});
            tile_id_input_values++;
            generate_index_tile<use_32bit_index>(cb_intermed_index, j);
            noc.async_read_barrier();
            input_values_cb.push_back(onetile);
        }
    }

    // input indices RM — push one stick per running core/user. Previously hard-coded to
    // Ht * tile_height (== 32); now `num_users` so fewer-than-32-user configs don't over-read.
    for (uint32_t j = 0; j < num_users; ++j) {
        input_indices_cb.reserve_back(onetile);
        noc.async_read(s1, input_indices_cb, input_indices_page_size, {.page_id = j}, {.offset_bytes = 0});
        noc.async_read_barrier();
        input_indices_cb.push_back(onetile);
    }
}
