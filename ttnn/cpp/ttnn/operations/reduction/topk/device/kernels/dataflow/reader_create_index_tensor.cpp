// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"

#include <stdint.h>

/**
 * @brief Generates an index tile and writes it to a circular buffer.
 *
 * This function creates a tile of indices based on the specified tile number and output format.
 * The generated indices follow a specific pattern across multiple nested loops and are written
 * to the circular buffer identified by cb_id.
 *
 * @param cb_id The circular buffer identifier where the index tile will be written
 * @param wt The tile number (width tile index) used to calculate the base offset for indices
 * @param uint16_output If true, generates 16-bit packed indices (two uint16_t values per uint32_t);
 *                      if false, generates 32-bit indices (one uint32_t value per entry)
 *
 * @note The function reserves space in the circular buffer, writes the generated indices,
 *       and then pushes the data back to the buffer.
 * @note For uint16_output mode, indices are packed as pairs with consecutive values.
 * @note The total number of indices generated is 1024 (2×2×16×16) regardless of output format.
 */
FORCE_INLINE void generate_index_tile(const uint32_t cb_id, const uint32_t wt, const bool uint16_output) {
    cb_reserve_back(cb_id, 1);
    uint32_t write_addr = get_write_ptr(cb_id);
    volatile tt_l1_ptr uint32_t* ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(write_addr);

    if (uint16_output) {
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
    } else {
        uint32_t wt_offset = wt << 5;

        uint32_t count = 0;
        for (uint32_t i = 0; i < 2; ++i) {
            for (uint32_t j = 0; j < 2; ++j) {
                for (uint32_t k = 0; k < 16; ++k) {
                    for (uint32_t l = 0; l < 16; l++) {
                        uint32_t value = l + 16 * j + wt_offset;
                        ptr[count] = value;
                        count++;
                    }
                }
            }
        }
    }
    cb_push_back(cb_id, 1);
}

void kernel_main() {
    // Runtime arguments
    const uint32_t src_addr = get_arg_val<uint32_t>(0);
    const uint32_t id = get_arg_val<uint32_t>(1);
    const uint32_t work_per_core = get_arg_val<uint32_t>(2);

    // Compile time arguments
    constexpr uint32_t cb_id_in0 = get_compile_time_arg_val(0);
    constexpr uint32_t cb_intermed_index = get_compile_time_arg_val(1);
    constexpr uint32_t Ht = get_compile_time_arg_val(2);
    constexpr uint32_t Wt = get_compile_time_arg_val(3);
    constexpr uint32_t total_number_of_cores = get_compile_time_arg_val(4);
    constexpr bool uint16_output = get_compile_time_arg_val(5) == 1;
    constexpr auto s_args = TensorAccessorArgs<6>();

    // Constants
    constexpr uint32_t onetile = 1;

    // Tensor accessor
    constexpr uint32_t tile_bytes = get_tile_size(cb_id_in0);
    const auto s = TensorAccessor(s_args, src_addr, tile_bytes);

    // Read data and generate indices
    for (uint32_t core_loop = 0; core_loop < work_per_core; core_loop++) {
        const uint32_t i = id + core_loop * total_number_of_cores;
        for (uint32_t j = 0; j < Wt; ++j) {
            cb_reserve_back(cb_id_in0, onetile);
            const uint32_t l1_write_addr = get_write_ptr(cb_id_in0);
            noc_async_read_tile(i * Wt + j, s, l1_write_addr);
            noc_async_read_barrier();
            cb_push_back(cb_id_in0, onetile);
            generate_index_tile(cb_intermed_index, j, uint16_output);
        }
    }
}
