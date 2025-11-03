// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include "dataflow_api.h"

// Compile-time recursively calculates floor(log2(n))
constexpr int log2(uint32_t n) { return (n <= 1) ? 0 : 1 + log2(n >> 1); }

// This function is templated to choose the pointer data-type based on 'val' size
// to avoid unaligned addresses and out-of-bounds access.
template <uint32_t val_size>
FORCE_INLINE void fill_with_val(uint32_t begin_addr, uint32_t n_bytes, uint32_t val) {
    static_assert(val_size == 2 || val_size == 4, "Unsupported val_size");
    using IntType = std::conditional_t<(val_size == 2), uint16_t, uint32_t>;

    auto* ptr = reinterpret_cast<volatile tt_l1_ptr IntType*>(begin_addr);
    IntType val_ = static_cast<IntType>(val);
    constexpr uint32_t val_size_log2 = log2(val_size);
    uint32_t n = n_bytes >> val_size_log2;  // = n_bytes / sizeof(val)
    for (uint32_t i = 0; i < n; ++i) {
        ptr[i] = val_;
    }
}

void kernel_main() {
    constexpr uint32_t cb_id_in0 = 0;
    constexpr uint32_t tile_height = 32;

    constexpr uint32_t tile_row_shift_bits = get_compile_time_arg_val(0);
    constexpr uint32_t unpadded_X_size = get_compile_time_arg_val(1);
    constexpr uint32_t elem_size = get_compile_time_arg_val(2);
    constexpr auto src_args = TensorAccessorArgs<3>();

    const uint32_t src_addr = get_arg_val<uint32_t>(0);
    const uint32_t padded_X_size = get_arg_val<uint32_t>(1);
    const uint32_t pad_value = get_arg_val<uint32_t>(2);
    const uint32_t start_stick_id = get_arg_val<uint32_t>(3);
    const uint32_t n_block_reps = get_arg_val<uint32_t>(4);

    const uint32_t num_tiles_per_row =
        padded_X_size >> tile_row_shift_bits;  // means / 64, assuming bfloat16, there are 64 bytes per tile row

    const auto s = TensorAccessor(src_args, src_addr, unpadded_X_size);

    auto pad_blocks = [&](uint32_t num_blocks) {
        for (uint32_t i = 0; i < num_blocks; i++) {
            cb_reserve_back(cb_id_in0, num_tiles_per_row);
            uint32_t l1_write_addr = get_write_ptr(cb_id_in0);
            // pad the tile by reading values from zero buffer in L1
            fill_with_val<elem_size>(l1_write_addr, padded_X_size << 5, pad_value);  // "<< 5" = "* tile_height"
            cb_push_back(cb_id_in0, num_tiles_per_row);
        }
    };

    auto read_block = [&](uint32_t base_stick_id, uint32_t num_rows) {
        uint32_t padding_rows = (tile_height - num_rows) & 31;
        bool has_rows = (num_rows + padding_rows) > 0;

        cb_reserve_back(cb_id_in0, num_tiles_per_row * has_rows);
        uint32_t l1_write_addr = get_write_ptr(cb_id_in0);
        for (uint32_t k = 0; k < num_rows; k++) {
            uint64_t src_noc_addr = get_noc_addr(base_stick_id + k, s);

            // Read from DRAM to tmp buffer
            noc_async_read(src_noc_addr, l1_write_addr, unpadded_X_size);

            fill_with_val<elem_size>(l1_write_addr + unpadded_X_size, padded_X_size - unpadded_X_size, pad_value);

            // Block before copying data from tmp to cb buffer
            noc_async_read_barrier();
            l1_write_addr += padded_X_size;
        }

        fill_with_val<elem_size>(l1_write_addr, padding_rows * padded_X_size, pad_value);
        cb_push_back(cb_id_in0, num_tiles_per_row * has_rows);
    };

    uint32_t stick_id = start_stick_id;
    uint32_t rt_arg_idx = 5;
    uint32_t count = 1;
    constexpr int32_t n_mixed_idx = 1;
    constexpr int32_t n_pad_idx = 2;
    constexpr int32_t times_idx = 3;
    constexpr uint32_t repeat_ct_idx = 4;
    constexpr int32_t num_rt_idx = 5;

    for (uint32_t block_rep_idx = 0; block_rep_idx < n_block_reps; ++block_rep_idx) {
        const uint32_t repeat_count =
            get_arg_val<uint32_t>(rt_arg_idx + repeat_ct_idx);  // number of times the same block representation is used
        const uint32_t n_data = get_arg_val<uint32_t>(rt_arg_idx);  // number of full tile-rows
        const uint32_t n_mixed =
            get_arg_val<uint32_t>(rt_arg_idx + n_mixed_idx);  // number of rows in a partially filled tile-row
        const uint32_t n_pads = get_arg_val<uint32_t>(rt_arg_idx + n_pad_idx);  // number of padding tile-rows
        const uint32_t times =
            get_arg_val<uint32_t>(rt_arg_idx + times_idx);  // number of times the pattern of tile-rows repeats
        if (count == repeat_count) {
            rt_arg_idx = rt_arg_idx + num_rt_idx;
            count = 1;
        } else {
            count++;
        }
        for (uint32_t t = 0; t < times; ++t) {
            for (uint32_t y_t = 0; y_t < n_data; y_t++) {
                read_block(stick_id, tile_height);
                stick_id += tile_height;
            }

            read_block(stick_id, n_mixed);
            stick_id += n_mixed;

            pad_blocks(n_pads);
        }
    }
}
