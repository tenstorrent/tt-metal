// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cmath>
#include <stdint.h>
#include "dataflow_api.h"
#include "ttnn/cpp/ttnn/operations/conv/conv2d/device/kernels/height_sharded_reader_common.hpp"

#define ALWI inline __attribute__((always_inline))

ALWI void fill_four_val(uint32_t begin_addr, uint16_t val, uint16_t val1, uint16_t val2, uint16_t val3) {
    volatile tt_l1_ptr uint32_t* ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(begin_addr);

    ptr[0] = (val | (val1 << 16));
    ptr[1] = (val2 | (val3 << 16));
}

inline uint16_t float_to_bfloat16(float value) {
    uint32_t tmp;
    std::memcpy(&tmp, &value, sizeof(tmp));
    return static_cast<uint16_t>(tmp >> 16);
}

inline float bfloat16_to_float(uint16_t bf16) {
    uint32_t tmp = static_cast<uint32_t>(bf16) << 16;
    float result;
    std::memcpy(&result, &tmp, sizeof(result));
    return result;
}

void kernel_main() {
    uint32_t input_addr = get_arg_val<uint32_t>(0);
    uint32_t grid_addr = get_arg_val<uint32_t>(1);
    uint32_t num_pages = get_arg_val<uint32_t>(2);
    uint32_t start_page_id = get_arg_val<uint32_t>(3);

    constexpr uint32_t input_cb_index = get_compile_time_arg_val(0);
    constexpr uint32_t grid_cb_index = get_compile_time_arg_val(1);
    constexpr uint32_t scalar_cb_index = get_compile_time_arg_val(2);
    constexpr uint32_t input_stick_nbytes = get_compile_time_arg_val(3);
    constexpr uint32_t grid_stick_nbytes = get_compile_time_arg_val(4);
    constexpr uint32_t input_height = get_compile_time_arg_val(5);
    constexpr uint32_t input_width = get_compile_time_arg_val(6);
    constexpr uint32_t output_hw_size = get_compile_time_arg_val(7);

    constexpr auto src_args = TensorAccessorArgs<8>();
    constexpr auto grid_args = TensorAccessorArgs<src_args.next_compile_time_args_offset()>();

    const auto s0 = TensorAccessor(grid_args, grid_addr, grid_stick_nbytes);
    const auto s1 = TensorAccessor(src_args, input_addr, input_stick_nbytes);

    const uint32_t end_id = start_page_id + num_pages;

    constexpr float input_height_f = float(input_height);
    constexpr float input_width_f = float(input_width);

    constexpr float height_scale = input_height_f * 0.5f;
    constexpr float height_offset = height_scale - 0.5f;

    constexpr float width_scale = input_width_f * 0.5f;
    constexpr float width_offset = width_scale - 0.5f;

    /*
    In the case of grid sampling, we need to account for the fact that the grid coordinates may fall outside the bounds
    of the input image. Since the padding mode is zero, we would simply set the weights for the appropriate sticks to
    zero in the for loop, and simply do not read from DRAM. In that case the stick we send to reduction would be the
    last pixel that we read for the appropriate location (SW, SE, NW, NE), but since weights are 0 this is not a
    problem.

    However, if there was no previous read for the appropriate stick, the memory in that location is invalid, and could
    include NaN and Inf values. For that reason we zero out the input_cb at the start.
    */

    zero_out_tiles<input_cb_index>();

    for (uint32_t i = start_page_id; i < end_id; ++i) {
        uint32_t l1_write_grid_addr = get_write_ptr(grid_cb_index);
        uint64_t grid_noc_addr = s0.get_noc_addr(i);

        noc_async_read(grid_noc_addr, l1_write_grid_addr, grid_stick_nbytes);
        noc_async_read_barrier();

        // Read the first two bfloat16 values (grid coordinates) from the L1 buffer

        volatile tt_l1_ptr uint16_t* grid_ptr = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(l1_write_grid_addr);
        uint16_t weight_nw_bf, weight_ne_bf, weight_sw_bf, weight_se_bf;
        int32_t h0, h1, w0, w1;

#ifdef USE_PRECOMPUTED_GRID
        int16_t h0_raw = *reinterpret_cast<volatile int16_t*>(&grid_ptr[0]);
        int16_t w0_raw = *reinterpret_cast<volatile int16_t*>(&grid_ptr[1]);

        h0 = static_cast<int32_t>(h0_raw);
        w0 = static_cast<int32_t>(w0_raw);
        h1 = h0 + 1;
        w1 = w0 + 1;

#else
        uint16_t h_coord_raw = grid_ptr[1];
        uint16_t w_coord_raw = grid_ptr[0];

        float h_coord_rel = bfloat16_to_float(h_coord_raw);
        float w_coord_rel = bfloat16_to_float(w_coord_raw);

        float h_coord_image = h_coord_rel * height_scale + height_offset;
        float w_coord_image = w_coord_rel * width_scale + width_offset;

        h0 = static_cast<int32_t>(floor(h_coord_image));
        h1 = h0 + 1;
        w0 = static_cast<int32_t>(floor(w_coord_image));
        w1 = w0 + 1;

#endif

        bool h0_valid = (h0 >= 0) && (h0 < static_cast<int32_t>(input_height));
        bool h1_valid = (h1 >= 0) && (h1 < static_cast<int32_t>(input_height));
        bool w0_valid = (w0 >= 0) && (w0 < static_cast<int32_t>(input_width));
        bool w1_valid = (w1 >= 0) && (w1 < static_cast<int32_t>(input_width));

        uint32_t curr_batch = i / output_hw_size;
        uint32_t batch_offset = curr_batch * input_height * input_width;

        cb_reserve_back(input_cb_index, 1);
        uint32_t l1_write_input_addr = get_write_ptr(input_cb_index);

        {
            DeviceZoneScopedN("Read input sticks with shard optimization");

            // Calculate stick indices for all 4 corners
            uint32_t north_west_stick_index = batch_offset + (h0 * input_width) + w0;
            uint32_t north_east_stick_index = batch_offset + (h0 * input_width) + w1;
            uint32_t south_west_stick_index = batch_offset + (h1 * input_width) + w0;
            uint32_t south_east_stick_index = batch_offset + (h1 * input_width) + w1;

            // Check if h0w0 and h0w1 are on the same shard (north row)
            bool north_same_shard = false;
            bool south_same_shard = false;

            if (h0_valid && w0_valid && w1_valid) {
                auto mapping_nw = s1.get_bank_and_offset(north_west_stick_index);
                auto mapping_ne = s1.get_bank_and_offset(north_east_stick_index);
                north_same_shard = (mapping_nw.bank_id == mapping_ne.bank_id);
            }

            if (h1_valid && w0_valid && w1_valid) {
                auto mapping_sw = s1.get_bank_and_offset(south_west_stick_index);
                auto mapping_se = s1.get_bank_and_offset(south_east_stick_index);
                south_same_shard = (mapping_sw.bank_id == mapping_se.bank_id);
            }

            // Optimized reading for north row (h0w0 and h0w1)
            if (north_same_shard && h0_valid && w0_valid && w1_valid) {
                // Both north corners are on same shard - check if they're contiguous
                uint64_t nw_addr = s1.get_noc_addr(north_west_stick_index);
                uint64_t ne_addr = s1.get_noc_addr(north_east_stick_index);

                if (ne_addr == nw_addr + input_stick_nbytes) {
                    // Contiguous - read both in one transaction
                    noc_async_read(nw_addr, l1_write_input_addr, input_stick_nbytes * 2);
                } else {
                    // Same shard but not contiguous - separate reads
                    noc_async_read(nw_addr, l1_write_input_addr, input_stick_nbytes);
                    noc_async_read(ne_addr, l1_write_input_addr + input_stick_nbytes, input_stick_nbytes);
                }
            } else {
                // Different shards or invalid - individual reads
                if (h0_valid && w0_valid) {
                    uint64_t nw_addr = s1.get_noc_addr(north_west_stick_index);
                    noc_async_read(nw_addr, l1_write_input_addr, input_stick_nbytes);
                }
                if (h0_valid && w1_valid) {
                    uint64_t ne_addr = s1.get_noc_addr(north_east_stick_index);
                    noc_async_read(ne_addr, l1_write_input_addr + input_stick_nbytes, input_stick_nbytes);
                }
            }

            l1_write_input_addr += input_stick_nbytes * 2;

            // Optimized reading for south row (h1w0 and h1w1)
            if (south_same_shard && h1_valid && w0_valid && w1_valid) {
                // Both south corners are on same shard - check if they're contiguous
                uint64_t sw_addr = s1.get_noc_addr(south_west_stick_index);
                uint64_t se_addr = s1.get_noc_addr(south_east_stick_index);

                if (se_addr == sw_addr + input_stick_nbytes) {
                    // Contiguous - read both in one transaction
                    noc_async_read(sw_addr, l1_write_input_addr, input_stick_nbytes * 2);
                } else {
                    // Same shard but not contiguous - separate reads
                    noc_async_read(sw_addr, l1_write_input_addr, input_stick_nbytes);
                    noc_async_read(se_addr, l1_write_input_addr + input_stick_nbytes, input_stick_nbytes);
                }
            } else {
                // Different shards or invalid - individual reads
                if (h1_valid && w0_valid) {
                    uint64_t sw_addr = s1.get_noc_addr(south_west_stick_index);
                    noc_async_read(sw_addr, l1_write_input_addr, input_stick_nbytes);
                }
                if (h1_valid && w1_valid) {
                    uint64_t se_addr = s1.get_noc_addr(south_east_stick_index);
                    noc_async_read(se_addr, l1_write_input_addr + input_stick_nbytes, input_stick_nbytes);
                }
            }
        }

        // Calculate bilinear interpolation weights

#ifdef USE_PRECOMPUTED_GRID
        // Weights are already in grid data
        weight_nw_bf = grid_ptr[2];
        weight_ne_bf = grid_ptr[3];
        weight_sw_bf = grid_ptr[4];
        weight_se_bf = grid_ptr[5];
#else
        float h0_f = static_cast<float>(h0);
        float w0_f = static_cast<float>(w0);

        float h_frac = h_coord_image - h0_f;
        float w_frac = w_coord_image - w0_f;
        float h_frac_inv = 1.0f - h_frac;
        float w_frac_inv = 1.0f - w_frac;

        float weight_nw = (h0_valid && w0_valid) ? (h_frac_inv * w_frac_inv) : 0.0f;  // North-West
        float weight_ne = (h0_valid && w1_valid) ? (h_frac_inv * w_frac) : 0.0f;      // North-East
        float weight_sw = (h1_valid && w0_valid) ? (h_frac * w_frac_inv) : 0.0f;      // South-West
        float weight_se = (h1_valid && w1_valid) ? (h_frac * w_frac) : 0.0f;          // South-East

        weight_nw_bf = float_to_bfloat16(weight_nw);
        weight_ne_bf = float_to_bfloat16(weight_ne);
        weight_sw_bf = float_to_bfloat16(weight_sw);
        weight_se_bf = float_to_bfloat16(weight_se);
#endif

        cb_reserve_back(scalar_cb_index, 1);

        fill_four_val(get_write_ptr(scalar_cb_index), weight_nw_bf, weight_ne_bf, weight_sw_bf, weight_se_bf);

        cb_push_back(scalar_cb_index, 1);

        noc_async_read_barrier();

        cb_push_back(input_cb_index, 1);
    }
}
