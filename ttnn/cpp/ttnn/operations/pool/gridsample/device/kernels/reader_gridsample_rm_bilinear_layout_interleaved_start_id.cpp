// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "dataflow_api.h"
#include <tuple>
#include <cmath>
#include <cstring>
#define ALWI inline __attribute__((always_inline))

ALWI bool fill_with_val(uint32_t begin_addr, float val) {
    volatile tt_l1_ptr float* ptr = reinterpret_cast<volatile tt_l1_ptr float*>(begin_addr);
    for (uint32_t i = 0; i < 32; ++i) {
        ptr[i] = (val);
    }
    return true;
}
float lies_in_boundary(float x_in, float y_in, int row, int col, int offset, uint32_t data) {
    if (x_in < 0 || y_in > row - 1 || y_in < 0 || x_in > col - 1) {
        return 0;
    }

    uint32_t cur_pos = offset + (int)y_in * col + (int)x_in;

    float* ptr = (float*)(data);

    float x = (ptr[cur_pos]);

    return x;
}

bool interpolate(float x_in, float y_in, int row, int col, int offset, uint32_t data_writer, uint32_t dest_addr) {
    if (x_in == std::numeric_limits<float>::max() || y_in == std::numeric_limits<float>::max()) {
        fill_with_val(dest_addr, (0));

        return true;
    }

    if (x_in - (int)x_in == 0 && y_in - (int)y_in == 0) {
        float val = lies_in_boundary((int)x_in, (int)y_in, row, col, offset, data_writer);
        fill_with_val(dest_addr, (val));
        return true;
    }

    float dy = y_in - floor(y_in);
    float dx = x_in - floor(x_in);

    std::tuple<float, float> top_left = std::make_tuple(floor(x_in), floor(y_in));
    std::tuple<float, float> top_right = std::make_tuple(ceil(x_in), floor(y_in));
    std::tuple<float, float> bottom_left = std::make_tuple(floor(x_in), ceil(y_in));
    std::tuple<float, float> bottom_right = std::make_tuple(ceil(x_in), ceil(y_in));

    float Q11 = lies_in_boundary(
        std::get<0>(top_left),
        std::get<1>(top_left),
        row,
        col,
        offset,
        data_writer);  // returns the respective value in the Input Tensor or returns 0
    float Q21 = lies_in_boundary(std::get<0>(top_right), std::get<1>(top_right), row, col, offset, data_writer);
    float Q12 = lies_in_boundary(std::get<0>(bottom_left), std::get<1>(bottom_left), row, col, offset, data_writer);
    float Q22 = lies_in_boundary(std::get<0>(bottom_right), std::get<1>(bottom_right), row, col, offset, data_writer);

    float val = (1 - dx) * (1 - dy) * Q11 + dx * (1 - dy) * Q21 + (1 - dx) * dy * Q12 +
                dx * dy * Q22;  // Formula to perform interpolate

    fill_with_val(dest_addr, (val));

    return true;
}

void kernel_main() {
    constexpr uint32_t cb_id_in0 = get_compile_time_arg_val(0);
    constexpr bool src0_is_dram = get_compile_time_arg_val(1) == 1;
    constexpr uint32_t channels = get_compile_time_arg_val(2);
    constexpr uint32_t batch_size = get_compile_time_arg_val(3);
    constexpr uint32_t total_count = get_compile_time_arg_val(4);
    constexpr uint32_t Gridoffset = get_compile_time_arg_val(5);
    constexpr uint32_t row = get_compile_time_arg_val(6);
    constexpr uint32_t col = get_compile_time_arg_val(7);

    uint32_t src_addr = get_arg_val<uint32_t>(0);   // Input address
    uint32_t dst_addr = get_arg_val<uint32_t>(1);   //  destination address
    uint32_t grid_addr = get_arg_val<uint32_t>(2);  // Grid address

    uint32_t volume_size = get_arg_val<uint32_t>(3);
    uint32_t element_size = get_arg_val<uint32_t>(4);
    uint32_t grid_size = get_arg_val<uint32_t>(5);

    const InterleavedAddrGen<src0_is_dram> s0 = {.bank_base_address = src_addr, .page_size = volume_size};  // source
    const InterleavedAddrGen<src0_is_dram> s1 = {
        .bank_base_address = dst_addr, .page_size = element_size};  // destination
    const InterleavedAddrGen<src0_is_dram> s2 = {.bank_base_address = grid_addr, .page_size = grid_size};  // grid

    uint32_t idx = 0;

    for (uint32_t b = 0; b < batch_size; ++b) {
        int offset = b * Gridoffset;  // move the offset to next batch

        for (uint32_t c = 0; c < channels; ++c) {
            for (uint32_t i = 0; i < total_count; i += 2) {
                cb_reserve_back(2, 1);

                uint32_t l1_grid_writer_addr = get_write_ptr(2);  // Read the entire input tensor in the cb : 2

                uint64_t grid_noc_addr = get_noc_addr(0, s2);

                noc_async_read(grid_noc_addr, l1_grid_writer_addr, grid_size);

                noc_async_read_barrier();

                float* grid_ptr = (float*)(l1_grid_writer_addr);

                float x = (grid_ptr[(offset + i)]);
                float y = (grid_ptr[(offset + i + 1)]);

                cb_pop_front(2, 1);

                // Starting point of the element in the tensor for particular batch size, channel

                int offset = b * (channels * row * col) + c * (row * col);

                cb_reserve_back(1, 1);

                uint32_t l1_src_writer_addr = get_write_ptr(1);  // Read the entire input tensor in the cb : 1

                uint64_t src_noc_addr = get_noc_addr(0, s0);

                noc_async_read(src_noc_addr, l1_src_writer_addr, volume_size);

                noc_async_read_barrier();

                uint32_t l1_dst_writer_addr = get_write_ptr(cb_id_in0);

                interpolate(x, y, row, col, offset, l1_src_writer_addr, l1_dst_writer_addr);

                uint32_t l1_read_addr = get_read_ptr(cb_id_in0);

                uint64_t dst_noc_addr = get_noc_addr(idx, s1);

                noc_async_write(l1_read_addr, dst_noc_addr, element_size);

                noc_async_write_barrier();
                idx++;

                cb_pop_front(1, 1);
            }
        }
    }
}
