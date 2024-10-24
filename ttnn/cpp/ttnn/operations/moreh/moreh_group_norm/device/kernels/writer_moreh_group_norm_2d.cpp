// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/cpp/ttnn/deprecated/tt_dnn/kernels/dataflow/moreh_common.hpp"

template <typename T>
FORCE_INLINE void write_output(
    uint32_t cb_id, T addrgen, uint32_t write_start, uint32_t write_end, uint32_t nt, uint32_t Ct, uint32_t N) {
    constexpr uint32_t onetile = 1;
    const auto elem_size = get_tile_size(cb_id) >> 10;

    cb_wait_front(cb_id, onetile);
    auto l1_read_addr = get_read_ptr(cb_id);

    uint32_t ct = write_start / TILE_WIDTH;
    auto noc_id = nt * Ct + ct;
    auto noc_addr = get_noc_addr(noc_id, addrgen);

    // write left face
    if (write_start % TILE_WIDTH < FACE_WIDTH) {
        // write [tile_start, tile_end)
        uint32_t tile_start = write_start;
        uint32_t tile_end = std::min(round_down(write_start, TILE_WIDTH) + FACE_WIDTH, write_end);

        uint32_t n_start = 0;
        uint32_t n_end = std::min((nt + 1) * TILE_HEIGHT, N);
        auto noc_write_size = (tile_end - tile_start) * elem_size;
        for (uint32_t n = n_start; n < n_end; n++) {
            uint32_t tilized_idx = get_tilized_idx(n, tile_start);
            uint32_t offset = tilized_idx * elem_size;

            noc_async_write(l1_read_addr + offset, noc_addr + offset, noc_write_size);
        }
    }  // write left face

    // write right face
    if ((write_end + TILE_WIDTH - 1) % TILE_WIDTH >= FACE_WIDTH) {
        // write [tile_start, tile_end)
        uint32_t tile_start = std::max(round_down(write_start, TILE_WIDTH) + FACE_WIDTH, write_start);
        uint32_t tile_end = write_end;

        uint32_t n_start = 0;
        uint32_t n_end = std::min((nt + 1) * TILE_HEIGHT, N);
        auto noc_write_size = (tile_end - tile_start) * elem_size;
        for (uint32_t n = n_start; n < n_end; n++) {
            uint32_t tilized_idx = get_tilized_idx(n, tile_start);
            uint32_t offset = tilized_idx * elem_size;

            noc_async_write(l1_read_addr + offset, noc_addr + offset, noc_write_size);
        }
    }  // write right face

    noc_async_write_barrier();

    cb_pop_front(cb_id, onetile);
}

void kernel_main() {
    int i{0};
    const auto output_addr = get_arg_val<uint32_t>(i++);
    const auto mean_addr = get_arg_val<uint32_t>(i++);
    const auto rstd_addr = get_arg_val<uint32_t>(i++);

    const auto unit_offset = get_arg_val<uint32_t>(i++);
    const auto num_units_per_core = get_arg_val<uint32_t>(i++);

    const auto N = get_arg_val<uint32_t>(i++);
    const auto C = get_arg_val<uint32_t>(i++);
    const auto num_groups = get_arg_val<uint32_t>(i++);

    // compile-time args
    constexpr bool output_is_dram = get_compile_time_arg_val(0) == 1;
    constexpr bool mean_has_value = get_compile_time_arg_val(1) == 1;
    constexpr bool mean_is_dram = get_compile_time_arg_val(2) == 1;
    constexpr bool rstd_has_value = get_compile_time_arg_val(3) == 1;
    constexpr bool rstd_is_dram = get_compile_time_arg_val(4) == 1;

    uint32_t cb_id = tt::CB::c_out0;
    const auto cb_output = cb_id++;
    const auto cb_mean = cb_id++;
    const auto cb_rstd = cb_id++;

    const auto Ct = div_up(C, TILE_WIDTH);

    // output
    const uint32_t output_tile_bytes = get_tile_size(cb_output);
    const auto output_data_format = get_dataformat(cb_output);
    const auto output_elem_size = get_tile_size(cb_output) >> 10;

    const InterleavedAddrGenFast<output_is_dram> output_addrgen = {
        .bank_base_address = output_addr, .page_size = output_tile_bytes, .data_format = output_data_format};

    constexpr uint32_t onetile = 1;

    for (uint32_t outer_idx = 0; outer_idx < num_units_per_core; ++outer_idx) {
        // input[N, C]
        // reshaped_input[N, num_groups, C / num_groups]

        auto unit_idx = unit_offset + outer_idx;
        auto group_idx = unit_idx % num_groups;
        auto nt = unit_idx / num_groups;

        uint32_t c_size_per_unit = C / num_groups;
        uint32_t c_start = c_size_per_unit * group_idx;
        uint32_t c_end = c_size_per_unit * (group_idx + 1);

        uint32_t write_start = c_start;
        uint32_t write_end = c_start;
        for (;;) {
            // write range [c_start, c_end)
            write_start = write_end;
            write_end = std::min(round_up(write_start + 1, TILE_WIDTH), c_end);

            if (write_start >= c_end) {
                break;
            }

            bool is_tile_end = (write_end % TILE_WIDTH == 0);
            bool is_last = (write_end == c_end);
            if (is_tile_end || is_last) {
                write_output(cb_output, output_addrgen, write_start, write_end, nt, Ct, N);
            }
        }

        // mean
        // shape: [N, num_groups]
        if (mean_has_value) {
            const uint32_t mean_tile_bytes = get_tile_size(cb_mean);
            const auto mean_data_format = get_dataformat(cb_mean);
            const auto mean_dtype_bytes = mean_tile_bytes / (TILE_HEIGHT * TILE_WIDTH);

            const InterleavedAddrGenFast<mean_is_dram> mean_addrg = {
                .bank_base_address = mean_addr, .page_size = mean_tile_bytes, .data_format = mean_data_format};

            cb_wait_front(cb_mean, onetile);
            const auto mean_l1_read_ptr = get_read_ptr(cb_mean);

            // shift [n, 0] -> [n, group_idx]
            if (group_idx % TILE_WIDTH != 0) {
                auto mean_ptr = reinterpret_cast<uint16_t *>(mean_l1_read_ptr);
                uint32_t n_start = 0;
                uint32_t n_end = std::min((nt + 1) * TILE_HEIGHT, N);
                for (uint32_t n = n_start; n < n_end; n++) {
                    uint32_t src_idx = get_tilized_idx(n, 0);
                    uint32_t dst_idx = get_tilized_idx(n, group_idx);

                    mean_ptr[dst_idx] = mean_ptr[src_idx];
                }
            }

            uint32_t noc_id = nt * div_up(num_groups, TILE_WIDTH) + (group_idx / TILE_WIDTH);
            const auto mean_noc_addr = get_noc_addr(noc_id, mean_addrg);
            uint32_t n_start = 0;
            uint32_t n_end = std::min((nt + 1) * TILE_HEIGHT, N);

            for (uint32_t n = n_start; n < n_end; n++) {
                uint32_t dst_idx = get_tilized_idx(n, group_idx);
                noc_async_write(
                    mean_l1_read_ptr + dst_idx * mean_dtype_bytes,
                    mean_noc_addr + dst_idx * mean_dtype_bytes,
                    mean_dtype_bytes);
            }
            noc_async_write_barrier();
            cb_pop_front(cb_mean, onetile);
        }

        // rstd
        // shape: [N, num_groups]
        if (rstd_has_value) {
            const uint32_t rstd_tile_bytes = get_tile_size(cb_rstd);
            const auto rstd_data_format = get_dataformat(cb_rstd);
            const auto rstd_dtype_bytes = rstd_tile_bytes / (TILE_HEIGHT * TILE_WIDTH);

            const InterleavedAddrGenFast<rstd_is_dram> rstd_addrg = {
                .bank_base_address = rstd_addr, .page_size = rstd_tile_bytes, .data_format = rstd_data_format};

            cb_wait_front(cb_rstd, onetile);
            const auto rstd_l1_read_ptr = get_read_ptr(cb_rstd);

            // shift [n, 0] -> [n, group_idx]
            if (group_idx % TILE_WIDTH != 0) {
                auto rstd_ptr = reinterpret_cast<uint16_t *>(rstd_l1_read_ptr);
                uint32_t n_start = 0;
                uint32_t n_end = std::min((nt + 1) * TILE_HEIGHT, N);
                for (uint32_t n = n_start; n < n_end; n++) {
                    uint32_t src_idx = get_tilized_idx(n, 0);
                    uint32_t dst_idx = get_tilized_idx(n, group_idx);

                    rstd_ptr[dst_idx] = rstd_ptr[src_idx];
                }
            }

            uint32_t noc_id = nt * div_up(num_groups, TILE_WIDTH) + (group_idx / TILE_WIDTH);
            const auto rstd_noc_addr = get_noc_addr(noc_id, rstd_addrg);
            uint32_t n_start = 0;
            uint32_t n_end = std::min((nt + 1) * TILE_HEIGHT, N);

            for (uint32_t n = n_start; n < n_end; n++) {
                uint32_t dst_idx = get_tilized_idx(n, group_idx);
                noc_async_write(
                    rstd_l1_read_ptr + dst_idx * rstd_dtype_bytes,
                    rstd_noc_addr + dst_idx * rstd_dtype_bytes,
                    rstd_dtype_bytes);
            }
            noc_async_write_barrier();
            cb_pop_front(cb_rstd, onetile);
        }
    }
}  // void kernel_main()
