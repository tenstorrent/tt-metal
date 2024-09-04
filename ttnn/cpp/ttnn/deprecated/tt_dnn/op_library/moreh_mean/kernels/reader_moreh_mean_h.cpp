// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/cpp/ttnn/deprecated/tt_dnn/kernels/dataflow/moreh_common.hpp"

void kernel_main() {
    uint32_t src_addr = get_arg_val<uint32_t>(0);
    uint32_t col_start_tile_id =
        get_arg_val<uint32_t>(1);  // Start id in column major order. This should be the start of a column
    uint32_t curr_col_in_batch = get_arg_val<uint32_t>(2);
    uint32_t num_cols = get_arg_val<uint32_t>(3);  // number of cols to read
    uint32_t mask_h = get_arg_val<uint32_t>(4);

    constexpr bool src_is_dram = get_compile_time_arg_val(0) == 1;
    constexpr uint32_t Ht = get_compile_time_arg_val(1);
    constexpr uint32_t Wt = get_compile_time_arg_val(2);
    constexpr uint32_t HtWt = get_compile_time_arg_val(3);

    constexpr uint32_t cb_id_in0 = tt::CB::c_in0;

    // ublocks size defined in tiles
    constexpr uint32_t onetile = 1;
    const uint32_t tile_bytes = get_tile_size(cb_id_in0);
    const DataFormat data_format = get_dataformat(cb_id_in0);

#ifdef REDUCE_SCALER
    constexpr uint32_t cb_id_in2 = tt::CB::c_in2;
    constexpr uint32_t scaler = get_compile_time_arg_val(4);
    cb_reserve_back(cb_id_in2, 1);
    constexpr uint32_t num_zeros_reads = 2048 / MEM_ZEROS_SIZE;
    uint64_t zeros_noc_addr = get_noc_addr(MEM_ZEROS_BASE);
    uint32_t write_addr = get_write_ptr(cb_id_in2);
    // Fill tile with zeros
    for (uint32_t i = 0; i < num_zeros_reads; ++i) {
        noc_async_read(zeros_noc_addr, write_addr, MEM_ZEROS_SIZE);
        write_addr += MEM_ZEROS_SIZE;
    }
    noc_async_read_barrier();
    if constexpr (scaler != 0) {
        volatile tt_l1_ptr uint32_t* ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_id_in2));
        uint32_t idx = 0;
        for (uint32_t k = 0; k < 4; ++k) {
            uint32_t curr_idx = idx;
            for (uint32_t j = 0; j < 8; ++j) {
                ptr[curr_idx] = scaler;
                curr_idx++;
            }
            idx += 128;
        }
    }
    cb_push_back(cb_id_in2, 1);
#endif

    constexpr uint32_t cb_id_mask_h = tt::CB::c_in3;
#ifdef DO_MASK_H
    generate_mask_h(cb_id_mask_h, mask_h);
#endif

    const InterleavedAddrGenFast<src_is_dram> s = {
        .bank_base_address = src_addr, .page_size = tile_bytes, .data_format = data_format};

    uint32_t w = curr_col_in_batch;

    // this reader will read a NHW tensor in NWH order
    for (uint32_t i = 0; i < num_cols; i++) {
        uint32_t curr_id = col_start_tile_id;
        for (uint32_t j = 0; j < Ht; j++) {
            cb_reserve_back(cb_id_in0, onetile);
            uint32_t l1_write_addr = get_write_ptr(cb_id_in0);
            noc_async_read_tile(curr_id, s, l1_write_addr);
            noc_async_read_barrier();
            cb_push_back(cb_id_in0, onetile);
            curr_id += Wt;  // stride in H
        }
        w++;
        if (w == Wt) {
            col_start_tile_id = curr_id - Wt + 1;
            w = 0;
        } else {
            col_start_tile_id++;
        }
    }
}
