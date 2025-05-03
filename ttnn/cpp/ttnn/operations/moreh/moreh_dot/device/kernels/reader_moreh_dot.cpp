// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include "dataflow_api.h"

void mask_tile_in_reader(uint32_t l1_addr, uint32_t mask_w = 32, uint32_t mask_h = 32) {
    union {
        float f;
        uint32_t u;
    } zero;
    zero.f = 0.0f;
    auto ptr = reinterpret_cast<uint16_t*>(l1_addr);
    for (uint32_t h = 0; h < 16; h++) {
        // sub tile 0
        {
            uint32_t mask_w_0 = (mask_w >= 16) ? 16 : mask_w;
            uint32_t mask_h_0 = (mask_h >= 16) ? 16 : mask_h;
            uint32_t w = (h >= mask_h_0) ? 0 : mask_w_0;
            for (; w < 16; w++) {
                ptr[h * 16 + w] = uint16_t(zero.u >> 16);
            }
        }
        // sub tile 1
        {
            uint32_t mask_w_1 = (mask_w < 16) ? 0 : mask_w - 16;
            uint32_t mask_h_0 = (mask_h >= 16) ? 16 : mask_h;
            uint32_t w = (h >= mask_h_0) ? 0 : mask_w_1;
            for (; w < 16; w++) {
                ptr[h * 16 + w + 256] = uint16_t(zero.u >> 16);
            }
        }
        // sub tile 2
        {
            uint32_t mask_w_0 = (mask_w >= 16) ? 16 : mask_w;
            uint32_t mask_h_1 = (mask_h < 16) ? 0 : mask_h - 16;
            uint32_t w = (h >= mask_h_1) ? 0 : mask_w_0;
            for (; w < 16; w++) {
                ptr[h * 16 + w + 512] = uint16_t(zero.u >> 16);
            }
        }
        // sub tile 3
        {
            uint32_t mask_w_1 = (mask_w < 16) ? 0 : mask_w - 16;
            uint32_t mask_h_1 = (mask_h < 16) ? 0 : mask_h - 16;
            uint32_t w = (h >= mask_h_1) ? 0 : mask_w_1;
            for (; w < 16; w++) {
                ptr[h * 16 + w + 768] = uint16_t(zero.u >> 16);
            }
        }
    }
}

void kernel_main() {
    // same arg indices as in reader_binary_diff_lenghts for compat
    uint32_t src0_addr = get_arg_val<uint32_t>(0);
    uint32_t src1_addr = get_arg_val<uint32_t>(1);
    uint32_t num_tiles = get_arg_val<uint32_t>(2);
    uint32_t start_id = get_arg_val<uint32_t>(3);
    uint32_t mask_h = get_arg_val<uint32_t>(4);
    uint32_t mask_w = get_arg_val<uint32_t>(5);

    constexpr bool src0_is_dram = get_compile_time_arg_val(0) == 1;
    constexpr bool src1_is_dram = get_compile_time_arg_val(1) == 1;
    constexpr uint32_t scaler = get_compile_time_arg_val(2);

    constexpr uint32_t cb_id_in0 = 0;
    constexpr uint32_t cb_id_in1 = 1;
    constexpr uint32_t cb_id_in2 = 2;
    cb_reserve_back(cb_id_in2, 1);
    if (scaler != 0) {
        auto ptr = reinterpret_cast<uint16_t*>(get_write_ptr(cb_id_in2));
        for (int j = 0; j < 1024; j++) {
            ptr[j] = uint16_t(0);
        }

        for (int k = 0; k < 4; k++) {
            for (int j = 0; j < 16; j++) {
                ptr[k * 256 + j] = uint16_t(scaler >> 16);
            }
        }
    }
    cb_push_back(cb_id_in2, 1);

    uint32_t l1_write_addr_in0;
    uint32_t src0_tile_bytes = get_tile_size(cb_id_in0);
    DataFormat src0_data_format = get_dataformat(cb_id_in0);
    const InterleavedAddrGenFast<src0_is_dram> s0 = {
        .bank_base_address = src0_addr, .page_size = src0_tile_bytes, .data_format = src0_data_format};
    uint32_t l1_write_addr_in1;
    uint32_t src1_tile_bytes = get_tile_size(cb_id_in1);
    DataFormat src1_data_format = get_dataformat(cb_id_in1);
    const InterleavedAddrGenFast<src1_is_dram> s1 = {
        .bank_base_address = src1_addr, .page_size = src1_tile_bytes, .data_format = src1_data_format};

    constexpr uint32_t onetile = 1;
    for (uint32_t i = start_id; i < start_id + num_tiles; i++) {
        bool last_tile = i == (start_id + num_tiles - 1);
        cb_reserve_back(cb_id_in0, onetile);
        l1_write_addr_in0 = get_write_ptr(cb_id_in0);
        noc_async_read_tile(i, s0, l1_write_addr_in0);

        cb_reserve_back(cb_id_in1, onetile);
        l1_write_addr_in1 = get_write_ptr(cb_id_in1);
        noc_async_read_tile(i, s1, l1_write_addr_in1);

        noc_async_read_barrier();

        if (last_tile) {
            mask_tile_in_reader(l1_write_addr_in0, mask_w, mask_h);
            mask_tile_in_reader(l1_write_addr_in1, mask_w, mask_h);
        }

        cb_push_back(cb_id_in0, onetile);
        cb_push_back(cb_id_in1, onetile);
    }
}
