// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "dataflow_api.h"
#include "debug/dprint.h"

inline void print_pages(uint32_t l1_addr, uint32_t pagelen, uint32_t npages, uint32_t start = 0) {
    volatile tt_l1_ptr uint16_t* ptr = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(l1_addr) + start * pagelen;
    for (uint32_t page = 0; page < npages; ++page) {
        DPRINT << start + page << ": ";
        for (uint32_t j = 0; j < pagelen; ++j, ++ptr) {
            DPRINT << BF16(*ptr) << " ";
        }
        DPRINT << ENDL();
    }
    DPRINT << ENDL();
}

void kernel_main() {
    constexpr bool src0_is_dram = (bool)get_compile_time_arg_val(0);
    constexpr uint32_t N = get_compile_time_arg_val(1);
    constexpr uint32_t page_size = get_compile_time_arg_val(2);
    constexpr uint32_t num_rows = get_compile_time_arg_val(3);
    constexpr uint32_t x_dim = get_compile_time_arg_val(4);

    const uint32_t src_addr = get_arg_val<uint32_t>(0);
    const DataFormat data_format = get_dataformat(tt::CBIndex::c_0);

    uint32_t input_shape[N], src_strides[N];
    for (uint32_t i = 1; i <= N; i++) {
        input_shape[i - 1] = get_arg_val<uint32_t>(i);
        src_strides[i - 1] = get_arg_val<uint32_t>(i + N);
    }

    uint32_t X = input_shape[x_dim];
    uint32_t X_stride = src_strides[x_dim];

    // for (uint32_t i = 0; i < N; i++) {
    //     DPRINT << "input_shape[" << i << "] = " << input_shape[i] << " ";
    // }
    // DPRINT << ENDL();
    // for (uint32_t i = 0; i < N; i++) {
    //     DPRINT << "src_strides[" << i << "] = " << src_strides[i] << " ";
    // }
    // DPRINT << ENDL();

    const InterleavedAddrGen<src0_is_dram> s0 = {.bank_base_address = src_addr, .page_size = page_size};

    uint32_t curr_addr = src_addr;
    // DPRINT << "Reading " << num_rows << " rows of " << X << " elements each" << ENDL();
    // DPRINT << "X dimension: " << x_dim << ENDL();
    for (uint32_t i = 0; i < num_rows/X; ++i) {
        uint32_t idxs[N];
        idxs[N - 1] = 0;
        uint32_t remainder = i;
        for (int32_t d = N - 2; d >= 0; --d) { // Exclude W dimension
            if (d == (int32_t)x_dim) {
                continue; // Skip X dimension
            }
            idxs[d] = remainder % input_shape[d];
            remainder /= input_shape[d];
        }
        cb_reserve_back(tt::CBIndex::c_0, X);
        uint32_t src_buffer_l1_addr = get_write_ptr(tt::CBIndex::c_0);
        for (uint32_t j = 0; j < X; ++j) {
            idxs[x_dim] = j;
            // for (uint32_t k = 0; k < N; ++k) {
            //     DPRINT << "idxs[" << k << "] = " << idxs[k] << " ";
            // }
            // Compute the address using indices and strides
            uint64_t addr_offset = 0;
            for (uint32_t d = 0; d < N; ++d) {
                addr_offset += idxs[d] * src_strides[d];
            }
            // DPRINT << "Reading page " << addr_offset << " into buffer " << ENDL();
            uint64_t src_noc_addr = get_noc_addr(addr_offset, s0);
            noc_async_read(src_noc_addr, src_buffer_l1_addr, page_size);
            src_buffer_l1_addr += page_size;
        }
        noc_async_read_barrier();
        // src_buffer_l1_addr = get_write_ptr(tt::CBIndex::c_0);
        // print_pages(src_buffer_l1_addr, 8, X, 0);
        cb_push_back(tt::CBIndex::c_0, X);
    }
}
