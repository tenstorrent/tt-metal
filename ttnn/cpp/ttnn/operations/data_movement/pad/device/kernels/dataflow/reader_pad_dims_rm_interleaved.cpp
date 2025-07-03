// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "dataflow_api.h"

inline __attribute__((always_inline)) void fill_with_val_async(
    const uint64_t in_noc_addr,
    const uint32_t begin_addr,
    const uint32_t begin_addr_aligned,
    uint32_t size_nbytes,
    uint32_t chunk_nbytes,
    uint16_t pad_value) {
    uint32_t curr_addr = begin_addr;
    while (curr_addr < begin_addr_aligned && size_nbytes > 0) {
        reinterpret_cast<uint16_t*>(curr_addr)[0] = pad_value;
        curr_addr += 2;
        size_nbytes -= 2;
    }
    uint32_t nchunks = size_nbytes / chunk_nbytes;
    uint32_t rem_nbytes = size_nbytes % chunk_nbytes;
    for (uint32_t i = 0; i < nchunks; ++i) {
        noc_async_read(in_noc_addr, curr_addr, chunk_nbytes);
        curr_addr += chunk_nbytes;
    }
    if (rem_nbytes > 0) {
        noc_async_read(in_noc_addr, curr_addr, rem_nbytes);
    }
}

void kernel_main() {
    const uint32_t src_addr = get_arg_val<uint32_t>(0);
    const uint32_t num_unpadded_W = get_arg_val<uint32_t>(2);
    const uint32_t num_total_W = get_arg_val<uint32_t>(3);
    const uint32_t num_unpadded_Z = get_arg_val<uint32_t>(4);
    const uint32_t num_total_Z = get_arg_val<uint32_t>(5);
    const uint32_t num_unpadded_Y = get_arg_val<uint32_t>(6);
    const uint32_t num_total_Y = get_arg_val<uint32_t>(7);
    const uint32_t unpadded_X_nbytes = get_arg_val<uint32_t>(10);
    const uint32_t padded_X_nbytes = get_arg_val<uint32_t>(11);
    const uint32_t padded_X_diff_nbytes = get_arg_val<uint32_t>(12);
    const uint32_t pad_value_const_buffer_addr = get_arg_val<uint32_t>(13);
    const uint32_t pad_value_const_buffer_nbytes = 64;  // assumed to be 64 bytes, fails on BH when > 64. TODO: generalize? (Issue #21978)
    const uint32_t pad_value_packed = get_arg_val<uint32_t>(15);
    const uint32_t start_src_stick_id = get_arg_val<uint32_t>(16);
    const uint32_t start_src_stick_wi = get_arg_val<uint32_t>(18);
    const uint32_t start_src_stick_offset = get_arg_val<uint32_t>(20);  // == start_src_stick_wi * elem_size
    const uint32_t num_local_Y = get_arg_val<uint32_t>(21);
    const uint32_t num_local_unpadded_Y = get_arg_val<uint32_t>(22);
    const uint32_t full_unpadded_X_nbytes = get_arg_val<uint32_t>(23);
    const uint32_t num_local_W = get_arg_val<uint32_t>(26);

    constexpr bool src0_is_dram = get_compile_time_arg_val(0) == 1;
    constexpr bool src_stick_size_is_pow2 = get_compile_time_arg_val(2) == 1;

    constexpr uint32_t cb_id = tt::CBIndex::c_0;

    // calculate the offset for alignment of padding in rows/sticks
    uint32_t l1_addr_partial = get_write_ptr(cb_id) + unpadded_X_nbytes;
    const uint32_t l1_addr_align_offset =
        32 - l1_addr_partial % 32;  // NOTE: this is fine with double buffering since offset will be same for each page

    // #if (src_stick_size_is_pow2)
    //     constexpr uint32_t src_log_base_2_of_page_size = get_compile_time_arg_val(3);
    //     const InterleavedPow2AddrGen<src0_is_dram> s0 = {
    //         .bank_base_address = src_addr,
    //         .log_base_2_of_page_size = src_log_base_2_of_page_size
    //     };
    // #else
    const InterleavedAddrGen<src0_is_dram> s0 = {.bank_base_address = src_addr, .page_size = full_unpadded_X_nbytes};
    // #endif

    const InterleavedPow2AddrGen<false> s_const = {
        .bank_base_address = pad_value_const_buffer_addr,
        .log_base_2_of_page_size = 6  // TODO: generalize. Currently hardcoded for 32 16b values (2^6 = 64)
    };
    const uint64_t const_buffer_noc_addr = get_noc_addr(0, s_const);

    uint16_t pad_value = pad_value_packed >> 16;

    uint32_t src_stick_id = start_src_stick_id;
    for (uint32_t w = 0; w < num_local_W; ++w) {
        for (uint32_t z = 0; z < num_total_Z; ++z) {
            for (uint32_t y = 0; y < num_local_Y; ++y) {
                cb_reserve_back(cb_id, 1);
                uint32_t l1_addr = get_write_ptr(cb_id);
                if (y >= num_local_unpadded_Y || z >= num_unpadded_Z || w >= num_unpadded_W) {
                    // this is fully padding
                    fill_with_val_async(
                        const_buffer_noc_addr,
                        l1_addr,
                        l1_addr,
                        padded_X_nbytes,
                        pad_value_const_buffer_nbytes,
                        pad_value);
                } else {
                    // this is a data row possibly with padding at end
                    uint64_t src_noc_addr = get_noc_addr(src_stick_id, s0, start_src_stick_offset);
                    noc_async_read(src_noc_addr, l1_addr, unpadded_X_nbytes);
                    l1_addr_partial = l1_addr + unpadded_X_nbytes;
                    fill_with_val_async(
                        const_buffer_noc_addr,
                        l1_addr_partial,
                        l1_addr_partial + l1_addr_align_offset,
                        padded_X_diff_nbytes,
                        pad_value_const_buffer_nbytes,
                        pad_value);
                    ++src_stick_id;
                }
                noc_async_read_barrier();
                cb_push_back(cb_id, 1);
            }
        }
    }
}
