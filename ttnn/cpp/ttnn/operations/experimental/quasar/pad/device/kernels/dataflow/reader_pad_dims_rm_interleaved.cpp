// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"

template <typename PadAccessor>
inline __attribute__((always_inline)) void fill_with_val_async(
    Noc& noc,
    const PadAccessor& s_const,
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
        CoreLocalMem<uint32_t> dst(curr_addr);
        noc.async_read(s_const, dst, chunk_nbytes, {.page_id = 0, .offset_bytes = 0}, {.offset_bytes = 0});
        curr_addr += chunk_nbytes;
    }
    if (rem_nbytes > 0) {
        CoreLocalMem<uint32_t> dst(curr_addr);
        noc.async_read(s_const, dst, rem_nbytes, {.page_id = 0, .offset_bytes = 0}, {.offset_bytes = 0});
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
    const uint32_t pad_value_const_buffer_nbytes =
        64;  // assumed to be 64 bytes, fails on BH when > 64. TODO: generalize? (Issue #21978)
    const uint32_t pad_value_packed = get_arg_val<uint32_t>(15);
    const uint32_t start_src_stick_id = get_arg_val<uint32_t>(16);
    const uint32_t start_src_stick_wi = get_arg_val<uint32_t>(18);
    const uint32_t start_src_stick_offset = get_arg_val<uint32_t>(20);  // == start_src_stick_wi * elem_size
    const uint32_t num_local_Y = get_arg_val<uint32_t>(21);
    const uint32_t num_local_unpadded_Y = get_arg_val<uint32_t>(22);
    const uint32_t full_unpadded_X_nbytes = get_arg_val<uint32_t>(23);
    const uint32_t num_local_W = get_arg_val<uint32_t>(26);

    constexpr auto src_args = TensorAccessorArgs<2>();
    constexpr auto dst_args = TensorAccessorArgs<src_args.next_compile_time_args_offset()>();
    constexpr auto pad_tensor_args = TensorAccessorArgs<dst_args.next_compile_time_args_offset()>();

    constexpr uint32_t cb_id = tt::CBIndex::c_0;
    CircularBuffer cb(cb_id);
    Noc noc;

    // calculate the offset for alignment of padding in rows/sticks
    uint32_t l1_addr_partial = cb.get_write_ptr() + unpadded_X_nbytes;
    const uint32_t l1_addr_align_offset =
        32 - l1_addr_partial % 32;  // NOTE: this is fine with double buffering since offset will be same for each page

    const auto s0 = TensorAccessor(src_args, src_addr);

    const auto s_const = TensorAccessor(pad_tensor_args, pad_value_const_buffer_addr);

    uint16_t pad_value = pad_value_packed >> 16;

    uint32_t src_stick_id = start_src_stick_id;
    for (uint32_t w = 0; w < num_local_W; ++w) {
        for (uint32_t z = 0; z < num_total_Z; ++z) {
            for (uint32_t y = 0; y < num_local_Y; ++y) {
                cb.reserve_back(1);
                uint32_t l1_addr = cb.get_write_ptr();
                if (y >= num_local_unpadded_Y || z >= num_unpadded_Z || w >= num_unpadded_W) {
                    // this is fully padding
                    fill_with_val_async(
                        noc, s_const, l1_addr, l1_addr, padded_X_nbytes, pad_value_const_buffer_nbytes, pad_value);
                } else {
                    // this is a data row possibly with padding at end
                    CoreLocalMem<uint32_t> dst(l1_addr);
                    noc.async_read(
                        s0,
                        dst,
                        unpadded_X_nbytes,
                        {.page_id = src_stick_id, .offset_bytes = start_src_stick_offset},
                        {.offset_bytes = 0});
                    l1_addr_partial = l1_addr + unpadded_X_nbytes;
                    fill_with_val_async(
                        noc,
                        s_const,
                        l1_addr_partial,
                        l1_addr_partial + l1_addr_align_offset,
                        padded_X_diff_nbytes,
                        pad_value_const_buffer_nbytes,
                        pad_value);
                    ++src_stick_id;
                }
                noc.async_read_barrier();
                cb.push_back(1);
            }
        }
    }
}
