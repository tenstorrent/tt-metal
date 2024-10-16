// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include <cstdint>

#include "dataflow_api.h"

#define ENABLE_DEBUG 0

#if ENABLE_DEBUG
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
}

inline void print_data_u16(uint32_t l1_addr, uint32_t pagelen, uint32_t npages, uint32_t start = 0) {
    volatile tt_l1_ptr uint16_t* ptr = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(l1_addr) + start * pagelen;
    for (uint32_t page = 0; page < npages; ++page) {
        DPRINT << start + page << ": ";
        for (uint32_t j = 0; j < pagelen; ++j, ++ptr) {
            DPRINT << uint16_t(*ptr) << " ";
        }
        DPRINT << ENDL();
    }
}
#endif

// Fill an L1 buffer with the given val
inline bool fill_with_val(uint32_t begin_addr, uint32_t n, uint16_t val) {
    // simplest impl:
    volatile tt_l1_ptr uint16_t* ptr = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(begin_addr);
    for (uint32_t i = 0; i < n; ++i) {
        ptr[i] = val;
    }
    return true;
}

template <uint32_t stick_nbytes, bool is_block_sharded, bool is_width_sharded, bool is_read, bool is_col_major>
void copy_sticks_async(tt_l1_ptr uint16_t const* config_data,
                       const uint16_t my_noc_x,
                       const uint16_t my_noc_y,
                       uint32_t const in_base_l1_addr,
                       uint32_t const out_base_l1_addr) {
    int i = 0;
    int length = config_data[i + 2];

    while (length) {
        uint16_t noc_x = ((is_block_sharded && !is_col_major) || is_width_sharded) ? my_noc_x : config_data[i + 0];
        uint16_t noc_y = ((is_block_sharded && is_col_major) || is_width_sharded) ? my_noc_y : config_data[i + 1];
        length = config_data[i + 2];
        i += 3;

        const uint64_t base_addr = get_noc_addr(noc_x, noc_y, is_read ? in_base_l1_addr : out_base_l1_addr);
        for (uint16_t j = 0; j < length; j += 3) {
            uint16_t src_local_idx = config_data[i + j + 0];
            uint16_t dst_local_idx = config_data[i + j + 1];
            uint16_t nsticks = config_data[i + j + 2];
            uint32_t size = nsticks * stick_nbytes;
            uint32_t dst_offset = dst_local_idx * stick_nbytes;
            uint32_t src_offset = src_local_idx * stick_nbytes;
            if constexpr (is_read) {
                uint32_t dst_addr = out_base_l1_addr + dst_offset;
                uint64_t src_addr = base_addr + src_offset;
                noc_async_read(src_addr, dst_addr, size);
            } else {
                uint64_t dst_addr = base_addr + dst_offset;
                uint32_t src_addr = in_base_l1_addr + src_offset;
                noc_async_write(src_addr, dst_addr, size);
            }
        }

        i += length;
    }
}

void kernel_main() {
    constexpr uint32_t padding_config_cb_id = get_compile_time_arg_val(0);  // has untilized input shard
    constexpr uint32_t local_config_cb_id = get_compile_time_arg_val(1);    // has untilized input shard
    constexpr uint32_t remote_config_cb_id = get_compile_time_arg_val(2);   // has untilized input shard
    constexpr uint32_t src_cb_id = get_compile_time_arg_val(3);             // has untilized input shard
    constexpr uint32_t in_cb_id = get_compile_time_arg_val(4);              // has untilized input shard
    constexpr uint32_t out_cb_id = get_compile_time_arg_val(5);     // output shard with padding and halo goes here
    constexpr uint32_t pad_cb_id = get_compile_time_arg_val(6);     // cb for const pad val buffer
    constexpr uint32_t pad_val_u32 = get_compile_time_arg_val(7);   // pad value to fill pad buffer with
    constexpr uint32_t in_nsticks = get_compile_time_arg_val(8);    // number of sticks
    constexpr uint32_t stick_nbytes = get_compile_time_arg_val(9);  // stick size in bytes (post untilize)
    constexpr uint32_t is_block_sharded = get_compile_time_arg_val(10);
    constexpr uint32_t remote_read = get_compile_time_arg_val(11);
    constexpr bool is_col_major = get_compile_time_arg_val(12) == 1;
    constexpr uint32_t is_width_sharded = get_compile_time_arg_val(13);

    constexpr uint32_t elem_nbytes = sizeof(uint16_t);
    constexpr uint16_t pad_core_id = 0xFFFF;

    const uint16_t my_noc_x = NOC_X(my_x[noc_index]);
    const uint16_t my_noc_y = NOC_Y(my_y[noc_index]);
    const uint32_t in_base_l1_addr = get_read_ptr(in_cb_id);
    const uint32_t out_base_l1_addr = get_write_ptr(out_cb_id);

    if constexpr (padding_config_cb_id) {
        // construct the pad stick in its buffer
        cb_reserve_back(pad_cb_id, 1);
        const uint16_t pad_val = pad_val_u32;
        fill_with_val(get_write_ptr(pad_cb_id), stick_nbytes / elem_nbytes, pad_val);
        cb_push_back(pad_cb_id, 1);

        uint32_t padding_config_l1_addr = get_read_ptr(padding_config_cb_id);
        volatile tt_l1_ptr uint16_t* config_data =
            reinterpret_cast<volatile tt_l1_ptr uint16_t*>(padding_config_l1_addr);
        // print_data_u16(padding_config_l1_addr, 1, 16);
        const uint64_t padding_l1_addr = get_noc_addr(my_noc_x, my_noc_y, get_read_ptr(pad_cb_id));
        const uint32_t dst_base_addr = out_base_l1_addr;
        uint16_t nsticks = 1;
        for (uint16_t j = 0; nsticks; j += 2) {
            uint16_t dst_local_idx = config_data[j + 0];
            nsticks = config_data[j + 1];

            uint64_t dst_addr = dst_base_addr + dst_local_idx * stick_nbytes;
            for (uint16_t k = 0; k < nsticks; ++k) {
                noc_async_read(padding_l1_addr, dst_addr, stick_nbytes);
                dst_addr += stick_nbytes;
            }
        }
    }

    // input shards
    if constexpr (local_config_cb_id) {
        cb_reserve_back(src_cb_id, in_nsticks);
        cb_push_back(src_cb_id, in_nsticks);
    }

    cb_wait_front(in_cb_id, in_nsticks);  // make sure untilized data is available
    if constexpr (remote_config_cb_id) {
        uint32_t config_data_l1_addr = get_read_ptr(remote_config_cb_id);
        tt_l1_ptr uint16_t const* config_data = reinterpret_cast<tt_l1_ptr uint16_t const*>(config_data_l1_addr);
        copy_sticks_async<stick_nbytes, is_block_sharded, is_width_sharded, remote_read, is_col_major>(
            config_data, my_noc_x, my_noc_y, in_base_l1_addr, out_base_l1_addr);
    }

    if constexpr (local_config_cb_id) {
        uint32_t config_data_l1_addr = get_read_ptr(local_config_cb_id);
        tt_l1_ptr uint16_t const* config_data = reinterpret_cast<tt_l1_ptr uint16_t const*>(config_data_l1_addr);
        copy_sticks_async<stick_nbytes, is_block_sharded, is_width_sharded, false, is_col_major>(
            config_data, my_noc_x, my_noc_y, in_base_l1_addr, out_base_l1_addr);
    }

    noc_async_read_barrier();
    noc_async_write_barrier();
}
