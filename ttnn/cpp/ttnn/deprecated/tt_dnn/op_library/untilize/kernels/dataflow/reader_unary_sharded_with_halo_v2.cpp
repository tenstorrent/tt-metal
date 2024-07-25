// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "dataflow_api.h"

// #include "debug/dprint.h"

// SliceRange srr = SliceRange{ .h0 = 0, .h1 = 1, .hs = 8, .w0 = 0, .w1 = 32, .ws = 1 };
// SliceRange srt = SliceRange{ .h0 = 0, .h1 = 8, .hs = 1, .w0 = 0, .w1 = 4, .ws = 1 };

// inline void print_sticks(uint32_t l1_addr, uint32_t stick_start, uint32_t nsticks, uint32_t stick_size = 64) {
//     for (uint32_t i = stick_start; i < stick_start + nsticks; ++ i) {
//         volatile tt_l1_ptr uint16_t* l1_ptr = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(l1_addr + i * stick_size * 2);
//         DPRINT << i << ": ";
//         for (uint32_t j = 0; j < stick_size; ++ j) {
//             DPRINT << BF16(l1_ptr[j]) << " ";
//         }
//         DPRINT << ENDL();
//     }
// }

// Fill an L1 buffer with the given val
inline bool fill_with_val(uint32_t begin_addr, uint32_t n, uint16_t val) {
    // simplest impl:
    volatile tt_l1_ptr uint16_t* ptr = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(begin_addr);
    for (uint32_t i = 0; i < n; ++ i) {
        ptr[i] = val;
    }
    return true;
}

// inline void push_to_neighbor_async(uint32_t noc_x,
//                                    uint32_t noc_y,
//                                    uint32_t data_ss_cb_id,
//                                    uint32_t data_nsegments,
//                                    uint32_t in_l1_addr,
//                                    uint32_t out_l1_addr,
//                                    uint32_t stick_nbytes) {
//     volatile tt_l1_ptr uint16_t* data_ss = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(get_read_ptr(data_ss_cb_id));
//     for (uint32_t i = 0; i < 2 * data_nsegments; i += 2) {
//         uint32_t seg_nbytes = data_ss[i + 1] * stick_nbytes;
//         uint32_t dst_addr = out_l1_addr + data_ss[i] * stick_nbytes;
//         noc_async_write(in_l1_addr, get_noc_addr(noc_x, noc_y, dst_addr), seg_nbytes);
//         in_l1_addr += seg_nbytes;
//     }
// }

void kernel_main() {
    constexpr uint32_t in_cb_id             = get_compile_time_arg_val(0);  // has untilized input shard
    constexpr uint32_t out_cb_id            = get_compile_time_arg_val(1);  // output shard with padding and halo goes here
    constexpr uint32_t pad_cb_id            = get_compile_time_arg_val(2);  // cb for const pad val buffer
    constexpr uint32_t local_pad_ss_cb_id   = get_compile_time_arg_val(3);  // cb for local pad config
    constexpr uint32_t local_data_ss_cb_id  = get_compile_time_arg_val(4);  // cb for local data config
    constexpr uint32_t ll_data_ss_cb_id     = get_compile_time_arg_val(5);  // cb for ll data config
    constexpr uint32_t l_data_ss_cb_id      = get_compile_time_arg_val(6);  // cb for l data config
    constexpr uint32_t r_data_ss_cb_id      = get_compile_time_arg_val(7);  // cb for r data config
    constexpr uint32_t rr_data_ss_cb_id     = get_compile_time_arg_val(8);  // cb for rr data config
    constexpr uint32_t pad_val_u32          = get_compile_time_arg_val(9);  // pad value to fill pad buffer with
    constexpr uint32_t pad_stick_len        = get_compile_time_arg_val(10); // pad stick size in nelems (post untilize)
    constexpr uint32_t stick_nbytes         = get_compile_time_arg_val(11); // stick size in bytes (post untilize)
    constexpr uint32_t stick_nbytes_log2    = get_compile_time_arg_val(12);
    constexpr uint32_t in_shard_cb_id       = get_compile_time_arg_val(13);

    constexpr uint32_t nbytes = 2;   // TODO: pass this in

    static_assert(stick_nbytes <= NOC_MAX_BURST_SIZE); // stick_nbytes used in noc_async_read_one_packet
    static_assert(stick_nbytes == pad_stick_len * nbytes);

    uint32_t in_nsticks = get_arg_val<uint32_t>(0);
    uint32_t has_ll     = get_arg_val<uint32_t>(1);
    uint32_t ll_noc_x   = get_arg_val<uint32_t>(2);
    uint32_t ll_noc_y   = get_arg_val<uint32_t>(3);
    uint32_t has_l      = get_arg_val<uint32_t>(4);
    uint32_t l_noc_x    = get_arg_val<uint32_t>(5);
    uint32_t l_noc_y    = get_arg_val<uint32_t>(6);
    uint32_t has_r      = get_arg_val<uint32_t>(7);
    uint32_t r_noc_x    = get_arg_val<uint32_t>(8);
    uint32_t r_noc_y    = get_arg_val<uint32_t>(9);
    uint32_t has_rr     = get_arg_val<uint32_t>(10);
    uint32_t rr_noc_x   = get_arg_val<uint32_t>(11);
    uint32_t rr_noc_y   = get_arg_val<uint32_t>(12);

    int32_t local_pad_nsegments            = get_arg_val<int32_t>(13);
    int32_t local_data_src_start_offset    = get_arg_val<int32_t>(14);
    int32_t local_data_nsegments           = get_arg_val<int32_t>(15);
    int32_t ll_data_src_start_offset       = get_arg_val<int32_t>(16);
    int32_t ll_data_nsegments              = get_arg_val<int32_t>(17);
    int32_t l_data_src_start_offset        = get_arg_val<int32_t>(18);
    int32_t l_data_nsegments               = get_arg_val<int32_t>(19);
    int32_t r_data_src_start_offset        = get_arg_val<int32_t>(20);
    int32_t r_data_nsegments               = get_arg_val<int32_t>(21);
    int32_t rr_data_src_start_offset       = get_arg_val<int32_t>(22);
    int32_t rr_data_nsegments              = get_arg_val<int32_t>(23);

    // input shards
    cb_reserve_back(in_shard_cb_id, in_nsticks);
    cb_push_back(in_shard_cb_id, in_nsticks);

    // construct the pad stick in its buffer
    cb_reserve_back(pad_cb_id, 1);
    const uint16_t pad_val = pad_val_u32;
    fill_with_val(get_write_ptr(pad_cb_id), pad_stick_len, pad_val);
    cb_push_back(pad_cb_id, 1);
    const uint64_t padding_noc_addr = get_noc_addr(get_read_ptr(pad_cb_id));

    const uint32_t in_base_l1_addr = get_read_ptr(in_cb_id);
    const uint32_t out_base_l1_addr = get_write_ptr(out_cb_id);

    // DPRINT << "IN:" << ENDL();
    // print_sticks(in_base_l1_addr, 0, 300, 32);

    uint32_t in_l1_addr = in_base_l1_addr;

    // insert all padding locally
    if (local_pad_nsegments > 0) {
        // cb_wait_front(local_pad_ss_cb_id, 1);
        uint32_t local_pad_ss_l1_addr = get_read_ptr(local_pad_ss_cb_id);
        volatile tt_l1_ptr uint16_t* local_pad_ss = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(local_pad_ss_l1_addr);
        for (int32_t i = 0; i < 2 * local_pad_nsegments; i += 2) {
            uint32_t dst_size = local_pad_ss[i + 1];
            uint32_t dst_addr = out_base_l1_addr + local_pad_ss[i] * stick_nbytes;
            for (uint32_t j = 0; j < dst_size; ++ j) {
                // noc_async_read(padding_noc_addr, dst_addr, stick_nbytes);
                noc_async_write(get_read_ptr(pad_cb_id), get_noc_addr(dst_addr), stick_nbytes);
                dst_addr += stick_nbytes;
            }
        }
    }

    cb_wait_front(in_cb_id, in_nsticks);    // make sure untilized data is available

    // then insert all local data
    if (local_data_nsegments > 0) {
        // cb_wait_front(local_data_ss_cb_id, 1);
        in_l1_addr = in_base_l1_addr + local_data_src_start_offset * stick_nbytes;
        uint32_t local_data_ss_l1_addr = get_read_ptr(local_data_ss_cb_id);
        volatile tt_l1_ptr uint16_t* local_data_ss = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(local_data_ss_l1_addr);
        for (int32_t i = 0; i < 2 * local_data_nsegments; i += 2) {
            uint32_t dst_size = local_data_ss[i + 1] * stick_nbytes;
            uint32_t dst_addr = out_base_l1_addr + local_data_ss[i] * stick_nbytes;
            noc_async_read(get_noc_addr(in_l1_addr), dst_addr, dst_size);
            // noc_async_write(in_l1_addr, get_noc_addr(dst_addr), dst_size);
            in_l1_addr += dst_size;
        }
    }

    // // push data to neighbors
    // if (has_ll && ll_data_nsegments > 0) {
    //     // cb_wait_front(ll_data_ss_cb_id, 1);
    //     push_to_neighbor_async(ll_noc_x, ll_noc_y, ll_data_ss_cb_id, ll_data_nsegments, in_base_l1_addr + ll_data_src_start_offset * stick_nbytes, out_base_l1_addr, stick_nbytes);
    // }
    // if (has_l && l_data_nsegments > 0) {
    //     // cb_wait_front(l_data_ss_cb_id, 1);
    //     push_to_neighbor_async(l_noc_x, l_noc_y, l_data_ss_cb_id, l_data_nsegments, in_base_l1_addr + l_data_src_start_offset * stick_nbytes, out_base_l1_addr, stick_nbytes);
    // }
    // if (has_r && r_data_nsegments > 0) {
    //     // cb_wait_front(r_data_ss_cb_id, 1);
    //     push_to_neighbor_async(r_noc_x, r_noc_y, r_data_ss_cb_id, r_data_nsegments, in_base_l1_addr + r_data_src_start_offset * stick_nbytes, out_base_l1_addr, stick_nbytes);
    // }
    // if (has_rr && rr_data_nsegments > 0) {
    //     // cb_wait_front(rr_data_ss_cb_id, 1);
    //     push_to_neighbor_async(rr_noc_x, rr_noc_y, rr_data_ss_cb_id, rr_data_nsegments, in_base_l1_addr + rr_data_src_start_offset * stick_nbytes, out_base_l1_addr, stick_nbytes);
    // }

    noc_async_read_barrier();
    noc_async_write_barrier();

    // DPRINT << "OUT:" << ENDL();
    // print_sticks(out_base_l1_addr, 0, 500, 32);
}
