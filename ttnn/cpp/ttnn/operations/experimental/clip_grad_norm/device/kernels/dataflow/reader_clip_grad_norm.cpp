// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "dataflow_api.h"
#include "hostdevcommon/common_values.hpp"
#include "debug/dprint.h"
#include "debug/waypoint.h"

namespace {
using L1Ptr = volatile tt_l1_ptr uint32_t*;

template <typename TensorAccessorType>
inline void read_tiles_from_dram(
    uint32_t cb_id_in0, uint32_t start_id, uint32_t num_tiles, const TensorAccessorType& s) {
    constexpr uint32_t onetile = 1;
    uint32_t end_id = start_id + num_tiles;
    for (uint32_t i = start_id; i < end_id; ++i) {
        cb_reserve_back(cb_id_in0, onetile);
        uint32_t l1_write_addr = get_write_ptr(cb_id_in0);
        noc_async_read_tile(i, s, l1_write_addr);
        noc_async_read_barrier();
        cb_push_back(cb_id_in0, onetile);
    }
}

inline void collect_all_partial_norms(
    uint32_t cb_norm_partial,
    uint32_t cb_norm_external,
    uint32_t num_cores,
    uint32_t single_tile_size_bytes,
    L1Ptr remote_noc_x,
    L1Ptr remote_noc_y) {
    uint32_t sender_partial_addr = get_read_ptr(cb_norm_partial);

    cb_reserve_back(cb_norm_external, num_cores);
    uint32_t l1_write_addr_external = get_write_ptr(cb_norm_external);

    // Copy our own partial first
    WAYPOINT("CP1");
    volatile tt_l1_ptr uint32_t* our_partial_src = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sender_partial_addr);
    volatile tt_l1_ptr uint32_t* our_partial_dst =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(l1_write_addr_external);
    for (uint32_t i = 0; i < single_tile_size_bytes / sizeof(uint32_t); ++i) {
        our_partial_dst[i] = our_partial_src[i];
    }
    l1_write_addr_external += single_tile_size_bytes;
    WAYPOINT("CP2");

    // Read remote partials
    // Note: All cores have the same CB layout, so cb_norm_partial is at the same L1 address on each core
    // get_noc_addr will convert the sender's L1 address to the remote core's L1 address
    for (uint32_t core = 0; core < num_cores - 1; ++core) {
        uint64_t noc_addr_partial = get_noc_addr(remote_noc_x[core], remote_noc_y[core], sender_partial_addr);
        noc_async_read_one_packet(noc_addr_partial, l1_write_addr_external, single_tile_size_bytes);
        l1_write_addr_external += single_tile_size_bytes;
    }
    WAYPOINT("CP3");
    noc_async_read_barrier();
    WAYPOINT("CP4");
    cb_push_back(cb_norm_external, num_cores);
    WAYPOINT("CP5");
}

inline void wait_local_norms_and_mcast_global_norm(
    uint32_t cb_norm_partial,
    uint32_t cb_norm_global,
    uint32_t cb_norm_external,
    uint32_t num_cores,
    uint32_t single_tile_size_bytes,
    volatile tt_l1_ptr uint32_t* reduce_sender_semaphore_addr_ptr,
    volatile tt_l1_ptr uint32_t* reduce_receiver_semaphore_addr_ptr,
    uint64_t multicast_data_noc,
    uint32_t reduce_sender_semaphore_addr,
    uint64_t reduce_sender_semaphore_noc_addr,
    L1Ptr remote_noc_x,
    L1Ptr remote_noc_y) {
    DPRINT << "READER_SENDER: signaling ready, waiting for " << (num_cores - 1) << " receivers" << ENDL();
    WAYPOINT("RS1");
    *reduce_sender_semaphore_addr_ptr = VALID;
    WAYPOINT("RS2");
    noc_semaphore_wait(reduce_receiver_semaphore_addr_ptr, num_cores - 1);
    WAYPOINT("RS3");
    DPRINT << "READER_SENDER: all receivers ready, collecting partials" << ENDL();
    noc_semaphore_set(reduce_receiver_semaphore_addr_ptr, 0);
    WAYPOINT("RS4");

    collect_all_partial_norms(
        cb_norm_partial, cb_norm_external, num_cores, single_tile_size_bytes, remote_noc_x, remote_noc_y);
    WAYPOINT("RS5");
    DPRINT << "READER_SENDER: collected all partials, waiting for compute to combine" << ENDL();

    // DO NOT pop cb_norm_partial - no need, data already copied to cb_norm_external
    WAYPOINT("RS6");
    cb_wait_front(cb_norm_global, 1);  // Wait for compute kernel to combine and push global norm
    WAYPOINT("RS7");
    DPRINT << "READER_SENDER: compute finished, multicasting global norm" << ENDL();

    uint32_t l1_read_addr_global = get_read_ptr(cb_norm_global);
    noc_async_write_multicast(
        l1_read_addr_global, multicast_data_noc | l1_read_addr_global, single_tile_size_bytes, num_cores - 1, true);
    noc_async_write_barrier();

    noc_semaphore_set_multicast(reduce_sender_semaphore_addr, reduce_sender_semaphore_noc_addr, num_cores - 1, false);

    // DO NOT pop cb_norm_global - compute kernel on sender may still need it
    // The multicast has already copied the data to receivers, so it's safe to leave it
}

inline void send_local_norm_to_collector(
    uint32_t cb_norm_partial,
    uint32_t cb_norm_global,
    uint32_t reduce_receiver_semaphore_addr,
    volatile tt_l1_ptr uint32_t* reduce_sender_semaphore_addr_ptr,
    uint32_t sender_noc_x,
    uint32_t sender_noc_y) {
    DPRINT << "READER_RECEIVER: notifying sender, waiting for global norm" << ENDL();
    WAYPOINT("RR1");
    DPRINT << "READER_RECEIVER: calculating semaphore NOC addr" << ENDL();
    uint64_t reduce_receiver_semaphore_noc_addr =
        get_noc_addr(sender_noc_x, sender_noc_y, reduce_receiver_semaphore_addr);
    DPRINT << "READER_RECEIVER: incrementing semaphore" << ENDL();
    noc_semaphore_inc(reduce_receiver_semaphore_noc_addr, 1);
    WAYPOINT("RR2");
    DPRINT << "READER_RECEIVER: semaphore incremented" << ENDL();

    // DO NOT pop cb_norm_partial yet - sender needs to read it via NOC first!
    // Reserve space for global norm
    cb_reserve_back(cb_norm_global, 1);
    WAYPOINT("RR3");

    // Wait for sender to complete multicast (this also ensures sender has collected our partial)
    noc_semaphore_wait(reduce_sender_semaphore_addr_ptr, VALID);
    WAYPOINT("RR4");
    DPRINT << "READER_RECEIVER: got global norm from sender" << ENDL();

    // DO NOT pop cb_norm_partial - no need, sender has already collected it via NOC

    // Push the received global norm
    cb_push_back(cb_norm_global, 1);
    WAYPOINT("RR5");
}

}  // namespace

void kernel_main() {
    constexpr uint32_t cb_id_in0 = get_compile_time_arg_val(0);
    constexpr auto src_args = TensorAccessorArgs<1>();

    uint32_t reduce_receiver_semaphore_addr = get_semaphore(get_compile_time_arg_val(2));
    uint32_t reduce_sender_semaphore_addr = get_semaphore(get_compile_time_arg_val(3));
    constexpr uint32_t num_cores = get_compile_time_arg_val(4);
    constexpr uint32_t p_bits = get_compile_time_arg_val(5);

    uint32_t src_addr = get_arg_val<uint32_t>(0);
    uint32_t num_tiles = get_arg_val<uint32_t>(1);
    uint32_t start_id = get_arg_val<uint32_t>(2);
    uint32_t is_sender = get_arg_val<uint32_t>(3);
    uint32_t mcast_dest_noc_start_x = get_arg_val<uint32_t>(4);
    uint32_t mcast_dest_noc_start_y = get_arg_val<uint32_t>(5);
    uint32_t mcast_dest_noc_end_x = get_arg_val<uint32_t>(6);
    uint32_t mcast_dest_noc_end_y = get_arg_val<uint32_t>(7);
    // For sender: remote_noc_x/y arrays start at arg[8], containing receiver core coordinates
    // For receiver: arg[8] and arg[9] are sender coordinates, followed by zeros
    L1Ptr remote_noc_x = (L1Ptr)(get_arg_addr(8));
    L1Ptr remote_noc_y = (L1Ptr)(get_arg_addr(8 + num_cores));

    constexpr uint32_t cb_norm_partial = tt::CBIndex::c_3;
    constexpr uint32_t cb_norm_global = tt::CBIndex::c_5;
    constexpr uint32_t cb_norm_external = tt::CBIndex::c_6;

    const uint32_t tile_bytes = get_tile_size(cb_id_in0);
    const auto s = TensorAccessor(src_args, src_addr, tile_bytes);
    const uint32_t single_tile_size_bytes = get_tile_size(cb_norm_partial);

    volatile tt_l1_ptr uint32_t* reduce_sender_semaphore_addr_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(reduce_sender_semaphore_addr);
    volatile tt_l1_ptr uint32_t* reduce_receiver_semaphore_addr_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(reduce_receiver_semaphore_addr);

    const uint64_t multicast_data_noc = get_noc_multicast_addr(
        mcast_dest_noc_start_x, mcast_dest_noc_start_y, mcast_dest_noc_end_x, mcast_dest_noc_end_y, 0);
    const uint64_t reduce_sender_semaphore_noc_addr = multicast_data_noc | reduce_sender_semaphore_addr;

    DPRINT << "READER: start, is_sender=" << (uint32_t)is_sender << " num_tiles=" << num_tiles
           << " start_id=" << start_id << ENDL();
    WAYPOINT("RST");

    read_tiles_from_dram(cb_id_in0, start_id, num_tiles, s);
    DPRINT << "READER: finished reading tiles, waiting for partial norm, num_tiles=" << num_tiles << ENDL();
    WAYPOINT("RWF");
    cb_wait_front(cb_norm_partial, 1);
    WAYPOINT("RWD");
    DPRINT << "READER: got partial norm, starting all-reduce" << ENDL();
    WAYPOINT("RAR");

    if (is_sender != 1) {
        DPRINT << "READER: receiver path, sending local norm to sender" << ENDL();
        // For receivers, sender coordinates are at args[8] and args[9]
        uint32_t sender_noc_x = get_arg_val<uint32_t>(8);
        uint32_t sender_noc_y = get_arg_val<uint32_t>(9);
        send_local_norm_to_collector(
            cb_norm_partial,
            cb_norm_global,
            reduce_receiver_semaphore_addr,
            reduce_sender_semaphore_addr_ptr,
            sender_noc_x,
            sender_noc_y);
        DPRINT << "READER: receiver finished, got global norm" << ENDL();
    } else {
        DPRINT << "READER: sender path, collecting all partials" << ENDL();
        wait_local_norms_and_mcast_global_norm(
            cb_norm_partial,
            cb_norm_global,
            cb_norm_external,
            num_cores,
            single_tile_size_bytes,
            reduce_sender_semaphore_addr_ptr,
            reduce_receiver_semaphore_addr_ptr,
            multicast_data_noc,
            reduce_sender_semaphore_addr,
            reduce_sender_semaphore_noc_addr,
            remote_noc_x,
            remote_noc_y);
        DPRINT << "READER: sender finished, multicasted global norm" << ENDL();
    }

    DPRINT << "READER: re-reading tiles for scaling phase" << ENDL();
    read_tiles_from_dram(cb_id_in0, start_id, num_tiles, s);
    DPRINT << "READER: done" << ENDL();
}
