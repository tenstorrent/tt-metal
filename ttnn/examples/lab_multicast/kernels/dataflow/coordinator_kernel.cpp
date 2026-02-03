// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "api/debug/dprint.h"  // required in all kernels using DPRINT
#include "api/dataflow/dataflow_api.h"

// Coordinator kernel: reads a tile from DRAM and multicasts it to receiver cores.
// This kernel runs on the sender core (e.g., logical core 0,0).
//
// Ensure DPRINT is enabled: export TT_METAL_DPRINT_CORES='(0,0)-(3,0)'
void kernel_main() {

    ////////// RUNTIME ARGS & VARS //////////
    uint32_t start_x = get_arg_val<uint32_t>(0);
    uint32_t start_y = get_arg_val<uint32_t>(1);
    uint32_t end_x = get_arg_val<uint32_t>(2);
    uint32_t end_y = get_arg_val<uint32_t>(3);
    uint32_t sender_addr = get_semaphore(get_arg_val<uint32_t>(4));
    uint32_t receiver_addr = get_semaphore(get_arg_val<uint32_t>(5));
    uint32_t src0_base_addr = get_arg_val<uint32_t>(6);
    uint32_t single_tile_size = get_arg_val<uint32_t>(7);
    uint32_t num_dests = get_arg_val<uint32_t>(8);

    ////////// BUFFER SETUP //////////
    // Use TensorAccessorArgs to handle DRAM addressing without needing to know bank IDs.
    // The layout parameters are passed as compile-time arguments.
    constexpr uint32_t cb_id_in0 = tt::CB::c_in0;  // index=0
    constexpr uint32_t tile_size_bytes = get_tile_size(cb_id_in0);
    uint32_t tile_l1_addr = get_write_ptr(cb_id_in0);

    // Create address generator for the input buffer using TensorAccessorArgs.
    // TensorAccessorArgs extracts data distribution details from compile-time arguments.
    constexpr auto src0_layout_args = TensorAccessorArgs<0>();
    const auto src0_addr_gen = TensorAccessor(src0_layout_args, src0_base_addr, tile_size_bytes);

    ////////// READ TILE DRAM->L1 //////////
    // Read tile index 0 (the only tile in the input tensor) using the address generator.
    noc_async_read_tile(0, src0_addr_gen, tile_l1_addr);
    noc_async_read_barrier();

    ////////// PRINT TILE SLICE //////////
    SliceRange sr = SliceRange{
        .h0 = static_cast<uint8_t>(0),
        .h1 = static_cast<uint8_t>(32),
        .hs = 8,
        .w0 = 0,
        .w1 = 32,
        .ws = 8};
    DPRINT << TileSlice(cb_id_in0, 0, sr, TSLICE_INPUT_CB, TSLICE_WR_PTR, true, false);

    cb_push_back(cb_id_in0, 1);

    ////////// SEMAPHORE SETUP //////////
    volatile tt_l1_ptr uint32_t* sender_addr_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sender_addr);
    volatile tt_l1_ptr uint32_t* receiver_addr_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(receiver_addr);

    ////////// WAIT FOR WORKER CORES' READINESS //////////
    noc_semaphore_wait(sender_addr_ptr, num_dests);

    ////////// RESET SEMAPHORE //////////
    // Set to 0 rather than INVALID because sender_addr_ptr can take on a value > 1
    noc_semaphore_set(sender_addr_ptr, 0);

    DPRINT << "CORE (" << (uint32_t)get_absolute_logical_x() << "," << (uint32_t)get_absolute_logical_y()
           << "): Tile ready for multicast. I am starting all inbound kernels in cores in given range." << ENDL()
           << ENDL();

    ////////// MULTICAST TILE TO RECEIVERS //////////
    uint64_t identity_tile_global_multicast_addr =
        get_noc_multicast_addr(start_x, start_y, end_x, end_y, tile_l1_addr);
    noc_async_write_multicast(tile_l1_addr, identity_tile_global_multicast_addr, single_tile_size, num_dests);

    ////////// MULTICAST 'VALID' STATE TO RECEIVERS //////////
    *(receiver_addr_ptr) = VALID;
    uint64_t validity_global_multicast_addr = get_noc_multicast_addr(start_x, start_y, end_x, end_y, receiver_addr);
    noc_semaphore_set_multicast(receiver_addr, validity_global_multicast_addr, num_dests);

    noc_async_write_barrier();

    // From this stage onwards, the coordinator can safely do more work.
    // The user is encouraged to play around with coordinator behavior (i.e., retrieve more tiles from DRAM,
    // aggregate tile work from worker cores, etc.) as an exercise.
}
