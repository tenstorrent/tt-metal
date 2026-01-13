// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    // Compile time arguments
    constexpr uint32_t num_experts = get_named_compile_time_arg_val("num_experts");

    constexpr auto in_args = TensorAccessorArgs<0>();
    constexpr auto w0_args = TensorAccessorArgs<in_args.next_compile_time_args_offset()>();
    constexpr auto w1_args = TensorAccessorArgs<w0_args.next_compile_time_args_offset()>();
    constexpr auto w2_args = TensorAccessorArgs<w1_args.next_compile_time_args_offset()>();
    constexpr auto out_args = TensorAccessorArgs<w2_args.next_compile_time_args_offset()>();

    // Run-time arguments
    uint32_t argidx = 0;
    const auto core_id = get_arg_val<uint32_t>(argidx++);
    const auto vchannel = get_arg_val<uint32_t>(argidx++);
    const auto in_addr = get_arg_val<uint32_t>(argidx++);
    const auto w0_addr = get_arg_val<uint32_t>(argidx++);
    const auto w1_addr = get_arg_val<uint32_t>(argidx++);
    const auto w2_addr = get_arg_val<uint32_t>(argidx++);
    const auto out_addr = get_arg_val<uint32_t>(argidx++);

    // CBs
    constexpr auto cb_r2c_w0 = tt::CBIndex::c_0;
    constexpr auto cb_s2c_in = tt::CBIndex::c_1;
    constexpr auto cb_c2c_mm0 = tt::CBIndex::c_2;
    constexpr auto cb_c2c_mm1 = tt::CBIndex::c_3;
    constexpr auto cb_c2w_elt = tt::CBIndex::c_4;
    constexpr auto cb_r2c_in2 = tt::CBIndex::c_5;
    constexpr auto cb_c2w_mm2 = tt::CBIndex::c_6;

    // CB Aliases
    constexpr auto cb_r2c_w1 = tt::CBIndex::c_0;
    constexpr auto cb_r2c_w2 = tt::CBIndex::c_0;

    // Tile sizes
    constexpr uint32_t in_tile_size = get_tile_size(cb_s2c_in);
    constexpr uint32_t w0_tile_size = get_tile_size(cb_r2c_w0);
    constexpr uint32_t w1_tile_size = get_tile_size(cb_r2c_w1);
    constexpr uint32_t w2_tile_size = get_tile_size(cb_r2c_w2);
    constexpr uint32_t out_tile_size = get_tile_size(cb_c2w_elt);

    // Tensor accessors
    const auto in_accessor = TensorAccessor(in_args, in_addr, in_tile_size);
    const auto w0_accessor = TensorAccessor(w0_args, w0_addr, w0_tile_size);
    const auto w1_accessor = TensorAccessor(w1_args, w1_addr, w1_tile_size);
    const auto w2_accessor = TensorAccessor(w2_args, w2_addr, w2_tile_size);
    const auto out_accessor = TensorAccessor(out_args, out_addr, out_tile_size);

    // Constants for MoE
    constexpr uint32_t num_w0_w1_tiles_h = 224;
    constexpr uint32_t num_w2_tiles_h = 64;

    const uint32_t num_w0_w1_tiles_w = (core_id < 8) ? 5 : 6;
    const uint32_t num_w2_tiles_w = (core_id < 8) ? 19 : 18;
    const uint32_t num_mm2_tiles = num_w2_tiles_w;

    constexpr uint32_t num_in2_tiles = 64;
    constexpr uint32_t num_elt_tiles = 1;

    constexpr uint32_t w0_w1_stride_w = 1;
    constexpr uint32_t w0_w1_stride_h = 64;
    constexpr uint32_t w2_stride_w = 1;
    constexpr uint32_t w2_stride_h = 224;

    const uint32_t w0_tile_id_start = (core_id < 8) ? (5 * core_id) : (5 * 8 + 6 * (core_id - 8));
    const uint32_t w1_tile_id_start = (core_id < 8) ? (5 * core_id) : (5 * 8 + 6 * (core_id - 8));
    const uint32_t w2_tile_id_start = (core_id < 8) ? (19 * core_id) : (19 * 8 + 18 * (core_id - 8));

    // // Read W0 and W1 from DRAM into CB
    uint32_t w0_tile_id = w0_tile_id_start;
    uint32_t w1_tile_id = w1_tile_id_start;
    uint32_t w2_tile_id = w2_tile_id_start;

    // DRAM Reading constants
    const uint32_t dram_bank_id = core_id;
    const uint64_t dram_noc_addr = get_noc_addr_from_bank_id<true>(dram_bank_id, /*bank_address_offset=*/0);
    uint32_t curr_trid = 1;
    uint32_t prev_trid = 2;

    // W0 and W1 reading constants
    constexpr uint32_t w0_w1_tiles_per_txn = 14;
    constexpr uint32_t w0_w1_bytes_per_txn = w0_w1_tiles_per_txn * w0_tile_size;  // 14 * 576B = 7488B
    const uint32_t w0_w1_txns =
        num_w0_w1_tiles_w * num_w0_w1_tiles_h / w0_w1_tiles_per_txn;  // (5|6 * 224) / 14 = 80|96

    // W2 reading constants
    // Total tiles of w2 does not divide evenly by w2_tiles_per_txn (14), so we do it in two steps
    constexpr uint32_t w2_tiles_per_txn = 14;
    constexpr uint32_t w2_bytes_per_txn = w2_tiles_per_txn * w2_tile_size;  // 14 * 576B = 7488B
    const uint32_t w2_txns = ((num_w2_tiles_h * num_w2_tiles_w) + w2_tiles_per_txn - 1) / w2_tiles_per_txn;  // round up

    //-------------------------------------------------------------------------
    // Read W0 with pipelined reads
    //-------------------------------------------------------------------------
    uint32_t w0_read_offset = 0;
    for (uint32_t expert_id = 0; expert_id < num_experts; ++expert_id) {
        noc_async_read_one_packet_set_state</*use_vc=*/true>(
            /*src_noc_addr=*/dram_noc_addr, /*size=*/w0_w1_bytes_per_txn, /*vc=*/vchannel);

        for (uint32_t txn = 0; txn < w0_w1_txns; ++txn) {
            // Issue reads with current trid
            cb_reserve_back(cb_r2c_w0, w0_w1_tiles_per_txn);
            uint32_t write_addr = get_write_ptr(cb_r2c_w0);

            noc_async_read_set_trid(curr_trid);
            noc_async_read_one_packet_with_state_with_trid</*skip_ptr_update=*/false, /*skip_cmdbuf_chk=*/true>(
                /*src_base_addr=*/dram_noc_addr, /*src_addr=*/w0_read_offset, /*dest_addr=*/write_addr, curr_trid);

            // After first block: wait for OTHER trid, push
            if (txn > 0) {
                noc_async_read_barrier_with_trid(prev_trid);
                cb_push_back(cb_r2c_w0, w0_w1_tiles_per_txn);
            }

            // Swap trids
            std::swap(curr_trid, prev_trid);

            // Increment read offset
            w0_read_offset += w0_w1_bytes_per_txn;
        }

        // Final cleanup
        noc_async_read_barrier_with_trid(prev_trid);
        cb_push_back(cb_r2c_w0, w0_w1_tiles_per_txn);

        //-------------------------------------------------------------------------
        // Read W1 with pipelined reads
        //-------------------------------------------------------------------------
        uint32_t w1_read_offset = w0_read_offset;

        noc_async_read_one_packet_set_state</*use_vc=*/true>(
            /*src_noc_addr=*/dram_noc_addr, /*size=*/w0_w1_bytes_per_txn, /*vc=*/vchannel);

        for (uint32_t txn = 0; txn < w0_w1_txns; ++txn) {
            // Issue reads with current trid
            cb_reserve_back(cb_r2c_w1, w0_w1_tiles_per_txn);
            uint32_t write_addr = get_write_ptr(cb_r2c_w1);

            noc_async_read_set_trid(curr_trid);
            noc_async_read_one_packet_with_state_with_trid</*skip_ptr_update=*/false, /*skip_cmdbuf_chk=*/true>(
                /*src_base_addr=*/dram_noc_addr, /*src_addr=*/w1_read_offset, /*dest_addr=*/write_addr, curr_trid);

            // After first block: wait for OTHER trid, push
            if (txn > 0) {
                noc_async_read_barrier_with_trid(prev_trid);
                cb_push_back(cb_r2c_w1, w0_w1_tiles_per_txn);
            }

            // Swap trids
            std::swap(curr_trid, prev_trid);

            // Increment read offset
            w1_read_offset += w0_w1_bytes_per_txn;
        }

        // Final cleanup
        noc_async_read_barrier_with_trid(prev_trid);
        cb_push_back(cb_r2c_w1, w0_w1_tiles_per_txn);

        //-------------------------------------------------------------------------
        // Read W2 with pipelined reads
        //-------------------------------------------------------------------------
        uint32_t w2_read_offset = w1_read_offset;

        noc_async_read_one_packet_set_state</*use_vc=*/true>(
            /*src_noc_addr=*/dram_noc_addr, /*size=*/w2_tiles_per_txn, /*vc=*/vchannel);

        for (uint32_t txn = 0; txn < w2_txns; ++txn) {
            // Issue reads with current trid
            cb_reserve_back(cb_r2c_w2, w2_tiles_per_txn);
            uint32_t write_addr = get_write_ptr(cb_r2c_w2);

            noc_async_read_set_trid(curr_trid);
            noc_async_read_one_packet_with_state_with_trid</*skip_ptr_update=*/false, /*skip_cmdbuf_chk=*/true>(
                /*src_base_addr=*/dram_noc_addr, /*src_addr=*/w2_read_offset, /*dest_addr=*/write_addr, curr_trid);

            // After first block: wait for OTHER trid, push
            if (txn > 0) {
                noc_async_read_barrier_with_trid(prev_trid);
                cb_push_back(cb_r2c_w2, w2_tiles_per_txn);
            }

            // Swap trids
            std::swap(curr_trid, prev_trid);

            // Increment read offset
            w2_read_offset += w2_bytes_per_txn;
        }

        // Final cleanup
        noc_async_read_barrier_with_trid(prev_trid);
        cb_push_back(cb_r2c_w2, w2_tiles_per_txn);
    }

    //-------------------------------------------------------------------------
}
