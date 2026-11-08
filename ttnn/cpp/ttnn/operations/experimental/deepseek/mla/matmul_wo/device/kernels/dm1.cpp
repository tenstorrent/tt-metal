// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/endpoints.h"
#include "api/dataflow/noc.h"
#include "matmul_wo_ring_common.h"

void kernel_main() {
    // Compile time arguments
    constexpr uint32_t layer_id = get_named_compile_time_arg_val("layer_id");
    constexpr uint32_t num_cores = get_named_compile_time_arg_val("num_cores");
    constexpr uint32_t reduce_semaphore_id = get_named_compile_time_arg_val("reduce_semaphore_id");

    constexpr auto in_args = TensorAccessorArgs<0>();
    constexpr auto w_args = TensorAccessorArgs<in_args.next_compile_time_args_offset()>();
    constexpr auto out_args = TensorAccessorArgs<w_args.next_compile_time_args_offset()>();

    // Run-time arguments
    uint32_t argidx = 0;
    const auto dram_bank_id = get_arg_val<uint32_t>(argidx++);
    const auto vchannel = get_arg_val<uint32_t>(argidx++);
    const auto in_addr = get_arg_val<uint32_t>(argidx++);
    const auto w_addr = get_arg_val<uint32_t>(argidx++);
    const auto out_addr = get_arg_val<uint32_t>(argidx++);

    // CBs
    constexpr auto cb_r2c_w_id = tt::CBIndex::c_0;
    constexpr auto cb_s2c_in_id = tt::CBIndex::c_1;
    constexpr auto cb_c2w_out_id = tt::CBIndex::c_2;
    constexpr auto cb_s2c_in2_id = tt::CBIndex::c_3;

    CircularBuffer cb_c2w_out(cb_c2w_out_id);
    CircularBuffer cb_s2c_in2(cb_s2c_in2_id);

    // Tile sizes
    constexpr uint32_t in_tile_size = get_tile_size(cb_s2c_in_id);
    constexpr uint32_t w_tile_size = get_tile_size(cb_r2c_w_id);
    constexpr uint32_t out_tile_size = get_tile_size(cb_c2w_out_id);

    // Constants for the kernel
    constexpr uint32_t num_w_tiles_w = matmul_wo_ring::NUM_W_TILES_W;
    constexpr uint32_t num_n_tiles_per_iter = matmul_wo_ring::N_TILES_PER_ITER;
    constexpr uint32_t max_num_tiles_h = matmul_wo_ring::MAX_K_TILES_PER_CORE;
    constexpr uint32_t num_iters = num_w_tiles_w / num_n_tiles_per_iter;
    const uint32_t num_tiles_h = matmul_wo_ring::K_TILES_PER_CORE_A[dram_bank_id];

    //-------------------------------------------------------------------------
    // Collector core
    //-------------------------------------------------------------------------
    constexpr uint32_t num_collectors = matmul_wo_ring::N_TILES_PER_ITER;
    constexpr uint8_t collector_core_coords[num_collectors][2] = COLLECTOR_CORE_COORDS;

    // Get dst address
    // The partials land in the 7 collector cores' c_3, at the address c_3 has on this core: c_3 is one CB
    // across both core sets, so its L1 address is the same on every core.
    const uint32_t local_collector_base_addr = cb_s2c_in2.get_write_ptr();

    constexpr uint32_t collector_dst_stride = num_cores * out_tile_size;
    const uint32_t collector_offset = dram_bank_id * out_tile_size;
    uint32_t local_collector_dst_addr = local_collector_base_addr + collector_offset;

    //-------------------------------------------------------------------------
    // W reading constants
    //-------------------------------------------------------------------------
    constexpr uint32_t w_txns_per_block = matmul_wo_ring::W_TXNS_PER_BLOCK;
    constexpr uint32_t w_tiles_per_txn = matmul_wo_ring::W_TILES_PER_TXN;
    constexpr uint32_t w_tiles_per_block = w_tiles_per_txn * w_txns_per_block;
    const uint32_t num_blocks_per_iter =
        (num_tiles_h * num_n_tiles_per_iter + w_tiles_per_block - 1) / w_tiles_per_block;
    const uint32_t w_total_blocks = num_blocks_per_iter * num_iters;

    //-------------------------------------------------------------------------
    // Reduction transactions
    //-------------------------------------------------------------------------
    // NoC 1 (this writer kernel's noc_index). Each partial tile is a posted write, followed by a posted increment
    // of that collector's semaphore on the same NoC and VC.
    const Noc noc;
    UnicastEndpoint collector;
    // The increment below needs the semaphore's raw L1 address, which a Device 2.0 Semaphore does not expose
    const uint32_t reduce_semaphore_addr = get_semaphore(reduce_semaphore_id);

    for (uint32_t iter_id = 0; iter_id < num_iters; ++iter_id) {
        cb_c2w_out.wait_front(num_n_tiles_per_iter);
        uint32_t src_offset = 0;

        for (uint32_t collector_idx = 0; collector_idx < num_collectors; ++collector_idx) {
            const uint32_t collector_x = collector_core_coords[collector_idx][0];
            const uint32_t collector_y = collector_core_coords[collector_idx][1];
            noc.set_async_write_state<NocOptions::POSTED | NocOptions::CUSTOM_VC, out_tile_size>(
                collector,
                out_tile_size,
                {.noc_x = collector_x, .noc_y = collector_y, .addr = local_collector_base_addr},
                {.vc = vchannel});
            noc.async_write_with_state<NocOptions::POSTED, out_tile_size>(
                cb_c2w_out, collector, out_tile_size, {.offset_bytes = src_offset}, {.addr = local_collector_dst_addr});
            // Stays on Device 1.0: Device 2.0 has no posted remote increment yet (Semaphore::up is non-posted only)
            noc_semaphore_inc</*posted=*/true>(
                collector.get_noc_unicast_addr(collector_x, collector_y, reduce_semaphore_addr, noc.get_noc_id()),
                /*incr=*/1,
                noc.get_noc_id(),
                /*vc=*/vchannel);

            src_offset += out_tile_size;
        }

        cb_c2w_out.pop_front(num_n_tiles_per_iter);

        local_collector_dst_addr += collector_dst_stride;
    }

    // Ensure the partial writes have left the core before the kernel exits. The posted increments are not tracked;
    // each collector waits for its own count.
    noc.async_writes_flushed<NocOptions::POSTED>();
}
