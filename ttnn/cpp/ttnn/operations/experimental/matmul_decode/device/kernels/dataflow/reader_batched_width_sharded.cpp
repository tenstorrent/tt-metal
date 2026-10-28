// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/endpoints.h"
#ifdef ENABLE_GLOBAL_CB
#include "api/remote_circular_buffer.h"
#endif

// Gathers this core's batch-block rows of width(K)-sharded A into full_in0 (sender-major layout).
//
// IN0_REPLICATED: A is ROW_MAJOR HEIGHT_SHARDED and already holds the full [M, batch * K] on this
// core (row m is every batch's K side by side, so M = 1 is also [batch, K]). Copy this core's Bc
// batches into full_in0 as [Bc][M][K] instead of the width-sharded gather.
//
// in1 (weights) arrive one of three ways:
//   - default: the in1 CB is globally allocated over the L1-resident weight shard, so the
//     reader only has to declare its tiles available.
//   - IN1_FROM_DRAM: the weight is DRAM ND-sharded, one [Bc*K, Nc] slab per core. The reader
//     issues the slab's read before the gather so the two overlap, and waits for it before
//     publishing full_in0: compute takes in1 only after full_in0, so that gates it.
//   - ENABLE_GLOBAL_CB: the weights are pushed into a DRAM-sender GlobalCircularBuffer by the
//     tensor prefetcher. This receiver's [Bc*K, Nc] slab arrives as num_k_blocks remote pages,
//     each a whole number of rows (num_k_blocks == 1 is the whole slab in one page). The reader
//     waits for a page, hands its tiles to compute through the local alias CB, and releases it
//     only after compute signals (via the sync CB) that it has finished reading -- releasing
//     earlier would let the prefetcher overwrite weights still in use. This kernel runs only on
//     the B core range set, so every core is a receiver.
void kernel_main() {
    // CB indices come in as named args so op fusion can remap them onto the hardware slots it
    // pool-allocates across phases (see models/experimental/ops/descriptors/fusion/docs/op_fusion.md).
    constexpr uint32_t in0_cb_index = get_named_compile_time_arg_val("cb_in0");
    constexpr uint32_t full_in0_cb_index = get_named_compile_time_arg_val("cb_full_in0");
    constexpr uint32_t in1_cb_index = get_named_compile_time_arg_val("cb_in1");
    constexpr uint32_t remote_cb_index = get_named_compile_time_arg_val("cb_in1_remote");
    constexpr uint32_t sync_cb_index = get_named_compile_time_arg_val("cb_sync");

    constexpr uint32_t block_slice_tiles = get_compile_time_arg_val(0);
    constexpr uint32_t tile_size_bytes = get_compile_time_arg_val(1);
    constexpr uint32_t num_senders = get_compile_time_arg_val(2);
    constexpr uint32_t in1_page_tiles = get_compile_time_arg_val(3);
    constexpr uint32_t num_k_blocks = get_compile_time_arg_val(4);
    constexpr uint32_t Bc = get_compile_time_arg_val(5);
    constexpr uint32_t inA_K_tiles_per_core = get_compile_time_arg_val(6);
    constexpr uint32_t M_tiles = get_compile_time_arg_val(7);
    constexpr uint32_t batch = get_compile_time_arg_val(8);
    constexpr uint32_t K_tiles = num_senders * inA_K_tiles_per_core;
    // One in0 tile row of this core's K slice (the whole K on the replicated path).
    constexpr uint32_t k_slice_bytes = inA_K_tiles_per_core * tile_size_bytes;

    const uint32_t b_idx = get_arg_val<uint32_t>(0);
#ifdef IN1_FROM_DRAM
    // Runtime args: [b_idx, weight address, weight shard, sender (x, y)...].
    constexpr auto in1_args = TensorAccessorArgs<9>();
    constexpr uint32_t sender_arg_base = 3;
#else
    // Runtime args: [b_idx, sender (x, y)...].
    constexpr uint32_t sender_arg_base = 1;
#endif

    const uint32_t block_slice_bytes = block_slice_tiles * tile_size_bytes;

    Noc noc;
    CircularBuffer in0_cb(in0_cb_index);
    CircularBuffer in1_cb(in1_cb_index);
    CircularBuffer full_in0_cb(full_in0_cb_index);
    UnicastEndpoint sender;

#ifdef ENABLE_GLOBAL_CB
    // Publish the first page before the gather so the prefetcher's transfer overlaps it.
    in1_cb.reserve_back(in1_page_tiles);
    experimental::remote_cb_wait_front(remote_cb_index, 1);
    in1_cb.push_back(in1_page_tiles);
#elif defined(IN1_FROM_DRAM)
    // TODO: compute waits for the whole slab, so the matmul does not overlap the DRAM read (o_a:
    // ~85 us read, then the per-row compute on top). Publishing the slab in K-blocks, as the GCB
    // path does, would hide the compute and make M > 1 cost about what M = 1 does.
    const auto in1 = TensorAccessor(in1_args, get_arg_val<uint32_t>(1), in1_cb.get_tile_size());
    in1_cb.reserve_back(in1_page_tiles);
    noc_async_read_shard(get_arg_val<uint32_t>(2), in1, in1_cb.get_write_ptr());
#else
    in1_cb.reserve_back(in1_page_tiles);
    in1_cb.push_back(in1_page_tiles);
#endif
    full_in0_cb.reserve_back(num_senders * block_slice_tiles);

    // in0 shard is at the same L1 offset on every core.
    const uint32_t src_addr = in0_cb.get_read_ptr() + b_idx * block_slice_bytes;
#ifdef IN0_REPLICATED
    // A is already the full [M, batch * K] replica on this core (ROW_MAJOR, 1x32 faces).
    // custom_mm and matmul_block both consume full_in0 as [bc][m][k] (inA_K_tiles_per_core is
    // the full K), so a single row is one contiguous copy and several are transposed row by row.
    if constexpr (M_tiles == 1) {
        noc_async_write(src_addr, get_noc_addr(full_in0_cb.get_write_ptr()), block_slice_bytes);
    } else {
        const uint32_t in0_base = in0_cb.get_read_ptr();
        const uint32_t dst_base = full_in0_cb.get_write_ptr();
        for (uint32_t m = 0; m < M_tiles; ++m) {
            for (uint32_t bc_i = 0; bc_i < Bc; ++bc_i) {
                const uint32_t src_row = m * batch + b_idx * Bc + bc_i;
                noc_async_write(
                    in0_base + src_row * k_slice_bytes,
                    get_noc_addr(dst_base + (bc_i * M_tiles + m) * k_slice_bytes),
                    k_slice_bytes);
            }
        }
    }
#elif defined(USE_CUSTOM_MM)
    // Restripe sender-major [sender][bc][m][k_local] into [bc][m][k_global] so custom_mm sees
    // kt_dim consecutive K tiles per in0 tile row.
    for (uint32_t s = 0; s < num_senders; ++s) {
        const uint32_t sender_x = get_arg_val<uint32_t>(sender_arg_base + 2 * s);
        const uint32_t sender_y = get_arg_val<uint32_t>(sender_arg_base + 1 + 2 * s);
        for (uint32_t row = 0; row < Bc * M_tiles; ++row) {
            noc.async_read(
                sender,
                full_in0_cb,
                k_slice_bytes,
                {.noc_x = sender_x, .noc_y = sender_y, .addr = src_addr + row * k_slice_bytes},
                {.offset_bytes = (row * K_tiles + s * inA_K_tiles_per_core) * tile_size_bytes});
        }
    }
#else
    for (uint32_t s = 0; s < num_senders; ++s) {
        const uint32_t sender_x = get_arg_val<uint32_t>(sender_arg_base + 2 * s);
        const uint32_t sender_y = get_arg_val<uint32_t>(sender_arg_base + 1 + 2 * s);
        noc.async_read(
            sender,
            full_in0_cb,
            block_slice_bytes,
            {.noc_x = sender_x, .noc_y = sender_y, .addr = src_addr},
            {.offset_bytes = s * block_slice_bytes});
    }
#endif
#ifdef IN0_REPLICATED
    noc.async_write_barrier();
#endif
#if !defined(IN0_REPLICATED) || defined(IN1_FROM_DRAM)
    noc.async_read_barrier();
#endif
#ifdef IN1_FROM_DRAM
    in1_cb.push_back(in1_page_tiles);
#endif

    full_in0_cb.push_back(num_senders * block_slice_tiles);

#ifdef ENABLE_GLOBAL_CB
    CircularBuffer sync_cb(sync_cb_index);
    // Page p-1's credit is returned only after page p has been published, so the reader runs a
    // page ahead of the credit return and the remote read pointer lags by one page. That is what
    // makes the wait below mean "the page I still owe a credit for, plus the new one": it can
    // never reach past this weight's own pages into a transfer nobody has queued.
    for (uint32_t page = 1; page < num_k_blocks; ++page) {
        in1_cb.reserve_back(in1_page_tiles);
        experimental::remote_cb_wait_front(remote_cb_index, 2);
        in1_cb.push_back(in1_page_tiles);
        sync_cb.wait_front(1);
        sync_cb.pop_front(1);
        experimental::remote_cb_pop_front(remote_cb_index, 1);
    }
    // Compute signals here once it has finished reading every in1 tile of the last page.
    sync_cb.wait_front(1);
    sync_cb.pop_front(1);
    experimental::remote_cb_pop_front(remote_cb_index, 1);
    // Persist the remote read pointer so the next invocation resumes at the right ring offset.
    experimental::update_remote_cb_config_in_l1(remote_cb_index);
    noc.async_atomic_barrier();
#endif
}
