// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/endpoints.h"

// Batched-width-sharded matmul activation (in0 / A) reader.
//
// Unlike PartialWidthSharded (which gathers the WHOLE activation onto every core), a batched core
// only needs the Bc batches in its own batch block. So this reader gathers just this core's
// batch-block rows of A -- shrinking full_in0 from batch*M*K to Bc*M*K tiles (a factor of b_blocks
// smaller in L1).
//
// A is width(K)-sharded across `num_senders` sender cores; sender s holds columns
// [s*inA_K_per_core, (s+1)*inA_K_per_core) for ALL batch*M rows, laid out [batch*M_tiles,
// inA_K_per_core] in TILE row-major order. The sharded activation is already resident in L1 before
// the program runs, so this core just NoC-reads the contiguous row range for its batch block
// (b_idx) out of every sender's shard -- no gather semaphores needed. The shard is allocated at the
// same L1 address on every core in the grid, so the local in0 read pointer doubles as the remote
// source address.
//
// full_in0 is SENDER-MAJOR (matching the compute kernel): sender s's block lands at offset
// s*block_slice_tiles, where block_slice_tiles = Bc*M_tiles*inA_K_per_core.
void kernel_main() {
    constexpr uint32_t in0_cb_index = get_compile_time_arg_val(0);       // this core's A shard (source, resident)
    constexpr uint32_t full_in0_cb_index = get_compile_time_arg_val(1);  // gathered batch-block A (destination)
    constexpr uint32_t block_slice_tiles = get_compile_time_arg_val(2);  // Bc*M_tiles*inA_K_per_core (per sender)
    constexpr uint32_t tile_size_bytes = get_compile_time_arg_val(3);
    constexpr uint32_t num_senders = get_compile_time_arg_val(4);
    constexpr uint32_t in1_cb_index = get_compile_time_arg_val(5);   // this core's resident weight block
    constexpr uint32_t in1_num_tiles = get_compile_time_arg_val(6);  // tiles of weights resident on this core

    const uint32_t b_idx = get_arg_val<uint32_t>(0);  // batch-block index owned by this core
    // Runtime args [1 .. 1 + 2*num_senders): interleaved (noc_x, noc_y) of each A sender core.

    const uint32_t block_slice_bytes = block_slice_tiles * tile_size_bytes;

    Noc noc;
    CircularBuffer in0_cb(in0_cb_index);
    CircularBuffer in1_cb(in1_cb_index);
    CircularBuffer full_in0_cb(full_in0_cb_index);
    UnicastEndpoint sender;

    // Weights (in1) are already resident in L1; just publish them to compute.
    in1_cb.reserve_back(in1_num_tiles);
    in1_cb.push_back(in1_num_tiles);

    full_in0_cb.reserve_back(num_senders * block_slice_tiles);

    // This batch block's rows occupy a contiguous tile range in every sender's shard.
    const uint32_t src_addr = in0_cb.get_read_ptr() + b_idx * block_slice_bytes;
    for (uint32_t s = 0; s < num_senders; ++s) {
        const uint32_t sender_x = get_arg_val<uint32_t>(1 + 2 * s);
        const uint32_t sender_y = get_arg_val<uint32_t>(2 + 2 * s);
        noc.async_read(
            sender,
            full_in0_cb,
            block_slice_bytes,
            {.noc_x = sender_x, .noc_y = sender_y, .addr = src_addr},
            {.offset_bytes = s * block_slice_bytes});
    }
    noc.async_read_barrier();

    full_in0_cb.push_back(num_senders * block_slice_tiles);
}
