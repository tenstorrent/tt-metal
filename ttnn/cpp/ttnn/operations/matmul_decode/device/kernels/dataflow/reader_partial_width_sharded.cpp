// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// matmul_decode targets a newer tt-metal where the OO dataflow API (Noc / CircularBuffer /
// Semaphore / UnicastEndpoint) lives in split api/dataflow/*.h headers. In this version those
// classes live under experimental/*.h in namespace experimental; api/dataflow/dataflow_api.h still
// provides the C-style primitives (noc_async_*, cb_*, get_arg_val, TensorAccessor).
#include "api/dataflow/dataflow_api.h"
#include "experimental/noc.h"
#include "experimental/circular_buffer.h"
#include "experimental/noc_semaphore.h"
using experimental::CircularBuffer;
using experimental::Noc;
using experimental::Semaphore;
using experimental::use;

// Full-width-sharded matmul activation (in0 / A) reader.
//
// Input A is width(K)-sharded across only a *subset* of the compute cores
// ("sender" cores), while every compute core holds a slice of B and produces a
// slice of the output.  Each compute core needs the *entire* A matrix to do its
// matmul, so this kernel gathers A onto every core using just two semaphores:
//
//   1. Every core reserves room for the full A in `full_in0_cb` (large enough
//      for M_tiles * K_tiles).
//   2. Each sender core broadcasts its own K-slice of A (its sharded `in0_cb`)
//      to all receiver cores (including itself) into the matching offset of
//      `full_in0_cb` via a loopback multicast, then atomically increments the
//      `gather` semaphore on the coordinator (first) core.
//   3. The coordinator core waits until the `gather` semaphore reaches
//      `num_senders` (i.e. every sender has broadcast its slice), then sets the
//      `done` semaphore on all cores (including itself) via multicast.
//   4. Every core waits on the `done` semaphore, then publishes the now
//      fully-populated `full_in0_cb` to the compute kernel.
//
// Uses the Device 2.0 data movement API (Noc / CircularBuffer / Semaphore).
void kernel_main() {
    // ---- Compile-time args ----
    constexpr uint32_t in0_cb_index = get_compile_time_arg_val(0);       // this core's sharded A slice (source)
    constexpr uint32_t full_in0_cb_index = get_compile_time_arg_val(1);  // gathered full A (destination)
    constexpr uint32_t shard_num_tiles = get_compile_time_arg_val(2);    // tiles per A slice (M_tiles * inA_K_per_core)
    constexpr uint32_t tile_size_bytes = get_compile_time_arg_val(3);
    constexpr uint32_t num_senders = get_compile_time_arg_val(4);    // # of A-holding cores
    constexpr uint32_t num_receivers = get_compile_time_arg_val(5);  // cores in the mcast rectangle (incl. self)
    uint32_t mcast_x_start = get_compile_time_arg_val(6);
    uint32_t mcast_y_start = get_compile_time_arg_val(7);
    uint32_t mcast_x_end = get_compile_time_arg_val(8);
    uint32_t mcast_y_end = get_compile_time_arg_val(9);
    constexpr uint32_t gather_sem_id = get_compile_time_arg_val(10);  // incremented by every sender on the coordinator
    constexpr uint32_t done_sem_id = get_compile_time_arg_val(11);    // broadcast by the coordinator to all cores
    constexpr uint32_t coordinator_noc_x = get_compile_time_arg_val(12);  // first core (gather/broadcast hub)
    constexpr uint32_t coordinator_noc_y = get_compile_time_arg_val(13);
    constexpr uint32_t in1_cb_index = get_compile_time_arg_val(14);   // this core's sharded B slice
    constexpr uint32_t in1_num_tiles = get_compile_time_arg_val(15);  // tiles of B resident on this core
    // reshard_input path: when set, senders NoC-read their K-slice of the INTERLEAVED A buffer into
    // in0_cb before the multicast gather (folding the caller's to_memory_config reshard into the op).
    constexpr uint32_t read_interleaved = get_compile_time_arg_val(16);
    constexpr uint32_t in0_M_tiles = get_compile_time_arg_val(17);           // A height in tiles
    constexpr uint32_t in0_K_tiles_per_core = get_compile_time_arg_val(18);  // this sender's K-slice width (tiles)
    constexpr uint32_t in0_K_tiles_total = get_compile_time_arg_val(19);     // global A width in tiles (page stride)
    // fused_residual path: base cores NoC-read their [M_tiles x Nc_tiles] N-slice of the interleaved
    // residual buffer into residual_cb for the gated-residual epilogue (out = residual + gate*(A@B)).
    constexpr uint32_t fused_residual = get_compile_time_arg_val(20);
    constexpr uint32_t residual_cb_index = get_compile_time_arg_val(21);
    constexpr uint32_t residual_Nc_tiles = get_compile_time_arg_val(22);
    constexpr uint32_t residual_N_tiles = get_compile_time_arg_val(23);  // residual width in tiles (page stride)
    constexpr uint32_t residual_tile_size_bytes = get_compile_time_arg_val(24);  // residual dtype tile size
    constexpr uint32_t gate_cb_index = get_compile_time_arg_val(25);             // buffer-backed gate (publish it)
    constexpr uint32_t gate_num_tiles = get_compile_time_arg_val(26);            // Nc_tiles gate tiles
    constexpr auto in0_args = TensorAccessorArgs<27>();
    constexpr auto residual_args = TensorAccessorArgs<in0_args.next_compile_time_args_offset()>();

    // ---- Runtime args ----
    const uint32_t is_sender = get_arg_val<uint32_t>(0);       // 1 if this core holds a slice of A
    const uint32_t sender_id = get_arg_val<uint32_t>(1);       // K-slice index (valid when is_sender)
    const uint32_t is_coordinator = get_arg_val<uint32_t>(2);  // 1 if this is the first core
    const uint32_t in0_buffer_addr = get_arg_val<uint32_t>(3);  // interleaved A base addr (read_interleaved only)
    const uint32_t is_base = get_arg_val<uint32_t>(4);          // 1 if this core owns an output N-slice
    const uint32_t res_n_idx = get_arg_val<uint32_t>(5);        // this base core's N-slice index
    const uint32_t residual_buffer_addr = get_arg_val<uint32_t>(6);  // interleaved residual base addr

    constexpr uint32_t full_num_tiles = num_senders * shard_num_tiles;
    const uint32_t shard_size_bytes = shard_num_tiles * tile_size_bytes;

    // NOC_1 uses an inverted coordinate system, so the rectangle corners swap.
    if (noc_index == 1) {
        std::swap(mcast_x_start, mcast_x_end);
        std::swap(mcast_y_start, mcast_y_end);
    }

    Noc noc;
    CircularBuffer in0_cb(in0_cb_index);
    CircularBuffer in1_cb(in1_cb_index);
    CircularBuffer full_in0_cb(full_in0_cb_index);
    Semaphore<> gather_sem(gather_sem_id);
    Semaphore<> done_sem(done_sem_id);

    // B (in1) is already resident in L1; just publish it to compute.
    in1_cb.reserve_back(in1_num_tiles);
    in1_cb.push_back(in1_num_tiles);

    // fused_residual: this base core NoC-reads its [in0_M_tiles x residual_Nc_tiles] N-slice of the
    // interleaved residual into residual_cb (page = mt*N_tiles + n_idx*Nc_tiles + nc -- identical to
    // the interleaved-output writer's scatter), so compute can add it after the gate multiply.
    if (fused_residual && is_base) {
        // gate is resident (buffer-backed); publish it to compute like in1.
        cb_reserve_back(gate_cb_index, gate_num_tiles);
        cb_push_back(gate_cb_index, gate_num_tiles);
        const auto res_acc = TensorAccessor(residual_args, residual_buffer_addr, residual_tile_size_bytes);
        const uint32_t res_num_tiles = in0_M_tiles * residual_Nc_tiles;
        cb_reserve_back(residual_cb_index, res_num_tiles);
        uint32_t res_l1_addr = get_write_ptr(residual_cb_index);
        for (uint32_t mt = 0; mt < in0_M_tiles; ++mt) {
            for (uint32_t nc = 0; nc < residual_Nc_tiles; ++nc) {
                const uint32_t page = mt * residual_N_tiles + res_n_idx * residual_Nc_tiles + nc;
                noc_async_read_tile(page, res_acc, res_l1_addr);
                res_l1_addr += residual_tile_size_bytes;
            }
        }
        noc_async_read_barrier();
        cb_push_back(residual_cb_index, res_num_tiles);
    }

    // reshard_input: this sender NoC-reads its K-slice of the interleaved A into in0_cb in the SAME
    // m-major contiguous order the buffer-backed shard used (shard is [M_tiles*th, in0_K_tiles_per_core],
    // tiles m-major), so the downstream multicast (which reads in0_cb's read_ptr) sees identical bytes.
    if (read_interleaved && is_sender) {
        const auto in0_acc = TensorAccessor(in0_args, in0_buffer_addr, tile_size_bytes);
        in0_cb.reserve_back(shard_num_tiles);
        uint32_t l1_write_addr = in0_cb.get_write_ptr();
        const uint32_t k_base = sender_id * in0_K_tiles_per_core;
        for (uint32_t m = 0; m < in0_M_tiles; ++m) {
            for (uint32_t kk = 0; kk < in0_K_tiles_per_core; ++kk) {
                const uint32_t page = m * in0_K_tiles_total + k_base + kk;
                noc_async_read_tile(page, in0_acc, l1_write_addr);
                l1_write_addr += tile_size_bytes;
            }
        }
        noc_async_read_barrier();
        in0_cb.push_back(shard_num_tiles);
    }

    // Reserve space for the whole A matrix; multicast writes land directly here.
    full_in0_cb.reserve_back(full_num_tiles);

    if (is_sender) {
        // Broadcast this core's contiguous A slice into every core's full_in0_cb
        // at the offset for this K-slice.  full_in0_cb is allocated identically
        // on all cores, so the destination L1 address is the same everywhere.
        const uint32_t dst_offset_bytes = sender_id * shard_size_bytes;
        noc.async_write_multicast<Noc::McastMode::INCLUDE_SRC>(
            use<CircularBuffer::AddrSelector::READ_PTR>(in0_cb),  // source: this core's A slice
            full_in0_cb,                                          // destination: gathered full A (mcast)
            shard_size_bytes,
            num_receivers,
            {.offset_bytes = 0},
            {.noc_x_start = mcast_x_start,
             .noc_y_start = mcast_y_start,
             .noc_x_end = mcast_x_end,
             .noc_y_end = mcast_y_end,
             .offset_bytes = dst_offset_bytes});
        // Ensure the broadcast data has landed at every core before reporting in.
        noc.async_write_barrier();

        if (num_senders > 1) {
            // // Report completion to the coordinator by bumping its gather semaphore.
            gather_sem.up(noc, coordinator_noc_x, coordinator_noc_y, 1);
            noc.async_atomic_barrier();
        } else {
            done_sem.set(1);
            done_sem.set_multicast<Noc::McastMode::INCLUDE_SRC>(
                noc, mcast_x_start, mcast_y_start, mcast_x_end, mcast_y_end, num_receivers);
            noc.async_write_barrier();
        }
    }

    if (is_coordinator && num_senders > 1) {
        // Wait for every sender to have broadcast its slice and reported in.
        gather_sem.wait(num_senders);

        // All of A is now resident on every core: tell everyone it is ready.
        done_sem.set(1);
        done_sem.set_multicast<Noc::McastMode::INCLUDE_SRC>(
            noc, mcast_x_start, mcast_y_start, mcast_x_end, mcast_y_end, num_receivers);
        noc.async_write_barrier();
    }

    // Wait until the coordinator signals that the full A matrix is available.
    done_sem.wait(1);

    // The full A matrix is now resident on this core; hand it to compute.
    full_in0_cb.push_back(full_num_tiles);
}
