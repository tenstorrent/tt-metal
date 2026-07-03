// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// all_gather — ring dataflow READER (NCRISC). ONE source, run on both cores of a
// device (core_fwd = (0,0), core_bwd = (0,1)); `direction` (CT arg) selects the
// forward vs backward channel. Pure data movement (no compute).
//
// It feeds the paired writer's fabric egress via cb_relay:
//   1. OWN slice  — read this device's input shard S_d -> cb_relay (once), so the
//      writer can locally place it in output slot d (fwd core) and/or forward it.
//   2. RELAYS     — store-and-forward: after the counting semaphore reports the
//      r-th neighbour slice has LANDED in this device's OWN output (data-before-inc
//      ordering, guaranteed by the fabric route), re-read that landed slice from
//      the local output tensor -> cb_relay so the writer forwards it one more hop.
//
// The op owns (per the CCL helper contract): the receive INGRESS (raw noc_async_read
// of the landed slice), TensorAccessor addressing, the WAITING half of the counting
// handshake (noc_semaphore_wait_min), and the cache-reuse re-arm (noc_semaphore_set).

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

// Stream n_tiles contiguous tiles [base_tile, base_tile + n_tiles) from `acc` into
// cb one tile at a time (small streaming CB; the writer pops them concurrently).
template <typename Acc>
FORCE_INLINE void stream_tiles(const Acc& acc, uint32_t base_tile, uint32_t n_tiles, uint32_t cb, uint32_t page_size) {
    for (uint32_t t = 0; t < n_tiles; ++t) {
        cb_reserve_back(cb, 1);
        const uint32_t l1_write_addr = get_write_ptr(cb);
        noc_async_read(acc.get_noc_addr(base_tile + t), l1_write_addr, page_size);
        noc_async_read_barrier();
        cb_push_back(cb, 1);
    }
}

// Concat-by-gather_dim output page for input page in_p of the slice from origin j
// (whole-page remap — op_design "Dataflow Strategy" stride table). block_in = input
// pages from the gather axis down; gather_dim=0 => block_in = pages_per_shard (contiguous).
FORCE_INLINE uint32_t concat_out_page(uint32_t in_p, uint32_t j, uint32_t block_in, uint32_t ring_size) {
    return (in_p / block_in) * (block_in * ring_size) + (in_p % block_in) + j * block_in;
}

void kernel_main() {
    constexpr uint32_t ring_size = get_compile_time_arg_val(0);
    constexpr uint32_t ring_index = get_compile_time_arg_val(1);
    constexpr uint32_t cb_relay = get_compile_time_arg_val(2);
    constexpr uint32_t direction = get_compile_time_arg_val(3);  // 0 = forward, 1 = backward
    constexpr uint32_t pages_per_shard = get_compile_time_arg_val(4);
    constexpr uint32_t page_size = get_compile_time_arg_val(5);
    constexpr uint32_t has_neighbor = get_compile_time_arg_val(6);     // this direction has a neighbour chip
    constexpr uint32_t num_recv_slices = get_compile_time_arg_val(7);  // counting-sem threshold / relay count
    constexpr uint32_t does_own_read = get_compile_time_arg_val(8);    // read own shard for the writer
    constexpr uint32_t block_in = get_compile_time_arg_val(9);         // input pages per outer block (concat stride)
    constexpr uint32_t sub_page = get_compile_time_arg_val(10);        // 1 = RM innermost gather (sub-page byte concat)
    constexpr uint32_t output_page_size = get_compile_time_arg_val(11);  // input page size, or N*input for sub-page
    constexpr auto input_args = TensorAccessorArgs<12>();
    constexpr auto output_args = TensorAccessorArgs<input_args.next_compile_time_args_offset()>();

    const uint32_t input_addr = get_arg_val<uint32_t>(0);
    const uint32_t output_addr = get_arg_val<uint32_t>(1);
    const uint32_t counting_sem_addr = get_arg_val<uint32_t>(2);

    const uint32_t input_page_size = page_size;  // relay CB page = one INPUT page
    const auto input_acc = TensorAccessor(input_args, input_addr, input_page_size);
    // Output accessor uses the OUTPUT page stride (== input for whole-page remap; N x larger
    // for the sub-page RM innermost path, where a page holds all N slices of a row).
    const auto output_acc = TensorAccessor(output_args, output_addr, output_page_size);

    // 1. Own slice: read this device's input shard S_d — always the contiguous input page
    //    range [0, pages_per_shard) (the input shard is a plain interleaved tensor).
    if constexpr (does_own_read) {
        stream_tiles(input_acc, 0, pages_per_shard, cb_relay, input_page_size);
    }

    auto* counting_sem = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(counting_sem_addr);

    // 2. Relays: gate each re-read on the counting semaphore (r-th landed slice), then
    //    re-read that landed slice from the LOCAL output (at its concat offset) so the
    //    writer forwards it one more hop.
    if constexpr (has_neighbor) {
        for (uint32_t r = 1; r <= num_recv_slices; ++r) {
            noc_semaphore_wait_min(counting_sem, r);
            // Forward channel receives S_{d-1}, S_{d-2}, ... (origin j = d - r).
            // Backward channel receives S_{d+1}, S_{d+2}, ... (origin j = d + r).
            const uint32_t j = (direction == 0) ? (ring_index - r) : (ring_index + r);
            for (uint32_t in_p = 0; in_p < pages_per_shard; ++in_p) {
                cb_reserve_back(cb_relay, 1);
                const uint32_t l1_write_addr = get_write_ptr(cb_relay);
                uint64_t src_noc;
                if constexpr (sub_page) {
                    // Slice j lives at byte offset j*input_page_size inside output page in_p.
                    src_noc = output_acc.get_noc_addr(in_p, j * input_page_size);
                } else {
                    src_noc = output_acc.get_noc_addr(concat_out_page(in_p, j, block_in, ring_size));
                }
                noc_async_read(src_noc, l1_write_addr, input_page_size);
                noc_async_read_barrier();
                cb_push_back(cb_relay, 1);
            }
        }
    } else if (num_recv_slices > 0) {
        // End device in this direction (no chip to forward to): it still RECEIVES all its
        // slices directly (written by the neighbour), so just count them before re-arming.
        noc_semaphore_wait_min(counting_sem, num_recv_slices);
    }

    // 3. Cache-reuse re-arm: reset AFTER the final wait (all incoming incs already counted).
    noc_semaphore_set(counting_sem, 0);
}
