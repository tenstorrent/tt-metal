// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// matmul_reduce_scatter — compute-core NCRISC: activation (A, in0) operand + hand-off bookkeeping.
//
// Per compute unit (one row-wave of a scatter block, in compute order) and per K-block:
//   stream_operand_block / load_resident_operand: the m-line injector (A_SENDS) reads its rows' K-block from DRAM
//   and multicasts it along its m-line (mcast_pipe SenderPipe); the other cores of the line receive it
//   (ReceiverPipe). Each unit carries its A row origin and a `fresh` flag from the host: when A is the
//   block-invariant operand (scatter_dim=-1) and resident (regime R1), only the first `waves` units (one slice per
//   wave) read; later units replay CB credits (reserve + push of the same pages; capacity = waves K passes exactly,
//   and units run wave 0..waves-1 per block, so the ring wraps onto the wave's own slice).
// handoff_block / release_block (raw semaphores; the mcast pipe is one-sender -> one-rectangle, the hand-off is
//   many producers -> few consumers with cumulative counters):
//   - when block `signalled` is fronted in cb_partial_handoff: one noc_semaphore_inc of sem_block_ready on each of
//     its consumers (the L ports of its direction, or the 2L finals for the chip's own block);
//   - when the ack counter of block `popped`'s consumer kind has reached that kind's cumulative ack count up to and
//     including the block: pop its hand-off slot. One counter per consumer kind (fwd ports, bwd ports, finals): each
//     kind acks its own blocks in order, but the kinds run independently, so a single shared counter could release a
//     slot on another kind's acks before its own readers are done.
// Both are non-blocking polls, run after every K-block transfer and inside every wait of this kernel, so a remote
// wait never stalls the operand stream; only the final drain blocks.

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "ttnn/cpp/ttnn/kernel_lib/mcast/kernel/mcast_args.hpp"
#include "matmul_reduce_scatter_injector.hpp"

using namespace dataflow_kernel_lib;

#ifndef MMRS_FINALS_PER_LINK
#define MMRS_FINALS_PER_LINK 2
#endif

void kernel_main() {
    constexpr uint32_t cb_act_operand = get_compile_time_arg_val(0);
    constexpr uint32_t cb_partial_handoff = get_compile_time_arg_val(1);
    constexpr uint32_t core_m_tiles = get_compile_time_arg_val(2);
    constexpr uint32_t k_block_tiles = get_compile_time_arg_val(3);
    constexpr uint32_t num_k_blocks = get_compile_time_arg_val(4);
    constexpr uint32_t num_blocks = get_compile_time_arg_val(5);   // compute units per call (G blocks x waves)
    constexpr uint32_t block_tiles = get_compile_time_arg_val(6);  // core_m_tiles * core_n_tiles (one hand-off slot)
    constexpr uint32_t a_tile_bytes = get_compile_time_arg_val(7);
    constexpr uint32_t a_sends = get_compile_time_arg_val(8);  // this core injects A along its m-line
    constexpr uint32_t num_links = get_compile_time_arg_val(9);
    constexpr bool read_ahead = get_compile_time_arg_val(10) != 0;  // injector DRAM read-ahead (injector header)
    constexpr uint32_t num_banks = get_compile_time_arg_val(11);    // DRAM banks A is interleaved over
    constexpr auto a_args = TensorAccessorArgs<12>();
    constexpr auto mc_a =
        McastArgs<get_named_compile_time_arg_val("a_ct_offset"), get_named_compile_time_arg_val("a_rt_offset")>();

    constexpr uint32_t kblock_pages = core_m_tiles * k_block_tiles;
    constexpr uint32_t kblock_bytes = kblock_pages * a_tile_bytes;

    size_t arg = 0;
    const uint32_t a_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t a_row_stride = get_arg_val<uint32_t>(arg++);  // Kt (A pages per tile-row)
    const uint32_t a_row0 = get_arg_val<uint32_t>(arg++);        // m-line's first row within a unit
    const uint32_t a_valid_rows = get_arg_val<uint32_t>(arg++);  // <= core_m_tiles (ragged last m-line)
    const uint32_t ready_sem_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t ack_sem_base = arg;  // 3 ack counters (L1 addresses), one per consumer kind
    arg += 3;
    // num_blocks x [unit's A row origin, consumer kind (0 fwd, 1 bwd, 2 finals), kind's cum. acks, A fresh]
    constexpr uint32_t order_stride = 4;
    const uint32_t order_idx = arg;
    arg += order_stride * num_blocks;
    constexpr uint32_t num_finals = MMRS_FINALS_PER_LINK * num_links;
    const uint32_t consumers_idx = arg;  // (2 + F) L packed NoC coords (x << 16 | y): fwd ports, bwd ports, finals
    arg += 2 * num_links + num_finals;

    const auto a_acc = TensorAccessor(a_args, a_addr, a_tile_bytes);

    uint32_t signalled = 0, popped = 0;
    auto kind_ack = [&](uint32_t kind) {
        return *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_arg_val<uint32_t>(ack_sem_base + kind));
    };
    // handoff_block: signal block `signalled` to its consumers once it is fronted.
    auto try_signal = [&]() {
        if (signalled >= num_blocks ||
            !cb_pages_available_at_front(cb_partial_handoff, (signalled - popped + 1) * block_tiles)) {
            return;
        }
        const uint32_t kind = get_arg_val<uint32_t>(order_idx + order_stride * signalled + 1);
        const uint32_t first = kind == 0 ? 0 : (kind == 1 ? num_links : 2 * num_links);
        const uint32_t count = kind == 2 ? num_finals : num_links;
        // A consumer counts ready signals cumulatively from every compute core, so no core may signal its next block
        // before the consumer is done with its current one (else a fast core's next-block signal could stand in for a
        // slow core's missing current-block signal): wait until this kind's consumers have acked every earlier block
        // of the kind.
        if (kind_ack(kind) < get_arg_val<uint32_t>(order_idx + order_stride * signalled + 2) - count) {
            return;
        }
        for (uint32_t c = 0; c < count; ++c) {
            const uint32_t xy = get_arg_val<uint32_t>(consumers_idx + first + c);
            noc_semaphore_inc(get_noc_addr(xy >> 16, xy & 0xFFFF, ready_sem_addr), 1);
        }
        ++signalled;
    };
    // release_block: pop block `popped` once its consumers have acked it.
    auto try_release = [&]() {
        if (popped < signalled && kind_ack(get_arg_val<uint32_t>(order_idx + order_stride * popped + 1)) >=
                                      get_arg_val<uint32_t>(order_idx + order_stride * popped + 2)) {
            cb_pop_front(cb_partial_handoff, block_tiles);
            ++popped;
        }
    };
    auto poll = [&]() {
        try_signal();
        try_release();
    };
    auto reserve_polling = [&](uint32_t pages) {
        while (!cb_pages_reservable_at_back(cb_act_operand, pages)) {
            poll();
        }
        cb_reserve_back(cb_act_operand, pages);
    };

    Noc noc;
    // K-block steps: step s = (unit s / num_k_blocks, K-block s % num_k_blocks)
    constexpr uint32_t steps = num_blocks * num_k_blocks;
    auto fresh = [&](uint32_t s) {
        return get_arg_val<uint32_t>(order_idx + order_stride * (s / num_k_blocks) + 3) != 0;
    };
    if constexpr (a_sends) {
        auto pipe = mc_a.sender(noc);
        // CB layout per K-block: [core_m_tiles][k_block_tiles]; padding rows of a ragged line stay unread (they only
        // feed output rows nobody reads).
        auto issue = [&](uint32_t s, uint32_t dst) {
            const uint32_t b = s / num_k_blocks, kb = s - b * num_k_blocks;
            const uint32_t row_base = get_arg_val<uint32_t>(order_idx + order_stride * b) + a_row0;
            // a row's K-block pages are consecutive: the ones num_banks apart share a bank, read them as one run
            for (uint32_t i = 0; i < a_valid_rows; ++i) {
                const uint32_t page0 = (row_base + i) * a_row_stride + kb * k_block_tiles;
                for (uint32_t r = 0; r < num_banks && r < k_block_tiles; ++r) {
                    mmrs::read_pages_strided(
                        a_acc,
                        page0 + r,
                        num_banks,
                        (k_block_tiles - r + num_banks - 1) / num_banks,
                        dst + (i * k_block_tiles + r) * a_tile_bytes,
                        num_banks * a_tile_bytes,
                        a_tile_bytes,
                        true);
                }
            }
        };
        mmrs::inject_operand<read_ahead>(
            cb_act_operand,
            steps,
            kblock_pages,
            kblock_bytes,
            fresh,
            issue,
            reserve_polling,
            [&](uint32_t dst) { pipe.send(dst, dst, kblock_bytes); },
            poll);
    } else {
        auto pipe = mc_a.receiver(noc);
        mmrs::receive_operand(
            cb_act_operand, steps, kblock_pages, fresh, reserve_polling, [&]() { pipe.receive(); }, poll);
    }

    {
        MaybeDeviceZoneScope("handoff_drain");  // last blocks' acks from the transport cores
        while (popped < num_blocks) {
            poll();
        }
    }
    // Re-arm: every semaphore is zero between calls.
    // re-arm each kind's counter by its total (the cumulative count of its last block)
    uint32_t totals[3] = {0, 0, 0};
    for (uint32_t b = 0; b < num_blocks; ++b) {
        totals[get_arg_val<uint32_t>(order_idx + order_stride * b + 1)] =
            get_arg_val<uint32_t>(order_idx + order_stride * b + 2);
    }
    for (uint32_t kind = 0; kind < 3; ++kind) {
        if (totals[kind] > 0) {
            noc_semaphore_inc(get_noc_addr(get_arg_val<uint32_t>(ack_sem_base + kind)), 0u - totals[kind]);
        }
    }
    noc_async_atomic_barrier();
}
