// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// matmul_reduce_scatter — compute-core NCRISC: activation (A, in0) operand + hand-off bookkeeping.
//
// Per scatter block (in compute order) and per K-block:
//   stream_operand_block / load_resident_operand: the m-line injector (A_SENDS) reads its rows' K-block from DRAM
//   and multicasts it along its m-line (mcast_pipe SenderPipe); the other cores of the line receive it
//   (ReceiverPipe). When A is the block-invariant operand (scatter_dim=-1) and resident (regime R1), blocks > 0
//   only replay CB credits (reserve + push of the same pages; capacity = one K pass exactly).
// handoff_block / release_block (raw semaphores; the mcast pipe is one-sender -> one-rectangle, the hand-off is
//   many producers -> few consumers with cumulative counters):
//   - when block `signalled` is fronted in cb_partial_handoff: one noc_semaphore_inc of sem_block_ready on each of
//     its consumers (the L ports of its direction, or the 2L finals for the chip's own block);
//   - when sem_block_ack has reached the cumulative ack count of block `popped`: pop its hand-off slot.
// Both are non-blocking polls, run after every K-block transfer and inside every wait of this kernel, so a remote
// wait never stalls the operand stream; only the final drain blocks.

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "ttnn/cpp/ttnn/kernel_lib/mcast/kernel/mcast_args.hpp"

using namespace dataflow_kernel_lib;

void kernel_main() {
    constexpr uint32_t cb_act_operand = get_compile_time_arg_val(0);
    constexpr uint32_t cb_partial_handoff = get_compile_time_arg_val(1);
    constexpr uint32_t core_m_tiles = get_compile_time_arg_val(2);
    constexpr uint32_t k_block_tiles = get_compile_time_arg_val(3);
    constexpr uint32_t num_k_blocks = get_compile_time_arg_val(4);
    constexpr uint32_t num_blocks = get_compile_time_arg_val(5);   // G scatter blocks per call
    constexpr uint32_t block_tiles = get_compile_time_arg_val(6);  // core_m_tiles * core_n_tiles (one hand-off slot)
    constexpr uint32_t a_tile_bytes = get_compile_time_arg_val(7);
    constexpr uint32_t a_sends = get_compile_time_arg_val(8);     // this core injects A along its m-line
    constexpr uint32_t a_resident = get_compile_time_arg_val(9);  // R1 with A invariant: replay after block 0
    constexpr uint32_t num_links = get_compile_time_arg_val(10);
    constexpr auto a_args = TensorAccessorArgs<11>();
    constexpr auto mc_a =
        McastArgs<get_named_compile_time_arg_val("a_ct_offset"), get_named_compile_time_arg_val("a_rt_offset")>();

    constexpr uint32_t kblock_pages = core_m_tiles * k_block_tiles;
    constexpr uint32_t kblock_bytes = kblock_pages * a_tile_bytes;

    size_t arg = 0;
    const uint32_t a_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t a_row_stride = get_arg_val<uint32_t>(arg++);      // Kt (A pages per tile-row)
    const uint32_t a_row0 = get_arg_val<uint32_t>(arg++);            // m-line's first row within a block
    const uint32_t a_valid_rows = get_arg_val<uint32_t>(arg++);      // <= core_m_tiles (ragged last m-line)
    const uint32_t a_block_row_step = get_arg_val<uint32_t>(arg++);  // blk_m_tiles for scatter_dim=-2, else 0
    const uint32_t ready_sem_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t ack_sem_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t order_idx = arg;  // num_blocks x [block j, consumer kind (0 fwd, 1 bwd, 2 finals), cumulative acks]
    arg += 3 * num_blocks;
    const uint32_t consumers_idx = arg;  // 4L packed NoC coords (x << 16 | y): fwd ports, bwd ports, finals
    arg += 4 * num_links;

    const auto a_acc = TensorAccessor(a_args, a_addr, a_tile_bytes);
    volatile tt_l1_ptr uint32_t* ack = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ack_sem_addr);

    uint32_t signalled = 0, popped = 0;
    // handoff_block + release_block, non-blocking.
    auto poll = [&]() {
        if (signalled < num_blocks &&
            cb_pages_available_at_front(cb_partial_handoff, (signalled - popped + 1) * block_tiles)) {
            const uint32_t kind = get_arg_val<uint32_t>(order_idx + 3 * signalled + 1);
            const uint32_t first = kind == 0 ? 0 : (kind == 1 ? num_links : 2 * num_links);
            const uint32_t count = kind == 2 ? 2 * num_links : num_links;
            for (uint32_t c = 0; c < count; ++c) {
                const uint32_t xy = get_arg_val<uint32_t>(consumers_idx + first + c);
                noc_semaphore_inc(get_noc_addr(xy >> 16, xy & 0xFFFF, ready_sem_addr), 1);
            }
            ++signalled;
        }
        if (popped < signalled && *ack >= get_arg_val<uint32_t>(order_idx + 3 * popped + 2)) {
            cb_pop_front(cb_partial_handoff, block_tiles);
            ++popped;
        }
    };
    auto reserve_polling = [&](uint32_t pages) {
        while (!cb_pages_reservable_at_back(cb_act_operand, pages)) {
            poll();
        }
        cb_reserve_back(cb_act_operand, pages);
    };

    Noc noc;
    // transfer(dst, row_base, kb): deliver one K-block of A into this core's CB at dst
    auto run = [&](auto&& transfer) {
        for (uint32_t b = 0; b < num_blocks; ++b) {
            const uint32_t j = get_arg_val<uint32_t>(order_idx + 3 * b);
            const uint32_t row_base = j * a_block_row_step + a_row0;
            for (uint32_t kb = 0; kb < num_k_blocks; ++kb) {
                reserve_polling(kblock_pages);
                if (!(a_resident && b > 0)) {
                    transfer(get_write_ptr(cb_act_operand), row_base, kb);
                }
                cb_push_back(cb_act_operand, kblock_pages);
                poll();
            }
        }
    };
    if constexpr (a_sends) {
        auto pipe = mc_a.sender(noc);
        run([&](uint32_t dst, uint32_t row_base, uint32_t kb) {
            // CB layout per K-block: [core_m_tiles][k_block_tiles]; padding rows of a ragged line stay unread (they
            // only feed output rows nobody reads).
            for (uint32_t i = 0; i < a_valid_rows; ++i) {
                const uint32_t page0 = (row_base + i) * a_row_stride + kb * k_block_tiles;
                for (uint32_t k = 0; k < k_block_tiles; ++k) {
                    noc_async_read(
                        a_acc.get_noc_addr(page0 + k), dst + (i * k_block_tiles + k) * a_tile_bytes, a_tile_bytes);
                }
            }
            noc_async_read_barrier();
            pipe.send(dst, dst, kblock_bytes);
        });
    } else {
        auto pipe = mc_a.receiver(noc);
        run([&](uint32_t, uint32_t, uint32_t) { pipe.receive(); });
    }

    while (popped < num_blocks) {
        poll();
    }
    // Re-arm: every semaphore is zero between calls.
    const uint32_t total_acks = get_arg_val<uint32_t>(order_idx + 3 * (num_blocks - 1) + 2);
    noc_semaphore_inc(get_noc_addr(ack_sem_addr), 0u - total_acks);
    noc_async_atomic_barrier();
}
