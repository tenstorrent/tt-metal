// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/endpoints.h"
#include "api/dataflow/noc_semaphore.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"
#include "api/tensor/tensor_accessor.h"
#include "api/compile_time_args.h"
#include "tools/profiler/kernel_profiler.hpp"

// Writer / group all-gather for fused_experts_prefill. One instance per worker core.
//
// For every owned expert with a non-zero count (same skip decision as reader and compute), per chunk of
// m_chunk tile rows:
//   1. wait for compute's SwiGLU block (cb_act_local, m_chunk * it_pc bf8 tiles, [r][jj]).
//   2. wait until every core of the group has released the previous chunk's gathered block
//      (sem_act_free), then unicast the mc * it_pc valid tiles into slot `c` of cb_act_full on ALL 8
//      cores of the group (itself included) and bump their sem_act_ready.
//   3. wait for the 8 blocks to have landed here (sem_act_ready) and publish cb_act_full to compute.
//   4. drain compute's untilized down output: per tile row, 32 row-major rows of this core's
//      out_seg_bytes slice (cb_outrm, nt_pc pages); every valid row (a token of the expert) goes to
//      partials row (slot * T + token) at byte offset c * out_seg_bytes. Then release the gathered block
//      to the group (sem_act_free).
//
// Both semaphores only ever count up (no resets), so a peer running ahead by one chunk is safe.
// cb_act_full holds exactly one block, so its write pointer is at the base address on every core
// whenever a block is being filled; the base is read once up front and used for the remote writes.
//
// Compile-time args:
//   0 cb_meta 1 cb_act_local 2 cb_act_full 3 cb_outrm
//   4 num_groups 5 nt_pc 6 it_pc 7 m_chunk 8 m_block
//   9 act_tile_bytes 10 sem_act_ready 11 sem_act_free 12 act_local_tiles 13 act_full_tiles
//   14 hdr_words 15 T 16 out_seg_bytes 17 bf16_tile_bytes 18 row_bytes
//   19+ TensorAccessorArgs: partials output
// Runtime args:
//   0 out (buffer address) 1 c (core index in group) 2 g (group) 3 n (experts owned by this group)
//   4 .. 4+2*8-1: NoC (x, y) of the group's 8 cores, indexed by core-in-group
void kernel_main() {
    constexpr uint32_t cb_meta_id = get_compile_time_arg_val(0);
    constexpr uint32_t cb_act_local_id = get_compile_time_arg_val(1);
    constexpr uint32_t cb_act_full_id = get_compile_time_arg_val(2);
    constexpr uint32_t cb_outrm_id = get_compile_time_arg_val(3);
    constexpr uint32_t nt_pc = get_compile_time_arg_val(5);
    constexpr uint32_t it_pc = get_compile_time_arg_val(6);
    constexpr uint32_t m_chunk = get_compile_time_arg_val(7);
    constexpr uint32_t act_tile_bytes = get_compile_time_arg_val(9);
    constexpr uint32_t sem_act_ready_id = get_compile_time_arg_val(10);
    constexpr uint32_t sem_act_free_id = get_compile_time_arg_val(11);
    constexpr uint32_t act_local_tiles = get_compile_time_arg_val(12);
    constexpr uint32_t act_full_tiles = get_compile_time_arg_val(13);
    constexpr uint32_t hdr_words = get_compile_time_arg_val(14);
    constexpr uint32_t T = get_compile_time_arg_val(15);
    constexpr uint32_t out_seg_bytes = get_compile_time_arg_val(16);
    constexpr uint32_t row_bytes = get_compile_time_arg_val(18);
    constexpr auto out_args = TensorAccessorArgs<19>();

    constexpr uint32_t kGroupCores = 8;
    constexpr uint32_t kPeerRt = 4;

    const uint32_t out_addr = get_arg_val<uint32_t>(0);
    const uint32_t c = get_arg_val<uint32_t>(1);
    const uint32_t n_owned = get_arg_val<uint32_t>(3);

    Noc noc;
    CircularBuffer cb_meta(cb_meta_id);
    CircularBuffer cb_act_local(cb_act_local_id);
    CircularBuffer cb_act_full(cb_act_full_id);
    CircularBuffer cb_outrm(cb_outrm_id);
    Semaphore<> sem_act_ready(sem_act_ready_id);
    Semaphore<> sem_act_free(sem_act_free_id);

    const auto out = TensorAccessor(out_args, out_addr, row_bytes);

    // The reader pushes the routing lists once (after the routing prologue); they stay resident.
    cb_meta.wait_front(1);
    const uint32_t meta_l1 = cb_meta.get_read_ptr();
    CoreLocalMem<volatile uint32_t> counts(meta_l1);
    CoreLocalMem<volatile uint32_t> entries(meta_l1 + hdr_words * 4);

    uint32_t peer_x[kGroupCores];
    uint32_t peer_y[kGroupCores];
    for (uint32_t p = 0; p < kGroupCores; ++p) {
        peer_x[p] = get_arg_val<uint32_t>(kPeerRt + 2 * p);
        peer_y[p] = get_arg_val<uint32_t>(kPeerRt + 2 * p + 1);
    }

    // cb_act_full is allocated identically on every core: its base address is valid on every peer.
    const uint32_t act_full_base = cb_act_full.get_write_ptr();
    const uint32_t my_block_bytes_offset = c * act_local_tiles * act_tile_bytes;

    uint32_t round = 0;
    for (uint32_t jl = 0; jl < n_owned; ++jl) {
        const uint32_t count = counts[jl];
        if (count == 0) {
            continue;
        }
        const uint32_t m = (count + 31) / 32;

        // One gather round per chunk of m_chunk tile rows (see the reader).
        for (uint32_t r0 = 0; r0 < m; r0 += m_chunk) {
            const uint32_t mc = (m - r0) < m_chunk ? (m - r0) : m_chunk;

            // ---- 1 + 2: all-gather this chunk's SwiGLU activations inside the group. ----
            {
                DeviceZoneScopedN("W_ACT_WAIT");  // compute's SwiGLU block
                cb_act_local.wait_front(act_local_tiles);
            }
            {
                DeviceZoneScopedN("W_FREE_WAIT");  // group released the previous gathered block
                sem_act_free.wait_min(kGroupCores * round);
            }
            {
                DeviceZoneScopedN("W_GATHER");  // unicast to the 8 cores + barrier + semaphore bumps
                const uint32_t send_bytes = mc * it_pc * act_tile_bytes;
                for (uint32_t p = 0; p < kGroupCores; ++p) {
                    noc.async_write(
                        cb_act_local,
                        UnicastEndpoint{},
                        send_bytes,
                        {.offset_bytes = 0},
                        {.noc_x = peer_x[p], .noc_y = peer_y[p], .addr = act_full_base + my_block_bytes_offset});
                }
                noc.async_write_barrier();
                for (uint32_t p = 0; p < kGroupCores; ++p) {
                    sem_act_ready.up(noc, peer_x[p], peer_y[p], 1);
                }
                cb_act_local.pop_front(act_local_tiles);
            }

            // ---- 3: all 8 blocks landed -> publish the gathered activations to compute. ----
            {
                DeviceZoneScopedN("W_READY_WAIT");
                sem_act_ready.wait_min(kGroupCores * (round + 1));
                cb_act_full.reserve_back(act_full_tiles);
                cb_act_full.push_back(act_full_tiles);
            }

            // ---- 4: scatter the down output rows, then release the gathered block. ----
            for (uint32_t r = 0; r < mc; ++r) {
                {
                    DeviceZoneScopedN("W_OUT_WAIT");  // compute's untilized rows
                    cb_outrm.wait_front(nt_pc);
                }
                DeviceZoneScopedN("W_OUT_WRITE");
                const uint32_t first = (r0 + r) * 32;
                const uint32_t nvalid = (count - first) < 32 ? (count - first) : 32;
                for (uint32_t rr = 0; rr < nvalid; ++rr) {
                    const uint32_t entry = entries[jl * T + first + rr];
                    const uint32_t token = entry & 0x1FFu;
                    const uint32_t slot = (entry >> 9) & 0xFu;
                    noc.async_write(
                        cb_outrm,
                        out,
                        out_seg_bytes,
                        {.offset_bytes = rr * out_seg_bytes},
                        {.page_id = slot * T + token, .offset_bytes = c * out_seg_bytes});
                }
                noc.async_write_barrier();
                cb_outrm.pop_front(nt_pc);
            }
            for (uint32_t p = 0; p < kGroupCores; ++p) {
                sem_act_free.up(noc, peer_x[p], peer_y[p], 1);
            }
            ++round;
        }
    }
    // The semaphore bumps are non-posted NoC atomics; the kernel must not exit with them in flight.
    noc.async_atomic_barrier();
}
