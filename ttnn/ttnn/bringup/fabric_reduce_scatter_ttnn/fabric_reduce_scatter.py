# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""fabric_reduce_scatter — a line / ring reduce-scatter over fabric, DRAM to DRAM, as one ttnn.generic_op, built on
fabric_all_gather's plan (groups: a mesh axis or a snake over the whole mesh, a line or ring, or two Hamiltonian cycles
on a torus; port placement under each link's Ethernet core, fabric connections, fence).

Input per chip: [1, 1, G * Sb, W] TILE (block j = rows j Sb ..), DRAM interleaved; output per chip: its block of the
sum over the group's G chips, [1, 1, Sb, W]. Blocks are output slots (a chip's row-major rank in its group), whatever
order the group travels in. Below, "block p" is the block of the chip at position p along the line (the line case;
a ring is in fabric_reduce_scatter's docstring).

Per chip p, per direction and per link, one *port* core (reader NCRISC / compute / sender BRISC):
  toward p+1 it sends the partial sums of blocks G-1, G-2, ..., p+1 (farthest first); toward p-1 blocks 0, 1, ..., p-1.
  A relay (every block but the line end's own) reads the chunk upstream sent into this chip's scratch, adds its own
  partial for it (compute) and sends the sum one hop on, into the same pages of the downstream chip's scratch.
  The last block a port sends is the downstream chip's own block: its chunks go to the downstream *final* core.
Per chip and per link two *final* cores, each owning half of the link's banks (bank set l + h L, stride 2 L): own block
= own partial + what arrived from p-1 + what arrived from p+1 (two or three inputs), written to the output. A final core
reads two input streams per chunk, twice what a port reads, and one core can't sustain that at the link's rate; halving
each final core's banks lets the senders run at the link's rate.
Arrival counters (as fabric_all_gather's): every 8th chunk of a stream (and its last) is a fused write + increment of
the receiving core's counter (the final core that owns the chunk's bank); a reader waits for chunk i of its own walk
once the counter covers it. Forward-arriving chunks count on
semaphore A, backward-arriving ones on B (only final cores receive both). The ready fence between calls is
fabric_all_gather's: no port writes into a neighbour's scratch before the neighbour has started the same call.
Scratch: [1, 1, (G + 1) Sb, W] per chip; forward chunks of block j land in slot j, backward chunks in slot j except
the receiver's own block (slot G).
"""

import math
import os

import ttnn
from ttnn.bringup.fabric_all_gather_ttnn.fabric_all_gather_py import (
    _CHUNK_WALK,
    NOC0,
    NOC1,
    _check_dram_interleaved,
    _chunks_per_shard,
    _needs_ready,
    _num_banks,
    _OPP,
    allowed_cores,
    plan,
)

CB_OWN, CB_IN, CB_IN2, CB_OUT = 0, 1, 2, 16

_PORT_READER = (
    r"""
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
"""
    + _CHUNK_WALK
    + r"""
// Port reader: per block it sends, per chunk: my partial (input) and, for a relayed block, what upstream sent (scratch,
// once the arrival counter covers it). The first `lead` blocks start their chain here (no upstream): my partial only,
// into c_0 for the compute's pass-through, or straight to the sender's c_16 when every block leads (a line end).
void kernel_main() {
    constexpr uint32_t page_bytes = get_compile_time_arg_val(0);
    constexpr uint32_t run_pages = get_compile_time_arg_val(1);
    constexpr uint32_t num_banks = get_compile_time_arg_val(2);
    constexpr uint32_t group = get_compile_time_arg_val(3);
    constexpr uint32_t inc_every = get_compile_time_arg_val(4);
    constexpr auto in_args = TensorAccessorArgs<5>();
    constexpr auto scr_args = TensorAccessorArgs<in_args.next_compile_time_args_offset()>();
    constexpr uint32_t chunk_bytes = run_pages * page_bytes;

    size_t a = 0;
    const uint32_t in_addr = get_arg_val<uint32_t>(a++);
    const uint32_t scr_addr = get_arg_val<uint32_t>(a++);
    const uint32_t blk_pages = get_arg_val<uint32_t>(a++);
    const uint32_t first_bank = get_arg_val<uint32_t>(a++);
    const uint32_t bank_stride = get_arg_val<uint32_t>(a++);
    const uint32_t arrival_addr = get_arg_val<uint32_t>(a++);
    const uint32_t full = get_arg_val<uint32_t>(a++);  // chunks of one block for this port
    const uint32_t lead = get_arg_val<uint32_t>(a++);  // leading blocks without upstream
    const uint32_t nblocks = get_arg_val<uint32_t>(a++);
    const uint32_t blocks_idx = a;
    const auto in = TensorAccessor(in_args, in_addr, page_bytes);
    const auto scr = TensorAccessor(scr_args, scr_addr, page_bytes);
    volatile tt_l1_ptr uint32_t* arrived = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(arrival_addr);

    // A batch is written contiguously from the CB's write pointer, so it must not cross the CB's end: the CBs hold
    // cap = 2 group chunks, and with two phases (or c_0 and c_1 starting at different offsets) the batches are not
    // group-aligned. o0 / o1: the chunk offset of the next write in the own-partial CB (c_0, or c_16) and in c_1.
    constexpr uint32_t cap = 2 * group;
    uint32_t o0 = 0, o1 = 0;
    auto room = [&](uint32_t o) { return cap - o < group ? cap - o : group; };

    // leading blocks: my partial only
    const uint32_t cb_lead = lead == nblocks ? 16 : 0;
    uint32_t chunks_left = lead * full;
    uint32_t batch = 0, limit = 0, wown = 0, win = 0;
    for (uint32_t k = 0; k < lead; ++k) {
        const uint32_t base = get_arg_val<uint32_t>(blocks_idx + k) * blk_pages;
        for_each_chunk(blk_pages, first_bank, bank_stride, num_banks, run_pages, 0, 0, [&](uint32_t page, uint32_t n, uint32_t) {
            if (batch == 0) {
                limit = room(o0);
                cb_reserve_back(cb_lead, run_pages * limit);
                wown = get_write_ptr(cb_lead);
            }
            noc_async_read(in.get_noc_addr(base + page), wown + batch * chunk_bytes, n * page_bytes);
            --chunks_left;
            if (++batch == limit || chunks_left == 0) {
                noc_async_read_barrier();
                cb_push_back(cb_lead, run_pages * batch);
                o0 = (o0 + batch) % cap;
                batch = 0;
            }
        });
    }
    if (cb_lead != 0) {
        o0 = 0;  // the relay phase's c_0 is untouched
    }
    // relayed blocks: my partial (c_0) and upstream's running sum (c_1); upstream's relay stream starts at my block lead
    chunks_left = (nblocks - lead) * full;
    for (uint32_t k = lead; k < nblocks; ++k) {
        const uint32_t base = get_arg_val<uint32_t>(blocks_idx + k) * blk_pages;
        for_each_chunk(blk_pages, first_bank, bank_stride, num_banks, run_pages, 0, 0, [&](uint32_t page, uint32_t n, uint32_t idx) {
            if (batch == 0) {
                limit = room(o0) < room(o1) ? room(o0) : room(o1);
                cb_reserve_back(0, run_pages * limit);
                wown = get_write_ptr(0);
                cb_reserve_back(1, run_pages * limit);
                win = get_write_ptr(1);
            }
            noc_async_read(in.get_noc_addr(base + page), wown + batch * chunk_bytes, n * page_bytes);
            noc_semaphore_wait_min(arrived, ((k - lead) * full + idx) / inc_every + 1);
            noc_async_read(scr.get_noc_addr(base + page), win + batch * chunk_bytes, n * page_bytes);
            --chunks_left;
            if (++batch == limit || chunks_left == 0) {
                noc_async_read_barrier();
                cb_push_back(0, run_pages * batch);
                cb_push_back(1, run_pages * batch);
                o0 = (o0 + batch) % cap;
                o1 = (o1 + batch) % cap;
                batch = 0;
            }
        });
    }
}
"""
)

_FINAL_READER = (
    r"""
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
"""
    + _CHUNK_WALK
    + r"""
// Final reader: per chunk of my own block (my link's banks): my partial, the forward arrival (scratch slot p, counter
// A) and the backward arrival (scratch slot G, counter B), each only if that direction has an upstream.
void kernel_main() {
    constexpr uint32_t page_bytes = get_compile_time_arg_val(0);
    constexpr uint32_t run_pages = get_compile_time_arg_val(1);
    constexpr uint32_t num_banks = get_compile_time_arg_val(2);
    constexpr uint32_t group = get_compile_time_arg_val(3);
    constexpr uint32_t inc_every = get_compile_time_arg_val(4);
    constexpr auto in_args = TensorAccessorArgs<5>();
    constexpr auto scr_args = TensorAccessorArgs<in_args.next_compile_time_args_offset()>();
    constexpr uint32_t chunk_bytes = run_pages * page_bytes;

    size_t a = 0;
    const uint32_t in_addr = get_arg_val<uint32_t>(a++);
    const uint32_t scr_addr = get_arg_val<uint32_t>(a++);
    const uint32_t blk_pages = get_arg_val<uint32_t>(a++);
    const uint32_t first_bank = get_arg_val<uint32_t>(a++);
    const uint32_t bank_stride = get_arg_val<uint32_t>(a++);
    const uint32_t own_base = get_arg_val<uint32_t>(a++) * blk_pages;
    const uint32_t fwd_base = get_arg_val<uint32_t>(a++) * blk_pages;
    const uint32_t bwd_base = get_arg_val<uint32_t>(a++) * blk_pages;
    const uint32_t has_fwd = get_arg_val<uint32_t>(a++);
    const uint32_t has_bwd = get_arg_val<uint32_t>(a++);
    const uint32_t a_addr = get_arg_val<uint32_t>(a++);
    const uint32_t b_addr = get_arg_val<uint32_t>(a++);
    uint32_t chunks_left = get_arg_val<uint32_t>(a++);
    const auto in = TensorAccessor(in_args, in_addr, page_bytes);
    const auto scr = TensorAccessor(scr_args, scr_addr, page_bytes);
    volatile tt_l1_ptr uint32_t* arr_a = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(a_addr);
    volatile tt_l1_ptr uint32_t* arr_b = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(b_addr);

    uint32_t batch = 0, w0 = 0, w1 = 0, w2 = 0;
    for_each_chunk(blk_pages, first_bank, bank_stride, num_banks, run_pages, 0, 0, [&](uint32_t page, uint32_t n, uint32_t idx) {
        if (batch == 0) {
            cb_reserve_back(0, run_pages * group);
            w0 = get_write_ptr(0);
            if (has_fwd) {
                cb_reserve_back(1, run_pages * group);
                w1 = get_write_ptr(1);
            }
            if (has_bwd) {
                cb_reserve_back(2, run_pages * group);
                w2 = get_write_ptr(2);
            }
        }
        noc_async_read(in.get_noc_addr(own_base + page), w0 + batch * chunk_bytes, n * page_bytes);
        if (has_fwd) {
            noc_semaphore_wait_min(arr_a, idx / inc_every + 1);
            noc_async_read(scr.get_noc_addr(fwd_base + page), w1 + batch * chunk_bytes, n * page_bytes);
        }
        if (has_bwd) {
            noc_semaphore_wait_min(arr_b, idx / inc_every + 1);
            noc_async_read(scr.get_noc_addr(bwd_base + page), w2 + batch * chunk_bytes, n * page_bytes);
        }
        --chunks_left;
        if (++batch == group || chunks_left == 0) {
            noc_async_read_barrier();
            cb_push_back(0, run_pages * batch);
            if (has_fwd) {
                cb_push_back(1, run_pages * batch);
            }
            if (has_bwd) {
                cb_push_back(2, run_pages * batch);
            }
            batch = 0;
        }
    });
}
"""
)

_ADD_COMPUTE = r"""
#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/tile_move_copy.h"

// run_pages tiles per chunk. First n_copy chunks: out(c_16) = c_0 (a ring port's leading block: it starts that block's
// chain). Then n chunks: out = c_0 + c_1 [+ c_2], or c_0 + c_2 (a final core without a forward upstream).
// Nothing (n_copy = n = 0): a line end's port, whose reader feeds the sender directly.
void kernel_main() {
    constexpr uint32_t R = get_compile_time_arg_val(0);
    constexpr uint32_t DST = 4;
    const uint32_t n = get_arg_val<uint32_t>(0);
    const uint32_t has_b = get_arg_val<uint32_t>(1);  // second input c_1
    const uint32_t has_c = get_arg_val<uint32_t>(2);  // third input c_2
    const uint32_t n_copy = get_arg_val<uint32_t>(3);
    if (n == 0 && n_copy == 0) {
        return;
    }
    const uint32_t cb_b = has_b ? 1 : 2;
    const bool three = has_b && has_c;
    binary_op_init_common(0, cb_b, 16);
    if (n_copy > 0) {
        copy_init(0);
        for (uint32_t i = 0; i < n_copy; ++i) {
            cb_wait_front(0, R);
            cb_reserve_back(16, R);
            for (uint32_t j0 = 0; j0 < R; j0 += DST) {
                const uint32_t m = R - j0 < DST ? R - j0 : DST;
                tile_regs_acquire();
                for (uint32_t j = 0; j < m; ++j) {
                    copy_tile(0, j0 + j, j);
                }
                tile_regs_commit();
                tile_regs_wait();
                for (uint32_t j = 0; j < m; ++j) {
                    pack_tile(j, 16);
                }
                tile_regs_release();
            }
            cb_push_back(16, R);
            PACK((RS_DBG(11, i + 1)));
            cb_pop_front(0, R);
        }
    }
    add_tiles_init(0, cb_b);
    for (uint32_t i = 0; i < n; ++i) {
        cb_wait_front(0, R);
        cb_wait_front(cb_b, R);
        if (three) {
            cb_wait_front(2, R);
        }
        cb_reserve_back(16, R);
        for (uint32_t j0 = 0; j0 < R; j0 += DST) {
            const uint32_t m = R - j0 < DST ? R - j0 : DST;
            tile_regs_acquire();
            if (three) {
                add_tiles_init(0, cb_b);
            }
            for (uint32_t j = 0; j < m; ++j) {
                add_tiles(0, cb_b, j0 + j, j0 + j, j);
            }
            if (three) {
                add_reuse_dest_init<EltwiseBinaryReuseDestType::DEST_TO_SRCA>(2);
                for (uint32_t j = 0; j < m; ++j) {
                    add_reuse_dest_tiles<EltwiseBinaryReuseDestType::DEST_TO_SRCA>(2, j0 + j, j);
                }
            }
            tile_regs_commit();
            tile_regs_wait();
            for (uint32_t j = 0; j < m; ++j) {
                pack_tile(j, 16);
            }
            tile_regs_release();
        }
        cb_push_back(16, R);
        PACK((RS_DBG(11, n_copy + i + 1)));
        cb_pop_front(0, R);
        cb_pop_front(cb_b, R);
        if (three) {
            cb_pop_front(2, R);
        }
    }
}
"""

_FINAL_WRITER = (
    r"""
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
"""
    + _CHUNK_WALK
    + r"""
// Final writer: my link's chunks of the summed own block into the output; then re-arm counters A / B.
void kernel_main() {
    constexpr uint32_t page_bytes = get_compile_time_arg_val(0);
    constexpr uint32_t run_pages = get_compile_time_arg_val(1);
    constexpr uint32_t num_banks = get_compile_time_arg_val(2);
    constexpr uint32_t group = get_compile_time_arg_val(3);
    constexpr auto out_args = TensorAccessorArgs<5>();
    constexpr uint32_t chunk_bytes = run_pages * page_bytes;

    size_t a = 0;
    const uint32_t out_addr = get_arg_val<uint32_t>(a++);
    const uint32_t blk_pages = get_arg_val<uint32_t>(a++);
    const uint32_t first_bank = get_arg_val<uint32_t>(a++);
    const uint32_t bank_stride = get_arg_val<uint32_t>(a++);
    uint32_t chunks_left = get_arg_val<uint32_t>(a++);
    const uint32_t a_addr = get_arg_val<uint32_t>(a++);
    const uint32_t expect_a = get_arg_val<uint32_t>(a++);
    const uint32_t b_addr = get_arg_val<uint32_t>(a++);
    const uint32_t expect_b = get_arg_val<uint32_t>(a++);
    const auto out = TensorAccessor(out_args, out_addr, page_bytes);

    uint32_t pending = 0;
    for_each_chunk(blk_pages, first_bank, bank_stride, num_banks, run_pages, 0, 0, [&](uint32_t page, uint32_t n, uint32_t) {
        cb_wait_front(16, run_pages * (pending + 1));
        noc_async_write(get_read_ptr(16) + pending * chunk_bytes, out.get_noc_addr(page), n * page_bytes);
        --chunks_left;
        if (++pending == group || chunks_left == 0) {
            noc_async_write_barrier();
            cb_pop_front(16, run_pages * pending);
            pending = 0;
        }
    });
    if (expect_a > 0) {
        volatile tt_l1_ptr uint32_t* c = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(a_addr);
        noc_semaphore_wait_min(c, expect_a);
        noc_semaphore_inc(get_noc_addr(a_addr), 0u - expect_a);
    }
    if (expect_b > 0) {
        volatile tt_l1_ptr uint32_t* c = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(b_addr);
        noc_semaphore_wait_min(c, expect_b);
        noc_semaphore_inc(get_noc_addr(b_addr), 0u - expect_b);
    }
    noc_async_atomic_barrier();
}
"""
)

_PORT_SENDER = (
    r"""
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "fabric/fabric_edm_packet_header.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/edm_fabric_worker_adapters.hpp"
#include "tt_metal/fabric/hw/inc/packet_header_pool.h"
#include "tt_metal/fabric/hw/inc/tt_fabric_api.h"

using namespace tt::tt_fabric;

inline void route_one_hop(volatile tt_l1_ptr PACKET_HEADER_TYPE* hdr, uint16_t chip_id, uint16_t mesh_id) {
#if (defined(ROUTING_MODE) && ((ROUTING_MODE & ROUTING_MODE_2D) != 0)) || defined(EMULE_FABRIC_2D)
    (void)fabric_set_unicast_route(hdr, chip_id, mesh_id);
#else
    (void)chip_id;
    (void)mesh_id;
    (void)fabric_set_unicast_route<false>(
        reinterpret_cast<volatile tt_l1_ptr LowLatencyPacketHeader*>(hdr), static_cast<uint16_t>(1));
#endif
}

FORCE_INLINE uint32_t cb_base(uint32_t cb) { return get_local_cb_interface(cb).fifo_limit - get_local_cb_interface(cb).fifo_size; }
"""
    + _CHUNK_WALK
    + r"""
// Port sender: every chunk of its stream into the downstream chip. The last block (the downstream chip's own) goes
// into its scratch, counted on the downstream final core that owns the chunk's bank (A forward, B backward). The relay
// blocks, by RS_MODE:
//   0 (single): the sums come from c_16 (this core's compute); relays go into the downstream scratch, counted on the
//      downstream port's counter A every inc_every-th chunk.
//   1 (split): the sums come from this core's ring (CB 3), written by the two helper cores (bank half h -> ring half
//      h, counter filled_h); relays go into the downstream scratch, counted on the downstream helper of the chunk's
//      bank half. Slots are handed back to helper h (its counter `consumed`) at every flush.
//   2 (l1): the sums come from c_16; the first scr_lead blocks (a ring's leading block: it must not wait for the
//      downstream, or the ring's senders wait for each other) go into the downstream scratch, counted on its port's
//      counter B every inc_every-th chunk; the other relays straight into the downstream port's c_1 (a ring of
//      `slots` chunks; c_1 position g = scr_lead * full + relay index, slot g % slots), counted on its counter A
//      every inc_every-th chunk and the last. Position g >= slots waits for the downstream's credit g - slots + 1 (counter C here). This core also
//      forwards the credits of the opposite-direction receiver on this chip (counter D here, counted by its reader) to
//      that receiver's upstream sender (counter C there), which is this core's downstream peer.
void kernel_main() {
    constexpr uint32_t page_bytes = get_compile_time_arg_val(0);
    constexpr uint32_t run_pages = get_compile_time_arg_val(1);
    constexpr uint32_t num_banks = get_compile_time_arg_val(2);
    constexpr uint32_t group = get_compile_time_arg_val(3);
    constexpr uint32_t inc_every = get_compile_time_arg_val(4);
    constexpr auto scr_args = TensorAccessorArgs<5>();
    constexpr uint32_t chunk_bytes = run_pages * page_bytes;
    constexpr uint32_t num_headers = group;

    size_t a = 0;
    const uint32_t scr_addr = get_arg_val<uint32_t>(a++);
    const uint32_t blk_pages = get_arg_val<uint32_t>(a++);
    const uint32_t first_bank = get_arg_val<uint32_t>(a++);
    const uint32_t bank_stride = get_arg_val<uint32_t>(a++);
    const uint32_t arrival_addr = get_arg_val<uint32_t>(a++);  // A: my counter, and the downstream port's
    const uint32_t final_sem_addr = get_arg_val<uint32_t>(a++);  // A or B on the downstream final core
    const uint32_t expect_in = get_arg_val<uint32_t>(a++);
    const uint32_t full = get_arg_val<uint32_t>(a++);
    const uint32_t ready_addr = get_arg_val<uint32_t>(a++);
    const uint32_t send_ready = get_arg_val<uint32_t>(a++);
    const uint32_t ready_x = get_arg_val<uint32_t>(a++);
    const uint32_t ready_y = get_arg_val<uint32_t>(a++);
    const uint32_t peer_x = get_arg_val<uint32_t>(a++);
    const uint32_t peer_y = get_arg_val<uint32_t>(a++);
    const uint32_t final_x = get_arg_val<uint32_t>(a++);
    const uint32_t final_y = get_arg_val<uint32_t>(a++);
    const uint32_t final_x1 = get_arg_val<uint32_t>(a++);  // the final core of the second half of my bank set
    const uint32_t final_y1 = get_arg_val<uint32_t>(a++);
    const uint32_t full_f0 = get_arg_val<uint32_t>(a++);  // chunks of one block in each half of my bank set
    const uint32_t full_f1 = get_arg_val<uint32_t>(a++);
    const uint16_t dst_mesh_id = static_cast<uint16_t>(get_arg_val<uint32_t>(a++));
    const uint16_t dst_chip_id = static_cast<uint16_t>(get_arg_val<uint32_t>(a++));
    uint32_t m[14];  // mode args
    for (uint32_t i = 0; i < 14; ++i) {
        m[i] = get_arg_val<uint32_t>(a++);
    }
    const uint32_t nblocks = get_arg_val<uint32_t>(a++);  // entries: block | slot << 16
    const uint32_t blocks_idx = a;
    a += nblocks;
    const auto scr = TensorAccessor(scr_args, scr_addr, page_bytes);
#if RS_MODE == 1
    // m: peer helper 0 / 1 (x, y), my helper 0 / 1 (x, y), filled_0, filled_1, consumed, my half totals 0 / 1, slots
    const uint64_t peer_h_noc[2] = {get_noc_addr(m[0], m[1], arrival_addr), get_noc_addr(m[2], m[3], arrival_addr)};
    const uint64_t my_h_noc[2] = {get_noc_addr(m[4], m[5], m[10]), get_noc_addr(m[6], m[7], m[10])};
    volatile tt_l1_ptr uint32_t* filled[2] = {
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(m[8]), reinterpret_cast<volatile tt_l1_ptr uint32_t*>(m[9])};
    const uint32_t slots = m[13];
    const uint32_t ring_base = cb_base(3);
    uint32_t got[2] = {0, 0}, ret[2] = {0, 0}, sent_h[2] = {0, 0};
    const bool need_conn = nblocks > 0 || send_ready;
#elif RS_MODE == 2
    // m: credit-in counter (C), credit-out counter (D), the receiver's upstream sender (x, y), credits in, credits out,
    // slots, scr_lead, B increments expected from upstream, B
    volatile tt_l1_ptr uint32_t* cred_in = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(m[0]);
    volatile tt_l1_ptr uint32_t* cred_out = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(m[1]);
    const uint32_t expect_cred_in = m[4], expect_cred_out = m[5], slots = m[6], scr_lead = m[7], expect_b = m[8];
    uint32_t sent_scr = 0;
    const uint32_t ring_base = cb_base(1);
    uint32_t fwd = 0;
    const bool need_conn = nblocks > 0 || send_ready || expect_cred_out > 0;
#else
    const bool need_conn = nblocks > 0 || send_ready;
#endif

    if (need_conn) {
        auto conn = WorkerToFabricEdmSender::build_from_args<ProgrammableCoreType::TENSIX>(a);
        volatile tt_l1_ptr PACKET_HEADER_TYPE* hdrs[num_headers];
        for (uint32_t h = 0; h < num_headers; ++h) {
            hdrs[h] = PacketHeaderPool::allocate_header();
            route_one_hop(hdrs[h], dst_chip_id, dst_mesh_id);
        }
#if RS_MODE == 2
        volatile tt_l1_ptr PACKET_HEADER_TYPE* cred_hdr = PacketHeaderPool::allocate_header();
        route_one_hop(cred_hdr, dst_chip_id, dst_mesh_id);
        const uint64_t up_cred_noc = get_noc_addr(m[2], m[3], m[0]);
#endif
        conn.open();
        if (send_ready) {
            hdrs[0]->to_noc_unicast_atomic_inc(NocUnicastAtomicIncCommandHeader{get_noc_addr(ready_x, ready_y, ready_addr), 1, true});
            conn.wait_for_empty_write_slot();
            conn.send_payload_flush_blocking_from_address(reinterpret_cast<uint32_t>(hdrs[0]), sizeof(PACKET_HEADER_TYPE));
        }
#if RS_MODE == 2
        auto service = [&]() {  // forward the opposite receiver's freed slots upstream
            if (expect_cred_out > 0) {
                const uint32_t c = *cred_out;
                if (c != fwd) {
                    cred_hdr->to_noc_unicast_atomic_inc(NocUnicastAtomicIncCommandHeader{up_cred_noc, c - fwd, true});
                    conn.wait_for_empty_write_slot();
                    conn.send_payload_flush_blocking_from_address(reinterpret_cast<uint32_t>(cred_hdr), sizeof(PACKET_HEADER_TYPE));
                    fwd = c;
                    RS_DBG(4, fwd);
                }
            }
        };
#endif
        if (nblocks > 0) {
            volatile tt_l1_ptr uint32_t* ready = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ready_addr);
            RS_DBG(0, 1);
#if RS_MODE == 2
            while (*ready < 1) {
                service();
            }
#else
            noc_semaphore_wait_min(ready, 1);
#endif
            noc_semaphore_inc(get_noc_addr(ready_addr), 0u - 1u);
            noc_async_atomic_barrier();
        }
        const uint64_t relay_noc = get_noc_addr(peer_x, peer_y, arrival_addr);
        const uint64_t final_noc = get_noc_addr(final_x, final_y, final_sem_addr);
        const uint64_t final_noc1 = get_noc_addr(final_x1, final_y1, final_sem_addr);
        const uint32_t relay_total = nblocks > 0 ? (nblocks - 1) * full : 0;
        uint32_t chunks_left = nblocks * full;
        uint32_t sent_relay = 0, sent_f0 = 0, sent_f1 = 0;
        uint32_t h = 0, unflushed = 0;
        for (uint32_t k = 0; k < nblocks; ++k) {
            const uint32_t entry = get_arg_val<uint32_t>(blocks_idx + k);
            const uint32_t base = (entry >> 16) * blk_pages;
            const bool fin = k + 1 == nblocks;
            RS_DBG(1, k);
            for_each_chunk(blk_pages, first_bank, bank_stride, num_banks, run_pages, 0, 0, [&](uint32_t page, uint32_t n, uint32_t) {
#if RS_MODE == 2
                service();
#endif
                // half h1 of my bank set: banks first_bank + (2 i + h1) * bank_stride
                const uint32_t h1 = (((page % num_banks) - first_bank) / bank_stride) & 1;
#if RS_MODE == 1
                noc_semaphore_wait_min(filled[h1], got[h1] + 1);
                const uint32_t src = ring_base + (h1 * slots + got[h1] % slots) * chunk_bytes;
                ++got[h1];
                ++ret[h1];
#elif RS_MODE == 2
                RS_DBG(0, 2);
                RS_DBG(5, chunks_left);
                while (!cb_pages_available_at_front(16, run_pages * (unflushed + 1))) {
                    service();
                }
                const uint32_t src = get_read_ptr(16) + unflushed * chunk_bytes;
#else
                cb_wait_front(16, run_pages * (unflushed + 1));
                const uint32_t src = get_read_ptr(16) + unflushed * chunk_bytes;
#endif
                const uint32_t bytes = n * page_bytes;
                uint64_t dst = scr.get_noc_addr(base + page, 0, 0);
                volatile tt_l1_ptr PACKET_HEADER_TYPE* hdr = hdrs[h];
                h = (h + 1 == num_headers) ? 0 : h + 1;
                bool inc;
                uint64_t ctr;
                if (fin) {
                    uint32_t& sent = h1 ? sent_f1 : sent_f0;
                    ++sent;
                    inc = (sent % inc_every) == 0 || sent == (h1 ? full_f1 : full_f0);
                    ctr = h1 ? final_noc1 : final_noc;
                } else {
#if RS_MODE == 1
                    ++sent_h[h1];
                    inc = (sent_h[h1] % inc_every) == 0 || sent_h[h1] == (nblocks - 1) * (h1 ? full_f1 : full_f0);
                    ctr = peer_h_noc[h1];
#elif RS_MODE == 2
                    if (k < scr_lead) {
                        ++sent_scr;
                        inc = (sent_scr % inc_every) == 0 || sent_scr == scr_lead * full;
                        ctr = get_noc_addr(peer_x, peer_y, m[9]);
                    } else {
                        const uint32_t g = scr_lead * full + sent_relay;
                        RS_DBG(2, g);
                        if (g >= slots) {
                            RS_DBG(0, 3);
                            RS_DBG(3, g - slots + 1);
                            while (*cred_in < g - slots + 1) {
                                service();
                            }
                        }
                        dst = get_noc_addr(peer_x, peer_y, ring_base + (g % slots) * chunk_bytes);
                        ++sent_relay;
                        inc = (sent_relay % inc_every) == 0 || sent_relay == relay_total - scr_lead * full;
                        ctr = relay_noc;
                    }
#else
                    ++sent_relay;
                    inc = (sent_relay % inc_every) == 0 || sent_relay == relay_total;
                    ctr = relay_noc;
#endif
                }
                if (inc) {
                    hdr->to_noc_fused_unicast_write_atomic_inc(NocUnicastAtomicIncFusedCommandHeader{dst, ctr, 1, true}, bytes);
                } else {
                    hdr->to_noc_unicast_write(NocUnicastCommandHeader{dst}, bytes);
                }
                RS_DBG(0, 4);
                conn.wait_for_empty_write_slot();
                conn.send_current_slot_non_blocking(src, bytes, reinterpret_cast<uint32_t>(hdr));
                --chunks_left;
                if (++unflushed == group || chunks_left == 0) {
                    noc_async_writes_flushed();
#if RS_MODE == 1
                    for (uint32_t q = 0; q < 2; ++q) {
                        if (ret[q] > 0) {
                            noc_semaphore_inc(my_h_noc[q], ret[q]);
                            ret[q] = 0;
                        }
                    }
#else
                    cb_pop_front(16, run_pages * unflushed);
#endif
                    unflushed = 0;
                }
            });
        }
#if RS_MODE == 2
        RS_DBG(0, 5);
        RS_DBG(3, expect_cred_out);
        while (fwd < expect_cred_out) {
            service();
        }
#endif
        conn.close();
    }
#if RS_MODE == 1
    for (uint32_t q = 0; q < 2; ++q) {  // re-arm filled_q (everything my helpers wrote, I sent)
        const uint32_t total = nblocks * m[11 + q];
        if (total > 0) {
            noc_semaphore_wait_min(filled[q], total);
            noc_semaphore_inc(get_noc_addr(m[8 + q]), 0u - total);
        }
    }
#elif RS_MODE == 2
    RS_DBG(0, 6);
    RS_DBG(3, expect_cred_in);
    if (expect_cred_in > 0) {
        noc_semaphore_wait_min(cred_in, expect_cred_in);
        noc_semaphore_inc(get_noc_addr(m[0]), 0u - expect_cred_in);
    }
    if (expect_cred_out > 0) {
        noc_semaphore_inc(get_noc_addr(m[1]), 0u - expect_cred_out);
    }
    RS_DBG(0, 7);
    RS_DBG(3, expect_b);
    if (expect_b > 0) {
        volatile tt_l1_ptr uint32_t* b = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(m[9]);
        noc_semaphore_wait_min(b, expect_b);
        noc_semaphore_inc(get_noc_addr(m[9]), 0u - expect_b);
    }
#endif
    RS_DBG(0, 8);
    RS_DBG(3, expect_in);
    if (expect_in > 0) {
        volatile tt_l1_ptr uint32_t* arrived = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(arrival_addr);
        noc_semaphore_wait_min(arrived, expect_in);
        noc_semaphore_inc(get_noc_addr(arrival_addr), 0u - expect_in);
    }
    noc_async_atomic_barrier();
    RS_DBG(0, 9);
}
"""
)

_HELPER_WRITER = r"""
#include <cstdint>
#include "api/dataflow/dataflow_api.h"

FORCE_INLINE uint32_t cb_base(uint32_t cb) { return get_local_cb_interface(cb).fifo_limit - get_local_cb_interface(cb).fifo_size; }

// Split-port helper writer: my sums (c_16, one chunk per run_pages tiles) into my half of the sender's ring (CB 3 on
// the sender core, same address as here): chunk k into slot k % slots once the sender has freed it (my counter
// `consumed` >= k - slots + 1), a batch at a time, then the sender's filled counter += batch. Finally re-arm
// `consumed` and my arrival counter (what upstream sent into the scratch for me).
void kernel_main() {
    constexpr uint32_t page_bytes = get_compile_time_arg_val(0);
    constexpr uint32_t run_pages = get_compile_time_arg_val(1);
    constexpr uint32_t group = get_compile_time_arg_val(3);
    constexpr uint32_t chunk_bytes = run_pages * page_bytes;
    size_t a = 0;
    const uint32_t total = get_arg_val<uint32_t>(a++);
    const uint32_t s_x = get_arg_val<uint32_t>(a++);
    const uint32_t s_y = get_arg_val<uint32_t>(a++);
    const uint32_t ring_off = get_arg_val<uint32_t>(a++);
    const uint32_t filled_addr = get_arg_val<uint32_t>(a++);
    const uint32_t consumed_addr = get_arg_val<uint32_t>(a++);
    const uint32_t slots = get_arg_val<uint32_t>(a++);
    const uint32_t expect_in = get_arg_val<uint32_t>(a++);
    const uint32_t arrival_addr = get_arg_val<uint32_t>(a++);
    volatile tt_l1_ptr uint32_t* consumed = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(consumed_addr);
    const uint32_t ring = cb_base(3) + ring_off;
    const uint64_t filled_noc = get_noc_addr(s_x, s_y, filled_addr);
    uint32_t pending = 0;
    for (uint32_t k = 0; k < total; ++k) {
        cb_wait_front(16, run_pages * (pending + 1));
        if (k >= slots) {
            noc_semaphore_wait_min(consumed, k - slots + 1);
        }
        noc_async_write(get_read_ptr(16) + pending * chunk_bytes, get_noc_addr(s_x, s_y, ring + (k % slots) * chunk_bytes), chunk_bytes);
        if (++pending == group || k + 1 == total) {
            noc_async_write_barrier();
            noc_semaphore_inc(filled_noc, pending);
            cb_pop_front(16, run_pages * pending);
            pending = 0;
        }
    }
    if (total > 0) {
        noc_semaphore_wait_min(consumed, total);
        noc_semaphore_inc(get_noc_addr(consumed_addr), 0u - total);
    }
    if (expect_in > 0) {
        volatile tt_l1_ptr uint32_t* arrived = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(arrival_addr);
        noc_semaphore_wait_min(arrived, expect_in);
        noc_semaphore_inc(get_noc_addr(arrival_addr), 0u - expect_in);
    }
    noc_async_atomic_barrier();
}
"""

_PORT_READER_L1 = (
    r"""
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
"""
    + _CHUNK_WALK
    + r"""
// L1-mode port reader: the leading blocks as the single-mode reader. The first scr_blocks relayed blocks (upstream's
// leading block, in a ring) as the single-mode reader, from the scratch (counter B). Then per relayed chunk r: my
// partial (input) into c_0, and c_1's next slot handed to the compute once upstream's sum has landed in it (upstream
// writes it there over the fabric and counts every inc_every-th chunk on A: a batch's slots are reserved, and
// credited, before waiting for it). Reserving c_1 position g means the compute has freed
// position g - slots: a credit, counted on the opposite-direction port core of this chip (counter D there), which
// forwards it to upstream. Credits go out every `group`, and before any wait for an arrival.
void kernel_main() {
    constexpr uint32_t page_bytes = get_compile_time_arg_val(0);
    constexpr uint32_t run_pages = get_compile_time_arg_val(1);
    constexpr uint32_t num_banks = get_compile_time_arg_val(2);
    constexpr uint32_t group = get_compile_time_arg_val(3);
    constexpr uint32_t inc_every = get_compile_time_arg_val(4);
    constexpr auto in_args = TensorAccessorArgs<5>();
    constexpr auto scr_args = TensorAccessorArgs<in_args.next_compile_time_args_offset()>();
    constexpr uint32_t chunk_bytes = run_pages * page_bytes;

    size_t a = 0;
    const uint32_t in_addr = get_arg_val<uint32_t>(a++);
    const uint32_t scr_addr = get_arg_val<uint32_t>(a++);
    const uint32_t blk_pages = get_arg_val<uint32_t>(a++);
    const uint32_t first_bank = get_arg_val<uint32_t>(a++);
    const uint32_t bank_stride = get_arg_val<uint32_t>(a++);
    const uint32_t arrival_addr = get_arg_val<uint32_t>(a++);
    const uint32_t full = get_arg_val<uint32_t>(a++);
    const uint32_t lead = get_arg_val<uint32_t>(a++);
    const uint32_t nblocks = get_arg_val<uint32_t>(a++);
    const uint32_t o_x = get_arg_val<uint32_t>(a++);
    const uint32_t o_y = get_arg_val<uint32_t>(a++);
    const uint32_t cred_out_addr = get_arg_val<uint32_t>(a++);
    const uint32_t slots = get_arg_val<uint32_t>(a++);
    const uint32_t scr_blocks = get_arg_val<uint32_t>(a++);
    const uint32_t b_addr = get_arg_val<uint32_t>(a++);
    const uint32_t blocks_idx = a;
    const auto in = TensorAccessor(in_args, in_addr, page_bytes);
    const auto scr = TensorAccessor(scr_args, scr_addr, page_bytes);
    volatile tt_l1_ptr uint32_t* arr_b = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(b_addr);
    volatile tt_l1_ptr uint32_t* arrived = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(arrival_addr);

    constexpr uint32_t cap = 2 * group;
    uint32_t o0 = 0, o1 = 0;
    auto room = [&](uint32_t o) { return cap - o < group ? cap - o : group; };

    const uint32_t cb_lead = lead == nblocks ? 16 : 0;
    uint32_t chunks_left = lead * full;
    uint32_t batch = 0, limit = 0, wown = 0;
    RS_DBG(6, 1);
    for (uint32_t k = 0; k < lead; ++k) {
        const uint32_t base = get_arg_val<uint32_t>(blocks_idx + k) * blk_pages;
        for_each_chunk(blk_pages, first_bank, bank_stride, num_banks, run_pages, 0, 0, [&](uint32_t page, uint32_t n, uint32_t) {
            if (batch == 0) {
                limit = room(o0);
                cb_reserve_back(cb_lead, run_pages * limit);
                wown = get_write_ptr(cb_lead);
            }
            noc_async_read(in.get_noc_addr(base + page), wown + batch * chunk_bytes, n * page_bytes);
            --chunks_left;
            if (++batch == limit || chunks_left == 0) {
                noc_async_read_barrier();
                cb_push_back(cb_lead, run_pages * batch);
                o0 = (o0 + batch) % cap;
                batch = 0;
            }
        });
    }
    if (cb_lead != 0) {
        return;
    }
    const uint64_t cred_noc = get_noc_addr(o_x, o_y, cred_out_addr);
    uint32_t g = 0, r = 0, credits = 0, win = 0, cred_total = 0;
    chunks_left = scr_blocks * full;
    for (uint32_t k = lead; k < lead + scr_blocks; ++k) {
        const uint32_t base = get_arg_val<uint32_t>(blocks_idx + k) * blk_pages;
        for_each_chunk(blk_pages, first_bank, bank_stride, num_banks, run_pages, 0, 0, [&](uint32_t page, uint32_t n, uint32_t idx) {
            if (batch == 0) {
                limit = room(o0) < room(o1) ? room(o0) : room(o1);
                cb_reserve_back(0, run_pages * limit);
                wown = get_write_ptr(0);
                cb_reserve_back(1, run_pages * limit);
                win = get_write_ptr(1);
            }
            noc_async_read(in.get_noc_addr(base + page), wown + batch * chunk_bytes, n * page_bytes);
            RS_DBG(6, 2);
            RS_DBG(7, k);
            RS_DBG(8, g);
            RS_DBG(9, ((k - lead) * full + idx) / inc_every + 1);
            noc_semaphore_wait_min(arr_b, ((k - lead) * full + idx) / inc_every + 1);
            noc_async_read(scr.get_noc_addr(base + page), win + batch * chunk_bytes, n * page_bytes);
            if (g >= slots) {
                ++credits;
            }
            ++g;
            --chunks_left;
            if (++batch == limit || chunks_left == 0) {
                noc_async_read_barrier();
                cb_push_back(0, run_pages * batch);
                cb_push_back(1, run_pages * batch);
                o0 = (o0 + batch) % cap;
                o1 = (o1 + batch) % cap;
                batch = 0;
                if (credits >= group) {
                    noc_semaphore_inc(cred_noc, credits);
                    cred_total += credits;
                    RS_DBG(10, cred_total);
                    credits = 0;
                }
            }
        });
    }
    chunks_left = (nblocks - lead - scr_blocks) * full;
    const uint32_t l1_total = chunks_left;
    for (uint32_t k = lead + scr_blocks; k < nblocks; ++k) {
        const uint32_t base = get_arg_val<uint32_t>(blocks_idx + k) * blk_pages;
        for_each_chunk(blk_pages, first_bank, bank_stride, num_banks, run_pages, 0, 0, [&](uint32_t page, uint32_t n, uint32_t) {
            RS_DBG(7, k);
            RS_DBG(8, g);
            if (batch == 0) {
                limit = room(o0);
                RS_DBG(6, 3);
                cb_reserve_back(0, run_pages * limit);
                wown = get_write_ptr(0);
            }
            noc_async_read(in.get_noc_addr(base + page), wown + batch * chunk_bytes, n * page_bytes);
            if (r % inc_every == 0) {  // upstream counts every inc_every-th chunk: take its batch's slots first
                const uint32_t nb = l1_total - r < inc_every ? l1_total - r : inc_every;
                RS_DBG(6, 4);
                cb_reserve_back(1, run_pages * nb);
                for (uint32_t q = 0; q < nb; ++q) {
                    credits += (g + q >= slots) ? 1 : 0;
                }
                if (credits > 0 && (credits >= group || *arrived < r / inc_every + 1)) {
                    noc_semaphore_inc(cred_noc, credits);
                    cred_total += credits;
                    RS_DBG(10, cred_total);
                    credits = 0;
                }
                RS_DBG(6, 5);
                RS_DBG(9, r / inc_every + 1);
                noc_semaphore_wait_min(arrived, r / inc_every + 1);
            }
            cb_push_back(1, run_pages);
            o1 = (o1 + 1) % cap;
            ++g;
            ++r;
            --chunks_left;
            if (++batch == limit || chunks_left == 0) {
                noc_async_read_barrier();
                cb_push_back(0, run_pages * batch);
                o0 = (o0 + batch) % cap;
                batch = 0;
            }
        });
    }
    if (credits > 0) {
        noc_semaphore_inc(cred_noc, credits);
        cred_total += credits;
        RS_DBG(10, cred_total);
    }
    noc_async_atomic_barrier();
    RS_DBG(6, 9);
}
"""
)


def _incs(chunks, inc_every):
    return -(-chunks // inc_every) if chunks else 0


def _walk(chips, coord, j, steps):
    """The chip `steps` hops from `coord` along ring j (forward for steps > 0; a ring wraps)."""
    for _ in range(abs(steps)):
        coord = chips[coord]["rings"][j]["next" if steps > 0 else "prev"]
    return coord


def _schedule(chips, coord, j, d, ring):
    """(blocks port d of `coord` on ring j sends, farthest first, as [(block, slot at the receiver)], lead).

    Blocks and scratch slots are output slots (row-major rank in the group, plan's "out"), so a snake travels in snake
    order but each chip owns its row-major block. Line: toward p+1 the blocks of positions G-1 .. p+1, toward p-1 those
    of 0 .. p-1; a line end starts every chain (lead = all), any other position relays every block (lead = 0).
    Ring: each block's partials arrive at its owner forward over a = G // 2 chips and backward over G - 1 - a, so a
    port sends a (or G - 1 - a) blocks; the first, farthest one starts its chain here (lead = 1), the rest are relays.
    The last block a port sends is the receiver's own: forward it lands in the receiver's own slot, backward in slot G."""
    rg = chips[coord]["rings"][j]
    G, p = chips[coord]["G"], rg["p"]
    peer = rg["next"] if d == "fwd" else rg["prev"]
    if peer is None:
        return [], 0
    out = lambda steps: chips[_walk(chips, coord, j, steps)]["out"]  # noqa: E731
    if ring:
        a = G // 2
        n = a if d == "fwd" else G - 1 - a
        steps = range(n, 0, -1) if d == "fwd" else range(-n, 0)
        lead = 1
    else:
        steps = range(G - 1 - p, 0, -1) if d == "fwd" else range(-p, 0)
        n = len(steps)
        upstream = rg["prev"] if d == "fwd" else rg["next"]
        lead = n if upstream is None else 0
    blocks = [(out(k), G if (d == "bwd" and k == -1) else out(k)) for k in steps]
    return blocks, min(lead, len(blocks))


PORT_MODES = {"single": 0, "split": 1, "l1": 2}
CB_RING = 3


def create_mesh_program_descriptor(
    mesh_device,
    input_tensor,
    scratch,
    output,
    sems,
    chips,
    *,
    num_links,
    chips_ring=False,
    port_mode="single",
    cb_bytes=128 * 1024,  # 8 chunks of 8 KiB per CB batch: a port reads two streams per chunk, so it needs the depth
    inc_every=8,
    dbg=(),
):
    """port_mode: "single" (a port core reads its partial and the upstream sum from the scratch, adds and sends),
    "split" (the port core only sends; two helper cores, one per half of its banks, read, add and fill its ring), "l1"
    (upstream writes the sum straight into the port core's c_1 over the fabric; the port reads only its partial)."""
    sem_a, sem_b, ready_addr, sem_c, sem_d = sems
    if dbg:  # RS_PROBE: progress words (one global semaphore each) for a hang dump
        pre = "#include <cstdint>\nconstexpr uint32_t RS_DBG_ADDR[] = {" + ", ".join(map(str, dbg)) + "};\n"
        pre += "#define RS_DBG(i, v) (*reinterpret_cast<volatile uint32_t*>(RS_DBG_ADDR[i]) = (v))\n"
    else:
        pre = "#define RS_DBG(i, v) ((void)0)\n"
    mode = PORT_MODES[port_mode]
    inc_every = int(os.environ.get("RS_INC_EVERY", inc_every))  # probe
    cb_bytes = int(os.environ.get("RS_CB_BYTES", cb_bytes))  # probe
    ring = chips_ring
    page_bytes = int(input_tensor.buffer_aligned_page_size())
    num_banks = _num_banks(mesh_device)
    G = next(iter(chips.values()))["G"]
    shape = list(input_tensor.padded_shape)
    blk_pages = (shape[-2] // G // 32) * (shape[-1] // 32)
    run_pages = max(1, ttnn.get_tt_fabric_max_payload_size_bytes() // page_bytes)
    chunk_bytes = run_pages * page_bytes
    group = max(1, min(8, cb_bytes // (2 * chunk_bytes)))
    slots = 2 * group  # a ring of chunk slots: the split sender's per helper, the l1 port's c_1
    if mode == 2:
        slots = int(os.environ.get("RS_L1_SLOTS", slots))  # probe
    in_ct = list(ttnn.TensorAccessorArgs(input_tensor).get_compile_time_args())
    scr_ct = list(ttnn.TensorAccessorArgs(scratch).get_compile_time_args())
    out_ct = list(ttnn.TensorAccessorArgs(output).get_compile_time_args())
    in_addr, scr_addr, out_addr = (int(t.buffer_address()) for t in (input_tensor, scratch, output))
    ct = [page_bytes, run_pages, num_banks, group, inc_every]
    virt = lambda c: mesh_device.worker_core_from_logical_core(c)
    node = lambda coord: mesh_device.get_fabric_node_id(ttnn.MeshCoordinate(*coord))
    dm = lambda risc, noc: ttnn.DataMovementConfigDescriptor(
        processor=getattr(ttnn.DataMovementProcessor, risc), noc=noc
    )
    src = ttnn.KernelDescriptor.SourceType.SOURCE_CODE
    cset = lambda cores: ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in cores])
    lanes_of = lambda ch: ch["n_rings"] * num_links

    def relays(coord, j, d):
        blocks, lead = _schedule(chips, coord, j, d, ring)
        return len(blocks) - lead

    def full_of(coord, j, l):
        return _chunks_per_shard(blk_pages, j * num_links + l, lanes_of(chips[coord]), num_banks, run_pages)

    def credits(coord, j, d, l):  # l1: slots the receiving port (coord, j, d, l) hands back to its upstream
        if mode != 2:
            return 0
        return max(0, relays(coord, j, d) * full_of(coord, j, l) - slots)

    def scr_lead(coord, j, d, l):
        """l1: the leading relay blocks port (coord, j, d, l) sends through the downstream scratch. A ring's leading
        block (its senders would otherwise wait for each other); every relay where a block is too short to keep a
        ring's senders apart (full chunks against ~6 CB batches of buffering per hop)."""
        if mode != 2 or not ring or coord is None:
            return 0
        blocks, lead = _schedule(chips, coord, j, d, ring)
        n_rel = max(0, len(blocks) - 1)
        if os.environ.get("RS_L1_SCR_ALL") == "1":
            return n_rel
        if os.environ.get("RS_L1_FORCE") == "1":  # probe: no all-scratch fallback for short blocks
            return min(n_rel, lead)
        return min(n_rel, lead if full_of(coord, j, l) >= 8 * group else n_rel)

    mesh_desc = ttnn.MeshProgramDescriptor()
    for coord, ch in chips.items():
        n_rings = ch["n_rings"]
        lanes = lanes_of(ch)  # lane (ring j, link l) = j L + l owns DRAM banks lane, lane + lanes, ...
        assert 2 * lanes <= num_banks, f"{n_rings} rings x {num_links} links need at least {2 * lanes} DRAM banks"
        program = ttnn.ProgramDescriptor()
        ports, finals = ch["ports"], ch["finals"]
        helpers = [c for hs in ch.get("helpers", {}).values() for c in hs] if mode == 1 else []
        workers = helpers if mode == 1 else list(ports.values())  # cores that read, add and feed a sender
        cbs = []
        if mode == 1:  # first, so that its address is the same on the senders and the helpers
            cbs.append(
                ttnn.CBDescriptor(
                    total_size=2 * slots * chunk_bytes,
                    core_ranges=cset(list(ports.values()) + helpers),
                    format_descriptors=[ttnn.CBFormatDescriptor(CB_RING, input_tensor.dtype, page_bytes)],
                )
            )
        for idx in (CB_OWN, CB_IN, CB_IN2, CB_OUT):
            cores = workers + finals if idx != CB_IN2 else finals
            cbs.append(
                ttnn.CBDescriptor(
                    total_size=(slots if mode == 2 and idx == CB_IN else 2 * group) * chunk_bytes,
                    core_ranges=cset(cores),
                    format_descriptors=[ttnn.CBFormatDescriptor(idx, input_tensor.dtype, page_bytes)],
                )
            )
        program.cbs = cbs
        pr_rt, ps_rt, pc_rt, hw_rt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
        for (j, d, l), core in ports.items():
            rg = ch["rings"][j]
            lane = j * num_links + l
            peer = rg["next"] if d == "fwd" else rg["prev"]
            blocks, lead = _schedule(chips, coord, j, d, ring)
            nrel = len(blocks) - lead
            full = _chunks_per_shard(blk_pages, lane, lanes, num_banks, run_pages)
            nf = [_chunks_per_shard(blk_pages, lane + h * lanes, 2 * lanes, num_banks, run_pages) for h in (0, 1)]
            packed = [b | (s_ << 16) for b, s_ in blocks]
            direct = lead == len(blocks)  # every block starts here: the reader feeds the writer / sender directly
            send_ready = _needs_ready(chips, coord, j, d)
            rc = virt(chips[peer]["ports"][(j, _OPP[d], l)]) if peer is not None else None
            opp = virt(ports[(j, _OPP[d], l)]) if (j, _OPP[d], l) in ports else None
            if mode == 1:
                for h, hc in enumerate(ch["helpers"][(j, d, l)]):
                    pr_rt[hc.x][hc.y] = [in_addr, scr_addr, blk_pages, lane + h * lanes, 2 * lanes, sem_a, nf[h], lead] + [
                        len(blocks)
                    ] + [b for b, _ in blocks]
                    pc_rt[hc.x][hc.y] = [0 if direct else nrel * nf[h], 1, 0, 0 if direct else lead * nf[h]]
                    hw_rt[hc.x][hc.y] = [
                        len(blocks) * nf[h],
                        virt(core).x,
                        virt(core).y,
                        h * slots * chunk_bytes,
                        sem_c if h == 0 else sem_d,
                        sem_b,
                        slots,
                        _incs(nrel * nf[h], inc_every),
                        sem_a,
                    ]
            else:
                rd = [in_addr, scr_addr, blk_pages, lane, lanes, sem_a, full, lead, len(blocks)]
                if mode == 2:
                    o = opp or ttnn.CoreCoord(0, 0)
                    upstream = rg["prev"] if d == "fwd" else rg["next"]
                    rd += [o.x, o.y, sem_d, slots, scr_lead(upstream, j, d, l), sem_b]
                pr_rt[core.x][core.y] = rd + [b for b, _ in blocks]
                pc_rt[core.x][core.y] = [0 if direct else nrel * full, 1, 0, 0 if direct else lead * full]
            # what upstream relays to me (the port itself counts it only in single / l1 mode)
            up = rg["prev"] if d == "fwd" else rg["next"]
            s0 = scr_lead(up, j, d, l) * full if mode == 2 else 0  # relayed chunks that come through the scratch
            expect_in = {0: _incs(nrel * full, inc_every), 1: 0, 2: _incs(nrel * full - s0, inc_every)}[mode]
            cred_out = credits(coord, j, _OPP[d], l)  # the opposite receiver's, forwarded to my peer
            cred_in = credits(peer, j, d, l) if peer is not None else 0
            if mode == 1 and peer is not None:
                ph = [virt(c) for c in chips[peer]["helpers"][(j, d, l)]]
                mh = [virt(c) for c in ch["helpers"][(j, d, l)]]
                margs = [ph[0].x, ph[0].y, ph[1].x, ph[1].y, mh[0].x, mh[0].y, mh[1].x, mh[1].y]
                margs += [sem_c, sem_d, sem_b, nf[0], nf[1], slots]
            elif mode == 1:
                mh = [virt(c) for c in ch["helpers"][(j, d, l)]]
                margs = [0, 0, 0, 0, mh[0].x, mh[0].y, mh[1].x, mh[1].y, sem_c, sem_d, sem_b, nf[0], nf[1], slots]
            elif mode == 2:
                margs = [sem_c, sem_d, rc.x if rc else 0, rc.y if rc else 0, cred_in, cred_out, slots]
                margs += [scr_lead(coord, j, d, l), _incs(s0, inc_every), sem_b] + [0] * 4
            else:
                margs = [0] * 14
            args = [scr_addr, blk_pages, lane, lanes, sem_a, sem_a if d == "fwd" else sem_b, expect_in, full, ready_addr]
            args += [1, rc.x, rc.y] if send_ready else [0, 0, 0]
            if rg.get(f"{d}_links") and blocks:
                pc = virt(chips[peer]["ports"][(j, d, l)])
                fc0 = virt(chips[peer]["finals"][2 * lane])
                fc1 = virt(chips[peer]["finals"][2 * lane + 1])
                pn = node(peer)
                args += [pc.x, pc.y, fc0.x, fc0.y, fc1.x, fc1.y, nf[0], nf[1], int(pn.mesh_id), int(pn.chip_id)]
                args += margs + [len(blocks)] + packed
                args += list(ttnn.setup_fabric_connection(node(coord), pn, rg[f"{d}_links"][l], program, core))
            elif send_ready or cred_out > 0:
                pn = node(peer)
                args += [0] * 8 + [int(pn.mesh_id), int(pn.chip_id)] + margs + [0]
                args += list(ttnn.setup_fabric_connection(node(coord), pn, rg[f"{d}_links"][l], program, core))
            else:
                args += [0] * 10 + margs + [0]
            ps_rt[core.x][core.y] = args
        fr_rt, fc_rt, fw_rt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
        own = ch["out"]
        for i, core in enumerate(finals):  # final core 2 lane + h: banks lane + h lanes, stride 2 lanes
            lane, h = divmod(i, 2)
            rg = ch["rings"][lane // num_links]
            # an arrival per direction where a chain ends here: a ring always (G >= 3), a line from each side it has
            has_f = int(rg["prev"] is not None and bool(_schedule(chips, rg["prev"], lane // num_links, "fwd", ring)[0]))
            has_b = int(rg["next"] is not None and bool(_schedule(chips, rg["next"], lane // num_links, "bwd", ring)[0]))
            first, fstride = lane + h * lanes, 2 * lanes
            n = _chunks_per_shard(blk_pages, first, fstride, num_banks, run_pages)
            fr_rt[core.x][core.y] = [
                in_addr,
                scr_addr,
                blk_pages,
                first,
                fstride,
                own,
                own,
                G,
                has_f,
                has_b,
                sem_a,
                sem_b,
                n,
            ]
            fc_rt[core.x][core.y] = [n, has_f, has_b, 0]
            fw_rt[core.x][core.y] = [
                out_addr,
                blk_pages,
                first,
                fstride,
                n,
                sem_a,
                _incs(n, inc_every) if has_f else 0,
                sem_b,
                _incs(n, inc_every) if has_b else 0,
            ]
        compute = lambda cores, rt: ttnn.KernelDescriptor(
            kernel_source=pre + _ADD_COMPUTE,
            source_type=src,
            core_ranges=cset(cores),
            compile_time_args=[run_pages],
            runtime_args=rt,
            config=ttnn.ComputeConfigDescriptor(),
        )
        kernels = [
            ttnn.KernelDescriptor(
                kernel_source=pre + (_PORT_READER_L1 if mode == 2 else _PORT_READER),
                source_type=src,
                core_ranges=cset(workers),
                compile_time_args=ct + in_ct + scr_ct,
                runtime_args=pr_rt,
                config=dm("RISCV_1", NOC0),
            ),
            compute(workers, pc_rt),
            ttnn.KernelDescriptor(
                kernel_source=pre + f"#define RS_MODE {mode}\n" + _PORT_SENDER,
                source_type=src,
                core_ranges=cset(ports.values()),
                compile_time_args=ct + scr_ct,
                runtime_args=ps_rt,
                config=dm("RISCV_0", NOC1),
            ),
        ]
        if mode == 1:
            kernels.append(
                ttnn.KernelDescriptor(
                    kernel_source=_HELPER_WRITER,
                    source_type=src,
                    core_ranges=cset(helpers),
                    compile_time_args=ct,
                    runtime_args=hw_rt,
                    config=dm("RISCV_0", NOC1),
                )
            )
        kernels += [
            ttnn.KernelDescriptor(
                kernel_source=_FINAL_READER,
                source_type=src,
                core_ranges=cset(finals),
                compile_time_args=ct + in_ct + scr_ct,
                runtime_args=fr_rt,
                config=dm("RISCV_1", NOC0),
            ),
            compute(finals, fc_rt),
            ttnn.KernelDescriptor(
                kernel_source=_FINAL_WRITER,
                source_type=src,
                core_ranges=cset(finals),
                compile_time_args=ct + out_ct,
                runtime_args=fw_rt,
                config=dm("RISCV_0", NOC1),
            ),
        ]
        program.kernels = kernels
        r, c = coord
        mesh_desc[ttnn.MeshCoordinateRange(ttnn.MeshCoordinate(r, c), ttnn.MeshCoordinate(r, c))] = program
    return mesh_desc


_SEM_CACHE, _SCRATCH_CACHE, _PLAN_CACHE = {}, {}, {}
_N_DBG = 12


def _write_probe_map(path, mesh_device, chips, sems, port_mode, ring):
    """RS_PROBE: where a hang dump (tools/triage/dump_rs_probe.py) finds every core's counters and progress words."""
    import json

    out = {"names": ["A", "B", "ready", "C", "D"] + [f"dbg{i}" for i in range(_N_DBG)], "addrs": list(sems)}
    out["port_mode"], out["ring"], out["chips"] = port_mode, bool(ring), []
    for coord, ch in chips.items():
        cores = [{"role": f"port j{j} {d} l{l}", "x": c.x, "y": c.y} for (j, d, l), c in ch["ports"].items()]
        cores += [{"role": f"final {i}", "x": c.x, "y": c.y} for i, c in enumerate(ch["finals"])]
        out["chips"].append(
            {"coord": list(coord), "dev": int(mesh_device.get_device_id(ttnn.MeshCoordinate(*coord))), "cores": cores}
        )
    with open(path, "w") as f:
        json.dump(out, f)


def _with_final_cores(mesh_device, chips, num_links, helpers=False):
    """Each chip's plan plus "finals": two final cores per lane (ring j, link l), [lane 0 half 0, lane 0 half 1, ...].
    A lane's half 0 is fabric_all_gather's copy core of its link (first ring) or the next free core of that row; half 1
    the next free core in that row (else any free core). helpers: also "helpers", two cores per port (split mode),
    in the port's row where free."""
    allowed = sorted(allowed_cores(mesh_device), key=lambda c: (c[1], c[0]))
    out = {}
    for coord, ch in chips.items():
        taken = {(c.x, c.y) for c in list(ch["ports"].values()) + ch["copy"]}

        def free(row, near=None):
            cand = [c for c in allowed if c[1] == row and c not in taken]
            if near is not None:
                cand.sort(key=lambda c: abs(c[0] - near))
            got = (cand[0] if cand else None) or next((c for c in allowed if c not in taken), None)
            if got is None:
                raise ValueError("fabric_reduce_scatter: the core grid has too few cores")
            taken.add(got)
            return ttnn.CoreCoord(*got)

        finals = []
        for j in range(ch["n_rings"]):
            for l in range(num_links):
                row = ch["copy"][l].y
                finals += [ch["copy"][l] if j == 0 else free(row), free(row)]
        extra = {"finals": finals}
        if helpers:
            extra["helpers"] = {key: [free(p.y, p.x), free(p.y, p.x)] for key, p in ch["ports"].items()}
        out[coord] = {**ch, **extra}
    return out


def fabric_reduce_scatter(
    input_tensor,
    *,
    cluster_axis=0,
    topology=ttnn.Topology.Linear,
    num_links=1,
    scheme="ring",
    output=None,
    placement="auto",
    port_mode="single",
):
    """Reduce-scatter `input_tensor` ([1, 1, G * Sb, W] TILE, DRAM interleaved, a partial per chip; block i = rows
    i Sb ..) over each group: every chip gets its block of the sum over the group, [1, 1, Sb, W] (the layout
    ttnn.reduce_scatter(dim=2) gives). The groups and orders are fabric_all_gather's:

    cluster_axis 0 / 1: a line or ring along that mesh axis (the other axis runs independent groups); None: one group
        over the whole mesh, a snake (line, or ring where the snake closes, e.g. 2 x 4).
    topology: Linear, or Ring (a group of 2 is a line). A ring sends each block's partials both ways round, so every
        link direction carries about half the bytes of a line's busiest.
    scheme "dual_cycles" (cluster_axis=None, Ring, a torus with both sides >= 3): two edge-disjoint Hamiltonian
        cycles, each reducing half of every block (half the DRAM banks).
    Whatever order a group travels in, chip (r, c) owns block (its row-major rank in the group), as the all-gather's
    output slots: block r C + c over the whole mesh, block r (c) along axis 0 (1).
    port_mode: how a port reads, adds and sends (create_mesh_program_descriptor)."""
    mesh_device = input_tensor.device()
    key = (id(mesh_device), cluster_axis, topology, num_links, scheme, placement, port_mode == "split")
    if key not in _PLAN_CACHE:
        chips, ring = plan(
            mesh_device,
            cluster_axis=cluster_axis,
            topology=topology,
            num_links=num_links,
            placement=placement,
            scheme=scheme,
        )
        _PLAN_CACHE[key] = (mesh_device, _with_final_cores(mesh_device, chips, num_links, helpers=port_mode == "split"), ring)
    ring = _PLAN_CACHE[key][2]
    chips = _PLAN_CACHE[key][1]
    G = next(iter(chips.values()))["G"]
    _check_dram_interleaved(input_tensor, "input")
    shape = list(input_tensor.shape)
    assert input_tensor.layout == ttnn.TILE_LAYOUT and shape[0] == shape[1] == 1, shape
    assert shape[2] % (32 * G) == 0, f"rows {shape[2]} must split into {G} tile-aligned blocks"
    Sb, W = shape[2] // G, shape[3]
    want = [1, 1, Sb, W]
    if output is None:
        output = ttnn.allocate_tensor_on_device(
            ttnn.Shape(want), input_tensor.dtype, ttnn.TILE_LAYOUT, mesh_device, ttnn.DRAM_MEMORY_CONFIG
        )
    assert list(output.shape) == want and output.dtype == input_tensor.dtype, (list(output.shape), want)
    skey = (id(mesh_device), G, Sb, W, input_tensor.dtype)
    if skey not in _SCRATCH_CACHE:
        scr = ttnn.allocate_tensor_on_device(
            ttnn.Shape([1, 1, (G + 1) * Sb, W]),
            input_tensor.dtype,
            ttnn.TILE_LAYOUT,
            mesh_device,
            ttnn.DRAM_MEMORY_CONFIG,
        )
        _SCRATCH_CACHE[skey] = (mesh_device, scr)
    scratch = _SCRATCH_CACHE[skey][1]
    cores = {
        (c.x, c.y)
        for ch in chips.values()
        for c in list(ch["ports"].values()) + ch["finals"] + [h for hs in ch.get("helpers", {}).values() for h in hs]
    }
    probe = os.environ.get("RS_PROBE")
    ckey = (id(mesh_device), tuple(sorted(cores)), bool(probe))
    if ckey not in _SEM_CACHE:
        crs = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(x, y), ttnn.CoreCoord(x, y)) for x, y in sorted(cores)])
        sems = tuple(ttnn.create_global_semaphore(mesh_device, crs, 0) for _ in range(5 + (_N_DBG if probe else 0)))
        ttnn.synchronize_device(mesh_device)  # every counter zero before any chip's first call increments a neighbour's
        _SEM_CACHE[ckey] = (mesh_device, sems)
    sems = tuple(int(ttnn.get_global_semaphore_address(s)) for s in _SEM_CACHE[ckey][1])
    if probe:
        _write_probe_map(probe, mesh_device, chips, sems, port_mode, ring)
    sems, dbg = sems[:5], sems[5:]
    desc = create_mesh_program_descriptor(
        mesh_device,
        input_tensor,
        scratch,
        output,
        sems,
        chips,
        num_links=num_links,
        chips_ring=ring,
        port_mode=port_mode,
        dbg=dbg,
    )
    return ttnn.generic_op([input_tensor, scratch, output], desc)
