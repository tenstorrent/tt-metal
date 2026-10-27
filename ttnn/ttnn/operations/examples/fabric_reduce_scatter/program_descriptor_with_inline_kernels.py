# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""fabric_reduce_scatter — a line / ring reduce-scatter over fabric, DRAM to DRAM, as one ttnn.generic_op, built on
fabric_all_gather's plan (groups, port placement under each link's Ethernet core, fabric connections, fence).

Input per chip: [1, 1, G * Sb, W] bf16 TILE (block j = rows j Sb ..), DRAM interleaved; output per chip p: block p of
the sum over the group's G chips, [1, 1, Sb, W]. The group is a line or ring along `cluster_axis` (chip p = position
p).

Per chip p, per direction and per link, one *port* core (reader NCRISC / compute / sender BRISC), sending blocks
farthest first:
  Line: toward p+1 the partial sums of blocks G-1, G-2, ..., p+1; toward p-1 blocks 0, 1, ..., p-1.
  Ring: toward p+1 the G//2 blocks p+G//2, ..., p+1; toward p-1 the other G-1-G//2 blocks, ..., p-1 (indices mod G);
  each block's partials are summed along the shorter arc to its owner. On an even G (balance) the far block is split
  in two halves of its chunk rounds, one half each way, so both directions carry G/2 - 1/2 blocks (see _entries).
  A relay block reads the chunk upstream sent into this chip's scratch, adds its own partial for it (compute) and sends
  the sum one hop on, into the same pages of the downstream chip's scratch. A block whose chain starts here (a line
  end's every block, a ring port's farthest) goes from the input straight to the sender (c_3), no compute.
  The last block a port sends is the downstream chip's own block: its chunks go to the downstream *final* core.
Per chip and per link two *final* cores, each owning half of the link's banks (bank set l + h L, stride 2 L): own block
= own partial + what arrived from p-1 + what arrived from p+1 (two or three inputs), written to the output. A final core
reads two input streams per chunk, twice what a port reads, and one core can't sustain that at the link's rate; halving
each final core's banks lets the senders run at the link's rate.
workers > 1 (a mux per link and direction): a relay port reads two DRAM streams per chunk through one reader, which runs
at about half the link's rate. With W workers per link and direction, worker w takes banks l + w L, stride W L, and
runs the port's reader / compute / sender on them; the linked port core becomes a fabric mux (tt_fabric_mux.cpp, on
NoC1: its router is one row up) that the W senders share, the router exposing one worker channel per link. Each link
then has W final cores, final w owning worker w's banks, so every arrival counter keeps a single sender per direction.
Arrival counters (as fabric_all_gather's): every 8th chunk of a stream (and its last) is a fused write + increment of
the receiving core's counter (the final core that owns the chunk's bank); a reader waits for chunk i of its own walk
once the counter covers it. Forward-arriving chunks count on
semaphore A, backward-arriving ones on B (only final cores receive both). The ready fence between calls is
fabric_all_gather's: no port writes into a neighbour's scratch before the neighbour has started the same call.
Scratch: [1, 1, (G + 1) Sb, W] per chip; forward chunks of block j land in slot j, backward chunks in slot j except
the receiver's own block (slot G).
"""

import math

import ttnn
from ttnn.operations.examples.fabric_all_gather.program_descriptor_with_inline_kernels import (
    _CHUNK_WALK,
    NOC0,
    NOC1,
    _check_dram_interleaved,
    _chip_maps,
    _chunks_per_shard,
    _noc_grid,
    _needs_ready,
    _num_banks,
    _OPP,
    allowed_cores,
    plan,
)

CB_OWN, CB_IN, CB_IN2, CB_PASS, CB_OUT = 0, 1, 2, 3, 16

_HALF_WALK = r"""
// One entry's walk: the whole block (half 2; for_each_chunk's order) or half 0 / 1 = the first / second half of its
// chunk rounds (m), every bank of the set each round. The final cores walk rounds-major too, so either half reaches
// each final core in its own walk order. One loop, one call site: the body is inlined once.
template <typename F>
FORCE_INLINE void for_each_chunk_half(
    uint32_t shard_pages, uint32_t first_bank, uint32_t bank_stride, uint32_t num_banks, uint32_t run_pages,
    uint32_t half, F&& f) {
    const uint32_t max_per_bank = (shard_pages + num_banks - 1) / num_banks;
    const uint32_t split = ((max_per_bank + run_pages - 1) / run_pages + 1) / 2 * run_pages;
    const uint32_t m0 = half == 1 ? split : 0, m1 = half == 0 ? split : max_per_bank;
    uint32_t idx = 0;
    for (uint32_t m = m0; m < m1; m += run_pages) {
        for (uint32_t b = first_bank; b < num_banks; b += bank_stride) {
            const uint32_t per_bank = b < shard_pages ? (shard_pages - b + num_banks - 1) / num_banks : 0;
            if (m >= per_bank) {
                continue;
            }
            const uint32_t n = (per_bank - m) < run_pages ? (per_bank - m) : run_pages;
            f(b + m * num_banks, n, idx++);
        }
    }
}
"""

_PORT_READER = (
    r"""
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
"""
    + _CHUNK_WALK
    + _HALF_WALK
    + r"""
// Port reader: per entry it sends (a block, or one half of it), per chunk: my partial (input) and, for a
// relay entry, what upstream sent (scratch, once the arrival counter covers it). An entry whose chain starts here (a
// line end's every block, a ring port's farthest) has no upstream: its partial goes straight to the sender (c_3).
// Each stream batches on its own.
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
    const uint32_t nentries = get_arg_val<uint32_t>(a++);  // entries: block | half << 12 | relay << 14
    const uint32_t entries_idx = a;
    const auto in = TensorAccessor(in_args, in_addr, page_bytes);
    const auto scr = TensorAccessor(scr_args, scr_addr, page_bytes);
    volatile tt_l1_ptr uint32_t* arrived = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(arrival_addr);

    // Per entry, into c_0 + c_1 (relay) or c_3 (chain start): a batch ends at the next multiple of `group` chunks in
    // the CB (2 group chunks; so a reservation never wraps) or at the end of the entry (a stream can pause for the other
    // one: an unpushed tail would stall the sender, and around a ring the whole ring).
    uint32_t pos[2] = {0, 0};
    uint32_t ro = 0;  // relay chunks so far: upstream sends exactly my relay entries, in this order
    for (uint32_t k = 0; k < nentries; ++k) {
        const uint32_t entry = get_arg_val<uint32_t>(entries_idx + k);
        const uint32_t base = (entry & 0xFFF) * blk_pages;
        const uint32_t half = (entry >> 12) & 3;
        const bool relay = (entry >> 14) & 1;
        const uint32_t cb_own = relay ? 0 : 3;
        uint32_t c = 0, need = 0, w0 = 0, w1 = 0, pushed = 0;
        const uint32_t p0 = pos[relay];
        for_each_chunk_half(blk_pages, first_bank, bank_stride, num_banks, run_pages, half, [&](uint32_t page, uint32_t n, uint32_t) {
            if (c == 0) {
                need = group - (p0 + pushed) % group;
                cb_reserve_back(cb_own, run_pages * need);
                w0 = get_write_ptr(cb_own);
                if (relay) {
                    cb_reserve_back(1, run_pages * need);
                    w1 = get_write_ptr(1);
                }
            }
            noc_async_read(in.get_noc_addr(base + page), w0 + c * chunk_bytes, n * page_bytes);
            if (relay) {
                noc_semaphore_wait_min(arrived, ro++ / inc_every + 1);
                noc_async_read(scr.get_noc_addr(base + page), w1 + c * chunk_bytes, n * page_bytes);
            }
            if (++c == need) {
                noc_async_read_barrier();
                cb_push_back(cb_own, run_pages * c);
                if (relay) {
                    cb_push_back(1, run_pages * c);
                }
                pushed += c;
                c = 0;
            }
        });
        if (c > 0) {
            noc_async_read_barrier();
            cb_push_back(cb_own, run_pages * c);
            if (relay) {
                cb_push_back(1, run_pages * c);
            }
            pushed += c;
        }
        pos[relay] = p0 + pushed;
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

// Sum of 2 or 3 inputs, run_pages tiles per chunk: out(c_16) = c_0 + c_1 [+ c_2]; or c_0 + c_2 (a final core without a
// forward upstream). n = 0: nothing to do (a port without relay blocks: its reader feeds the sender directly).
void kernel_main() {
    constexpr uint32_t R = get_compile_time_arg_val(0);
    constexpr uint32_t DST = 4;
    const uint32_t n = get_arg_val<uint32_t>(0);
    const uint32_t has_b = get_arg_val<uint32_t>(1);  // second input c_1
    const uint32_t has_c = get_arg_val<uint32_t>(2);  // third input c_2
    if (n == 0) {
        return;
    }
    const uint32_t cb_b = has_b ? 1 : 2;
    const bool three = has_b && has_c;
    binary_op_init_common(0, cb_b, 16);
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
#ifdef FRS_MUX
#include "tt_metal/fabric/hw/inc/tt_fabric_mux_interface.hpp"
#endif

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
"""
    + _CHUNK_WALK
    + _HALF_WALK
    + r"""
// Port sender: every chunk of c_16 (a relay entry's sums) or c_3 (a chain start's partials) into the downstream chip's
// scratch. Three increment streams: the entries the downstream relays count on the downstream port's counter (A), the
// downstream chip's own block (`fin` entries) on the counter (A forward, B backward) of the downstream final core that
// owns the chunk's bank; every inc_every-th chunk and each stream's last carries one. After sending: wait for
// everything upstream sends me, then re-arm my counter. FRS_MUX: this core is one of several workers of its link and
// direction, each on its own share of the link's banks; they reach the link through a fabric mux core (connect,
// send, disconnect; worker 0 then terminates the mux once the others have disconnected).
void kernel_main() {
    constexpr uint32_t page_bytes = get_compile_time_arg_val(0);
    constexpr uint32_t run_pages = get_compile_time_arg_val(1);
    constexpr uint32_t num_banks = get_compile_time_arg_val(2);
    constexpr uint32_t group = get_compile_time_arg_val(3);
    constexpr uint32_t inc_every = get_compile_time_arg_val(4);
    constexpr auto scr_args = TensorAccessorArgs<5>();
    constexpr uint32_t chunk_bytes = run_pages * page_bytes;
    constexpr uint32_t num_headers = group;
#ifdef FRS_MUX
    constexpr uint32_t mux_ct = scr_args.next_compile_time_args_offset();  // ccl::fabric_mux_connection_ct_args
    constexpr uint8_t mux_num_buffers = get_compile_time_arg_val(mux_ct + 0);
    constexpr size_t mux_buffer_bytes = get_compile_time_arg_val(mux_ct + 1);
    constexpr size_t mux_status_address = get_compile_time_arg_val(mux_ct + 2);
    constexpr size_t mux_termination_address = get_compile_time_arg_val(mux_ct + 3);
    constexpr uint32_t mux_num_clients = get_compile_time_arg_val(mux_ct + 4);
#endif

    size_t a = 0;
    const uint32_t scr_addr = get_arg_val<uint32_t>(a++);
    const uint32_t blk_pages = get_arg_val<uint32_t>(a++);
    const uint32_t first_bank = get_arg_val<uint32_t>(a++);
    const uint32_t bank_stride = get_arg_val<uint32_t>(a++);
    const uint32_t arrival_addr = get_arg_val<uint32_t>(a++);  // A: my counter, and the downstream port's
    const uint32_t final_sem_addr = get_arg_val<uint32_t>(a++);  // A or B on the downstream final core
    const uint32_t expect_in = get_arg_val<uint32_t>(a++);
    const uint32_t relay_total = get_arg_val<uint32_t>(a++);  // chunks of the entries the downstream relays
    const uint32_t ready_addr = get_arg_val<uint32_t>(a++);
    const uint32_t send_ready = get_arg_val<uint32_t>(a++);
    const uint32_t ready_x = get_arg_val<uint32_t>(a++);
    const uint32_t ready_y = get_arg_val<uint32_t>(a++);
    const uint32_t peer_x = get_arg_val<uint32_t>(a++);
    const uint32_t peer_y = get_arg_val<uint32_t>(a++);
    // The downstream's own block goes to its final cores: final h owns banks link_first + (F i + h) link_stride
    const uint32_t nfin = get_arg_val<uint32_t>(a++);  // F (<= 8)
    const uint32_t link_first = get_arg_val<uint32_t>(a++);
    const uint32_t link_stride = get_arg_val<uint32_t>(a++);
    uint32_t fin_xy[8][2], full_f[8];  // per final core: NoC x, y and the chunks I send it
    for (uint32_t i = 0; i < nfin; ++i) {
        fin_xy[i][0] = get_arg_val<uint32_t>(a++);
        fin_xy[i][1] = get_arg_val<uint32_t>(a++);
        full_f[i] = get_arg_val<uint32_t>(a++);
    }
    const uint16_t dst_mesh_id = static_cast<uint16_t>(get_arg_val<uint32_t>(a++));
    const uint16_t dst_chip_id = static_cast<uint16_t>(get_arg_val<uint32_t>(a++));
    const uint32_t nblocks = get_arg_val<uint32_t>(a++);  // entries: slot | half << 12 | relay << 14 | fin << 15
    const uint32_t blocks_idx = a;
    a += nblocks;
    const auto scr = TensorAccessor(scr_args, scr_addr, page_bytes);

    if (nblocks > 0 || send_ready) {
#ifdef FRS_MUX
        // ccl::fabric_mux_connection_rt_args, 17 args
        a++;  // connection valid (always)
        const bool term_master = get_arg_val<uint32_t>(a++);
        const uint8_t mux_x = get_arg_val<uint32_t>(a++);
        const uint8_t mux_y = get_arg_val<uint32_t>(a++);
        const size_t mux_channel_base = get_arg_val<uint32_t>(a++);
        const size_t mux_connection_info = get_arg_val<uint32_t>(a++);
        const size_t mux_handshake = get_arg_val<uint32_t>(a++);
        const size_t mux_flow_control = get_arg_val<uint32_t>(a++);
        const size_t mux_buffer_index = get_arg_val<uint32_t>(a++);
        const uint8_t mux_channel_id = get_arg_val<uint32_t>(a++);
        const uint32_t term_sync = get_semaphore(get_arg_val<uint32_t>(a++));
        const uint32_t local_status = get_semaphore(get_arg_val<uint32_t>(a++));
        const uint32_t local_flow_control = get_semaphore(get_arg_val<uint32_t>(a++));
        const uint32_t local_teardown = get_semaphore(get_arg_val<uint32_t>(a++));
        const uint32_t local_buffer_index = get_semaphore(get_arg_val<uint32_t>(a++));
        const uint8_t term_master_x = get_arg_val<uint32_t>(a++);
        const uint8_t term_master_y = get_arg_val<uint32_t>(a++);
        auto conn = build_connection_to_fabric_endpoint<mux_num_buffers>(
            mux_x, mux_y, mux_channel_id, mux_num_buffers, mux_buffer_bytes, mux_channel_base, mux_connection_info,
            mux_handshake, mux_flow_control, mux_buffer_index, local_flow_control, local_teardown, local_buffer_index);
#else
        auto conn = WorkerToFabricEdmSender::build_from_args<ProgrammableCoreType::TENSIX>(a);
#endif
        volatile tt_l1_ptr PACKET_HEADER_TYPE* hdrs[num_headers];
        for (uint32_t h = 0; h < num_headers; ++h) {
            hdrs[h] = PacketHeaderPool::allocate_header();
            route_one_hop(hdrs[h], dst_chip_id, dst_mesh_id);
        }
#ifdef FRS_MUX
        wait_for_fabric_endpoint_ready(mux_x, mux_y, mux_status_address, local_status);
        fabric_client_connect(conn);
#else
        conn.open();
#endif
        if (send_ready) {
            hdrs[0]->to_noc_unicast_atomic_inc(NocUnicastAtomicIncCommandHeader{get_noc_addr(ready_x, ready_y, ready_addr), 1, true});
            conn.wait_for_empty_write_slot();
            conn.send_payload_flush_blocking_from_address(reinterpret_cast<uint32_t>(hdrs[0]), sizeof(PACKET_HEADER_TYPE));
        }
        if (nblocks > 0) {
            volatile tt_l1_ptr uint32_t* ready = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ready_addr);
            noc_semaphore_wait_min(ready, 1);
            noc_semaphore_inc(get_noc_addr(ready_addr), 0u - 1u);
            noc_async_atomic_barrier();
        }
        const uint64_t relay_noc = get_noc_addr(peer_x, peer_y, arrival_addr);
        uint64_t final_noc[8];
        for (uint32_t i = 0; i < nfin; ++i) {
            final_noc[i] = get_noc_addr(fin_xy[i][0], fin_xy[i][1], final_sem_addr);
        }
        // Per entry, from c_16 (relay) or c_3 (chain start), as the reader batches: pop at each multiple of `group`
        // chunks in the CB (reads never wrap) and at the end of the entry.
        uint32_t rpos[2] = {0, 0};
        uint32_t sent_relay = 0, sent_f[8] = {0, 0, 0, 0, 0, 0, 0, 0};
        uint32_t h = 0;
        for (uint32_t k = 0; k < nblocks; ++k) {
            const uint32_t entry = get_arg_val<uint32_t>(blocks_idx + k);
            const uint32_t base = (entry & 0xFFF) * blk_pages;
            const uint32_t half = (entry >> 12) & 3;
            const uint32_t s = (entry >> 14) & 1;
            const uint32_t cb = s ? 16 : 3;
            const bool fin = (entry >> 15) & 1;
            uint32_t u = 0, lim = group - rpos[s] % group, popped = 0;
            for_each_chunk_half(blk_pages, first_bank, bank_stride, num_banks, run_pages, half, [&](uint32_t page, uint32_t n, uint32_t) {
                cb_wait_front(cb, run_pages * (u + 1));
                const uint32_t src = get_read_ptr(cb) + u * chunk_bytes;
                const uint32_t bytes = n * page_bytes;
                const uint64_t dst = scr.get_noc_addr(base + page, 0, 0);
                volatile tt_l1_ptr PACKET_HEADER_TYPE* hdr = hdrs[h];
                h = (h + 1 == num_headers) ? 0 : h + 1;
                bool inc;
                uint64_t ctr;
                if (fin) {
                    const uint32_t fh = (((page % num_banks) - link_first) / link_stride) % nfin;
                    const uint32_t sent = ++sent_f[fh];
                    inc = (sent % inc_every) == 0 || sent == full_f[fh];
                    ctr = final_noc[fh];
                } else {
                    ++sent_relay;
                    inc = (sent_relay % inc_every) == 0 || sent_relay == relay_total;
                    ctr = relay_noc;
                }
                if (inc) {
                    hdr->to_noc_fused_unicast_write_atomic_inc(NocUnicastAtomicIncFusedCommandHeader{dst, ctr, 1, true}, bytes);
                } else {
                    hdr->to_noc_unicast_write(NocUnicastCommandHeader{dst}, bytes);
                }
                conn.wait_for_empty_write_slot();
                conn.send_current_slot_non_blocking(src, bytes, reinterpret_cast<uint32_t>(hdr));
                if (++u == lim) {
                    noc_async_writes_flushed();
                    cb_pop_front(cb, run_pages * u);
                    popped += u;
                    u = 0;
                    lim = group;
                }
            });
            if (u > 0) {
                noc_async_writes_flushed();
                cb_pop_front(cb, run_pages * u);
                popped += u;
            }
            rpos[s] += popped;
        }
#ifdef FRS_MUX
        noc_async_write_barrier();
        fabric_client_disconnect(conn);
        if (term_master) {
            noc_semaphore_wait(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(term_sync), mux_num_clients - 1);
            fabric_endpoint_terminate(mux_x, mux_y, mux_termination_address);
        } else {
            noc_semaphore_inc(get_noc_addr(term_master_x, term_master_y, term_sync), 1);
            noc_async_atomic_barrier();
        }
#else
        conn.close();
#endif
    }
    if (expect_in > 0) {
        volatile tt_l1_ptr uint32_t* arrived = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(arrival_addr);
        noc_semaphore_wait_min(arrived, expect_in);
        noc_semaphore_inc(get_noc_addr(arrival_addr), 0u - expect_in);
        noc_async_atomic_barrier();
    }
}
"""
)


def _incs(chunks, inc_every):
    return -(-chunks // inc_every) if chunks else 0


def _chunks_half(shard_pages, first_bank, stride, num_banks, run_pages, half):
    """Chunks of one entry (replica of for_each_chunk_half): the whole block, or half 0 / 1 of its chunk rounds."""
    if half == 2:
        return _chunks_per_shard(shard_pages, first_bank, stride, num_banks, run_pages)
    max_per_bank = -(-shard_pages // num_banks)
    split = (-(-max_per_bank // run_pages) + 1) // 2 * run_pages
    total = 0
    for b in range(first_bank, num_banks, stride):
        per_bank = -(-(shard_pages - b) // num_banks) if b < shard_pages else 0
        lo, hi = (split, per_bank) if half else (0, min(split, per_bank))
        total += -(-(hi - lo) // run_pages) if hi > lo else 0
    return total


def _entries(p, G, d, ring=False, balance=False):
    """What port d of position p sends, in order (farthest block first): [(block, half, slot at the receiver, relay,
    fin)]. half 2 = the whole block, 0 / 1 = the first / second half of its chunk rounds (for_each_chunk_half). relay: p has an upstream for it (adds what arrived to its own partial); else p starts its chain.
    fin: the downstream chip's own block (to its final cores; backward it lands in slot G, as slot j holds the forward
    arrival); every other entry the downstream relays, in this order.

    Line: toward p+1 blocks G-1 .. p+1 (relays unless p is the first chip), toward p-1 blocks 0 .. p-1.
    Ring: toward p+1 the G//2 blocks p+G//2 .. p+1, toward p-1 the other G-1-G//2 (indices mod G); each block's
    partials are summed along the shorter arc to its owner. balance (even G >= 4): the far block is split instead, half 0
    the long way forward and half 1 the long way backward, so each direction carries G/2 - 1/2 blocks, not G/2 and
    G/2 - 1; every block then travels as two halves, so what a port sends and what its downstream relays match."""
    step = 1 if d == "fwd" else -1
    out = []
    if ring and balance and G % 2 == 0 and G >= 4:
        h_long = 0 if d == "fwd" else 1
        for i in range(G // 2, 0, -1):
            for h in (0, 1):
                start = G // 2 if h == h_long else G // 2 - 1  # distance of the half's chain start from its owner
                if i <= start:
                    out.append(((p + step * i) % G, h, i < start))
    elif ring:
        n = G // 2 if d == "fwd" else G - 1 - G // 2
        out = [((p + step * i) % G, 2, i < n) for i in range(n, 0, -1)]
    elif d == "fwd":
        out = [(j, 2, p > 0) for j in range(G - 1, p, -1)]
    else:
        out = [(j, 2, p < G - 1) for j in range(0, p)]
    own = (p + step) % G
    return [(j, h, G if d == "bwd" and j == own else j, relay, j == own) for j, h, relay in out]


def create_mesh_program_descriptor(
    mesh_device,
    input_tensor,
    scratch,
    output,
    sems,
    chips,
    *,
    num_links,
    ring=False,
    balance=True,
    workers=1,
    finals=2,
    mux_buffers=2,
    mux_noc=1,
    cb_bytes=112 * 1024,
    inc_every=8,
):
    sem_a, sem_b, ready_addr = sems
    page_bytes = int(input_tensor.buffer_aligned_page_size())
    num_banks = _num_banks(mesh_device)
    G = next(iter(chips.values()))["G"]
    shape = list(input_tensor.padded_shape)
    blk_pages = (shape[-2] // G // 32) * (shape[-1] // 32)
    run_pages = max(1, ttnn.get_tt_fabric_max_payload_size_bytes() // page_bytes)
    chunk_bytes = run_pages * page_bytes
    group = max(1, min(8, cb_bytes // (2 * chunk_bytes)))
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
    stride = num_links
    src = ttnn.KernelDescriptor.SourceType.SOURCE_CODE
    cset = lambda cores: ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in cores])
    W = workers
    F = finals if W == 1 else W  # final cores per link: with W > 1, one per worker (its banks)
    assert F * stride <= num_banks, f"{F} final cores x {stride} links need at least {F * stride} DRAM banks"
    wstride = W * stride  # worker w of link l: banks l + w L, l + w L + W L, ...
    if W > 1:
        fm = ttnn.experimental.fabric_mux
        full = fm.ChannelType.FULL_SIZE
        mux_cfg = fm.Config(
            num_full_size_channels=W,
            num_header_only_channels=0,
            num_buffers_per_full_size_channel=mux_buffers,
            num_buffers_per_header_only_channel=0,
            full_size_channel_buffer_size_bytes=fm.channel_buffer_size_bytes(),
            base_l1_address=ttnn._ttnn.device.GetAllocatorBaseAddress(mesh_device, ttnn.BufferType.L1),
        )
        mux_ct = list(fm.client_compile_time_args(num_clients=W, channel_type=full, config=mux_cfg))

    mesh_desc = ttnn.MeshProgramDescriptor()
    for coord, ch in chips.items():
        rg = ch["rings"][0]
        p = rg["p"]
        program = ttnn.ProgramDescriptor()
        wk, finals = ch["workers"], ch["finals"]
        cbs = []
        for idx in (CB_OWN, CB_IN, CB_IN2, CB_PASS, CB_OUT):
            cores = finals if idx == CB_IN2 else list(wk.values()) if idx == CB_PASS else list(wk.values()) + finals
            cbs.append(
                ttnn.CBDescriptor(
                    total_size=2 * group * chunk_bytes,
                    core_ranges=cset(cores),
                    format_descriptors=[ttnn.CBFormatDescriptor(idx, input_tensor.dtype, page_bytes)],
                )
            )
        program.cbs = cbs
        pr_rt, ps_rt, pc_rt, mux_rt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
        term_sem = {}  # (d, l): worker 0's termination semaphore id

        def _connection(program, coord, peer_node, link, core, d, l, w):
            """The sender's fabric connection args: straight to the link's router (1 worker), or to the mux (W > 1)."""
            if W == 1:
                return list(ttnn.setup_fabric_connection(node(coord), peer_node, link, program, core))
            master = wk[(d, l, 0)]
            rt = list(
                fm.client_runtime_args(
                    connection_valid=True,
                    is_termination_master=w == 0,
                    channel_type=full,
                    mux_virtual_core=virt(ch["muxes"][(d, l)]),
                    client_index=w,
                    client_logical_core=core,
                    config=mux_cfg,
                    program_descriptor=program,
                    termination_master_virtual_core=virt(master),
                    termination_master_semaphore_id=term_sem.get((d, l)),
                )
            )
            if w == 0:
                term_sem[(d, l)] = rt[10]
            return rt

        for (d, l, w), core in sorted(wk.items()):
            j = 0
            peer = rg["next"] if d == "fwd" else rg["prev"]
            first = l + w * stride
            entries = _entries(p, G, d, ring, balance) if peer is not None else []
            nch = lambda h: _chunks_half(blk_pages, first, wstride, num_banks, run_pages, h)
            n_relay = sum(nch(h) for _, h, _, relay, _ in entries if relay)
            relay_total = sum(nch(h) for _, h, _, _, fin in entries if not fin)
            pr_rt[core.x][core.y] = [in_addr, scr_addr, blk_pages, first, wstride, sem_a, len(entries)] + [
                b | (h << 12) | (int(relay) << 14) for b, h, _, relay, _ in entries
            ]
            pc_rt[core.x][core.y] = [n_relay, 1, 0]
            # what upstream sends my port: every entry but its fin ones (mine, which go to my final cores), which are
            # exactly my relay entries, in the order I relay them
            expect_in = _incs(n_relay, inc_every)
            send_ready = _needs_ready(chips, coord, j, d)
            rc = virt(chips[peer]["workers"][(_OPP[d], l, w)]) if send_ready else None
            args = [scr_addr, blk_pages, first, wstride, sem_a, sem_a if d == "fwd" else sem_b, expect_in, relay_total]
            args += [ready_addr] + ([1, rc.x, rc.y] if send_ready else [0, 0, 0])
            if rg.get(f"{d}_links") and entries:
                # the downstream worker that relays what I send; a line end relays nothing (all I send it is its
                # own block), and with W > 1 its outward side has no workers
                pw = chips[peer]["workers"].get((d, l, w))
                assert pw is not None or relay_total == 0, (coord, d, l, w)
                pc = virt(pw) if pw is not None else ttnn.CoreCoord(0, 0)
                pn = node(peer)
                if W == 1:
                    # the link's F final cores split its banks (final h: banks l + h L, stride F L)
                    fins = [
                        (
                            virt(chips[peer]["finals"][F * l + h]),
                            _chunks_per_shard(blk_pages, l + h * stride, F * stride, num_banks, run_pages),
                        )
                        for h in range(F)
                    ]
                    route = [F, l, stride]
                else:
                    # final core w of the link owns exactly my banks, so each final counter has one sender per
                    # direction (its arrivals come in its own walk order); route stride num_banks: every chunk -> it
                    fins = [
                        (
                            virt(chips[peer]["finals"][W * l + w]),
                            _chunks_per_shard(blk_pages, first, wstride, num_banks, run_pages),
                        )
                    ]
                    route = [1, first, num_banks]
                args += [pc.x, pc.y, *route] + [v for c, n in fins for v in (c.x, c.y, n)]
                args += [int(pn.mesh_id), int(pn.chip_id)]
                args += [len(entries)]
                args += [sl | (h << 12) | (int(relay) << 14) | (int(fin) << 15) for _, h, sl, relay, fin in entries]
                args += _connection(program, coord, pn, rg[f"{d}_links"][l], core, d, l, w)
            elif send_ready:
                args += [0] * 5 + [*(lambda n: (int(n.mesh_id), int(n.chip_id)))(node(peer)), 0]
                args += _connection(program, coord, node(peer), rg[f"{d}_links"][l], core, d, l, w)
            else:
                args += [0] * 8
            ps_rt[core.x][core.y] = args
        fr_rt, fc_rt, fw_rt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
        has_f, has_b = int(rg["prev"] is not None), int(rg["next"] is not None)
        nfin = F  # final cores per link
        for i, core in enumerate(finals):  # final core nfin l + h: banks l + h L, stride nfin L
            first, fstride = i // nfin + (i % nfin) * stride, nfin * stride
            n = _chunks_per_shard(blk_pages, first, fstride, num_banks, run_pages)
            fr_rt[core.x][core.y] = [
                in_addr,
                scr_addr,
                blk_pages,
                first,
                fstride,
                p,
                p,
                G,
                has_f,
                has_b,
                sem_a,
                sem_b,
                n,
            ]
            fc_rt[core.x][core.y] = [n, has_f, has_b]
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
            kernel_source=_ADD_COMPUTE,
            source_type=src,
            core_ranges=cset(cores),
            compile_time_args=[run_pages],
            runtime_args=rt,
            config=ttnn.ComputeConfigDescriptor(),
        )
        kernels = [
            ttnn.KernelDescriptor(
                kernel_source=_PORT_READER,
                source_type=src,
                core_ranges=cset(wk.values()),
                compile_time_args=ct + in_ct + scr_ct,
                runtime_args=pr_rt,
                config=dm("RISCV_1", NOC0),
            ),
            compute(wk.values(), pc_rt),
            ttnn.KernelDescriptor(
                kernel_source=_PORT_SENDER,
                source_type=src,
                core_ranges=cset(wk.values()),
                compile_time_args=ct + scr_ct + (mux_ct if W > 1 else []),
                defines=[("FRS_MUX", "1")] if W > 1 else [],
                runtime_args=ps_rt,
                config=dm("RISCV_0", NOC1),
            ),
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
        if ch["muxes"]:
            for (d, l), core in ch["muxes"].items():
                peer = rg["next"] if d == "fwd" else rg["prev"]
                mux_rt[core.x][core.y] = list(
                    mux_cfg.kernel_runtime_args(
                        source_node_id=node(coord),
                        destination_node_id=node(peer),
                        link_index=rg[f"{d}_links"][l],
                        program_descriptor=program,
                        mux_logical_core=core,
                    )
                )
            kernels.append(
                ttnn.KernelDescriptor(
                    kernel_source="tt_metal/fabric/impl/kernels/tt_fabric_mux.cpp",
                    source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
                    core_ranges=cset(ch["muxes"].values()),
                    compile_time_args=list(mux_cfg.kernel_compile_time_args()),
                    runtime_args=mux_rt,
                    opt_level=ttnn.experimental.fabric_mux.KernelBuildOptLevel.O3,
                    # NoC1 (-Y, then -X): the Ethernet core is one row up from the mux; NoC0 would wrap the column
                    config=dm("RISCV_0", NOC1 if mux_noc == 1 else NOC0),
                )
            )
        program.kernels = kernels
        r, c = coord
        mesh_desc[ttnn.MeshCoordinateRange(ttnn.MeshCoordinate(r, c), ttnn.MeshCoordinate(r, c))] = program
    return mesh_desc


_SEM_CACHE, _SCRATCH_CACHE, _PLAN_CACHE, _DRAM_EP_CACHE = {}, {}, {}, {}


def _chunks_on_banks(shard_pages, banks, num_banks, run_pages):
    """Chunks of one shard that sit in `banks` (each bank's pages in runs of run_pages)."""
    return sum(-(-(-(-(shard_pages - b) // num_banks)) // run_pages) for b in banks if b < shard_pages)


def _dram_noc0_endpoints(mesh_device):
    """[physical NoC (x, y) a worker reads DRAM bank b from on NoC0], from the arch SoC descriptor's dram views."""
    if id(mesh_device) not in _DRAM_EP_CACHE:
        import os
        import yaml

        root = os.environ.get("TT_METAL_HOME", os.getcwd())
        arch_yaml = {"BLACKHOLE": "blackhole_140_arch.yaml", "WORMHOLE_B0": "wormhole_b0_80_arch.yaml"}[
            str(mesh_device.arch()).split(".")[-1]
        ]
        soc = yaml.safe_load(open(os.path.join(root, "tt_metal", "soc_descriptors", arch_yaml)))
        eps = []
        for view in soc["dram_views"]:
            ep = soc["dram"][int(view["channel"])][int(view["worker_endpoint"][0])]
            eps.append(tuple(int(v) for v in ep.split("-")))
        _DRAM_EP_CACHE[id(mesh_device)] = (mesh_device, eps)
    return _DRAM_EP_CACHE[id(mesh_device)][1]


def _with_workers(mesh_device, chips, num_links, workers, worker_placement="spread"):
    """Each chip's plan plus "workers" {(direction, link, w): core} and "muxes" {(direction, link): core}.

    workers = 1: the port cores are the workers and talk to the routers directly (no mux).
    workers > 1: each linked port core becomes the link's fabric mux (it sits by the Ethernet core); its W workers take
    banks l + w L, stride W L. Placement of the reading cores (workers, then the final cores, which it replaces):
      "spread" (default): the fewest NoC1 hops (-Y, then -X) from a worker to its mux, plus 4 hops for every
                 reader already in the core's physical column. NoC0 read data leaves the DRAM endpoint along its
                 row and turns south only in the reader's column, so readers in distinct columns share no link
                 there; readers stacked in one column share its links (see examples/noc_placement). Sharing is a
                 cost, not a ban: NoC1 runs west only, so a worker must sit east of its mux to be near it, and with
                 2 links the columns east of the westmost mux run out (a strict ban sent a worker 8 hops along the
                 muxes' row and was 13% slower). The final cores are placed last, on the least used columns.
      "mux":    the fewest NoC1 hops to the mux only.
      "dram":   the fewest NoC0 hops from the worker's DRAM banks (x2: a relay reads two streams) plus the hops to the
                 mux. Kept for comparison: it stacks workers in the columns beside DRAM, the noc_placement trap.
    Ties: lower row, then lower column."""
    out = {}
    if workers == 1:
        for coord, ch in chips.items():
            out[coord] = {**ch, "workers": {(d, l, 0): c for (_, d, l), c in ch["ports"].items()}, "muxes": {}}
        return out
    gx, gy = _noc_grid(mesh_device)
    eps = _dram_noc0_endpoints(mesh_device)
    nb = _num_banks(mesh_device)
    allowed = sorted(allowed_cores(mesh_device))
    spread = worker_placement == "spread"
    penalty = 4  # hops a shared column costs (measured best of 2, 4, 8 and strict)
    for coord, ch in chips.items():
        rg = ch["rings"][0]
        t2p = _chip_maps(mesh_device, coord)[2]

        def phys(xy):
            v = mesh_device.worker_core_from_logical_core(ttnn.CoreCoord(*xy))
            return t2p.get(v.x, v.x), v.y

        muxes = {}
        for (_, d, l), mux in sorted(ch["ports"].items(), key=lambda kv: (kv[0][1], kv[0][2])):
            if (rg["next"] if d == "fwd" else rg["prev"]) is not None:  # a line end's outward port has no link
                muxes[(d, l)] = mux
        taken = {(c.x, c.y) for c in muxes.values()}
        if not spread:
            taken |= {(c.x, c.y) for c in ch["finals"]}
        readers = {}  # physical column -> reading cores placed in it

        def pick(score):
            best = None
            for xy in allowed:
                if xy in taken:
                    continue
                px, py = phys(xy)
                sc = score(px, py)
                if spread:  # a reader already in the column costs `penalty` hops
                    sc = (sc[0] + penalty * readers.get(px, 0) if sc else readers.get(px, 0),) + sc[1:]
                key = sc + (py, px)
                if best is None or key < best[0]:
                    best = (key, xy)
            if best is None:
                raise ValueError(f"fabric_reduce_scatter: too few free cores for {workers} workers per link")
            taken.add(best[1])
            px = phys(best[1])[0]
            readers[px] = readers.get(px, 0) + 1
            return ttnn.CoreCoord(*best[1])

        wk = {}
        for (d, l), mux in muxes.items():
            mx, my = phys((mux.x, mux.y))
            for w in range(workers):
                banks = [b for b in range(nb) if b % (workers * num_links) == l + w * num_links]

                def score(px, py):
                    sends = (py - my) % gy + (px - mx) % gx
                    if worker_placement != "dram":
                        return (sends,)
                    reads = sum((px - eps[b][0]) % gx + (py - eps[b][1]) % gy for b in banks) / max(1, len(banks))
                    return (2 * reads + sends,)

                wk[(d, l, w)] = pick(score)
        finals = [pick(lambda px, py: ()) for _ in ch["finals"]] if spread else ch["finals"]
        out[coord] = {**ch, "workers": wk, "muxes": muxes, "finals": finals}
    return out


def _with_final_cores(mesh_device, chips, num_links, nfin=2):
    """Each chip's plan plus "finals": nfin final cores per link, [link 0 core 0 .. nfin - 1, link 1 core 0, ...].
    Core 0 is fabric_all_gather's copy core of the link; the others the next free cores in its row (else any)."""
    allowed = sorted(allowed_cores(mesh_device), key=lambda c: (c[1], c[0]))
    out = {}
    for coord, ch in chips.items():
        taken = {(c.x, c.y) for c in list(ch["ports"].values()) + ch["copy"]}
        finals = []
        for l in range(num_links):
            row = ch["copy"][l].y
            finals.append(ch["copy"][l])
            for _ in range(nfin - 1):
                extra = next((c for c in allowed if c[1] == row and c not in taken), None) or next(
                    (c for c in allowed if c not in taken), None
                )
                if extra is None:
                    raise ValueError(
                        f"fabric_reduce_scatter: the core grid has too few cores for {nfin} final cores per link"
                    )
                taken.add(extra)
                finals.append(ttnn.CoreCoord(*extra))
        out[coord] = {**ch, "finals": finals}
    return out


def fabric_reduce_scatter(
    input_tensor,
    *,
    cluster_axis=0,
    num_links=1,
    topology=ttnn.Topology.Linear,
    balance=True,
    workers=1,
    finals=2,
    mux_buffers=2,
    mux_noc=1,
    worker_placement="spread",
    output=None,
    placement="auto",
    inc_every=8,
    cb_bytes=112 * 1024,
):
    """Reduce-scatter `input_tensor` ([1, 1, G * Sb, W] bf16 TILE, DRAM interleaved, a partial per chip) over each group
    along `cluster_axis` (None: one snake over the whole mesh), as a line or a ring (`topology`; a 2-chip ring is a
    line): chip p gets block p of the sum, [1, 1, Sb, W] (the layout ttnn.reduce_scatter(dim=2) gives). balance: on a
    ring of even G >= 4, split the far block between the two directions (see _entries). workers: cores per link and
    direction that read, add and send; above 1 they share the link through a fabric mux (see _with_workers).
    finals: final cores per link with workers=1 (final h owns the link's banks l + h L, stride finals L); with workers >
    1 there is one final core per worker.
    """
    mesh_device = input_tensor.device()
    key = (id(mesh_device), cluster_axis, num_links, placement, topology, workers, worker_placement, finals)
    if key not in _PLAN_CACHE:
        chips, ring = plan(
            mesh_device,
            cluster_axis=cluster_axis,
            topology=topology,
            num_links=num_links,
            placement=placement,
        )
        chips = _with_final_cores(mesh_device, chips, num_links, finals if workers == 1 else workers)
        chips = _with_workers(mesh_device, chips, num_links, workers, worker_placement)
        _PLAN_CACHE[key] = (mesh_device, chips, ring)
    _, chips, ring = _PLAN_CACHE[key]
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
    cores = {(c.x, c.y) for ch in chips.values() for c in list(ch["workers"].values()) + ch["finals"]}
    ckey = (id(mesh_device), tuple(sorted(cores)))
    if ckey not in _SEM_CACHE:
        crs = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(x, y), ttnn.CoreCoord(x, y)) for x, y in sorted(cores)])
        sems = tuple(ttnn.create_global_semaphore(mesh_device, crs, 0) for _ in range(3))
        ttnn.synchronize_device(mesh_device)  # every counter zero before any chip's first call increments a neighbour's
        _SEM_CACHE[ckey] = (mesh_device, sems)
    sems = tuple(int(ttnn.get_global_semaphore_address(s)) for s in _SEM_CACHE[ckey][1])
    desc = create_mesh_program_descriptor(
        mesh_device,
        input_tensor,
        scratch,
        output,
        sems,
        chips,
        num_links=num_links,
        ring=ring,
        balance=balance,
        workers=workers,
        finals=finals,
        mux_buffers=mux_buffers,
        mux_noc=mux_noc,
        cb_bytes=cb_bytes,
        inc_every=inc_every,
    )
    return ttnn.generic_op([input_tensor, scratch, output], desc)
