# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""fabric_reduce_scatter — a line reduce-scatter over fabric, DRAM to DRAM, as one ttnn.generic_op, built on
fabric_all_gather's plan (groups, port placement under each link's Ethernet core, fabric connections, fence).

Input per chip: [1, 1, G * Sb, W] bf16 TILE (block j = rows j Sb ..), DRAM interleaved; output per chip p: block p of
the sum over the group's G chips, [1, 1, Sb, W]. The group is a line along `cluster_axis` (chip p = position p).

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

import ttnn
from ttnn.operations.examples.fabric_all_gather.program_descriptor_with_inline_kernels import (
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
// Port reader: per block it sends, per chunk: my partial (input) and, as a relay, what upstream sent (scratch, once the
// arrival counter covers it). A line end (no upstream) pushes its partial straight to the sender's CB.
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
    const uint32_t relay = get_arg_val<uint32_t>(a++);
    const uint32_t nblocks = get_arg_val<uint32_t>(a++);
    const uint32_t blocks_idx = a;
    const auto in = TensorAccessor(in_args, in_addr, page_bytes);
    const auto scr = TensorAccessor(scr_args, scr_addr, page_bytes);
    volatile tt_l1_ptr uint32_t* arrived = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(arrival_addr);
    const uint32_t cb_own = relay ? 0 : 16;  // a line end feeds the sender directly

    uint32_t chunks_left = nblocks * full;
    uint32_t batch = 0, wown = 0, win = 0;
    for (uint32_t k = 0; k < nblocks; ++k) {
        const uint32_t base = (get_arg_val<uint32_t>(blocks_idx + k) & 0xFFFF) * blk_pages;
        for_each_chunk(blk_pages, first_bank, bank_stride, num_banks, run_pages, 0, 0, [&](uint32_t page, uint32_t n, uint32_t idx) {
            if (batch == 0) {
                cb_reserve_back(cb_own, run_pages * group);
                wown = get_write_ptr(cb_own);
                if (relay) {
                    cb_reserve_back(1, run_pages * group);
                    win = get_write_ptr(1);
                }
            }
            noc_async_read(in.get_noc_addr(base + page), wown + batch * chunk_bytes, n * page_bytes);
            if (relay) {
                noc_semaphore_wait_min(arrived, (k * full + idx) / inc_every + 1);
                noc_async_read(scr.get_noc_addr(base + page), win + batch * chunk_bytes, n * page_bytes);
            }
            --chunks_left;
            if (++batch == group || chunks_left == 0) {
                noc_async_read_barrier();
                cb_push_back(cb_own, run_pages * batch);
                if (relay) {
                    cb_push_back(1, run_pages * batch);
                }
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

// Sum of 2 or 3 inputs, run_pages tiles per chunk: out(c_16) = c_0 + c_1 [+ c_2]; or c_0 + c_2 (a final core without a
// forward upstream). n = 0: nothing to do (a line end's port: its reader feeds the sender directly).
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
    + r"""
// Port sender: every chunk of c_16 (the sums, or a line end's partials) into the downstream chip's scratch. Three
// increment streams: the relay blocks count on the downstream port's counter (A), the last block (the downstream
// chip's own) on the counter (A forward, B backward) of the downstream final core that owns the chunk's bank; every
// inc_every-th chunk and each stream's last carries one. After sending: wait for everything upstream sends me, then re-arm my counter.
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
    const uint32_t full_f0 = get_arg_val<uint32_t>(a++);  // chunks of the last block for each final core
    const uint32_t full_f1 = get_arg_val<uint32_t>(a++);
    const uint16_t dst_mesh_id = static_cast<uint16_t>(get_arg_val<uint32_t>(a++));
    const uint16_t dst_chip_id = static_cast<uint16_t>(get_arg_val<uint32_t>(a++));
    const uint32_t nblocks = get_arg_val<uint32_t>(a++);  // entries: block | slot << 16
    const uint32_t blocks_idx = a;
    a += nblocks;
    const auto scr = TensorAccessor(scr_args, scr_addr, page_bytes);

    if (nblocks > 0 || send_ready) {
        auto conn = WorkerToFabricEdmSender::build_from_args<ProgrammableCoreType::TENSIX>(a);
        volatile tt_l1_ptr PACKET_HEADER_TYPE* hdrs[num_headers];
        for (uint32_t h = 0; h < num_headers; ++h) {
            hdrs[h] = PacketHeaderPool::allocate_header();
            route_one_hop(hdrs[h], dst_chip_id, dst_mesh_id);
        }
        conn.open();
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
        const uint64_t final_noc = get_noc_addr(final_x, final_y, final_sem_addr);
        const uint64_t final_noc1 = get_noc_addr(final_x1, final_y1, final_sem_addr);
        const uint32_t relay_total = (nblocks - 1) * full;
        uint32_t chunks_left = nblocks * full;
        uint32_t sent_relay = 0, sent_f0 = 0, sent_f1 = 0;
        uint32_t h = 0, unflushed = 0;
        for (uint32_t k = 0; k < nblocks; ++k) {
            const uint32_t entry = get_arg_val<uint32_t>(blocks_idx + k);
            const uint32_t base = (entry >> 16) * blk_pages;
            const bool fin = k + 1 == nblocks;
            for_each_chunk(blk_pages, first_bank, bank_stride, num_banks, run_pages, 0, 0, [&](uint32_t page, uint32_t n, uint32_t) {
                cb_wait_front(16, run_pages * (unflushed + 1));
                const uint32_t src = get_read_ptr(16) + unflushed * chunk_bytes;
                const uint32_t bytes = n * page_bytes;
                const uint64_t dst = scr.get_noc_addr(base + page, 0, 0);
                volatile tt_l1_ptr PACKET_HEADER_TYPE* hdr = hdrs[h];
                h = (h + 1 == num_headers) ? 0 : h + 1;
                bool inc;
                uint64_t ctr;
                if (fin) {
                    // final core h owns banks first_bank + (2 i + h) * bank_stride of my set
                    const bool h1 = (((page % num_banks) - first_bank) / bank_stride) & 1;
                    uint32_t& sent = h1 ? sent_f1 : sent_f0;
                    ++sent;
                    inc = (sent % inc_every) == 0 || sent == (h1 ? full_f1 : full_f0);
                    ctr = h1 ? final_noc1 : final_noc;
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
                --chunks_left;
                if (++unflushed == group || chunks_left == 0) {
                    noc_async_writes_flushed();
                    cb_pop_front(16, run_pages * unflushed);
                    unflushed = 0;
                }
            });
        }
        conn.close();
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


def _blocks(p, G, d):
    """Blocks port d of line position p sends, in order (farthest first): [(block, slot at the receiver)]."""
    if d == "fwd":
        return [(j, j) for j in range(G - 1, p, -1)]
    blocks = list(range(0, p))
    return [(j, G if j == p - 1 else j) for j in blocks]


def create_mesh_program_descriptor(
    mesh_device, input_tensor, scratch, output, sems, chips, *, num_links, cb_bytes=112 * 1024, inc_every=8
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

    mesh_desc = ttnn.MeshProgramDescriptor()
    for coord, ch in chips.items():
        rg = ch["rings"][0]
        p = rg["p"]
        program = ttnn.ProgramDescriptor()
        ports, finals = ch["ports"], ch["finals"]
        cbs = []
        for idx in (CB_OWN, CB_IN, CB_IN2, CB_OUT):
            cores = list(ports.values()) + finals if idx != CB_IN2 else finals
            cbs.append(
                ttnn.CBDescriptor(
                    total_size=2 * group * chunk_bytes,
                    core_ranges=cset(cores),
                    format_descriptors=[ttnn.CBFormatDescriptor(idx, input_tensor.dtype, page_bytes)],
                )
            )
        program.cbs = cbs
        pr_rt, ps_rt, pc_rt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
        for (j, d, l), core in ports.items():
            peer = rg["next"] if d == "fwd" else rg["prev"]
            blocks = _blocks(p, G, d) if peer is not None else []
            up = rg["prev"] if d == "fwd" else rg["next"]
            relay = up is not None and len(blocks) > 0
            full = _chunks_per_shard(blk_pages, l, stride, num_banks, run_pages)
            packed = [b | (s << 16) for b, s in blocks]
            pr_rt[core.x][core.y] = [in_addr, scr_addr, blk_pages, l, stride, sem_a, full, int(relay), len(blocks)] + [
                b for b, _ in blocks
            ]
            pc_rt[core.x][core.y] = [len(blocks) * full if relay else 0, 1, 0]
            # what upstream relays to me: my relay blocks (all but the downstream's own, which I send last)
            expect_in = _incs(len(blocks) * full, inc_every) if relay else 0
            send_ready = _needs_ready(chips, coord, j, d)
            rc = virt(chips[peer]["ports"][(j, _OPP[d], l)]) if send_ready else None
            args = [scr_addr, blk_pages, l, stride, sem_a, sem_a if d == "fwd" else sem_b, expect_in, full, ready_addr]
            args += [1, rc.x, rc.y] if send_ready else [0, 0, 0]
            if rg.get(f"{d}_links") and blocks:
                pc = virt(chips[peer]["ports"][(j, d, l)])
                fc0 = virt(chips[peer]["finals"][2 * l])
                fc1 = virt(chips[peer]["finals"][2 * l + 1])
                nf0 = _chunks_per_shard(blk_pages, l, 2 * stride, num_banks, run_pages)
                nf1 = _chunks_per_shard(blk_pages, l + stride, 2 * stride, num_banks, run_pages)
                pn = node(peer)
                args += [
                    pc.x,
                    pc.y,
                    fc0.x,
                    fc0.y,
                    fc1.x,
                    fc1.y,
                    nf0,
                    nf1,
                    int(pn.mesh_id),
                    int(pn.chip_id),
                    len(blocks),
                ] + packed
                args += list(ttnn.setup_fabric_connection(node(coord), pn, rg[f"{d}_links"][l], program, core))
            elif send_ready:
                args += [0] * 8 + [*(lambda n: (int(n.mesh_id), int(n.chip_id)))(node(peer)), 0]
                args += list(ttnn.setup_fabric_connection(node(coord), node(peer), rg[f"{d}_links"][l], program, core))
            else:
                args += [0] * 11
            ps_rt[core.x][core.y] = args
        fr_rt, fc_rt, fw_rt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
        has_f, has_b = int(rg["prev"] is not None), int(rg["next"] is not None)
        for i, core in enumerate(finals):  # final core 2 l + h: banks l + h L, stride 2 L
            first, fstride = i // 2 + (i % 2) * stride, 2 * stride
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
        program.kernels = [
            ttnn.KernelDescriptor(
                kernel_source=_PORT_READER,
                source_type=src,
                core_ranges=cset(ports.values()),
                compile_time_args=ct + in_ct + scr_ct,
                runtime_args=pr_rt,
                config=dm("RISCV_1", NOC0),
            ),
            compute(ports.values(), pc_rt),
            ttnn.KernelDescriptor(
                kernel_source=_PORT_SENDER,
                source_type=src,
                core_ranges=cset(ports.values()),
                compile_time_args=ct + scr_ct,
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
        r, c = coord
        mesh_desc[ttnn.MeshCoordinateRange(ttnn.MeshCoordinate(r, c), ttnn.MeshCoordinate(r, c))] = program
    return mesh_desc


_SEM_CACHE, _SCRATCH_CACHE, _PLAN_CACHE = {}, {}, {}


def _with_final_cores(mesh_device, chips, num_links):
    """Each chip's plan plus "finals": two final cores per link, [link 0 half 0, link 0 half 1, link 1 half 0, ...].
    Half 0 is fabric_all_gather's copy core of the link; half 1 the next free core in its row (else any free core)."""
    allowed = sorted(allowed_cores(mesh_device), key=lambda c: (c[1], c[0]))
    out = {}
    for coord, ch in chips.items():
        taken = {(c.x, c.y) for c in list(ch["ports"].values()) + ch["copy"]}
        finals = []
        for l in range(num_links):
            row = ch["copy"][l].y
            extra = next((c for c in allowed if c[1] == row and c not in taken), None) or next(
                (c for c in allowed if c not in taken), None
            )
            if extra is None:
                raise ValueError("fabric_reduce_scatter: the core grid has too few cores for two final cores per link")
            taken.add(extra)
            finals += [ch["copy"][l], ttnn.CoreCoord(*extra)]
        out[coord] = {**ch, "finals": finals}
    return out


def fabric_reduce_scatter(input_tensor, *, cluster_axis=0, num_links=1, output=None, placement="auto"):
    """Reduce-scatter `input_tensor` ([1, 1, G * Sb, W] bf16 TILE, DRAM interleaved, a partial per chip) over each line
    along `cluster_axis`: chip p gets block p of the sum, [1, 1, Sb, W] (the layout ttnn.reduce_scatter(dim=2) gives).
    """
    mesh_device = input_tensor.device()
    key = (id(mesh_device), cluster_axis, num_links, placement)
    if key not in _PLAN_CACHE:
        chips, _ = plan(
            mesh_device,
            cluster_axis=cluster_axis,
            topology=ttnn.Topology.Linear,
            num_links=num_links,
            placement=placement,
        )
        _PLAN_CACHE[key] = (mesh_device, _with_final_cores(mesh_device, chips, num_links))
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
    cores = {(c.x, c.y) for ch in chips.values() for c in list(ch["ports"].values()) + ch["finals"]}
    ckey = (id(mesh_device), tuple(sorted(cores)))
    if ckey not in _SEM_CACHE:
        crs = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(x, y), ttnn.CoreCoord(x, y)) for x, y in sorted(cores)])
        sems = tuple(ttnn.create_global_semaphore(mesh_device, crs, 0) for _ in range(3))
        ttnn.synchronize_device(mesh_device)  # every counter zero before any chip's first call increments a neighbour's
        _SEM_CACHE[ckey] = (mesh_device, sems)
    sems = tuple(int(ttnn.get_global_semaphore_address(s)) for s in _SEM_CACHE[ckey][1])
    desc = create_mesh_program_descriptor(mesh_device, input_tensor, scratch, output, sems, chips, num_links=num_links)
    return ttnn.generic_op([input_tensor, scratch, output], desc)
