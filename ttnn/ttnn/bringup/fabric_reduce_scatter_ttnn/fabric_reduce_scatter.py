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

CB_OWN, CB_IN, CB_IN2, CB_LAND, CB_OUT = 0, 1, 2, 3, 16

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
    a += nblocks;
    const uint32_t direct = get_arg_val<uint32_t>(a++);  // every block leads: my partials straight into c_16
    const uint32_t has_fin = get_arg_val<uint32_t>(a++);  // after the sends, my own block's sum (final="bwd_port")
    const uint32_t own_blk = get_arg_val<uint32_t>(a++);
    const uint32_t bwd_slot = get_arg_val<uint32_t>(a++);  // G: the backward arrival of my own block
    const uint32_t b_addr = get_arg_val<uint32_t>(a++);  // counter B: the forward arrival of my own block
    const uint32_t land = get_arg_val<uint32_t>(a++);  // the forward arrival lands in c_3 (L1), not in scratch
    const uint32_t cap_land = get_arg_val<uint32_t>(a++);  // c_3's chunks
    const auto in = TensorAccessor(in_args, in_addr, page_bytes);
    const auto scr = TensorAccessor(scr_args, scr_addr, page_bytes);
    volatile tt_l1_ptr uint32_t* arrived = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(arrival_addr);

    // A batch is written contiguously from the CB's write pointer, so it must not cross the CB's end: the CBs hold
    // cap = 2 group chunks, and with two phases (or c_0 and c_1 starting at different offsets) the batches are not
    // group-aligned. o0 / o1: the chunk offset of the next write in the own-partial CB (c_0, or c_16) and in c_1.
    constexpr uint32_t cap = 2 * group;
    uint32_t o0 = 0, o1 = 0;
    auto room = [&](uint32_t o) { return cap - o < group ? cap - o : group; };
    auto room3 = [&](uint32_t o) { return cap_land - o < group ? cap_land - o : group; };

    // leading blocks: my partial only
    const uint32_t cb_lead = direct ? 16 : 0;
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
    // my own block: my partial (c_0), the forward arrival (c_1: scratch slot own, or c_3 with land; counter B) and
    // the backward one (c_2: scratch slot G, counter A, after the relays)
    if (has_fin) {
        const uint32_t cb_fwd = land ? 3 : 1;
        uint32_t o3 = 0;
        volatile tt_l1_ptr uint32_t* arr_b = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(b_addr);
        const uint32_t own_base = own_blk * blk_pages, bwd_base = bwd_slot * blk_pages;
        const uint32_t a_before = (nblocks - lead) * full;
        uint32_t o2 = 0, w2 = 0;
        chunks_left = full;
        for_each_chunk(blk_pages, first_bank, bank_stride, num_banks, run_pages, 0, 0, [&](uint32_t page, uint32_t n, uint32_t idx) {
            if (batch == 0) {
                limit = room(o0) < room(o2) ? room(o0) : room(o2);
                const uint32_t rf = land ? room3(o3) : room(o1);
                limit = rf < limit ? rf : limit;
                cb_reserve_back(0, run_pages * limit);
                wown = get_write_ptr(0);
                cb_reserve_back(cb_fwd, run_pages * limit);
                win = get_write_ptr(cb_fwd);
                cb_reserve_back(2, run_pages * limit);
                w2 = get_write_ptr(2);
            }
            noc_async_read(in.get_noc_addr(own_base + page), wown + batch * chunk_bytes, n * page_bytes);
            noc_semaphore_wait_min(arr_b, idx / inc_every + 1);
            if (!land) {
                noc_async_read(scr.get_noc_addr(own_base + page), win + batch * chunk_bytes, n * page_bytes);
            }
            noc_semaphore_wait_min(arrived, (a_before + idx) / inc_every + 1);
            noc_async_read(scr.get_noc_addr(bwd_base + page), w2 + batch * chunk_bytes, n * page_bytes);
            --chunks_left;
            if (++batch == limit || chunks_left == 0) {
                noc_async_read_barrier();
                cb_push_back(0, run_pages * batch);
                cb_push_back(cb_fwd, run_pages * batch);
                cb_push_back(2, run_pages * batch);
                o0 = (o0 + batch) % cap;
                if (land) {
                    o3 = (o3 + batch) % cap_land;
                } else {
                    o1 = (o1 + batch) % cap;
                }
                o2 = (o2 + batch) % cap;
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
// chain). Then n chunks: out = c_0 + c_1 [+ c_2], or c_0 + c_2 (a final core without a forward upstream). Then n_fin
// chunks: out = c_0 + c_1 (or c_3) + c_2 (a backward port's own block, final="bwd_port" / "bwd_port_l1").
// Nothing (n_copy = n = n_fin = 0): a line end's port, whose reader feeds the sender directly.
void kernel_main() {
    constexpr uint32_t R = get_compile_time_arg_val(0);
    constexpr uint32_t DST = 4;
    const uint32_t n = get_arg_val<uint32_t>(0);
    const uint32_t has_b = get_arg_val<uint32_t>(1);  // second input c_1
    const uint32_t has_c = get_arg_val<uint32_t>(2);  // third input c_2
    const uint32_t n_copy = get_arg_val<uint32_t>(3);
    const uint32_t n_fin = get_arg_val<uint32_t>(4);
    const uint32_t fin_b = get_arg_val<uint32_t>(5);  // the forward arrival: c_1, or c_3 (landed in L1)
    if (n == 0 && n_copy == 0 && n_fin == 0) {
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
            cb_pop_front(0, R);
        }
    }
    auto adds = [&](uint32_t count, uint32_t in_b, bool tri) {
        add_tiles_init(0, in_b);
        for (uint32_t i = 0; i < count; ++i) {
            cb_wait_front(0, R);
            cb_wait_front(in_b, R);
            if (tri) {
                cb_wait_front(2, R);
            }
            cb_reserve_back(16, R);
            for (uint32_t j0 = 0; j0 < R; j0 += DST) {
                const uint32_t m = R - j0 < DST ? R - j0 : DST;
                tile_regs_acquire();
                if (tri) {
                    add_tiles_init(0, in_b);
                }
                for (uint32_t j = 0; j < m; ++j) {
                    add_tiles(0, in_b, j0 + j, j0 + j, j);
                }
                if (tri) {
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
            cb_pop_front(in_b, R);
            if (tri) {
                cb_pop_front(2, R);
            }
        }
    };
    adds(n, cb_b, three);
    if (n_fin > 0) {
        adds(n_fin, fin_b, true);
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
    constexpr auto out_args = TensorAccessorArgs<scr_args.next_compile_time_args_offset()>();
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
    // fin_mode: where the last block (the downstream chip's own) counts. 0: the downstream final core owning the
    // chunk's bank; 1: one counter (final_x, final_y); 2: the relay stream's counter, continued
    const uint32_t fin_mode = get_arg_val<uint32_t>(a++);
    const uint32_t has_fin = get_arg_val<uint32_t>(a++);  // then my own block's sum out of c_16 into the output
    const uint32_t out_addr = get_arg_val<uint32_t>(a++);
    const uint32_t b_addr = get_arg_val<uint32_t>(a++);  // my counter B (the forward arrival of my own block)
    const uint32_t expect_b = get_arg_val<uint32_t>(a++);  // B's re-arm (with fin_mode 3: the credits, full)
    // fin_mode 3: the last block into the receiver's L1 landing buffer (land_addr on (final_x, final_y), cap_land
    // chunks), a slot reused once B (my credit counter) covers it. credit_x/y: has_fin with a landing buffer: return a
    // credit per landing chunk the compute has popped to the upstream forward port (counter B there)
    const uint32_t land_addr = get_arg_val<uint32_t>(a++);
    const uint32_t cap_land = get_arg_val<uint32_t>(a++);
    const uint32_t credit_on = get_arg_val<uint32_t>(a++);
    const uint32_t credit_x = get_arg_val<uint32_t>(a++);
    const uint32_t credit_y = get_arg_val<uint32_t>(a++);
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
        volatile tt_l1_ptr uint32_t* credits = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(b_addr);
        const uint64_t relay_noc = get_noc_addr(peer_x, peer_y, arrival_addr);
        const uint64_t final_noc = get_noc_addr(final_x, final_y, final_sem_addr);
        const uint64_t final_noc1 = get_noc_addr(final_x1, final_y1, final_sem_addr);
        const uint32_t relay_total = (fin_mode == 2 ? nblocks : nblocks - 1) * full;
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
                uint64_t dst = scr.get_noc_addr(base + page, 0, 0);
                if (fin && fin_mode == 3) {
                    if (sent_f0 >= cap_land) {
                        noc_semaphore_wait_min(credits, sent_f0 - cap_land + 1);
                    }
                    dst = get_noc_addr(final_x, final_y, land_addr + (sent_f0 % cap_land) * chunk_bytes);
                }
                volatile tt_l1_ptr PACKET_HEADER_TYPE* hdr = hdrs[h];
                h = (h + 1 == num_headers) ? 0 : h + 1;
                bool inc;
                uint64_t ctr;
                if (fin && fin_mode != 2) {
                    // final core h owns banks first_bank + (2 i + h) * bank_stride of my set
                    const bool h1 = fin_mode == 0 && ((((page % num_banks) - first_bank) / bank_stride) & 1);
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
        // my own block's sum (after the sums I sent, in c_16) into the output, before the re-arm: my reader is past
        // every wait on A and B once the compute has produced all of it. With a landing buffer: seeing c_16 chunk
        // j + 1 means the compute popped landing chunk j (it unpacks j + 1 after popping j), so credit up to there
        if (has_fin) {
            const auto out = TensorAccessor(out_args, out_addr, page_bytes);
            constexpr uint32_t cap = 2 * group;
            uint32_t o = (nblocks * full) % cap, pending = 0, left = full, seen = 0, credited = 0;
            volatile tt_l1_ptr PACKET_HEADER_TYPE* chdr = nullptr;
            if (credit_on) {
                chdr = PacketHeaderPool::allocate_header();
                route_one_hop(chdr, dst_chip_id, dst_mesh_id);
            }
            const uint64_t credit_noc = get_noc_addr(credit_x, credit_y, b_addr);
            auto credit = [&](uint32_t upto) {
                chdr->to_noc_unicast_atomic_inc(NocUnicastAtomicIncCommandHeader{credit_noc, upto - credited, true});
                conn.wait_for_empty_write_slot();
                conn.send_payload_flush_blocking_from_address(reinterpret_cast<uint32_t>(chdr), sizeof(PACKET_HEADER_TYPE));
                credited = upto;
            };
            for_each_chunk(blk_pages, first_bank, bank_stride, num_banks, run_pages, 0, 0, [&](uint32_t page, uint32_t n, uint32_t) {
                cb_wait_front(16, run_pages * (pending + 1));
                ++seen;
                if (credit_on && seen - 1 >= credited + group) {
                    credit(seen - 1);
                }
                noc_async_write(get_read_ptr(16) + pending * chunk_bytes, out.get_noc_addr(page), n * page_bytes);
                --left;
                if (++pending == group || left == 0 || o + pending == cap) {
                    noc_async_write_barrier();
                    cb_pop_front(16, run_pages * pending);
                    o = (o + pending) % cap;
                    pending = 0;
                }
            });
            if (credit_on && credited < full) {  // the upstream writes nothing more: the rest only re-arms its counter
                credit(full);
            }
        }
        conn.close();
    }
    if (expect_in > 0) {
        volatile tt_l1_ptr uint32_t* arrived = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(arrival_addr);
        noc_semaphore_wait_min(arrived, expect_in);
        noc_semaphore_inc(get_noc_addr(arrival_addr), 0u - expect_in);
        noc_async_atomic_barrier();
    }
    if (expect_b > 0) {
        volatile tt_l1_ptr uint32_t* arr_b = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(b_addr);
        noc_semaphore_wait_min(arr_b, expect_b);
        noc_semaphore_inc(get_noc_addr(b_addr), 0u - expect_b);
        noc_async_atomic_barrier();
    }
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
    The last block a port sends is the receiver's own: forward it lands in the receiver's own slot, backward in slot G.
    """
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
    on_port=False,
    land=None,
    cb_bytes=128 * 1024,  # 8 chunks of 8 KiB per CB batch: a port reads two streams per chunk, so it needs the depth
    inc_every=8,
):
    sem_a, sem_b, ready_addr = sems
    ring = chips_ring
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
    # land: the L1 landing buffer of final="bwd_port_l1" (a tensor sharded over every chip's backward ports)
    land_addr = int(land.buffer_address()) if land is not None else 0
    cap_land = 0
    if land is not None:
        cap_land = land.memory_config().shard_spec.shape[0] // 32 * page_bytes // chunk_bytes
        # the reader hands c_3 to the compute a batch (<= group chunks) at a time, once B covers the whole batch, and
        # credits lag by up to a batch: under 2 batches deadlocks; 2 crawls (128 KiB: 605 us vs 358 at 256 KiB, 2x2)
        assert cap_land >= 3 * group, f"the landing buffer needs at least {3 * group} chunks, has {cap_land}"
    virt = lambda c: mesh_device.worker_core_from_logical_core(c)
    node = lambda coord: mesh_device.get_fabric_node_id(ttnn.MeshCoordinate(*coord))
    dm = lambda risc, noc: ttnn.DataMovementConfigDescriptor(
        processor=getattr(ttnn.DataMovementProcessor, risc), noc=noc
    )
    src = ttnn.KernelDescriptor.SourceType.SOURCE_CODE
    cset = lambda cores: ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in cores])

    mesh_desc = ttnn.MeshProgramDescriptor()
    for coord, ch in chips.items():
        n_rings = ch["n_rings"]
        lanes = n_rings * num_links  # lane (ring j, link l) = j L + l owns DRAM banks lane, lane + lanes, ...
        assert 2 * lanes <= num_banks, f"{n_rings} rings x {num_links} links need at least {2 * lanes} DRAM banks"
        program = ttnn.ProgramDescriptor()
        ports, finals = ch["ports"], ch["finals"]
        own = ch["out"]
        cbs = []
        for idx in (CB_OWN, CB_IN, CB_IN2, CB_OUT):
            cores = list(ports.values()) + finals if idx != CB_IN2 or on_port else finals
            cbs.append(
                ttnn.CBDescriptor(
                    total_size=2 * group * chunk_bytes,
                    core_ranges=cset(cores),
                    format_descriptors=[ttnn.CBFormatDescriptor(idx, input_tensor.dtype, page_bytes)],
                )
            )
        if land is not None:
            bwd = [c for (j, d, l), c in ports.items() if d == "bwd"]
            cbs.append(
                ttnn.cb_descriptor_from_sharded_tensor(
                    CB_LAND, land, total_size=cap_land * chunk_bytes, core_ranges=cset(bwd)
                )
            )
        program.cbs = cbs
        pr_rt, ps_rt, pc_rt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
        for (j, d, l), core in ports.items():
            rg = ch["rings"][j]
            lane = j * num_links + l
            peer = rg["next"] if d == "fwd" else rg["prev"]
            blocks, lead = _schedule(chips, coord, j, d, ring)
            relays = len(blocks) - lead
            full = _chunks_per_shard(blk_pages, lane, lanes, num_banks, run_pages)
            packed = [b | (s_ << 16) for b, s_ in blocks]
            # on_port: the backward port also sums my own block after its sends (the forward arrival counts on its B,
            # the backward arrival continues its relay stream on A)
            has_fin = on_port and d == "bwd"
            direct = lead == len(blocks) and not has_fin  # every block starts here: the reader feeds the sender
            pr_rt[core.x][core.y] = (
                [in_addr, scr_addr, blk_pages, lane, lanes, sem_a, full, lead, len(blocks)]
                + [b for b, _ in blocks]
                + [int(direct), int(has_fin), own, G, sem_b, int(has_fin and land is not None), cap_land]
            )
            pc_rt[core.x][core.y] = [
                0 if direct else relays * full,
                1,
                0,
                0 if direct else lead * full,
                full if has_fin else 0,
                CB_LAND if land is not None else CB_IN,
            ]
            # what upstream relays to me: my relayed blocks, and with has_fin my own block (else it goes to a final core)
            expect_in = _incs((relays + has_fin) * full, inc_every)
            send_ready = _needs_ready(chips, coord, j, d)
            rc = virt(chips[peer]["ports"][(j, _OPP[d], l)]) if send_ready else None
            fin_sem = (sem_b if d == "fwd" else sem_a) if on_port else (sem_a if d == "fwd" else sem_b)
            args = [scr_addr, blk_pages, lane, lanes, sem_a, fin_sem, expect_in, full, ready_addr]
            args += [1, rc.x, rc.y] if send_ready else [0, 0, 0]
            fin_mode = ((3 if land is not None else 1) if d == "fwd" else 2) if on_port else 0
            # B: a backward port's forward arrival of its own block; a forward port's landing credits (fin_mode 3)
            expect_b = _incs(full, inc_every) if has_fin else (full if fin_mode == 3 and blocks else 0)
            args += [fin_mode, int(has_fin), out_addr, sem_b, expect_b]
            credit_on = has_fin and land is not None
            cc = virt(chips[rg["prev"]]["ports"][(j, "fwd", l)]) if credit_on else None
            args += [land_addr if fin_mode == 3 else 0, cap_land, int(credit_on), cc.x if cc else 0, cc.y if cc else 0]
            if rg.get(f"{d}_links") and blocks:
                pc = virt(chips[peer]["ports"][(j, d, l)])
                if on_port:  # the receiver's backward port sums its own block
                    fc0 = fc1 = virt(chips[peer]["ports"][(j, "bwd", l)])
                    nf0, nf1 = full, 0
                else:
                    fc0 = virt(chips[peer]["finals"][2 * lane])
                    fc1 = virt(chips[peer]["finals"][2 * lane + 1])
                    nf0 = _chunks_per_shard(blk_pages, lane, 2 * lanes, num_banks, run_pages)
                    nf1 = _chunks_per_shard(blk_pages, lane + lanes, 2 * lanes, num_banks, run_pages)
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
        for i, core in enumerate(finals):  # final core 2 lane + h: banks lane + h lanes, stride 2 lanes
            lane, h = divmod(i, 2)
            rg = ch["rings"][lane // num_links]
            # an arrival per direction where a chain ends here: a ring always (G >= 3), a line from each side it has
            has_f = int(
                rg["prev"] is not None and bool(_schedule(chips, rg["prev"], lane // num_links, "fwd", ring)[0])
            )
            has_b = int(
                rg["next"] is not None and bool(_schedule(chips, rg["next"], lane // num_links, "bwd", ring)[0])
            )
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
            fc_rt[core.x][core.y] = [n, has_f, has_b, 0, 0, CB_IN]
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
                compile_time_args=ct + scr_ct + out_ct,
                runtime_args=ps_rt,
                config=dm("RISCV_0", NOC1),
            ),
        ]
        if finals:
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


_SEM_CACHE, _SCRATCH_CACHE, _PLAN_CACHE, _LAND_CACHE = {}, {}, {}, {}


def _with_final_cores(mesh_device, chips, num_links, on_port=False):
    """Each chip's plan plus "finals": two final cores per lane (ring j, link l), [lane 0 half 0, lane 0 half 1, ...];
    none with on_port (the backward ports sum the own block).
    A lane's half 0 is fabric_all_gather's copy core of its link (first ring) or the next free core of that row; half 1
    the next free core in that row (else any free core)."""
    allowed = sorted(allowed_cores(mesh_device), key=lambda c: (c[1], c[0]))
    out = {}
    for coord, ch in chips.items():
        if on_port:
            out[coord] = {**ch, "finals": []}
            continue
        taken = {(c.x, c.y) for c in list(ch["ports"].values()) + ch["copy"]}

        def free(row):
            got = next((c for c in allowed if c[1] == row and c not in taken), None) or next(
                (c for c in allowed if c not in taken), None
            )
            if got is None:
                raise ValueError("fabric_reduce_scatter: the core grid has too few cores for two final cores per lane")
            taken.add(got)
            return ttnn.CoreCoord(*got)

        finals = []
        for j in range(ch["n_rings"]):
            for l in range(num_links):
                row = ch["copy"][l].y
                finals += [ch["copy"][l] if j == 0 else free(row), free(row)]
        out[coord] = {**ch, "finals": finals}
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
    final="cores",
    land_kib=256,
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
    final "cores": two extra cores per link sum each chip's own block; "bwd_port": on an even ring (G >= 4) the
        backward port does it after its sends (it sends one block fewer than the forward port), no extra cores; any
        other group falls back to "cores". "bwd_port_l1": as "bwd_port", and the forward arrival of the own block
        lands in the backward port's L1 (land_kib per port, a ring of slots behind credits) instead of DRAM."""
    mesh_device = input_tensor.device()
    assert final in ("cores", "bwd_port", "bwd_port_l1"), final
    key = (id(mesh_device), cluster_axis, topology, num_links, scheme, placement, final)
    if key not in _PLAN_CACHE:
        chips, ring = plan(
            mesh_device,
            cluster_axis=cluster_axis,
            topology=topology,
            num_links=num_links,
            placement=placement,
            scheme=scheme,
        )
        G = next(iter(chips.values()))["G"]
        on_port = final in ("bwd_port", "bwd_port_l1") and bool(ring) and G % 2 == 0 and G >= 4
        _PLAN_CACHE[key] = (mesh_device, _with_final_cores(mesh_device, chips, num_links, on_port), ring, on_port)
    ring, on_port = _PLAN_CACHE[key][2], _PLAN_CACHE[key][3]
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
    land = None
    if on_port and final == "bwd_port_l1":
        bwd = sorted({(c.x, c.y) for ch in chips.values() for (j, d, l), c in ch["ports"].items() if d == "bwd"})
        tiles = land_kib * 1024 // int(input_tensor.buffer_aligned_page_size())
        lkey = (id(mesh_device), tuple(bwd), tiles, input_tensor.dtype)
        if lkey not in _LAND_CACHE:
            grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(x, y), ttnn.CoreCoord(x, y)) for x, y in bwd])
            spec = ttnn.ShardSpec(grid, [tiles * 32, 32], ttnn.ShardOrientation.ROW_MAJOR)
            mc = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, spec)
            t = ttnn.allocate_tensor_on_device(
                ttnn.Shape([1, 1, len(bwd) * tiles * 32, 32]), input_tensor.dtype, ttnn.TILE_LAYOUT, mesh_device, mc
            )
            _LAND_CACHE[lkey] = (mesh_device, t)
        land = _LAND_CACHE[lkey][1]
    desc = create_mesh_program_descriptor(
        mesh_device,
        input_tensor,
        scratch,
        output,
        sems,
        chips,
        num_links=num_links,
        chips_ring=ring,
        on_port=on_port,
        land=land,
    )
    return ttnn.generic_op([input_tensor, scratch, output], desc)
