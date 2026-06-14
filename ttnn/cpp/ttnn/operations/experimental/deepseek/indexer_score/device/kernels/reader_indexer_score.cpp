// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Reader for indexer_score: walks this core's flat span of work units
// (q_tiles_per_unit q-tile-rows x up-to-k_tiles_per_unit k-tiles). On a new
// q-row-group pushes the resident w group and (when all heads fit) the q group;
// per unit pushes the k chunk. Builds the [diag, full] -inf mask tiles once.
//
// Grid-aligned multicast (INDEXER_DATAMOVEMENT.md): when the dense deal lands on
// the physical grid, cores in a grid ROW share q/w and cores in a grid COLUMN
// share the k-band. One core per row/column reads from DRAM and multicasts to its
// peers (role 1 = sender), the rest receive L1->L1 (role 2), so each input is read
// from DRAM once per line instead of once per core. role 0 = no mcast (DRAM read).
// Q/W (row) and K (column) are independent; either may be off (then role 0).

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/noc_semaphore.h"
#include "api/dataflow/endpoints.h"
#include "api/dataflow/circular_buffer.h"
#include "api/core_local_mem.h"
#include "ttnn/operations/transformer/sdpa/device/kernels/dataflow/dataflow_common.hpp"

#include "indexer_score_common.hpp"

constexpr uint32_t cb_q = tt::CBIndex::c_0;
constexpr uint32_t cb_k = tt::CBIndex::c_1;
constexpr uint32_t cb_w = tt::CBIndex::c_2;
constexpr uint32_t cb_mask = tt::CBIndex::c_3;

constexpr uint32_t tile_bytes = get_tile_size(cb_q);    // q/w: bf16
constexpr uint32_t k_tile_bytes = get_tile_size(cb_k);  // k: bf16 or bfp8_b (smaller tile)

// Compile-time arg layout after the 8 common args: q/k/w TensorAccessors, reader_dma_off, then the
// 8 multicast args. Hoisted to file scope so the semaphore ids are usable as template parameters.
constexpr auto q_args = TensorAccessorArgs<num_common_ct_args>();
constexpr auto k_args = TensorAccessorArgs<q_args.next_compile_time_args_offset()>();
constexpr auto w_args = TensorAccessorArgs<k_args.next_compile_time_args_offset()>();
constexpr uint32_t dma_off_ct = w_args.next_compile_time_args_offset();  // reader_dma_off slot

// ---- multicast compile-time config (semaphore ids; on/off per direction) ------------------
constexpr uint32_t mc_ct_base = dma_off_ct + 1;
constexpr uint32_t k_mcast_on = get_compile_time_arg_val(mc_ct_base + 0);
constexpr uint32_t q_mcast_on = get_compile_time_arg_val(mc_ct_base + 1);
constexpr uint32_t k_send_sem = get_compile_time_arg_val(mc_ct_base + 2);
constexpr uint32_t k_recv_sem = get_compile_time_arg_val(mc_ct_base + 3);
constexpr uint32_t k_valid_sem = get_compile_time_arg_val(mc_ct_base + 4);
constexpr uint32_t q_send_sem = get_compile_time_arg_val(mc_ct_base + 5);
constexpr uint32_t q_recv_sem = get_compile_time_arg_val(mc_ct_base + 6);
constexpr uint32_t q_valid_sem = get_compile_time_arg_val(mc_ct_base + 7);
// q and w share the row mcast but handshake independently, so each can be forwarded on its own
// (diagnostic: INDEXER_NO_QFWD / INDEXER_NO_WFWD force just that input onto the per-core DRAM path).
constexpr uint32_t q_fwd_on = get_compile_time_arg_val(mc_ct_base + 8);
constexpr uint32_t w_fwd_on = get_compile_time_arg_val(mc_ct_base + 9);

// Receiver rectangle / sender coords for one mcast direction (physical NoC), set per core on host.
struct McastDir {
    uint32_t role;            // 0 none (DRAM read), 1 sender (read + mcast), 2 receiver (wait for mcast)
    uint32_t xs, ys, xe, ye;  // receiver rectangle (sender excluded by default mcast opts)
    uint32_t sx, sy;          // sender physical coord (receivers signal it ready)
    uint32_t ndst;            // number of receivers
};

/** Sender: data already sits at `addr` (just read from DRAM); wait for all receivers to be ready,
 *  multicast the block, then relay the valid flag into their recv semaphore. Mirrors chain_link. */
template <uint32_t send_sem, uint32_t recv_sem, uint32_t valid_sem>
inline void mcast_send(Noc noc, const McastDir& d, uint32_t addr, uint32_t bytes) {
    Semaphore<> s(send_sem);
    s.wait(d.ndst);  // every receiver reserved its slot and signaled ready
    s.set(0);
    noc.async_write_multicast(
        CoreLocalMem<uint32_t>(addr),
        MulticastEndpoint{},
        bytes,
        d.ndst,
        {},
        {.noc_x_start = d.xs, .noc_y_start = d.ys, .noc_x_end = d.xe, .noc_y_end = d.ye, .addr = addr},
        /*linked=*/true);
    // back-to-back after the linked data write (a flush/barrier between them would deadlock it)
    Semaphore<>(valid_sem).relay_multicast(
        noc, Semaphore<>(recv_sem), d.xs, d.ys, d.xe, d.ye, d.ndst, /*linked=*/false);
    noc.async_writes_flushed();
}

/** Receiver: slot already reserved at `addr`; set recv=INVALID, signal sender ready, wait VALID. */
template <uint32_t send_sem, uint32_t recv_sem>
inline void mcast_recv(Noc noc, const McastDir& d) {
    Semaphore<> r(recv_sem);
    r.set(0);
    Semaphore<>(send_sem).up(noc, d.sx, d.sy, 1);
    r.wait(1);
}

inline void build_mask_tiles(Noc noc) {
    CircularBuffer cb(cb_mask);
    cb.reserve_back(2);
    fill_causal_diagonal_tile_bf16<tile_bytes>(noc, cb_mask, /*tile_id=*/0);
    fill_neginf_tile<tile_bytes>(cb_mask, /*tile_id=*/1);
    cb.push_back(2);
}

/** q head-group block [q_tiles_per_unit][heads_per_group][head_dim_tiles], page id
 *  h*q_len_tiles*head_dim_tiles + (q_row_start+r)*head_dim_tiles + d. role-aware (q row mcast).
 *  Read as ONE block / ONE mcast handshake: the startup (unit 0, nothing to overlap) is bound by the
 *  COUNT of mcast rendezvous on the critical path (each ~a few us of cross-core sync), so the whole
 *  block in one handshake beats per-row streaming (which multiplies the rendezvous count). */
template <bool dma_off, typename QAcc>
inline void read_q_block(Noc noc, const QAcc& q_acc, uint32_t q_row_start, uint32_t first_head, const McastDir& qd) {
    CircularBuffer cb(cb_q);
    cb.reserve_back(q_group_tiles);
    const uint32_t addr = cb.get_write_ptr();
    if constexpr (q_fwd_on) {
        if (qd.role == 2) {  // receiver: wait for the row sender's mcast into this slot
            mcast_recv<q_send_sem, q_recv_sem>(noc, qd);
            cb.push_back(q_group_tiles);
            return;
        }
    }
    if constexpr (!dma_off) {
        uint32_t ptr = addr;
        for (uint32_t r = 0; r < q_tiles_per_unit; ++r) {
            for (uint32_t h = first_head; h < first_head + heads_per_group; ++h) {
                const uint32_t base = h * q_len_tiles * head_dim_tiles + (q_row_start + r) * head_dim_tiles;
                for (uint32_t d = 0; d < head_dim_tiles; ++d) {
                    noc.async_read(q_acc, CoreLocalMem<uint32_t>(ptr), tile_bytes, {.page_id = base + d}, {});
                    ptr += tile_bytes;
                }
            }
        }
        noc.async_read_barrier();
    }
    if constexpr (q_fwd_on) {
        if (qd.role == 1) {  // sender: broadcast the whole block to the rest of the grid row
            mcast_send<q_send_sem, q_recv_sem, q_valid_sem>(noc, qd, addr, q_group_tiles * tile_bytes);
        }
    }
    cb.push_back(q_group_tiles);
}

/** Resident-heads q, read/mcast ONE q-row at a time (q_tiles_per_unit pushes of heads_per_group*Dt
 *  tiles each) instead of the whole block in one shot. The first matmul needs only row 0's tiles, so
 *  pushing row 0 the moment it lands lets compute start its row-0 matmuls while row 1 is still draining
 *  from DRAM. Costs one mcast rendezvous PER ROW (QC total) vs the whole-block's one -- QC=2 -> 2, still
 *  near the 3-rendezvous startup optimum (per-tile streaming's 19 was the regression). For QC==1 this is
 *  byte-identical to read_q_block (one row == the whole block). first_head is always 0 here (resident). */
template <bool dma_off, typename QAcc>
inline void read_q_rows(Noc noc, const QAcc& q_acc, uint32_t q_row_start, const McastDir& qd) {
    constexpr uint32_t row_tiles = heads_per_group * head_dim_tiles;  // one q-row across all resident heads
    CircularBuffer cb(cb_q);
    cb.reserve_back(q_group_tiles);
    const uint32_t base = cb.get_write_ptr();
    for (uint32_t r = 0; r < q_tiles_per_unit; ++r) {
        const uint32_t row_addr = base + r * row_tiles * tile_bytes;
        if constexpr (q_fwd_on) {
            if (qd.role == 2) {  // receiver: one mcast handshake per row, data lands at row_addr
                mcast_recv<q_send_sem, q_recv_sem>(noc, qd);
                cb.push_back(row_tiles);
                continue;
            }
        }
        if constexpr (!dma_off) {
            uint32_t ptr = row_addr;
            for (uint32_t h = 0; h < heads_per_group; ++h) {
                const uint32_t pbase = h * q_len_tiles * head_dim_tiles + (q_row_start + r) * head_dim_tiles;
                for (uint32_t d = 0; d < head_dim_tiles; ++d) {
                    noc.async_read(q_acc, CoreLocalMem<uint32_t>(ptr), tile_bytes, {.page_id = pbase + d}, {});
                    ptr += tile_bytes;
                }
            }
            noc.async_read_barrier();
        }
        if constexpr (q_fwd_on) {
            if (qd.role == 1) {  // sender: broadcast this row to the rest of the grid row
                mcast_send<q_send_sem, q_recv_sem, q_valid_sem>(noc, qd, row_addr, row_tiles * tile_bytes);
            }
        }
        cb.push_back(row_tiles);
    }
}

/** resident w group [q_tiles_per_unit][num_heads], page id h*q_len_tiles + q_row_start + r. */
template <bool dma_off, typename WAcc>
inline void read_w_group(Noc noc, const WAcc& w_acc, uint32_t q_row_start, const McastDir& qd) {
    CircularBuffer cb(cb_w);
    cb.reserve_back(w_group_tiles);
    const uint32_t addr = cb.get_write_ptr();
    if constexpr (w_fwd_on) {
        if (qd.role == 2) {
            mcast_recv<q_send_sem, q_recv_sem>(noc, qd);
            cb.push_back(w_group_tiles);
            return;
        }
    }
    if constexpr (!dma_off) {
        uint32_t ptr = addr;
        for (uint32_t r = 0; r < q_tiles_per_unit; ++r) {
            for (uint32_t h = 0; h < num_heads; ++h) {
                noc.async_read(
                    w_acc, CoreLocalMem<uint32_t>(ptr), tile_bytes, {.page_id = h * q_len_tiles + q_row_start + r}, {});
                ptr += tile_bytes;
            }
        }
        noc.async_read_barrier();
    }
    if constexpr (w_fwd_on) {
        if (qd.role == 1) {
            mcast_send<q_send_sem, q_recv_sem, q_valid_sem>(noc, qd, addr, w_group_tiles * tile_bytes);
        }
    }
    cb.push_back(w_group_tiles);
}

/** k chunk [k_tiles_in_unit][head_dim_tiles], page id (k_tile_start+c)*head_dim_tiles + d. Always
 *  reserves/pushes the full k_chunk_tiles so the 2-chunk ring stays half-aligned. role-aware (k col
 *  mcast). Read as ONE chunk / ONE mcast handshake (see read_q_block: minimize startup rendezvous). */
template <bool dma_off, typename KAcc>
inline void read_k_chunk(
    Noc noc, const KAcc& k_acc, uint32_t k_tile_start, uint32_t k_tiles_in_unit, const McastDir& kd) {
    CircularBuffer cb(cb_k);
    cb.reserve_back(k_chunk_tiles);
    const uint32_t addr = cb.get_write_ptr();
    if constexpr (k_mcast_on) {
        if (kd.role == 2) {  // receiver: wait for the column sender's mcast into this slot
            mcast_recv<k_send_sem, k_recv_sem>(noc, kd);
            cb.push_back(k_chunk_tiles);
            return;
        }
    }
    if constexpr (!dma_off) {
        uint32_t ptr = addr;
        for (uint32_t c = 0; c < k_tiles_in_unit; ++c) {
            for (uint32_t d = 0; d < head_dim_tiles; ++d) {
                noc.async_read(
                    k_acc,
                    CoreLocalMem<uint32_t>(ptr),
                    k_tile_bytes,
                    {.page_id = (k_tile_start + c) * head_dim_tiles + d},
                    {});
                ptr += k_tile_bytes;
            }
        }
        noc.async_read_barrier();
    }
    if constexpr (k_mcast_on) {
        if (kd.role == 1) {  // sender: broadcast the full chunk down the column
            mcast_send<k_send_sem, k_recv_sem, k_valid_sem>(noc, kd, addr, k_chunk_tiles * k_tile_bytes);
        }
    }
    cb.push_back(k_chunk_tiles);
}

void kernel_main() {
    const uint32_t q_addr = get_arg_val<uint32_t>(0);
    const uint32_t k_addr = get_arg_val<uint32_t>(1);
    const uint32_t w_addr = get_arg_val<uint32_t>(2);
    const uint32_t flat_start = get_arg_val<uint32_t>(3);
    const uint32_t flat_count = get_arg_val<uint32_t>(4);
    const McastDir kd{
        get_arg_val<uint32_t>(5),
        get_arg_val<uint32_t>(6),
        get_arg_val<uint32_t>(7),
        get_arg_val<uint32_t>(8),
        get_arg_val<uint32_t>(9),
        get_arg_val<uint32_t>(10),
        get_arg_val<uint32_t>(11),
        get_arg_val<uint32_t>(12)};
    const McastDir qd{
        get_arg_val<uint32_t>(13),
        get_arg_val<uint32_t>(14),
        get_arg_val<uint32_t>(15),
        get_arg_val<uint32_t>(16),
        get_arg_val<uint32_t>(17),
        get_arg_val<uint32_t>(18),
        get_arg_val<uint32_t>(19),
        get_arg_val<uint32_t>(20)};

    // DMA-off bitmask (bit0=q, bit1=k, bit2=w) sits right after the three TensorAccessors.
    constexpr uint32_t dma_mask = get_compile_time_arg_val(dma_off_ct);
    constexpr bool q_off = (dma_mask & 0b001u) != 0;
    constexpr bool k_off = (dma_mask & 0b010u) != 0;
    constexpr bool w_off = (dma_mask & 0b100u) != 0;
    const auto q_acc = TensorAccessor(q_args, q_addr, tile_bytes);
    const auto k_acc = TensorAccessor(k_args, k_addr, k_tile_bytes);
    const auto w_acc = TensorAccessor(w_args, w_addr, tile_bytes);

    Noc noc;

    build_mask_tiles(noc);

    WorkUnitSpan span;
    span.start(flat_start);

    // Order per group (resident-heads path): q -> k -> w. The gates w are consumed only in the mul
    // phase (after the unit's matmuls), so they are read LAST -- behind the latency-critical q/k that
    // gate the first matmul -- and compute waits w at the mul phase rather than at unit start. (Finer
    // per-row/per-col streaming was tried and reverted: the unit-0 startup is bound by the COUNT of
    // mcast rendezvous, which streaming multiplies; whole-block q + whole-chunk k = the two minimal.)
    //
    // Head-STREAMING path (heads_per_group < num_heads): q is streamed per output tile and compute's
    // mul consumes those q tiles, so w MUST be available before compute starts draining the streamed q
    // (otherwise compute blocks on w while the reader blocks on the full q CB -> deadlock). So w is
    // read FIRST for that path; the deferral applies only to the resident-heads (production) path.
    bool need_group = true;
    for (uint32_t i = 0; i < flat_count; ++i) {
        const bool group_start = need_group;
        if (group_start && stream_heads) {
            read_w_group<w_off>(noc, w_acc, span.q_tile_start(), qd);  // gates before the streamed q
        }
        // k FIRST on the resident path: the first matmul needs k AND q-row0, and compute waits the whole
        // k chunk before any row. Reading k ahead of q means it is ready when q-row0 lands, so the split
        // q-row0 push actually unblocks the first matmul (otherwise the k wait would re-serialize it).
        read_k_chunk<k_off>(noc, k_acc, span.k_tile_start(), span.k_tiles(), kd);
        if (group_start && !stream_heads) {
            read_q_rows<q_off>(noc, q_acc, span.q_tile_start(), qd);  // per-row: compute starts on row 0
        }
        if constexpr (stream_heads) {
            // one q-block per (r, c) output tile per head group; must match compute's tile order
            for (uint32_t tile_idx = 0; tile_idx < q_tiles_per_unit * span.k_tiles(); ++tile_idx) {
                for (uint32_t first_head = 0; first_head < num_heads; first_head += heads_per_group) {
                    read_q_block<q_off>(noc, q_acc, span.q_tile_start(), first_head, qd);
                }
            }
        }
        if (group_start && !stream_heads) {
            read_w_group<w_off>(noc, w_acc, span.q_tile_start(), qd);  // gates deferred behind q/k
        }
        need_group = span.advance();
    }
}
