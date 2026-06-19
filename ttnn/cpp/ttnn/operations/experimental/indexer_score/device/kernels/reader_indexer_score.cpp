// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Reader for indexer_score (DMA bottleneck). Walks this core's flat span of work units
// (QC q-rows x up-to-KC k-tiles): per group pushes resident w + (if all heads fit) q,
// per unit pushes the k chunk. Builds the [diag, full] -inf mask tiles once.
//
// Grid-aligned multicast: grid ROW shares q/w, grid COLUMN shares the k-band. role 1
// (sender) reads DRAM + mcasts; role 2 (receiver) takes the L1->L1 copy; role 0 plain
// DRAM read. Q/W (row) and K (column) mcast are independent; either may be off (role 0).

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/noc_semaphore.h"
#include "api/dataflow/endpoints.h"
#include "api/dataflow/circular_buffer.h"
#include "api/core_local_mem.h"
#include "ttnn/operations/transformer/sdpa/device/kernels/dataflow/dataflow_common.hpp"

#include "indexer_score_common.hpp"  // shared CB indices, compile-time dims, work-unit walk

constexpr uint32_t q_tile_bytes = get_tile_size(cb_q);     // q: bf16 or bfp8_b (smaller tile)
constexpr uint32_t bf16_tile_bytes = get_tile_size(cb_w);  // w / mask: always bf16
constexpr uint32_t k_tile_bytes = get_tile_size(cb_k);     // k: bf16 or bfp8_b (smaller tile)

// CT arg layout after the common args: q/k/w TensorAccessors, then 8 multicast args.
// File-scope so the semaphore ids work as template parameters.
constexpr auto q_args = TensorAccessorArgs<num_common_ct_args>();
constexpr auto k_args = TensorAccessorArgs<q_args.next_compile_time_args_offset()>();
constexpr auto w_args = TensorAccessorArgs<k_args.next_compile_time_args_offset()>();

// ---- multicast CT config (semaphore ids; on/off per direction) -------------------
constexpr uint32_t mc_ct_base = w_args.next_compile_time_args_offset();
constexpr uint32_t k_mcast_on = get_compile_time_arg_val(mc_ct_base + 0);
constexpr uint32_t q_mcast_on = get_compile_time_arg_val(mc_ct_base + 1);  // covers q and w (shared row mcast)
constexpr uint32_t k_send_sem = get_compile_time_arg_val(mc_ct_base + 2);
constexpr uint32_t k_recv_sem = get_compile_time_arg_val(mc_ct_base + 3);
constexpr uint32_t k_valid_sem = get_compile_time_arg_val(mc_ct_base + 4);
constexpr uint32_t q_send_sem = get_compile_time_arg_val(mc_ct_base + 5);
constexpr uint32_t q_recv_sem = get_compile_time_arg_val(mc_ct_base + 6);
constexpr uint32_t q_valid_sem = get_compile_time_arg_val(mc_ct_base + 7);

// Receiver rectangle / sender coords for one mcast direction (physical NoC), set per core on host.
struct McastDir {
    uint32_t role;            // McastRole: none (DRAM read), sender (read + mcast), receiver (wait for mcast)
    uint32_t xs, ys, xe, ye;  // receiver rectangle (sender excluded by default mcast opts)
    uint32_t sx, sy;          // sender physical coord (receivers signal it ready)
    uint32_t ndst;            // number of receivers
};

/** Unpack a McastDir from runtime args [base, base+8) in the host's push order. */
inline McastDir read_mcast_dir(uint32_t base) {
    return McastDir{
        get_arg_val<uint32_t>(base + 0),
        get_arg_val<uint32_t>(base + 1),
        get_arg_val<uint32_t>(base + 2),
        get_arg_val<uint32_t>(base + 3),
        get_arg_val<uint32_t>(base + 4),
        get_arg_val<uint32_t>(base + 5),
        get_arg_val<uint32_t>(base + 6),
        get_arg_val<uint32_t>(base + 7)};
}

/** Sender: data already at `addr` from DRAM; wait all receivers ready, mcast the block,
 *  then relay the valid flag into their recv semaphore. Mirrors chain_link. */
template <uint32_t send_sem, uint32_t recv_sem, uint32_t valid_sem>
inline void mcast_send(Noc noc, const McastDir& d, uint32_t addr, uint32_t bytes) {
    Semaphore<> s(send_sem);
    s.wait(d.ndst);  // all receivers reserved their slot and signaled ready
    s.set(0);
    noc.async_write_multicast(
        CoreLocalMem<uint32_t>(addr),
        MulticastEndpoint{},
        bytes,
        d.ndst,
        {},
        {.noc_x_start = d.xs, .noc_y_start = d.ys, .noc_x_end = d.xe, .noc_y_end = d.ye, .addr = addr},
        /*linked=*/true);
    // back-to-back after the linked data write (a flush/barrier between would deadlock)
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

/** Role-aware block read shared by the q-block, w-group and k-chunk readers. Reserve `ntiles` of cb_id;
 *  receiver waits the sender's mcast and returns; everyone else runs `read_into(addr)` (DRAM read loop),
 *  barriers, and (sender only) broadcasts `bytes`. read_q_rows has its own per-row variant. */
template <uint32_t cb_id, uint32_t mcast_on, uint32_t send_sem, uint32_t recv_sem, uint32_t valid_sem, typename ReadFn>
inline void read_block_or_mcast(Noc noc, uint32_t ntiles, uint32_t bytes, const McastDir& dir, ReadFn&& read_into) {
    CircularBuffer cb(cb_id);
    cb.reserve_back(ntiles);
    const uint32_t addr = cb.get_write_ptr();
    if constexpr (mcast_on) {
        if (dir.role == iscore::mcast_role_receiver) {  // wait the sender's mcast into this slot
            mcast_recv<send_sem, recv_sem>(noc, dir);
            cb.push_back(ntiles);
            return;
        }
    }
    read_into(addr);
    noc.async_read_barrier();
    if constexpr (mcast_on) {
        if (dir.role == iscore::mcast_role_sender) {  // broadcast the block to the rest of the rect
            mcast_send<send_sem, recv_sem, valid_sem>(noc, dir, addr, bytes);
        }
    }
    cb.push_back(ntiles);
}

inline void build_mask_tiles(Noc noc) {
    CircularBuffer cb(cb_mask);
    cb.reserve_back(num_mask_tiles);
    fill_causal_diagonal_tile_bf16<bf16_tile_bytes>(noc, cb_mask, /*tile_id=*/0);  // diagonal strict-upper -inf
    fill_neginf_tile<bf16_tile_bytes>(cb_mask, /*tile_id=*/1);                     // full -inf
    cb.push_back(num_mask_tiles);
}

/** Read ONE q-row (heads_per_group heads x head_dim_tiles tiles, heads starting at first_head) from
 *  DRAM into L1 at `ptr`; returns the advanced write pointer. Shared inner loop of the resident
 *  (read_q_rows, first_head=0) and head-streaming (read_q_block, varying first_head) paths -- the q
 *  page layout is [Hi][q_len_tiles][head_dim_tiles]. */
template <typename QAcc>
inline uint32_t read_q_row_into(Noc noc, const QAcc& q_acc, uint32_t ptr, uint32_t q_row_abs, uint32_t first_head) {
    for (uint32_t h = first_head; h < first_head + heads_per_group; ++h) {
        const uint32_t base = h * q_len_tiles * head_dim_tiles + q_row_abs * head_dim_tiles;
        for (uint32_t d = 0; d < head_dim_tiles; ++d) {
            noc.async_read(q_acc, CoreLocalMem<uint32_t>(ptr), q_tile_bytes, {.page_id = base + d}, {});
            ptr += q_tile_bytes;
        }
    }
    return ptr;
}

/** q head-group block [QC][heads_per_group][head_dim_tiles], role-aware (q row mcast).
 *  ONE block / ONE mcast handshake: unit-0 startup is bound by the COUNT of mcast rendezvous,
 *  so one handshake beats per-row streaming (which multiplies the count). */
template <typename QAcc>
inline void read_q_block(Noc noc, const QAcc& q_acc, uint32_t q_row_start, uint32_t first_head, const McastDir& q_dir) {
    read_block_or_mcast<cb_q, q_mcast_on, q_send_sem, q_recv_sem, q_valid_sem>(
        noc, q_group_tiles, q_group_tiles * q_tile_bytes, q_dir, [&](uint32_t addr) {
            uint32_t ptr = addr;
            for (uint32_t r = 0; r < q_tiles_per_unit; ++r) {
                ptr = read_q_row_into(noc, q_acc, ptr, q_row_start + r, first_head);
            }
        });
}

/** Resident-heads q, read/mcast ONE q-row at a time (QC pushes) so compute starts row-0 matmuls while
 *  row 1 still drains. Costs one mcast rendezvous PER ROW (QC) vs the block's one, still near the startup
 *  optimum. QC==1 is byte-identical to read_q_block. first_head always 0 (resident). */
template <typename QAcc>
inline void read_q_rows(Noc noc, const QAcc& q_acc, uint32_t q_row_start, const McastDir& q_dir) {
    constexpr uint32_t row_tiles = heads_per_group * head_dim_tiles;  // one q-row across all resident heads
    CircularBuffer cb(cb_q);
    cb.reserve_back(q_group_tiles);
    const uint32_t base = cb.get_write_ptr();
    for (uint32_t r = 0; r < q_tiles_per_unit; ++r) {
        const uint32_t row_addr = base + r * row_tiles * q_tile_bytes;
        if constexpr (q_mcast_on) {
            if (q_dir.role == iscore::mcast_role_receiver) {  // one mcast handshake per row -> row_addr
                mcast_recv<q_send_sem, q_recv_sem>(noc, q_dir);
                cb.push_back(row_tiles);
                continue;
            }
        }
        read_q_row_into(noc, q_acc, row_addr, q_row_start + r, /*first_head=*/0);
        noc.async_read_barrier();
        if constexpr (q_mcast_on) {
            if (q_dir.role == iscore::mcast_role_sender) {  // broadcast this row to the rest of the grid row
                mcast_send<q_send_sem, q_recv_sem, q_valid_sem>(noc, q_dir, row_addr, row_tiles * q_tile_bytes);
            }
        }
        cb.push_back(row_tiles);
    }
}

/** resident w (gates) group [q_tiles_per_unit][num_heads], role-aware (q row mcast). */
template <typename WAcc>
inline void read_w_group(Noc noc, const WAcc& w_acc, uint32_t q_row_start, const McastDir& q_dir) {
    read_block_or_mcast<cb_w, q_mcast_on, q_send_sem, q_recv_sem, q_valid_sem>(
        noc, w_group_tiles, w_group_tiles * bf16_tile_bytes, q_dir, [&](uint32_t addr) {
            uint32_t ptr = addr;
            for (uint32_t r = 0; r < q_tiles_per_unit; ++r) {
                for (uint32_t h = 0; h < num_heads; ++h) {
                    noc.async_read(
                        w_acc,
                        CoreLocalMem<uint32_t>(ptr),
                        bf16_tile_bytes,
                        {.page_id = h * q_len_tiles + q_row_start + r},
                        {});
                    ptr += bf16_tile_bytes;
                }
            }
        });
}

/** k chunk [k_tiles_in_unit][head_dim_tiles], role-aware (k col mcast). ONE chunk / ONE mcast
 *  handshake to minimize startup rendezvous. */
template <typename KAcc>
inline void read_k_chunk(
    Noc noc, const KAcc& k_acc, uint32_t k_tile_start, uint32_t k_tiles_in_unit, const McastDir& k_dir) {
    // Reserves/pushes the full k_chunk_tiles to keep the 2-chunk ring half-aligned, but reads only the
    // k_tiles_in_unit valid columns (pad slots stay stale; compute masks them).
    read_block_or_mcast<cb_k, k_mcast_on, k_send_sem, k_recv_sem, k_valid_sem>(
        noc, k_chunk_tiles, k_chunk_tiles * k_tile_bytes, k_dir, [&](uint32_t addr) {
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
        });
}

void kernel_main() {
    const uint32_t q_addr = get_arg_val<uint32_t>(0);
    const uint32_t k_addr = get_arg_val<uint32_t>(1);
    const uint32_t w_addr = get_arg_val<uint32_t>(2);
    const uint32_t flat_start = get_arg_val<uint32_t>(3);
    const uint32_t flat_count = get_arg_val<uint32_t>(4);
    const McastDir k_dir = read_mcast_dir(5);   // K column mcast: args [5, 13)
    const McastDir q_dir = read_mcast_dir(13);  // Q/W row mcast: args [13, 21)

    const auto q_acc = TensorAccessor(q_args, q_addr, q_tile_bytes);
    const auto k_acc = TensorAccessor(k_args, k_addr, k_tile_bytes);
    const auto w_acc = TensorAccessor(w_args, w_addr, bf16_tile_bytes);

    Noc noc;

    build_mask_tiles(noc);

    WorkUnitSpan span;
    span.start(flat_start);

    // Resident-heads path order: k -> q -> w. w (gates) is consumed only in the mul phase, so read it
    // LAST behind the latency-critical q/k. Streaming path reads w FIRST: compute's mul drains streamed
    // q, so w must be present or compute blocks on w while the reader blocks on the full q CB => deadlock.
    bool need_group = true;
    for (uint32_t i = 0; i < flat_count; ++i) {
        const bool group_start = need_group;
        if (group_start && stream_heads) {
            read_w_group(noc, w_acc, span.q_tile_start(), q_dir);  // gates before the streamed q
        }
        // k FIRST: compute waits the whole k chunk before any row, so reading k ahead of q lets the
        // split q-row0 push unblock the first matmul (else the k wait re-serializes it).
        read_k_chunk(noc, k_acc, span.k_tile_start(), span.k_tiles(), k_dir);
        if (group_start && !stream_heads) {
            read_q_rows(noc, q_acc, span.q_tile_start(), q_dir);  // per-row: compute starts on row 0
        }
        if constexpr (stream_heads) {
            // one q-block per (r, c) output tile per head group; must match compute's tile order, which
            // walks the FULL k_tiles_per_unit columns (compute masks the padded tail of a partial last
            // unit). Using span.k_tiles() here would under-produce q blocks on a partial unit and hang
            // compute, which still waits/pops a q block for every padded column.
            for (uint32_t tile_idx = 0; tile_idx < q_tiles_per_unit * k_tiles_per_unit; ++tile_idx) {
                for (uint32_t first_head = 0; first_head < num_heads; first_head += heads_per_group) {
                    read_q_block(noc, q_acc, span.q_tile_start(), first_head, q_dir);
                }
            }
        }
        if (group_start && !stream_heads) {
            read_w_group(noc, w_acc, span.q_tile_start(), q_dir);  // gates deferred behind q/k
        }
        need_group = span.advance();
    }
}
