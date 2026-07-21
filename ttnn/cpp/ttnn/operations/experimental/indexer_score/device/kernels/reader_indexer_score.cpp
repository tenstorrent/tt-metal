// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Reader for indexer_score (DMA bottleneck). Walks this core's (group-phase x k-band) rectangle: per
// group pushes resident w + (if all heads fit) q, per band pushes the k chunk. Builds the [diag, full]
// -inf mask tiles once, plus a 1.0 reduce-scaler when block-max-pooling. G-agnostic.
//
// Banded-product multicast: a grid ROW shares q/w (q-mcast), a COLUMN shares the k-band (k-mcast). role
// sender reads DRAM + mcasts; receiver takes the L1->L1 copy; none is a plain DRAM read. q/w (row) and k
// (column) mcast are independent; either may be off.

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/noc_semaphore.h"
#include "api/dataflow/endpoints.h"
#include "api/dataflow/circular_buffer.h"
#include "api/core_local_mem.h"
#include "ttnn/operations/transformer/sdpa/device/kernels/dataflow/dataflow_common.hpp"
#include "ttnn/operations/transformer/sdpa/device/kernels/dataflow/block_cyclic_remap.hpp"  // shared invP remap
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"  // block-max-pool: calculate_and_prepare_reduce_scaler

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
// Fused single-head: read q+w first (the matmul gate needs them), then k.
constexpr uint32_t fuse_single = get_compile_time_arg_val(mc_ct_base + 8);
// Fused + no mcast: STREAM k in column sub-chunks (overlap the DRAM read). With mcast, read whole.
constexpr uint32_t fused_stream_k = get_compile_time_arg_val(mc_ct_base + 9);
// MSA constant gate: fill cb_w with gate_scale in L1 (no DRAM read, no mcast) instead of reading weights.
constexpr uint32_t synthesize_gate = get_compile_time_arg_val(mc_ct_base + 10);
constexpr uint32_t gate_scale_bits = get_compile_time_arg_val(mc_ct_base + 11);  // bf16 pair (two per word)
constexpr uint32_t bc_ct_base = mc_ct_base + 12;
constexpr bool block_cyclic = get_compile_time_arg_val(bc_ct_base) != 0;
constexpr uint32_t bc_chunk_local = get_compile_time_arg_val(bc_ct_base + 1);
constexpr uint32_t bc_sp = get_compile_time_arg_val(bc_ct_base + 2);
constexpr uint32_t bc_shard_stride_gap = get_compile_time_arg_val(bc_ct_base + 3);
constexpr uint32_t bc_slab_stride_gap = get_compile_time_arg_val(bc_ct_base + 4);

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

/** Read ONE q-row (heads_per_group heads x head_dim_tiles tiles from first_head) into L1 at `ptr`;
 *  returns the advanced ptr. Shared by resident (read_q_rows) and streaming (read_q_block) paths.
 *  q page layout is [Hi][q_len_tiles][head_dim_tiles]. */
template <typename QAcc>
inline uint32_t read_q_row_into(Noc noc, const QAcc& q_acc, uint32_t ptr, uint32_t q_row_abs, uint32_t first_head) {
    for (uint32_t head = first_head; head < first_head + heads_per_group; ++head) {
        const uint32_t base = head * q_len_tiles * head_dim_tiles + q_row_abs * head_dim_tiles;
        for (uint32_t dim_tile = 0; dim_tile < head_dim_tiles; ++dim_tile) {
            noc.async_read(q_acc, CoreLocalMem<uint32_t>(ptr), q_tile_bytes, {.page_id = base + dim_tile}, {});
            ptr += q_tile_bytes;
        }
    }
    return ptr;
}

/** q head-group block [QC][heads_per_group][head_dim_tiles], role-aware (q row mcast). ONE block / ONE
 *  mcast handshake (unit-0 startup is bound by the rendezvous count). */
template <typename QAcc>
inline void read_q_block(Noc noc, const QAcc& q_acc, uint32_t q_row_start, uint32_t first_head, const McastDir& q_dir) {
    read_block_or_mcast<cb_q, q_mcast_on, q_send_sem, q_recv_sem, q_valid_sem>(
        noc, q_group_tiles, q_group_tiles * q_tile_bytes, q_dir, [&](uint32_t addr) {
            uint32_t ptr = addr;
            for (uint32_t q_row = 0; q_row < q_tiles_per_unit; ++q_row) {
                ptr = read_q_row_into(noc, q_acc, ptr, q_row_start + q_row, first_head);
            }
        });
}

/** Resident-heads q, read/mcast ONE q-row at a time (QC pushes) so compute starts row-0 matmuls while
 *  row 1 drains. One mcast rendezvous per row. QC==1 is byte-identical to read_q_block. */
template <typename QAcc>
inline void read_q_rows(Noc noc, const QAcc& q_acc, uint32_t q_row_start, const McastDir& q_dir) {
    constexpr uint32_t row_tiles = heads_per_group * head_dim_tiles;  // one q-row across all resident heads
    CircularBuffer cb(cb_q);
    cb.reserve_back(q_group_tiles);
    const uint32_t base = cb.get_write_ptr();
    for (uint32_t q_row = 0; q_row < q_tiles_per_unit; ++q_row) {
        const uint32_t row_addr = base + q_row * row_tiles * q_tile_bytes;
        if constexpr (q_mcast_on) {
            if (q_dir.role == iscore::mcast_role_receiver) {  // one handshake per row -> row_addr
                mcast_recv<q_send_sem, q_recv_sem>(noc, q_dir);
                cb.push_back(row_tiles);
                continue;
            }
        }
        read_q_row_into(noc, q_acc, row_addr, q_row_start + q_row, /*first_head=*/0);
        noc.async_read_barrier();
        if constexpr (q_mcast_on) {
            if (q_dir.role == iscore::mcast_role_sender) {  // broadcast this row down the grid row
                mcast_send<q_send_sem, q_recv_sem, q_valid_sem>(noc, q_dir, row_addr, row_tiles * q_tile_bytes);
            }
        }
        cb.push_back(row_tiles);
    }
}

/** MSA constant gate: fill the resident w group with gate_scale in L1 (no DRAM read, no mcast). Every core
 *  fills its own cb_w -- the gate is the same scalar for every (head, query). Mirrors the mask/scaler fills. */
inline void fill_w_group_const() {
    CircularBuffer cb(cb_w);
    cb.reserve_back(w_group_tiles);
    volatile tt_l1_ptr uint32_t* ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cb.get_write_ptr());
    constexpr uint32_t total_words = w_group_tiles * (bf16_tile_bytes / sizeof(uint32_t));
    for (uint32_t i = 0; i < total_words; ++i) {
        ptr[i] = gate_scale_bits;
    }
    cb.push_back(w_group_tiles);
}

/** resident w (gates) group [q_tiles_per_unit][num_heads], role-aware (q row mcast). MSA fills a constant
 *  scale in L1 instead (no weights tensor); the q placeholder accessor is then unused. */
template <typename WAcc>
inline void read_w_group(Noc noc, const WAcc& w_acc, uint32_t q_row_start, const McastDir& q_dir) {
    if constexpr (synthesize_gate) {
        fill_w_group_const();
        return;
    }
    read_block_or_mcast<cb_w, q_mcast_on, q_send_sem, q_recv_sem, q_valid_sem>(
        noc, w_group_tiles, w_group_tiles * bf16_tile_bytes, q_dir, [&](uint32_t addr) {
            uint32_t ptr = addr;
            for (uint32_t q_row = 0; q_row < q_tiles_per_unit; ++q_row) {
                for (uint32_t head = 0; head < num_heads; ++head) {
                    noc.async_read(
                        w_acc,
                        CoreLocalMem<uint32_t>(ptr),
                        bf16_tile_bytes,
                        {.page_id = head * q_len_tiles + q_row_start + q_row},
                        {});
                    ptr += bf16_tile_bytes;
                }
            }
        });
}

/** k chunk [k_tiles_in_unit][head_dim_tiles], role-aware (k col mcast). ONE chunk / ONE mcast handshake. */
template <typename KAcc>
inline void read_k_chunk(
    Noc noc,
    const KAcc& k_acc,
    uint32_t k_tile_start,
    uint32_t k_tiles_in_unit,
    const McastDir& k_dir,
    uint32_t k_batch_page_offset) {
    // Reserves/pushes the full k_chunk_tiles (keeps the 2-chunk ring half-aligned) but reads only the
    // k_tiles_in_unit valid cols (pad slots stale, compute masks them). k_batch_page_offset = indexed-cache
    // slot shift; 0 when not indexed.
    read_block_or_mcast<cb_k, k_mcast_on, k_send_sem, k_recv_sem, k_valid_sem>(
        noc, k_chunk_tiles, k_chunk_tiles * k_tile_bytes, k_dir, [&](uint32_t addr) {
            uint32_t ptr = addr;
            for (uint32_t k_col = 0; k_col < k_tiles_in_unit; ++k_col) {
                const uint32_t seq_tile = tt::block_cyclic::logical_to_physical_page<
                    block_cyclic,
                    bc_chunk_local,
                    bc_sp,
                    bc_shard_stride_gap,
                    bc_slab_stride_gap>(k_tile_start + k_col);
                for (uint32_t dim_tile = 0; dim_tile < head_dim_tiles; ++dim_tile) {
                    noc.async_read(
                        k_acc,
                        CoreLocalMem<uint32_t>(ptr),
                        k_tile_bytes,
                        {.page_id = k_batch_page_offset + seq_tile * head_dim_tiles + dim_tile},
                        {});
                    ptr += k_tile_bytes;
                }
            }
        });
}

/** Fused path: read the k chunk in mm_col_batch sub-chunks, pushing each as it lands so compute matmuls
 *  it while the next reads (overlap). Pushes the full k_chunk_tiles (pad cols stale, compute masks them).
 *  No mcast. mm_col_batch is shared with the compute kernel's DEST column batch (indexer_score_common.hpp). */
template <typename KAcc>
inline void read_k_chunk_streaming(
    Noc noc, const KAcc& k_acc, uint32_t k_tile_start, uint32_t k_tiles_in_unit, uint32_t k_batch_page_offset) {
    // k_batch_page_offset = indexed-cache slot shift (0 when not indexed), applied on top of the (possibly
    // block-cyclic remapped) intra-slot page -- identical to read_k_chunk; the fused-streaming path must not
    // drop it, or an indexed slot would silently read slot 0.
    CircularBuffer cb(cb_k);
    for (uint32_t cbase = 0; cbase < k_tiles_per_unit; cbase += mm_col_batch) {
        const uint32_t c_end = (cbase + mm_col_batch <= k_tiles_per_unit) ? (cbase + mm_col_batch) : k_tiles_per_unit;
        const uint32_t sub_tiles = (c_end - cbase) * head_dim_tiles;
        cb.reserve_back(sub_tiles);
        uint32_t ptr = cb.get_write_ptr();
        for (uint32_t c = cbase; c < c_end; ++c) {
            if (c < k_tiles_in_unit) {
                const uint32_t seq_tile = tt::block_cyclic::logical_to_physical_page<
                    block_cyclic,
                    bc_chunk_local,
                    bc_sp,
                    bc_shard_stride_gap,
                    bc_slab_stride_gap>(k_tile_start + c);
                for (uint32_t d = 0; d < head_dim_tiles; ++d) {
                    noc.async_read(
                        k_acc,
                        CoreLocalMem<uint32_t>(ptr),
                        k_tile_bytes,
                        {.page_id = k_batch_page_offset + seq_tile * head_dim_tiles + d},
                        {});
                    ptr += k_tile_bytes;
                }
            }
        }
        noc.async_read_barrier();
        cb.push_back(sub_tiles);
    }
}

void kernel_main() {
    const uint32_t q_addr = get_arg_val<uint32_t>(0);
    const uint32_t k_addr = get_arg_val<uint32_t>(1);
    const uint32_t w_addr = get_arg_val<uint32_t>(2);
    // Banded schedule: groups -> grid rows (q/w shared = q_dir mcast), k-bands -> columns (k shared = k_dir mcast).
    const uint32_t row_group0 = get_arg_val<uint32_t>(3);
    const uint32_t group_stride = get_arg_val<uint32_t>(4);
    const uint32_t num_groups = get_arg_val<uint32_t>(5);
    const uint32_t band0 = get_arg_val<uint32_t>(6);
    const uint32_t num_bands = get_arg_val<uint32_t>(7);
    const uint32_t max_bands = get_arg_val<uint32_t>(8);  // row's widest column; streaming pads q to this
    const McastDir k_dir = read_mcast_dir(9);             // K column mcast: args [9, 17)
    const McastDir q_dir = read_mcast_dir(17);            // Q/W row mcast: args [17, 25)
    // Persistent-cache args (hash-excluded, re-applied each dispatch), after the mcast tuples.
    const uint32_t k_batch_page_offset = get_arg_val<uint32_t>(25);  // indexed-cache page offset; 0 when not indexed
    const uint32_t kv_len_tiles = get_arg_val<uint32_t>(26);         // valid KV length in tiles (full when unset)

    const auto q_acc = TensorAccessor(q_args, q_addr, q_tile_bytes);
    const auto k_acc = TensorAccessor(k_args, k_addr, k_tile_bytes);
    const auto w_acc = TensorAccessor(w_args, w_addr, bf16_tile_bytes);

    Noc noc;

    build_mask_tiles(noc);
    if constexpr (block_pool) {
        // 1.0 reduce-MAX scaler for the block-max-pool (row-0 fill, the layout reduce_block_max_row expects).
        dataflow_kernel_lib::
            calculate_and_prepare_reduce_scaler<cb_scaler, ckernel::PoolType::MAX, ckernel::ReduceDim::REDUCE_ROW>();
    }

    WorkUnitSpan span;
    span.set_valid_k_len_tiles(kv_len_tiles);

    // group-OUTER, band-INNER. Read order within a group: resident reads k -> q -> w (gates last, behind
    // latency-critical q/k); streaming reads w FIRST (else compute's mul drains streamed q with no w =>
    // deadlock); fused reads q+w FIRST (gates q before the matmul), then k (streamed when no mcast).
    // Streaming pads the band loop to max_bands so every core in a row issues the same q reads (q-mcast
    // lockstep): a phantom band [num_bands, max_bands) re-issues only the band-independent q (no k/output).
    // Resident reads q once per group, so it never pads.
    const uint32_t band_iters = stream_heads ? max_bands : num_bands;
    for (uint32_t phase = 0; phase < num_groups; ++phase) {
        const uint32_t group = row_group0 + phase * group_stride;
        const uint32_t q_row_start = group * q_tiles_per_unit;
        if constexpr (fuse_single) {
            // Fused: q+w FIRST (the matmul gate needs them), once per group.
            read_q_rows(noc, q_acc, q_row_start, q_dir);
            read_w_group(noc, w_acc, q_row_start, q_dir);
        }
        if constexpr (stream_heads) {
            read_w_group(noc, w_acc, q_row_start, q_dir);  // gates before the streamed q (once per group)
        }
        for (uint32_t band = 0; band < band_iters; ++band) {
            const bool real_band = band < num_bands;  // phantom bands (streaming pad) carry q-mcast only
            if (real_band) {
                span.set(group, band0 + band);
                if constexpr (fuse_single && fused_stream_k) {
                    read_k_chunk_streaming(
                        noc, k_acc, span.k_tile_start(), span.k_tiles(), k_batch_page_offset);  // no mcast: stream
                } else {
                    // k FIRST: compute waits the whole k chunk, so reading k ahead lets the split q-row0 push
                    // unblock the first matmul.
                    read_k_chunk(noc, k_acc, span.k_tile_start(), span.k_tiles(), k_dir, k_batch_page_offset);
                }
                if (band == 0 && !stream_heads && !fuse_single) {
                    // Non-fused resident: q/w deferred behind q/k here.
                    read_q_rows(noc, q_acc, q_row_start, q_dir);
                    read_w_group(noc, w_acc, q_row_start, q_dir);
                }
            }
            if constexpr (stream_heads) {
                // one q-block per output tile per head group, matching compute's order over the FULL
                // k_tiles_per_unit cols (compute masks a partial last band's tail). span.k_tiles() here would
                // under-produce q and hang compute. q is band-independent -> the phantom band re-issues this.
                for (uint32_t tile_idx = 0; tile_idx < q_tiles_per_unit * k_tiles_per_unit; ++tile_idx) {
                    for (uint32_t first_head = 0; first_head < num_heads; first_head += heads_per_group) {
                        read_q_block(noc, q_acc, q_row_start, first_head, q_dir);
                    }
                }
            }
        }
    }
}
