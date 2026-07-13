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
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"  // block-max-pool: calculate_and_prepare_reduce_scaler

#include "indexer_score_common.hpp"  // shared CB indices, compile-time dims, work-unit walk

#ifdef FUSED_RING
// Ring-fused indexer: the fused all-gather producer signals per-slab arrival into two direction semaphores;
// the reader gates each band on ONLY the SP-shards that band's tiles land in (fine-grained overlap, see the
// per-band gate in kernel_main) so scoring of already-arrived shards runs while farther slabs are in flight.
// Reuses the ring-joint-SDPA receiver so the crossed direction-index swap + asymmetric thresholds stay identical.
#include "ttnn/operations/transformer/sdpa/device/kernels/dataflow/fused_op_receiver.hpp"
#endif

constexpr uint32_t q_tile_bytes = get_tile_size(cb_q);     // q: bf16 or bfp8_b (smaller tile)
constexpr uint32_t bf16_tile_bytes = get_tile_size(cb_w);  // w / mask: always bf16
constexpr uint32_t k_tile_bytes = get_tile_size(cb_k);     // k: bf16 or bfp8_b (smaller tile)

// CT arg layout after the common args: q/k/w TensorAccessors, then (fused only) the k_local accessor, then
// 8 multicast args. File-scope so the semaphore ids work as template parameters.
constexpr auto q_args = TensorAccessorArgs<num_common_ct_args>();
constexpr auto k_args = TensorAccessorArgs<q_args.next_compile_time_args_offset()>();
constexpr auto w_args = TensorAccessorArgs<k_args.next_compile_time_args_offset()>();
#ifdef FUSED_RING
// Fused ring: the SP-sharded LOCAL K shard (the all-gather input) is read directly for this device's own slab
// (the all-gather does NOT write the local band into the gathered buffer). Its accessor args sit after w's.
constexpr auto kl_args = TensorAccessorArgs<w_args.next_compile_time_args_offset()>();
constexpr uint32_t mc_ct_base = kl_args.next_compile_time_args_offset();
#else
constexpr uint32_t mc_ct_base = w_args.next_compile_time_args_offset();
#endif
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

template <uint32_t ChunkLocal, uint32_t Sp, uint32_t ShardStrideGap, uint32_t SlabStrideGap>
FORCE_INLINE uint32_t logical_to_chunked_physical(uint32_t n) {
    const uint32_t block_idx = n / ChunkLocal;
    const uint32_t slab = block_idx / Sp;
    const uint32_t shard = block_idx - slab * Sp;
    return n + shard * ShardStrideGap - slab * SlabStrideGap;
}

template <bool BlockCyclic, uint32_t ChunkLocal, uint32_t Sp, uint32_t ShardStrideGap, uint32_t SlabStrideGap>
FORCE_INLINE uint32_t logical_to_physical_page(uint32_t page) {
    if constexpr (BlockCyclic) {
        return logical_to_chunked_physical<ChunkLocal, Sp, ShardStrideGap, SlabStrideGap>(page);
    } else {
        return page;
    }
}

// Thin alias over the compile-time-constexpr logical->physical map above: identity for contiguous K, invP for
// the per-SP-shard block-cyclic layout. The ring-fused K-gather paths key shard ownership on this map, so keep
// one name shared between the non-fused reader and the fused dual-source reads (both read byte-identically).
#define BC_KTILE(L) \
    logical_to_physical_page<block_cyclic, bc_chunk_local, bc_sp, bc_shard_stride_gap, bc_slab_stride_gap>(L)

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

/** Read one k tile's head_dim_tiles pages [base, base+head_dim_tiles) from `acc` into the CB at `ptr`,
 *  advancing ptr. The per-tile page loop shared by read_k_chunk and both (local / remote) branches of
 *  read_k_chunk_fused -- only the accessor and the page base differ. */
template <typename Acc>
inline void read_ktile_dims(Noc noc, const Acc& acc, uint32_t& ptr, uint32_t base) {
    for (uint32_t dim_tile = 0; dim_tile < head_dim_tiles; ++dim_tile) {
        noc.async_read(acc, CoreLocalMem<uint32_t>(ptr), k_tile_bytes, {.page_id = base + dim_tile}, {});
        ptr += k_tile_bytes;
    }
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
                // BC_KTILE: identity for contiguous K; invP(logical seq tile) for the per-SP-shard block-cyclic layout.
                const uint32_t seq_tile = BC_KTILE(k_tile_start + k_col);
                read_ktile_dims(noc, k_acc, ptr, k_batch_page_offset + seq_tile * head_dim_tiles);
            }
        });
}

#ifdef FUSED_RING
/** Ring-fused k chunk: dual-source per tile. The gathered buffer (k_acc) holds every REMOTE SP-shard's slab
 *  (the all-gather wrote them); this device's OWN shard is read straight from the SP-sharded local cache
 *  (k_local_acc) -- the all-gather omits the local band from the gathered buffer. shard(logical tile L) =
 *  BC_KTILE(L)/tiles_per_shard; local page = k_batch_page_offset/ring_size +
 *  (BC_KTILE(L) - ring_index*tiles_per_shard)*head_dim_tiles (the slot offset is 1/ring of the gathered
 *  buffer's since k_local holds only sll=T/ring keys per slot). Same mcast wrapper + pad/stale as read_k_chunk. */
template <typename KAcc, typename KLocalAcc>
inline void read_k_chunk_fused(
    Noc noc,
    const KAcc& k_acc,
    const KLocalAcc& k_local_acc,
    uint32_t ring_index,
    uint32_t ring_size,
    uint32_t tiles_per_shard,
    uint32_t k_tile_start,
    uint32_t k_tiles_in_unit,
    const McastDir& k_dir,
    uint32_t k_batch_page_offset) {
    read_block_or_mcast<cb_k, k_mcast_on, k_send_sem, k_recv_sem, k_valid_sem>(
        noc, k_chunk_tiles, k_chunk_tiles * k_tile_bytes, k_dir, [&](uint32_t addr) {
            uint32_t ptr = addr;
            for (uint32_t k_col = 0; k_col < k_tiles_in_unit; ++k_col) {
                const uint32_t seq_tile = BC_KTILE(k_tile_start + k_col);
                const uint32_t shard = seq_tile / tiles_per_shard;
                if (shard == ring_index) {
                    // Indexed multi-user cache: the remote branch adds k_batch_page_offset (= cache_batch_idx *
                    // Tt * Dt, a full-T slot stride) into the gathered [B,1,T,D] buffer. k_local is [B,1,sll,D]
                    // with sll = T/ring_size, so its per-slot stride is 1/ring_size of the gathered one; T =
                    // ring_size * sll EXACTLY (validate-pinned), so the division is integral. Offset 0 when not
                    // indexed. ASSERT the integrality as a REQUIREMENT (not just an observation): if the offset
                    // were ever derived from a non-slot-aligned quantity the local pages would silently shift.
                    ASSERT(k_batch_page_offset % ring_size == 0);
                    const uint32_t local_batch_offset = k_batch_page_offset / ring_size;
                    const uint32_t local_base =
                        local_batch_offset + (seq_tile - ring_index * tiles_per_shard) * head_dim_tiles;
                    read_ktile_dims(noc, k_local_acc, ptr, local_base);  // OWN shard from the SP-local cache
                } else {
                    read_ktile_dims(noc, k_acc, ptr, k_batch_page_offset + seq_tile * head_dim_tiles);  // remote slab
                }
            }
        });
}

/** Per-band all-gather gate for the ring-fused path: encapsulates the fine-grained overlap machinery so
 *  kernel_main stays linear. Instead of waiting for the WHOLE gather, each band waits only on the SP-shards its
 *  tiles land in, so a band whose shards have already arrived scores immediately while farther slabs are still
 *  in flight. shard(logical tile L) = BC_KTILE(L)/tiles_per_shard (the gathered buffer stores shard c at the
 *  physical band [c*tps,(c+1)*tps)). "shard c arrived" == semaphore[dir_c] >= val_c, where (dir_c,val_c) is
 *  what RingIdSequencer emits for the iteration delivering c -- we REPLAY it once to index (dir,val) by shard
 *  (inheriting the crossed direction swap + asymmetric fwd/bwd + forward-writer +1 pre-signal verbatim).
 *  wait_min is monotone/non-destructive, so re-waiting a resident shard is a no-op and the gate cannot deadlock
 *  (edge-device empty directions are never required -- a band never lands in a shard the device does not
 *  receive). */
struct FusedRingGate {
    static constexpr uint32_t max_ring_size = 32;  // bounds the largest supported SP ring
    using KLocalAcc = decltype(TensorAccessor(kl_args, uint32_t{}, uint32_t{}));

    uint32_t ring_index;
    uint32_t ring_size;
    uint32_t tiles_per_shard;           // tiles per SP shard in the gathered buffer
    uint32_t sem_id[2];                 // the two direction semaphore ids
    KLocalAcc k_local_acc;              // local SP shard cache (the all-gather INPUT; AG omits it from gathered)
    uint32_t perm_base;                 // rt slot of the band-visit permutation (one entry per band)
    uint32_t shard_dir[max_ring_size];  // shard -> direction semaphore index
    uint32_t shard_val[max_ring_size];  // shard -> wait threshold

    // recv has already consumed the 6-arg fused block (waiting for the op signal) and advanced argidx; take the
    // k_local addr from the next slot and leave argidx at the band-perm base.
    FusedRingGate(const RingSDPAOpReceiver& recv, uint32_t& argidx) :
        ring_index(recv.seq.ring_index),
        ring_size(recv.seq.ring_size),
        tiles_per_shard(k_len_tiles / recv.seq.ring_size),
        sem_id{recv.signal_op_semaphore_ids[0], recv.signal_op_semaphore_ids[1]},
        k_local_acc(TensorAccessor(kl_args, get_arg_val<uint32_t>(argidx++), k_tile_bytes)),
        perm_base(argidx),
        shard_dir{},
        shard_val{} {
        RingIdSequencer s = recv.seq;  // fresh copy (received={0,0}); replay to index (dir,val) by shard id
        for (uint32_t i = 0; i < ring_size; ++i) {
            uint32_t cap_dir = 0, cap_val = 0;
            const uint32_t rid = s.get_next_ring_id([&](uint32_t d, uint32_t v) {
                cap_dir = d;
                cap_val = v;
            });
            shard_dir[rid] = cap_dir;
            shard_val[rid] = cap_val;
        }
    }

    // Absolute band index this column visits at iteration band_i (striped set, band0=0).
    uint32_t band(uint32_t band_i) const { return get_arg_val<uint32_t>(perm_base + band_i); }

    // Wait on each distinct non-local SP-shard the band [k_tile_start, +k_tiles_in_unit) lands in. Shards form
    // contiguous runs over the tiles (change only at block boundaries), so wait once per run start.
    void gate_band(uint32_t k_tile_start, uint32_t k_tiles_in_unit) const {
        uint32_t prev_shard = 0xFFFFFFFFu;
        for (uint32_t c = 0; c < k_tiles_in_unit; ++c) {
            const uint32_t shard = BC_KTILE(k_tile_start + c) / tiles_per_shard;
            if (shard != prev_shard) {
                prev_shard = shard;
                if (shard != ring_index) {
                    Semaphore<>(sem_id[shard_dir[shard]]).wait_min(shard_val[shard]);
                }
            }
        }
    }

    // Gate the band (sender / no-mcast role only -- receivers get K via the already-gated column mcast), then
    // dual-source read: own shard from k_local, remote shards from the gathered buffer.
    template <typename KAcc>
    void read_k(
        Noc noc,
        const KAcc& k_acc,
        uint32_t k_tile_start,
        uint32_t k_tiles_in_unit,
        const McastDir& k_dir,
        uint32_t k_batch_page_offset) const {
        if (k_mcast_on == 0 || k_dir.role == iscore::mcast_role_sender) {
            gate_band(k_tile_start, k_tiles_in_unit);
        }
        read_k_chunk_fused(
            noc,
            k_acc,
            k_local_acc,
            ring_index,
            ring_size,
            tiles_per_shard,
            k_tile_start,
            k_tiles_in_unit,
            k_dir,
            k_batch_page_offset);
    }
};
#endif

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
                const uint32_t seq_tile = logical_to_physical_page<
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

#ifdef FUSED_RING
    // FUSED_RING is DSA-only: the per-band ring gate assumes all heads resident and no single-head matmul
    // fuse. The factory enforces this (TT_FATAL HB==Hi, block_size==0); lock the invariant at compile time too
    // -- stream_heads would read the band perm out of bounds, and fuse_single would double-reserve cb_q/cb_w
    // (the fuse_single block above ALSO reads them) -> CB overflow / hang.
    static_assert(!stream_heads, "FUSED_RING requires all heads resident (no head streaming)");
    static_assert(fuse_single == 0, "FUSED_RING is incompatible with the single-head matmul fuse");
    // Per-band all-gather gate (fine-grained overlap, replaces the coarse barrier). The 6-arg fused block
    // {ring_size, ring_index, fwd_writes, bwd_writes, sem0, sem1} sits at slot 27; the receiver consumes it
    // (waiting for the op signal) and advances argidx, then FusedRingGate consumes the k_local addr, builds the
    // shard->(dir,val) table, and records the band-perm base. See FusedRingGate above.
    uint32_t fused_argidx = 27;
    RingSDPAOpReceiver fused_recv(/*wait_for_op_signal=*/true, fused_argidx);
    const FusedRingGate gate(fused_recv, fused_argidx);
#endif

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
#ifdef FUSED_RING
        // Fused ring (DSA: !fuse_single, !stream_heads): read q + gates FIRST, UNGATED, so the q/w row-mcast
        // rendezvous completes immediately (decoupled from the per-band fabric gate) and every reordered band's
        // compute has its resident q/w ready.
        read_q_rows(noc, q_acc, q_row_start, q_dir);
        read_w_group(noc, w_acc, q_row_start, q_dir);
#endif
        for (uint32_t band_i = 0; band_i < band_iters; ++band_i) {
#ifdef FUSED_RING
            // Reordered band-visit order (local-first, then remote by ring arrival): ABSOLUTE owned band
            // indices from the perm slots, IDENTICAL to compute/writer so the cb_k / cb_out FIFOs stay in
            // lockstep. Every iterated band is real (stream_heads -- the only phantom-band source -- is
            // disallowed on the fused path), so no real_band guard.
            const uint32_t band = gate.band(band_i);
            span.set(group, band0 + band);
            gate.read_k(noc, k_acc, span.k_tile_start(), span.k_tiles(), k_dir, k_batch_page_offset);
#else
            const uint32_t band = band_i;
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
                    // Non-fused resident: q/w deferred behind q/k here. (Fused reads q/w before the band loop.)
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
#endif
        }
    }
}
