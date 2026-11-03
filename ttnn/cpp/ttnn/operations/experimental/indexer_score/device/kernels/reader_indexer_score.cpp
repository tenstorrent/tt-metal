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

#include <tt-metalium/constants.hpp>

#include "indexer_score_runtime_args.hpp"
#include "indexer_schedule.hpp"

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
constexpr auto q_args = TensorAccessorArgs<num_common_ct_args>();
constexpr auto k_args = TensorAccessorArgs<q_args.next_compile_time_args_offset()>();
constexpr auto w_args = TensorAccessorArgs<k_args.next_compile_time_args_offset()>();
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
// Thin alias over the shared block-cyclic invP map (tt::block_cyclic, block_cyclic_remap.hpp): identity for
// contiguous K, invP for the per-SP-shard block-cyclic layout.
FORCE_INLINE uint32_t bc_ktile(uint32_t L) {
    return tt::block_cyclic::
        logical_to_physical_page<block_cyclic, bc_chunk_local, bc_sp, bc_shard_stride_gap, bc_slab_stride_gap>(L);
}
// Receiver rectangle / sender coordinates derived from the shared physical axes and core identity.
struct McastDir {
    uint32_t role;            // McastRole: none (DRAM read), sender (read + mcast), receiver (wait for mcast)
    uint32_t xs, ys, xe, ye;  // receiver rectangle (sender excluded by default mcast opts)
    uint32_t sx, sy;          // sender physical coord (receivers signal it ready)
    uint32_t ndst;            // number of receivers
};

/** Sender: data already at `addr` from DRAM; wait all receivers ready, mcast the block,
 *  then relay the valid flag into their recv semaphore. */
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

// Prepare gate columns after the first K read so their local preparation overlaps QK matmul.
// The weight row multicast must still complete before any column multicast.
inline void expand_w_group() {
    constexpr uint32_t head_tiles = (num_heads + tt::constants::TILE_WIDTH - 1) / tt::constants::TILE_WIDTH;
    using namespace tt::constants;
    constexpr uint32_t bf16_per_word = sizeof(uint32_t) / sizeof(uint16_t);
    constexpr uint32_t face_row_words = FACE_WIDTH / bf16_per_word;
    constexpr uint32_t face_words = FACE_HW / bf16_per_word;
    constexpr uint32_t tile_words = TILE_HW / bf16_per_word;
    constexpr uint32_t bf16_bits = sizeof(uint16_t) * 8;
    // read_w_group reserved the output group; compute waits until expansion finishes.
    // Walk input tiles and head pairs backward: expanded destinations are at or after their
    // source tile. The final head-0 store aliases its original column without changing it.
    CircularBuffer gates(cb_w);
    const uint32_t addr = gates.get_write_ptr();
    for (uint32_t row_end = q_tiles_per_unit; row_end > 0; --row_end) {
        const uint32_t q_row = row_end - 1;
        for (uint32_t tile_end = head_tiles; tile_end > 0; --tile_end) {
            const uint32_t ht = tile_end - 1;
            auto* src =
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(addr + (q_row * head_tiles + ht) * bf16_tile_bytes);
            const uint32_t heads_in_tile = std::min<uint32_t>(TILE_WIDTH, num_heads - ht * TILE_WIDTH);
            for (uint32_t pair_end = (heads_in_tile + bf16_per_word - 1) / bf16_per_word; pair_end > 0; --pair_end) {
                const uint32_t h = bf16_per_word * (pair_end - 1);
                auto* dst = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
                    addr + (q_row * num_heads + ht * TILE_WIDTH + h) * bf16_tile_bytes);
                // Extract two BF16 heads per word. Word stores also set the unused column 1;
                // this avoids narrow L1 accesses.
                for (uint32_t r = 0; r < TILE_HEIGHT; ++r) {
                    const uint32_t row_offset =
                        (r / FACE_HEIGHT) * (TILE_WIDTH / FACE_WIDTH) * face_words + (r % FACE_HEIGHT) * face_row_words;
                    const uint32_t pair =
                        src[row_offset + (h / FACE_WIDTH) * face_words + (h % FACE_WIDTH) / bf16_per_word];
                    dst[row_offset] = pair;
                    if (ht * TILE_WIDTH + h + 1 < num_heads) {
                        dst[tile_words + row_offset] = pair >> bf16_bits;
                    }
                }
            }
        }
    }
    gates.push_back(w_group_tiles);
}

/** resident w (gates) group [q_tiles_per_unit][num_heads], role-aware (q row mcast). MSA fills a constant
 *  scale in L1 instead (no weights tensor); the q placeholder accessor is then unused. */
template <typename WAcc>
inline void read_w_group(Noc noc, const WAcc& w_acc, uint32_t q_row_start, const McastDir& q_dir) {
    if constexpr (synthesize_gate) {
        fill_w_group_const();
        return;
    }
    // Transport [query, head] weight tiles, then prepare each core's resident gate columns.
    // Multicasting before expansion avoids broadcasting 31 padding columns per head.
    constexpr uint32_t head_tiles = (num_heads + tt::constants::TILE_WIDTH - 1) / tt::constants::TILE_WIDTH;
    constexpr uint32_t input_tiles = q_tiles_per_unit * head_tiles;
    CircularBuffer cb(cb_w);
    cb.reserve_back(w_group_tiles);
    const uint32_t addr = cb.get_write_ptr();
    if (q_mcast_on && q_dir.role == iscore::mcast_role_receiver) {
        mcast_recv<q_send_sem, q_recv_sem>(noc, q_dir);
    } else {
        for (uint32_t tile = 0; tile < input_tiles; ++tile) {
            noc.async_read(
                w_acc,
                CoreLocalMem<uint32_t>(addr + tile * bf16_tile_bytes),
                bf16_tile_bytes,
                {.page_id = q_row_start * head_tiles + tile},
                {});
        }
        noc.async_read_barrier();
        if (q_mcast_on && q_dir.role == iscore::mcast_role_sender) {
            mcast_send<q_send_sem, q_recv_sem, q_valid_sem>(noc, q_dir, addr, input_tiles * bf16_tile_bytes);
        }
    }
    if constexpr (stream_heads || fuse_single) {
        expand_w_group();
    }
}

/** Read one k tile's head_dim_tiles pages [base, base+head_dim_tiles) from `acc` into the CB at `ptr`,
 *  advancing ptr. */
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
                // bc_ktile: identity for contiguous K; invP(logical seq tile) for the per-SP-shard block-cyclic layout.
                const uint32_t seq_tile = bc_ktile(k_tile_start + k_col);
                read_ktile_dims(noc, k_acc, ptr, k_batch_page_offset + seq_tile * head_dim_tiles);
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
    constexpr uint32_t schedule_blocks = get_named_compile_time_arg_val("schedule_blocks");
    constexpr uint32_t schedule_cols = get_named_compile_time_arg_val("schedule_cols");
    constexpr uint32_t schedule_groups = get_named_compile_time_arg_val("schedule_groups");
    constexpr uint32_t schedule_group_rows = get_named_compile_time_arg_val("schedule_group_rows");
    constexpr uint32_t schedule_max_bands = get_named_compile_time_arg_val("schedule_max_bands");
    constexpr uint32_t schedule_units = get_named_compile_time_arg_val("schedule_units");
    const uint32_t q_addr = get_common_arg_val<uint32_t>(indexer_common::reader::Q);
    const uint32_t k_addr = get_common_arg_val<uint32_t>(indexer_common::reader::K);
    const uint32_t w_addr = get_common_arg_val<uint32_t>(indexer_common::reader::W);
    // Banded schedule: groups -> grid rows (q/w shared = q_dir mcast), k-bands -> columns (k shared = k_dir mcast).
    const uint32_t core_id = get_arg_val<uint32_t>(0);
    constexpr uint32_t group_stride = schedule_group_rows;
    constexpr uint32_t num_groups = schedule_groups;
    const auto schedule =
        indexer_schedule::for_core(core_id, group_stride, schedule_units, schedule_blocks, schedule_cols);
    const uint32_t row_group0 = schedule.row_group;
    const uint32_t band0 = schedule.band_start;
    const uint32_t num_bands = schedule.band_count;
    constexpr uint32_t max_bands = schedule_max_bands;
    constexpr uint32_t x_base = indexer_common::reader::Count;
    constexpr uint32_t y_base = x_base + schedule_cols;
    const uint32_t block = schedule.block;
    const uint32_t col = schedule.col;
    const uint32_t first_row = block * group_stride;
    const uint32_t row = first_row + row_group0;
    const auto x = [](uint32_t c) { return get_common_arg_val<uint32_t>(x_base + c); };
    const auto y = [](uint32_t r) { return get_common_arg_val<uint32_t>(y_base + r); };
    uint32_t xs = x(0), xe = xs;
    for (uint32_t c = 1; c < schedule_cols; ++c) {
        const uint32_t px = x(c);
        xs = px < xs ? px : xs;
        xe = px > xe ? px : xe;
    }
    uint32_t ys = y(first_row), ye = ys;
    for (uint32_t r = first_row + 1; r < first_row + group_stride; ++r) {
        const uint32_t py = y(r);
        ys = py < ys ? py : ys;
        ye = py > ye ? py : ye;
    }
    const uint32_t diag = row < schedule_cols ? row : schedule_cols - 1;
    const McastDir k_dir{
        k_mcast_on ? (row == first_row ? iscore::mcast_role_sender : iscore::mcast_role_receiver)
                   : iscore::mcast_role_none,
        x(col),
        ys,
        x(col),
        ye,
        x(col),
        y(first_row),
        group_stride - 1};
    const McastDir q_dir{
        q_mcast_on ? (col == diag ? iscore::mcast_role_sender : iscore::mcast_role_receiver) : iscore::mcast_role_none,
        xs,
        y(row),
        xe,
        y(row),
        x(diag),
        y(row),
        schedule_cols - 1};
    // Persistent-cache args remain common and are re-applied on every dispatch.
    const uint32_t k_batch_page_offset = get_common_arg_val<uint32_t>(
        indexer_common::reader::BatchOffset);  // indexed-cache page offset; 0 when not indexed
    const uint32_t kv_len_tiles =
        get_common_arg_val<uint32_t>(indexer_common::reader::KvLength);  // valid KV length in tiles (full when unset)

    const auto q_acc = TensorAccessor(q_args, q_addr, q_tile_bytes);
    const auto k_acc = TensorAccessor(k_args, k_addr, k_tile_bytes);
    const auto w_acc = TensorAccessor(w_args, w_addr, bf16_tile_bytes);

    Noc noc;
    build_mask_tiles(noc);
    if constexpr (block_pool) {
        dataflow_kernel_lib::
            calculate_and_prepare_reduce_scaler<cb_scaler, ckernel::PoolType::MAX, ckernel::ReduceDim::REDUCE_ROW>();
    }

    WorkUnitSpan span;
    span.set_valid_k_len_tiles(kv_len_tiles);
    const uint32_t band_iters = stream_heads ? max_bands : num_bands;
    for (uint32_t phase = 0; phase < num_groups; ++phase) {
        const uint32_t group = row_group0 + phase * group_stride;
        const uint32_t q_row_start = group * q_tiles_per_unit;
        if constexpr (fuse_single) {
            read_q_rows(noc, q_acc, q_row_start, q_dir);
            read_w_group(noc, w_acc, q_row_start, q_dir);
        }
        if constexpr (stream_heads) {
            read_w_group(noc, w_acc, q_row_start, q_dir);
        }
        for (uint32_t band_i = 0; band_i < band_iters; ++band_i) {
            const uint32_t band = band_i;
            const bool real_band = band < num_bands;
            if (real_band) {
                span.set(group, band0 + band);
                // Resident q/w are multicasted once at each core's first band. This must
                // happen even if that core's band falls past kv_len, since another column
                // may still be waiting at the shared row rendezvous.
                if (band == 0 && !stream_heads && !fuse_single) {
                    read_q_rows(noc, q_acc, q_row_start, q_dir);
                    read_w_group(noc, w_acc, q_row_start, q_dir);
                }
                // Head streaming emits the q blocks below even for an empty K band; the
                // compute kernel drains them before moving to the next band.
                if (span.k_tiles() != 0) {
                    if constexpr (fuse_single && fused_stream_k) {
                        read_k_chunk_streaming(noc, k_acc, span.k_tile_start(), span.k_tiles(), k_batch_page_offset);
                    } else {
                        read_k_chunk(noc, k_acc, span.k_tile_start(), span.k_tiles(), k_dir, k_batch_page_offset);
                    }
                }
            }
            if constexpr (stream_heads) {
                for (uint32_t tile_idx = 0; tile_idx < q_tiles_per_unit * k_tiles_per_unit; ++tile_idx) {
                    for (uint32_t first_head = 0; first_head < num_heads; first_head += heads_per_group) {
                        read_q_block(noc, q_acc, q_row_start, first_head, q_dir);
                    }
                }
            }
            if constexpr (!synthesize_gate && !stream_heads && !fuse_single) {
                if (band_i == 0) {
                    expand_w_group();
                }
            }
        }
    }
}
