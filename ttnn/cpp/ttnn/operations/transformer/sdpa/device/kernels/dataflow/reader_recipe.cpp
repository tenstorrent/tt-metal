// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Private noncausal, per-head KV unicast-chain reader. Every core reads its own Q;
// only rank zero reads K/V from DRAM.
//
// CTA: [q_tiles, k_chunks, q_chunks_per_head, primary/joint Q rows, primary/joint KV rows, accessors...]
// Named CTAs (host: run_recipe_segments, sdpa_recipe.cpp): see recipe_reader below.
// Runtime args:
//   0 Q address, 1 K address, 2 V address, 3 first_flat_Q_job, 4 local_Q_jobs,
//   5 chain_rank, 6 chain_length, 7 upstream_physical_x, 8 upstream_physical_y,
//   9 downstream_physical_x, 10 downstream_physical_y, 11 downstream_Q_jobs.
// Semaphores on EVERY participating core: 0 ready=0, 1 received=0, 2 valid=1.
// For singleton chains, rank=0, length=1, downstream_Q_jobs=0; coordinates unused.
//
// Host invariants:
// - One head per core; one chain per head; positive, nonincreasing Q-job counts.
// - Identical CB allocation/format/capacity on every core, including K/V slots.
// - No dummy or skipped K/V rounds; all active links traverse K then V for each
//   (local_Q_ordinal, K_chunk) in the same order. Global Q offsets may differ.
// These make the reserved K/V write pointer identical across an active link:
// base + ((local_Q_ordinal*k_chunks + K_chunk) % slots)*kv_tiles*tile_bytes.
// When a shorter downstream core finishes, forwarding stops BEFORE any pointer
// equality would be assumed for an event it no longer receives.

#include "api/core_local_mem.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/tensor/noc_traits.h"
#include "ttnn/kernel/dataflow/generate_bcast_scalar.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"
#include "ttnn/operations/transformer/sdpa/device/kernels/dataflow/chain_link.hpp"
#include "sequence_accessor.hpp"
#include "tile_padding.hpp"

namespace recipe_reader {
// Joint SDPA: three joint accessors follow the primary Q/K/V ones, and the joint addresses runtime args 12-14.
constexpr bool has_joint = get_named_compile_time_arg_val("joint") != 0;
constexpr uint32_t k_chunk_tiles = get_named_compile_time_arg_val("k_chunk_tiles");
constexpr uint32_t d_tiles = get_named_compile_time_arg_val("d_tiles");
// Zero Q rows padding a paired recipe's odd chunk (host: recipe_compute_q_tiles).
constexpr uint32_t q_pad_tiles = get_named_compile_time_arg_val("q_pad_tiles");
// Grouped-query attention: query heads per KV head (1: one KV head per query head).
constexpr uint32_t q_per_kv_head = get_named_compile_time_arg_val("q_per_kv_head");
// Additive attn_mask (dense only): its accessor follows the primary ones, its address is runtime arg 12.
constexpr bool has_mask = get_named_compile_time_arg_val("mask") != 0;
constexpr uint32_t mask_cb = get_named_compile_time_arg_val("mask_cb");
constexpr uint32_t mask_group_rows = get_named_compile_time_arg_val("mask_group_rows");
constexpr uint32_t mask_q_tiles = get_named_compile_time_arg_val("mask_q_tiles");
constexpr uint32_t mask_k_tiles = get_named_compile_time_arg_val("mask_k_tiles");
constexpr uint32_t mask_heads = get_named_compile_time_arg_val("mask_heads");
constexpr bool mask_bcast_batch = get_named_compile_time_arg_val("mask_bcast_batch") != 0;
constexpr bool mask_bcast_heads = get_named_compile_time_arg_val("mask_bcast_heads") != 0;
static_assert(!(has_mask && has_joint), "SDPA recipe masks are dense-only");
}  // namespace recipe_reader
using namespace recipe_reader;

// DRAM read barrier every 16 K/V tiles keeps enough reads in flight for the chain head (every 2 tiles serialized
// the fetch).
constexpr uint32_t reader_barrier_tiles = 16;
constexpr uint32_t kv_tiles = k_chunk_tiles * d_tiles;

template <uint32_t tile_bytes, bool transpose, typename Accessor>
FORCE_INLINE void read_kv_from_dram(const Noc& noc, const Accessor& tensor, uint32_t first_page, uint32_t write_ptr) {
    // Sequential source requests distribute traffic over the interleaved banks;
    // K scatters the tile grid in L1, without transposing individual tiles.
    for (uint32_t p = 0; p < kv_tiles; ++p) {
        const uint32_t dst_tile = transpose ? (p % d_tiles) * k_chunk_tiles + p / d_tiles : p;
        const CoreLocalMem<uint32_t> destination(write_ptr + dst_tile * tile_bytes);
        if (!tensor.visit(first_page + p, [&](const auto& source, uint32_t page) {
                noc.async_read(source, destination, tile_bytes, {.page_id = page}, {});
            })) {
            noc.async_write_zeros(destination, tile_bytes);
        }
        if ((p + 1) % reader_barrier_tiles == 0) {
            noc.async_read_barrier();
        }
    }
    noc.async_read_barrier();
    if constexpr (Accessor::has_partial_rows) {
        for (uint32_t p = 0; p < kv_tiles; ++p) {
            const uint32_t rows = tensor.valid_rows(first_page + p);
            if (rows > 0 && rows < 32) {
                const uint32_t dst_tile = transpose ? (p % d_tiles) * k_chunk_tiles + p / d_tiles : p;
                zero_tile_padding<tile_bytes>(write_ptr + dst_tile * tile_bytes, rows);
            }
        }
    }
}

// Additive attn_mask [1|B, 1|H, Sq, Sk]: stream one Q chunk x K chunk of tiles, one Q tile row
// (k_chunk_tiles tiles) at a time, so compute can apply each QK row group as it lands.
// The mask CB holds one or two compute row groups (mask_group_rows rows each).
// Tiles outside the mask (Q rows past Sq, K columns past Sk) are zero: the recipe's pack-thread
// tail hook already stamps -inf on padded K columns, and padded Q rows are never written.
template <uint32_t q_tiles, typename Accessor>
FORCE_INLINE void read_mask_chunk(
    const Noc& noc, const Accessor& mask, CircularBuffer& cb, uint32_t head, uint32_t q_tile0, uint32_t k_tile0) {
    constexpr uint32_t bytes = get_tile_size(mask_cb);
    constexpr uint32_t mask_head_count = mask_bcast_heads ? 1 : mask_heads;
    const uint32_t batch = mask_bcast_batch ? 0 : head / mask_heads;
    const uint32_t mask_head = mask_bcast_heads ? 0 : head % mask_heads;
    const uint32_t base = (batch * mask_head_count + mask_head) * mask_q_tiles * mask_k_tiles;
    // Whole compute row groups: an odd chunk's last group gets a zero padding row.
    constexpr uint32_t rows = (q_tiles + mask_group_rows - 1) / mask_group_rows * mask_group_rows;
    for (uint32_t row = 0; row < rows; ++row) {
        const uint32_t q_tile = row < q_tiles ? q_tile0 + row : mask_q_tiles;
        cb.reserve_back(k_chunk_tiles);
        const uint32_t ptr = cb.get_write_ptr();
        bool zeroed = false;
        for (uint32_t col = 0; col < k_chunk_tiles; ++col) {
            const uint32_t k_tile = k_tile0 + col;
            const CoreLocalMem<uint32_t> destination(ptr + col * bytes);
            if (q_tile < mask_q_tiles && k_tile < mask_k_tiles) {
                noc.async_read(mask, destination, bytes, {.page_id = base + q_tile * mask_k_tiles + k_tile}, {});
            } else {
                noc.async_write_zeros(destination, bytes);
                zeroed = true;
            }
        }
        if (zeroed) {
            noc.write_zeros_l1_barrier();
        }
        noc.async_read_barrier();
        cb.push_back(k_chunk_tiles);
    }
}

// Q, K or V as one sequence: the primary tensor's accessor (args at args_offset) and, in joint builds, the joint
// segment's: the index-th of the three joint accessors whose args start at joint_args.
template <uint32_t args, uint32_t index>
constexpr uint32_t joint_args_offset() {
    if constexpr (index == 0) {
        return args;
    } else {
        return joint_args_offset<TensorAccessorArgs<args>().next_compile_time_args_offset(), index - 1>();
    }
}

template <
    uint32_t primary_rows,
    uint32_t joint_rows,
    uint32_t chunk_rows,
    uint32_t args_offset,
    uint32_t joint_args,
    uint32_t index>
FORCE_INLINE auto recipe_sequence(uint32_t address_arg) {
    constexpr auto primary_args = TensorAccessorArgs<args_offset>();
    if constexpr (has_joint) {
        constexpr auto segment_args = TensorAccessorArgs<joint_args_offset<joint_args, index>()>();
        return sequence_accessor<primary_rows, joint_rows, chunk_rows, d_tiles>(
            TensorAccessor(primary_args, get_arg_val<uint32_t>(address_arg)),
            TensorAccessor(segment_args, get_arg_val<uint32_t>(12 + index)));
    } else {
        return sequence_accessor<primary_rows, chunk_rows, d_tiles>(
            TensorAccessor(primary_args, get_arg_val<uint32_t>(address_arg)));
    }
}

// The attn_mask accessor (args at args_offset, address in runtime arg 12); 0 without a mask.
template <uint32_t args_offset>
FORCE_INLINE auto recipe_mask_accessor() {
    if constexpr (has_mask) {
        constexpr auto ma = TensorAccessorArgs<args_offset>();
        return TensorAccessor(ma, get_arg_val<uint32_t>(12));
    } else {
        return 0;
    }
}

void kernel_main() {
    constexpr uint32_t q_tiles = get_compile_time_arg_val(0);
    constexpr uint32_t k_chunks = get_compile_time_arg_val(1);
    constexpr uint32_t queries_per_head = get_compile_time_arg_val(2);
    static_assert(q_tiles > 0);
    static_assert(k_chunks > 0 && queries_per_head > 0);
    constexpr uint32_t q_primary_rows = get_compile_time_arg_val(3);
    constexpr uint32_t q_joint_rows = get_compile_time_arg_val(4);
    constexpr uint32_t kv_primary_rows = get_compile_time_arg_val(5);
    constexpr uint32_t kv_joint_rows = get_compile_time_arg_val(6);
    constexpr auto qa = TensorAccessorArgs<7>();
    constexpr auto ka = TensorAccessorArgs<qa.next_compile_time_args_offset()>();
    constexpr auto va = TensorAccessorArgs<ka.next_compile_time_args_offset()>();
    constexpr uint32_t after_primary = va.next_compile_time_args_offset();
    const auto q = recipe_sequence<q_primary_rows, q_joint_rows, q_tiles * 32, 7, after_primary, 0>(0);
    const auto k = recipe_sequence<
        kv_primary_rows,
        kv_joint_rows,
        k_chunk_tiles * 32,
        qa.next_compile_time_args_offset(),
        after_primary,
        1>(1);
    const auto v = recipe_sequence<
        kv_primary_rows,
        kv_joint_rows,
        k_chunk_tiles * 32,
        ka.next_compile_time_args_offset(),
        after_primary,
        2>(2);
    const auto mask_accessor = recipe_mask_accessor<after_primary>();
    CircularBuffer mcb(mask_cb);
    const uint32_t first_job = get_arg_val<uint32_t>(3);
    const uint32_t jobs = get_arg_val<uint32_t>(4);
    const uint32_t rank = get_arg_val<uint32_t>(5);
    const uint32_t chain_length = get_arg_val<uint32_t>(6);
    const uint32_t prev_x = get_arg_val<uint32_t>(7);
    const uint32_t prev_y = get_arg_val<uint32_t>(8);
    const uint32_t next_x = get_arg_val<uint32_t>(9);
    const uint32_t next_y = get_arg_val<uint32_t>(10);
    const uint32_t next_jobs = get_arg_val<uint32_t>(11);
    const uint32_t head = first_job / queries_per_head;
    ASSERT(jobs > 0 && chain_length > 0 && rank < chain_length);
    ASSERT((first_job + jobs - 1) / queries_per_head == head);
    ASSERT(next_jobs <= jobs);
    ASSERT((rank + 1 == chain_length) ? next_jobs == 0 : next_jobs > 0);

    constexpr uint32_t ready_sem = 0, received_sem = 1, valid_sem = 2;
    constexpr uint32_t qbytes = get_tile_size(0);
    constexpr uint32_t kbytes = get_tile_size(1);
    constexpr uint32_t vbytes = get_tile_size(2);
    Noc noc;
    CircularBuffer qcb(0), kcb(1), vcb(2);
    const ChainLink<false, true> link(
        chain_length > 1,
        rank == 0,
        rank + 1 == chain_length,
        ready_sem,
        received_sem,
        valid_sem,
        prev_x,
        prev_y,
        next_x,
        next_y,
        0,
        0,
        0,
        0,
        0,  // Multicast rectangle/destination count: unused for unicast.
        kv_tiles,
        kbytes,
        head,
        next_jobs);

    // Do not locally reinitialize ready/received here: a downstream reader may
    // already have sent readiness. Descriptor initialization precedes all kernels.
    dataflow_kernel_lib::calculate_and_prepare_reduce_scaler<
        3,
        ckernel::PoolType::MAX,
        ckernel::ReduceDim::REDUCE_ROW,
        dataflow_kernel_lib::SUM_AND_MAX_REDUCE_FACTOR>();
    generate_bcast_col_scalar_zeroed(CircularBuffer(4), 0x3f803f80);

    const uint32_t kv_head = head / q_per_kv_head;
    const uint32_t kvbase = kv_head * k_chunks * kv_tiles;
    for (uint32_t qi = 0; qi < jobs; ++qi) {
        const uint32_t qbase = (first_job + qi) * q_tiles * d_tiles;
        // Paired recipes pad an odd chunk with q_pad_tiles zero rows (host: recipe_compute_q_tiles).
        constexpr uint32_t q_push_tiles = (q_tiles + q_pad_tiles) * d_tiles;
        qcb.reserve_back(q_push_tiles);
        const uint32_t qptr = qcb.get_write_ptr();
        bool zeroed = q_pad_tiles > 0;
        for (uint32_t p = 0; p < q_tiles * d_tiles; ++p) {
            const CoreLocalMem<uint32_t> destination(qptr + p * qbytes);
            if (!q.visit(qbase + p, [&](const auto& source, uint32_t page) {
                    noc.async_read(source, destination, qbytes, {.page_id = page}, {});
                })) {
                noc.async_write_zeros(destination, qbytes);
                zeroed = true;
            }
        }
        for (uint32_t p = q_tiles * d_tiles; p < q_push_tiles; ++p) {
            noc.async_write_zeros(CoreLocalMem<uint32_t>(qptr + p * qbytes), qbytes);
        }
        if (zeroed) {
            noc.write_zeros_l1_barrier();
        }
        noc.async_read_barrier();
        if constexpr (decltype(q)::has_partial_rows) {
            for (uint32_t p = 0; p < q_tiles * d_tiles; ++p) {
                const uint32_t rows = q.valid_rows(qbase + p);
                if (rows > 0 && rows < 32) {
                    zero_tile_padding<qbytes>(qptr + p * qbytes, rows);
                }
            }
        }
        qcb.push_back(q_push_tiles);

        const bool receive = link.should_receive(head);
        const bool forward = link.should_forward(head, qi);
        for (uint32_t ki = 0; ki < k_chunks; ++ki) {
            const uint32_t first_kv_page = kvbase + ki * kv_tiles;

            kcb.reserve_back(kv_tiles);
            const uint32_t kptr = kcb.get_write_ptr();
            if (receive) {
                link.receive(noc);
            } else {
                read_kv_from_dram<kbytes, true>(noc, k, first_kv_page, kptr);
            }
            if (forward) {
                link.forward(noc, kptr, kv_tiles, kbytes);
            }
            // Publish on every rank only AFTER forwarding has released its source.
            // Crucially, K is published BEFORE reserving V, retaining K lookahead
            // when the previous iteration still occupies the one-slot V buffer.
            kcb.push_back(kv_tiles);
            if constexpr (has_mask) {
                // Mask after K, before V: QK (phase one) consumes it; PV (phase two) needs only V.
                read_mask_chunk<q_tiles>(
                    noc, mask_accessor, mcb, head, ((first_job + qi) % queries_per_head) * q_tiles, ki * k_chunk_tiles);
            }

            vcb.reserve_back(kv_tiles);
            const uint32_t vptr = vcb.get_write_ptr();
            if (receive) {
                link.receive(noc);
            } else {
                read_kv_from_dram<vbytes, false>(noc, v, first_kv_page, vptr);
            }
            if (forward) {
                link.forward(noc, vptr, kv_tiles, vbytes);
            }
            vcb.push_back(kv_tiles);
        }
    }
    noc.async_write_barrier();
}
