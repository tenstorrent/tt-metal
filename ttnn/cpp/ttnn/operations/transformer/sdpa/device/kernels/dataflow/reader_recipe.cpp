// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// Private noncausal, per-head KV unicast-chain reader. Compute and writer are
// unchanged. Every core reads its own Q; only rank zero reads K/V from DRAM.
//
// CTA: [q_tiles, k_chunks, q_chunks_per_head, Q accessor..., K accessor..., V accessor...]
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

#ifndef SDPA_READER_BARRIER_TILES
#define SDPA_READER_BARRIER_TILES 2
#endif
#ifndef SDPA_K_CHUNK_TILES
#define SDPA_K_CHUNK_TILES 16
#endif
constexpr uint32_t kv_tiles = SDPA_K_CHUNK_TILES * 4;

template <uint32_t tile_bytes, bool transpose, typename Accessor>
FORCE_INLINE void read_kv_from_dram(const Noc& noc, const Accessor& tensor, uint32_t first_page, uint32_t write_ptr) {
    // Sequential source requests distribute traffic over the interleaved banks;
    // K scatters the tile grid in L1, without transposing individual tiles.
    for (uint32_t p = 0; p < kv_tiles; ++p) {
        const uint32_t dst_tile = transpose ? (p % 4) * SDPA_K_CHUNK_TILES + p / 4 : p;
        const CoreLocalMem<uint32_t> destination(write_ptr + dst_tile * tile_bytes);
        if (!tensor.visit(first_page + p, [&](const auto& source, uint32_t page) {
                noc.async_read(source, destination, tile_bytes, {.page_id = page}, {});
            })) {
            noc.async_write_zeros(destination, tile_bytes);
        }
#if SDPA_READER_BARRIER_TILES > 0
        if ((p + 1) % SDPA_READER_BARRIER_TILES == 0) {
            noc.async_read_barrier();
        }
#endif
    }
    noc.async_read_barrier();
}

void kernel_main() {
    constexpr uint32_t q_tiles = get_compile_time_arg_val(0);
    constexpr uint32_t k_chunks = get_compile_time_arg_val(1);
    constexpr uint32_t queries_per_head = get_compile_time_arg_val(2);
    static_assert(q_tiles > 0 && q_tiles % 2 == 0);
    static_assert(k_chunks > 0 && queries_per_head > 0);
#ifdef SDPA_JOINT
    constexpr uint32_t q_primary_pages = get_compile_time_arg_val(3);
    constexpr uint32_t q_joint_pages = get_compile_time_arg_val(4);
    constexpr uint32_t kv_primary_pages = get_compile_time_arg_val(5);
    constexpr uint32_t kv_joint_pages = get_compile_time_arg_val(6);
    constexpr auto qa = TensorAccessorArgs<7>();
#else
    constexpr auto qa = TensorAccessorArgs<3>();
#endif
    constexpr auto ka = TensorAccessorArgs<qa.next_compile_time_args_offset()>();
    constexpr auto va = TensorAccessorArgs<ka.next_compile_time_args_offset()>();
#ifdef SDPA_JOINT
    constexpr auto jqa = TensorAccessorArgs<va.next_compile_time_args_offset()>();
    constexpr auto jka = TensorAccessorArgs<jqa.next_compile_time_args_offset()>();
    constexpr auto jva = TensorAccessorArgs<jka.next_compile_time_args_offset()>();
    const auto q = sequence_accessor<q_primary_pages, q_joint_pages, queries_per_head * q_tiles * 4>(
        TensorAccessor(qa, get_arg_val<uint32_t>(0)), TensorAccessor(jqa, get_arg_val<uint32_t>(12)));
    const auto k = sequence_accessor<kv_primary_pages, kv_joint_pages, k_chunks * kv_tiles>(
        TensorAccessor(ka, get_arg_val<uint32_t>(1)), TensorAccessor(jka, get_arg_val<uint32_t>(13)));
    const auto v = sequence_accessor<kv_primary_pages, kv_joint_pages, k_chunks * kv_tiles>(
        TensorAccessor(va, get_arg_val<uint32_t>(2)), TensorAccessor(jva, get_arg_val<uint32_t>(14)));
#else
    const auto q = sequence_accessor(TensorAccessor(qa, get_arg_val<uint32_t>(0)));
    const auto k = sequence_accessor(TensorAccessor(ka, get_arg_val<uint32_t>(1)));
    const auto v = sequence_accessor(TensorAccessor(va, get_arg_val<uint32_t>(2)));
#endif
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
    generate_bcast_col_scalar(CircularBuffer(4), 0x3f803f80);

    const uint32_t kvbase = head * k_chunks * kv_tiles;
    for (uint32_t qi = 0; qi < jobs; ++qi) {
        const uint32_t qbase = (first_job + qi) * q_tiles * 4;
        qcb.reserve_back(q_tiles * 4);
        const uint32_t qptr = qcb.get_write_ptr();
        for (uint32_t p = 0; p < q_tiles * 4; ++p) {
            const CoreLocalMem<uint32_t> destination(qptr + p * qbytes);
            if (!q.visit(qbase + p, [&](const auto& source, uint32_t page) {
                    noc.async_read(source, destination, qbytes, {.page_id = page}, {});
                })) {
                noc.async_write_zeros(destination, qbytes);
            }
        }
        noc.async_read_barrier();
        qcb.push_back(q_tiles * 4);

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
                // Production ChainLink waits for downstream reservation, sends
                // to (next_x,next_y,kptr), flushes its source, then relays valid.
                link.forward(noc, kptr, kv_tiles, kbytes);
            }
            // Publish on every rank only AFTER forwarding has released its source.
            // Crucially, K is published BEFORE reserving V, retaining K lookahead
            // when the previous iteration still occupies the one-slot V buffer.
            kcb.push_back(kv_tiles);

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
