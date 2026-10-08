// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Private noncausal, per-head KV unicast-chain reader. Compute and writer are
// unchanged. Every core reads its own Q; only rank zero reads K/V from DRAM.
//
// CTA: [q_tiles, k_chunks, q_chunks_per_head, primary/joint Q rows, primary/joint KV rows, accessors...]
// Runtime args:
//   0 Q address, 1 K address, 2 V address, 3 first_flat_Q_job, 4 local_Q_jobs,
//   5 chain_rank, 6 chain_length, 7 upstream_physical_x, 8 upstream_physical_y,
//   9 downstream_physical_x, 10 downstream_physical_y, 11 downstream_Q_jobs.
// Semaphores on EVERY participating core: 0 ready=0, 1 received=0, 2 valid=1.
// For singleton chains, rank=0, length=1, downstream_Q_jobs=0; coordinates unused.
// Key ranges (SDPA_RECIPE_KRANGE: causal, sliding window, chunked, windowed; recipe_key_range.hpp): singleton
// chains; args 3/4 are a range of positions in the heads' zigzag orders; 12 scalar Q offset, 13-15 the Q offset,
// cu_window_seqlens and page table addresses (0 when absent). Each Q chunk reads only its K range; the writer
// generates the edge chunks' masks.
//
// Host invariants:
// - Chains (length > 1): one head per core; one chain per head; positive, nonincreasing Q-job counts.
//   Singleton chains may hold several heads' jobs (more batch/heads than cores); each job reads its own head.
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
#ifdef SDPA_RECIPE_KRANGE
#include "recipe_key_range.hpp"
#endif

// DRAM read-barrier interval for the chain head's K/V fetch. Barriering every two
// tiles serialized the fetch and left multi-core chains 12-30% behind legacy SDPA;
// 16 tiles keeps enough reads in flight (measured 10x8192x8192 D128, full grid:
// legacy-numerics recipe Q256 2.16 -> 1.78 ms, Q128 4.12 -> 2.45 ms) and is neutral on 1-4 cores.
// Key ranges (no chain: every core reads) pass a smaller interval for many cores (host: run_recipe_segments).
#ifdef SDPA_RECIPE_READ_BARRIER_TILES
constexpr uint32_t reader_barrier_tiles = SDPA_RECIPE_READ_BARRIER_TILES;
#else
constexpr uint32_t reader_barrier_tiles = 16;
#endif
constexpr uint32_t kv_tiles = SDPA_K_CHUNK_TILES * SDPA_RECIPE_DHT;

template <uint32_t tile_bytes, bool transpose, typename Accessor>
FORCE_INLINE void read_kv_from_dram(const Noc& noc, const Accessor& tensor, uint32_t first_page, uint32_t write_ptr) {
    // Sequential source requests distribute traffic over the interleaved banks;
    // K scatters the tile grid in L1, without transposing individual tiles.
    for (uint32_t p = 0; p < kv_tiles; ++p) {
        const uint32_t dst_tile = transpose ? (p % SDPA_RECIPE_DHT) * SDPA_K_CHUNK_TILES + p / SDPA_RECIPE_DHT : p;
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
                const uint32_t dst_tile = transpose ? (p % SDPA_RECIPE_DHT) * SDPA_K_CHUNK_TILES + p / SDPA_RECIPE_DHT : p;
                zero_tile_padding<tile_bytes>(write_ptr + dst_tile * tile_bytes, rows);
            }
        }
    }
}

#ifdef SDPA_RECIPE_MASK
// Additive attn_mask [1|B, 1|H, Sq, Sk]: stream one Q chunk x K chunk of tiles, one Q tile row
// (SDPA_K_CHUNK_TILES tiles) at a time, so compute can apply each QK row group as it lands.
// The mask CB holds one or two compute row groups (SDPA_RECIPE_MASK_GROUP_ROWS rows each).
// Tiles outside the mask (Q rows past Sq, K columns past Sk) are zero: the recipe's pack-thread
// tail hook already stamps -inf on padded K columns, and padded Q rows are never written.
template <uint32_t q_tiles, typename Accessor>
FORCE_INLINE void read_mask_chunk(
    const Noc& noc, const Accessor& mask, CircularBuffer& cb, uint32_t head, uint32_t q_tile0, uint32_t k_tile0) {
    constexpr uint32_t bytes = get_tile_size(SDPA_RECIPE_MASK_CB);
    constexpr uint32_t mask_heads = SDPA_RECIPE_MASK_BCAST_HEADS ? 1 : SDPA_RECIPE_MASK_HEADS;
    const uint32_t batch = SDPA_RECIPE_MASK_BCAST_BATCH ? 0 : head / SDPA_RECIPE_MASK_HEADS;
    const uint32_t mask_head = SDPA_RECIPE_MASK_BCAST_HEADS ? 0 : head % SDPA_RECIPE_MASK_HEADS;
    const uint32_t base = (batch * mask_heads + mask_head) * SDPA_RECIPE_MASK_Q_TILES * SDPA_RECIPE_MASK_K_TILES;
    // Whole compute row groups: an odd chunk's last group gets a zero padding row.
    constexpr uint32_t rows = (q_tiles + SDPA_RECIPE_MASK_GROUP_ROWS - 1) / SDPA_RECIPE_MASK_GROUP_ROWS *
                              SDPA_RECIPE_MASK_GROUP_ROWS;
    for (uint32_t row = 0; row < rows; ++row) {
        const uint32_t q_tile = row < q_tiles ? q_tile0 + row : SDPA_RECIPE_MASK_Q_TILES;
        cb.reserve_back(SDPA_K_CHUNK_TILES);
        const uint32_t ptr = cb.get_write_ptr();
        bool zeroed = false;
        for (uint32_t col = 0; col < SDPA_K_CHUNK_TILES; ++col) {
            const uint32_t k_tile = k_tile0 + col;
            const CoreLocalMem<uint32_t> destination(ptr + col * bytes);
            if (q_tile < SDPA_RECIPE_MASK_Q_TILES && k_tile < SDPA_RECIPE_MASK_K_TILES) {
                noc.async_read(
                    mask, destination, bytes, {.page_id = base + q_tile * SDPA_RECIPE_MASK_K_TILES + k_tile}, {});
            } else {
                noc.async_write_zeros(destination, bytes);
                zeroed = true;
            }
        }
        if (zeroed) {
            noc.write_zeros_l1_barrier();
        }
        noc.async_read_barrier();
        cb.push_back(SDPA_K_CHUNK_TILES);
    }
}
#endif

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
#ifdef SDPA_JOINT
    constexpr auto jqa = TensorAccessorArgs<va.next_compile_time_args_offset()>();
    constexpr auto jka = TensorAccessorArgs<jqa.next_compile_time_args_offset()>();
    constexpr auto jva = TensorAccessorArgs<jka.next_compile_time_args_offset()>();
    const auto q = sequence_accessor<q_primary_rows, q_joint_rows, q_tiles * 32, SDPA_RECIPE_DHT>(
        TensorAccessor(qa, get_arg_val<uint32_t>(0)), TensorAccessor(jqa, get_arg_val<uint32_t>(12)));
    const auto k = sequence_accessor<kv_primary_rows, kv_joint_rows, SDPA_K_CHUNK_TILES * 32, SDPA_RECIPE_DHT>(
        TensorAccessor(ka, get_arg_val<uint32_t>(1)), TensorAccessor(jka, get_arg_val<uint32_t>(13)));
    const auto v = sequence_accessor<kv_primary_rows, kv_joint_rows, SDPA_K_CHUNK_TILES * 32, SDPA_RECIPE_DHT>(
        TensorAccessor(va, get_arg_val<uint32_t>(2)), TensorAccessor(jva, get_arg_val<uint32_t>(14)));
#else
    const auto q = sequence_accessor<q_primary_rows, q_tiles * 32, SDPA_RECIPE_DHT>(TensorAccessor(qa, get_arg_val<uint32_t>(0)));
    const auto k =
        sequence_accessor<kv_primary_rows, SDPA_K_CHUNK_TILES * 32, SDPA_RECIPE_DHT>(TensorAccessor(ka, get_arg_val<uint32_t>(1)));
    const auto v =
        sequence_accessor<kv_primary_rows, SDPA_K_CHUNK_TILES * 32, SDPA_RECIPE_DHT>(TensorAccessor(va, get_arg_val<uint32_t>(2)));
#endif
#ifdef SDPA_RECIPE_MASK
#ifdef SDPA_JOINT
#error "SDPA recipe masks are dense-only"
#endif
    constexpr auto ma = TensorAccessorArgs<va.next_compile_time_args_offset()>();
    const auto mask = TensorAccessor(ma, get_arg_val<uint32_t>(12));
    CircularBuffer mcb(SDPA_RECIPE_MASK_CB);
#endif
#ifdef SDPA_RECIPE_KRANGE
#ifdef SDPA_JOINT
#error "SDPA recipe key ranges are dense-only"
#endif
    // Optional device tensors, in this order: the Q offset, cu_window_seqlens, the page table.
    constexpr uint32_t kr_cta0 = va.next_compile_time_args_offset();
#ifdef SDPA_RECIPE_Q_OFFSET_PAGE
    constexpr auto offset_args = TensorAccessorArgs<kr_cta0>();
    constexpr uint32_t kr_cta1 = offset_args.next_compile_time_args_offset();
#else
    constexpr uint32_t kr_cta1 = kr_cta0;
#endif
#ifdef SDPA_RECIPE_SEGMENTS_PAGE
    constexpr auto segment_args = TensorAccessorArgs<kr_cta1>();
    constexpr uint32_t kr_cta2 = segment_args.next_compile_time_args_offset();
#else
    constexpr uint32_t kr_cta2 = kr_cta1;
#endif
#ifdef SDPA_RECIPE_PAGE_TABLE_PAGE
    constexpr auto page_table_args = TensorAccessorArgs<kr_cta2>();
#endif
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
    ASSERT(chain_length == 1 || (first_job + jobs - 1) / queries_per_head == head);
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

#ifdef SDPA_RECIPE_KRANGE
    // Runtime args 12-15: the scalar Q offset, then the Q offset, cu_window_seqlens and page table addresses. The
    // tensors land in this kernel's half of the scratch CB (recipe_key_range.hpp: RecipeScratch).
    RecipeKeyRange keys{.q_offset = get_arg_val<uint32_t>(12), .k_rows = kv_primary_rows};
    [[maybe_unused]] const uint32_t scratch = CircularBuffer(SDPA_RECIPE_SCRATCH_CB).get_write_ptr();
#ifdef SDPA_RECIPE_Q_OFFSET_PAGE
    keys.q_offset = recipe_read_index_page(
        noc, TensorAccessor(offset_args, get_arg_val<uint32_t>(13)), 0, SDPA_RECIPE_Q_OFFSET_PAGE, scratch);
#endif
#ifdef SDPA_RECIPE_SEGMENTS_PAGE
    recipe_read_index_page(
        noc,
        TensorAccessor(segment_args, get_arg_val<uint32_t>(14)),
        0,
        SDPA_RECIPE_SEGMENTS_PAGE,
        scratch + RecipeScratch::Segments);
    keys.segments = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scratch + RecipeScratch::Segments);
#endif
#ifdef SDPA_RECIPE_PAGE_TABLE_PAGE
    const auto page_table = TensorAccessor(page_table_args, get_arg_val<uint32_t>(15));
    uint32_t page_table_batch = UINT32_MAX, block = 0;
#endif
#endif
    for (uint32_t qi = 0; qi < jobs; ++qi) {
#ifdef SDPA_RECIPE_KRANGE
        // Position first_job + qi of the heads' zigzag orders (host: run_recipe_segments).
        const uint32_t z = first_job + qi;
        const uint32_t job = z - z % queries_per_head + recipe_zigzag_job(z % queries_per_head, queries_per_head);
#else
        const uint32_t job = first_job + qi;
#endif
        const uint32_t job_head = job / queries_per_head;
#ifdef SDPA_RECIPE_Q_PER_KV_HEAD
        uint32_t kv_head = job_head / SDPA_RECIPE_Q_PER_KV_HEAD;
#else
        uint32_t kv_head = job_head;
#endif
#ifdef SDPA_RECIPE_PAGE_TABLE_PAGE
        // Chunked prefill with one cache block per sequence: K/V head = block * KV heads + head within the batch.
        // A multi-block page table would translate each K tile row in read_kv_from_dram instead.
        const uint32_t batch = job_head / SDPA_RECIPE_Q_HEADS;
        if (batch != page_table_batch) {
            block = recipe_read_index_page(
                noc, page_table, batch, SDPA_RECIPE_PAGE_TABLE_PAGE, scratch + RecipeScratch::PageTable);
            page_table_batch = batch;
        }
        kv_head = block * SDPA_RECIPE_KV_HEADS + kv_head % SDPA_RECIPE_KV_HEADS;
#endif
        const uint32_t kvbase = kv_head * k_chunks * kv_tiles;
        const uint32_t qbase = job * q_tiles * SDPA_RECIPE_DHT;
        // Paired recipes pad an odd chunk with SDPA_RECIPE_Q_PAD_TILES zero rows (host: recipe_compute_q_tiles).
        constexpr uint32_t q_push_tiles = (q_tiles + SDPA_RECIPE_Q_PAD_TILES) * SDPA_RECIPE_DHT;
        qcb.reserve_back(q_push_tiles);
        const uint32_t qptr = qcb.get_write_ptr();
        bool zeroed = SDPA_RECIPE_Q_PAD_TILES > 0;
        for (uint32_t p = 0; p < q_tiles * SDPA_RECIPE_DHT; ++p) {
            const CoreLocalMem<uint32_t> destination(qptr + p * qbytes);
            if (!q.visit(qbase + p, [&](const auto& source, uint32_t page) {
                    noc.async_read(source, destination, qbytes, {.page_id = page}, {});
                })) {
                noc.async_write_zeros(destination, qbytes);
                zeroed = true;
            }
        }
        for (uint32_t p = q_tiles * SDPA_RECIPE_DHT; p < q_push_tiles; ++p) {
            noc.async_write_zeros(CoreLocalMem<uint32_t>(qptr + p * qbytes), qbytes);
        }
        if (zeroed) {
            noc.write_zeros_l1_barrier();
        }
        noc.async_read_barrier();
        if constexpr (decltype(q)::has_partial_rows) {
            for (uint32_t p = 0; p < q_tiles * SDPA_RECIPE_DHT; ++p) {
                const uint32_t rows = q.valid_rows(qbase + p);
                if (rows > 0 && rows < 32) {
                    zero_tile_padding<qbytes>(qptr + p * qbytes, rows);
                }
            }
        }
        qcb.push_back(q_push_tiles);

#ifdef SDPA_RECIPE_KRANGE
        // K chunks of this Q chunk (recipe_key_range.hpp; the writer sends compute the range and the edge masks).
        // The chain is off: every core reads its own K/V.
        const uint32_t q_row0 = (job % queries_per_head) * q_tiles * 32;
        const uint32_t q_row_end = q_row0 + q_tiles * 32 < q_primary_rows ? q_row0 + q_tiles * 32 : q_primary_rows;
        const RecipeChunkRange range = keys.chunks(q_row0, q_row_end, SDPA_K_CHUNK_TILES * 32, k_chunks);
        for (uint32_t ki = range.first; ki < range.end; ++ki) {
            const uint32_t first_kv_page = kvbase + ki * kv_tiles;
            kcb.reserve_back(kv_tiles);
            read_kv_from_dram<kbytes, true>(noc, k, first_kv_page, kcb.get_write_ptr());
            kcb.push_back(kv_tiles);
            vcb.reserve_back(kv_tiles);
            read_kv_from_dram<vbytes, false>(noc, v, first_kv_page, vcb.get_write_ptr());
            vcb.push_back(kv_tiles);
        }
#else
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
#ifdef SDPA_RECIPE_MASK
            // Mask after K, before V: QK (phase one) consumes it; PV (phase two) needs only V.
            read_mask_chunk<q_tiles>(
                noc, mask, mcb, job_head, (job % queries_per_head) * q_tiles, ki * SDPA_K_CHUNK_TILES);
#endif

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
#endif
    }
    noc.async_write_barrier();
}
