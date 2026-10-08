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
// Key ranges (SDPA_RECIPE_KRANGE: causal, sliding window, chunked, windowed; recipe_key_range.hpp): no chain link;
// arg 3 is the core's index in the snake deal of all heads' Q chunks, 4 its Q chunk count, 7-10 its K/V sharing
// partners (SDPA_RECIPE_KV_SHARE: RecipeKvShare, semaphore 3 too), 12 the scalar Q offset, 13-15 the Q offset,
// cu_window_seqlens and page table addresses (0 when absent); with Q slabs (ring-distributed SDPA) 16-17 the slabs'
// first Q chunks. Each Q chunk reads only its K range; the writer generates the edge chunks' masks.
// Attention sink (SDPA_RECIPE_SINK_CB): runtime arg SDPA_RECIPE_SINK_ARG is the sink tensor's address; each Q chunk
// gets a page holding its head's sink logit.
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
// K/V sharing (RecipeKvShare): one core per head and round reads the shared chunks, so it keeps more reads in flight
// (measured 10 heads x 8192^2 causal: STANDARD 1.56 -> 1.49 ms, FAST BFP8 0.96 -> 0.88 ms; 16 tiles on every read
// slows sliding windows and GQA instead).
#ifdef SDPA_RECIPE_SHARED_READ_BARRIER_TILES
constexpr uint32_t shared_barrier_tiles = SDPA_RECIPE_SHARED_READ_BARRIER_TILES;
#else
constexpr uint32_t shared_barrier_tiles = reader_barrier_tiles;
#endif
constexpr uint32_t kv_tiles = SDPA_K_CHUNK_TILES * SDPA_RECIPE_DHT;
#ifdef SDPA_RECIPE_V_DHT
constexpr uint32_t v_tiles = SDPA_K_CHUNK_TILES * SDPA_RECIPE_V_DHT;
#else
constexpr uint32_t v_tiles = kv_tiles;
#endif

template <uint32_t tile_bytes, bool transpose, typename Accessor>
FORCE_INLINE void read_kv_from_dram(
    const Noc& noc,
    const Accessor& tensor,
    uint32_t first_page,
    uint32_t write_ptr,
    uint32_t barrier_tiles = reader_barrier_tiles) {
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
        if ((p + 1) % barrier_tiles == 0) {
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

#ifdef SDPA_RECIPE_KV_ROWS
// One K chunk of K or V, `width` tiles of K tile row t starting at page row_page(t) (paged K/V: through the page table;
// MLA: V narrower than its source rows, or read from K). Tile rows past the SDPA_RECIPE_KV_ROWS keys are zero, as is
// the padding of a partial last row.
template <uint32_t tile_bytes, bool transpose, uint32_t width, typename Accessor, typename RowPage>
FORCE_INLINE void read_kv_rows(
    const Noc& noc,
    const Accessor& tensor,
    uint32_t row0,
    const RowPage& row_page,
    uint32_t write_ptr,
    uint32_t barrier_tiles = reader_barrier_tiles) {
    constexpr uint32_t row_tiles = (SDPA_RECIPE_KV_ROWS + 31) / 32;
    auto tile = [](uint32_t r, uint32_t c) { return transpose ? c * SDPA_K_CHUNK_TILES + r : r * width + c; };
    uint32_t issued = 0;
    bool zeroed = false;
    for (uint32_t r = 0; r < SDPA_K_CHUNK_TILES; ++r) {
        const uint32_t t = row0 + r;
        if (t >= row_tiles) {
            for (uint32_t c = 0; c < width; ++c) {
                noc.async_write_zeros(CoreLocalMem<uint32_t>(write_ptr + tile(r, c) * tile_bytes), tile_bytes);
            }
            zeroed = true;
            continue;
        }
        const uint32_t page = row_page(t);
        for (uint32_t c = 0; c < width; ++c) {
            noc.async_read(
                tensor,
                CoreLocalMem<uint32_t>(write_ptr + tile(r, c) * tile_bytes),
                tile_bytes,
                {.page_id = page + c},
                {});
            if (++issued % barrier_tiles == 0) {
                noc.async_read_barrier();
            }
        }
    }
    if (zeroed) {
        noc.write_zeros_l1_barrier();
    }
    noc.async_read_barrier();
    if constexpr (SDPA_RECIPE_KV_ROWS % 32 != 0) {
        if (row0 < row_tiles && row_tiles <= row0 + SDPA_K_CHUNK_TILES) {
            for (uint32_t c = 0; c < width; ++c) {
                zero_tile_padding<tile_bytes>(
                    write_ptr + tile(row_tiles - 1 - row0, c) * tile_bytes, SDPA_RECIPE_KV_ROWS % 32);
            }
        }
    }
}
#endif

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

#ifdef SDPA_RECIPE_KV_SHARE
// K/V sharing between the cores running one head's Q chunks in the same snake round. In round r, deal position p runs
// Q chunk j and position p + heads (downstream) chunk j - 1 of the same head; position p's core is core (r even) or
// cores - 1 - core (r odd), so a core's two partners are cores core -/+ heads (runtime args 7-10), downstream the
// higher one in even rounds. A full K chunk of both this core's and upstream's Q chunk comes from upstream, other full
// chunks are read here (all of them if this core leads the round: p < heads), and full chunks of both this core's and
// downstream's Q chunk go downstream; both ends process their full chunks in ascending order. (Both ends of the key
// interval rise with the row, so neighbouring Q chunks share most of their full chunks.) Store and forward: the
// receiver passes its CB write pointer in the sender's ready semaphore (one per direction, so a partner already in the
// next round cannot be mistaken for this one), then waits for the received semaphore.
struct RecipeKvShare {
    static constexpr uint32_t ready_from_lower = 0, received = 1, valid = 2, ready_from_higher = 3;
    // Full chunks of upstream's and downstream's Q chunks (empty without that partner).
    uint32_t up_begin = 0, up_end = 0, down_begin = 0, down_end = 0;
    uint32_t up_x = 0, up_y = 0, down_x = 0, down_y = 0;
    uint32_t ready = 0;  // semaphore id: the upstream core's for this core's ready, this core's for downstream's

    template <typename ChunksOf>
    RecipeKvShare(uint32_t core, uint32_t round, uint32_t total_jobs, uint32_t job, const ChunksOf& chunks_of) {
        constexpr uint32_t cores = SDPA_RECIPE_CORES, heads = SDPA_RECIPE_BATCH_HEADS;
        const uint32_t p = round % 2 == 0 ? core : cores - 1 - core;
        const uint32_t active = total_jobs - round * cores < cores ? total_jobs - round * cores : cores;
        const bool lower_is_up = round % 2 == 0;
        const uint32_t lower_x = get_arg_val<uint32_t>(7), lower_y = get_arg_val<uint32_t>(8);
        const uint32_t higher_x = get_arg_val<uint32_t>(9), higher_y = get_arg_val<uint32_t>(10);
        up_x = lower_is_up ? lower_x : higher_x;
        up_y = lower_is_up ? lower_y : higher_y;
        down_x = lower_is_up ? higher_x : lower_x;
        down_y = lower_is_up ? higher_y : lower_y;
        // This core is on upstream's higher side in even rounds, and downstream on this core's.
        ready = lower_is_up ? ready_from_higher : ready_from_lower;
        // Upstream runs this head's next Q chunk, downstream its previous one (recipe_snake_job: entries s -/+ heads).
        if (p >= heads) {
            const RecipeChunkRange up = chunks_of(job + 1);
            up_begin = up.full_begin;
            up_end = up.full_end;
        }
        if (p + heads < active) {
            const RecipeChunkRange down = chunks_of(job - 1);
            down_begin = down.full_begin;
            down_end = down.full_end;
        }
    }

    bool receives(uint32_t k) const { return k >= up_begin && k < up_end; }
    bool forwards(uint32_t k) const { return k >= down_begin && k < down_end; }

    void receive(const Noc& noc, uint32_t address) const {
        Semaphore<> done(received);
        done.set(0);
        noc.inline_dw_write<NocOptions::INLINE_L1>(
            UnicastEndpoint{}, address, {.noc_x = up_x, .noc_y = up_y, .addr = get_semaphore(ready)});
        done.wait(1);
    }

    void forward(const Noc& noc, uint32_t address, uint32_t bytes) const {
        Semaphore<> posted(ready);
        posted.wait_min(1);
        const uint32_t destination = posted.value();
        posted.set(0);
        noc.async_write(
            CoreLocalMem<uint32_t>(address),
            UnicastEndpoint{},
            bytes,
            {.offset_bytes = 0},
            {.noc_x = down_x, .noc_y = down_y, .addr = destination});
        noc.async_writes_flushed();
        Semaphore<>(valid).relay_unicast(noc, Semaphore<>(received), down_x, down_y);
    }
};
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
#ifdef SDPA_RECIPE_V_IS_K
    // MLA without a V tensor: V is K's first SDPA_RECIPE_V_DHT tile columns (runtime arg 2 is K's address).
    constexpr auto va = ka;
#else
    constexpr auto va = TensorAccessorArgs<ka.next_compile_time_args_offset()>();
#endif
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
#ifdef SDPA_RECIPE_KV_ROWS
    const auto k_tensor = TensorAccessor(ka, get_arg_val<uint32_t>(1));
    const auto v_tensor = TensorAccessor(va, get_arg_val<uint32_t>(2));
#endif
#ifdef SDPA_RECIPE_SINK_CB
    const auto sink =
        TensorAccessor(TensorAccessorArgs<SDPA_RECIPE_SINK_CTA>(), get_arg_val<uint32_t>(SDPA_RECIPE_SINK_ARG));
    CircularBuffer scb(SDPA_RECIPE_SINK_CB);
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
    // The current batch's page-table row: its cache blocks in sequence order, after both scratch halves.
    const auto page_table = TensorAccessor(page_table_args, get_arg_val<uint32_t>(15));
    uint32_t page_table_batch = UINT32_MAX;
    const auto* blocks = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scratch + SDPA_RECIPE_PAGE_TABLE_OFFSET);
#endif
#ifdef SDPA_RECIPE_Q_SLAB_JOBS
    const RecipeQSlabs slabs{{get_arg_val<uint32_t>(16), get_arg_val<uint32_t>(17)}};
#endif
#endif
    for (uint32_t qi = 0; qi < jobs; ++qi) {
#ifdef SDPA_RECIPE_KRANGE
        // Runtime arg 3 is this core's index in the snake deal (recipe_key_range.hpp: recipe_snake_job).
        const uint32_t job =
            recipe_snake_job(first_job, qi, SDPA_RECIPE_CORES, SDPA_RECIPE_BATCH_HEADS, queries_per_head);
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
        // Chunked prefill: K/V tile row t of the sequence is row t % block of cache block blocks[t / block], and
        // kv_head the head within the batch.
        const uint32_t batch = job_head / SDPA_RECIPE_Q_HEADS;
        if (batch != page_table_batch) {
            recipe_read_index_page(
                noc, page_table, batch, SDPA_RECIPE_PAGE_TABLE_PAGE, scratch + SDPA_RECIPE_PAGE_TABLE_OFFSET);
            page_table_batch = batch;
        }
        kv_head %= SDPA_RECIPE_KV_HEADS;
        constexpr uint32_t block_tiles = SDPA_RECIPE_PAGE_BLOCK_TILES;
        auto kv_row = [&](uint32_t t) {
            return (blocks[t / block_tiles] * SDPA_RECIPE_KV_HEADS + kv_head) * block_tiles + t % block_tiles;
        };
#elif defined(SDPA_RECIPE_KV_ROWS)
        auto kv_row = [&](uint32_t t) { return kv_head * ((SDPA_RECIPE_KV_ROWS + 31) / 32) + t; };
#endif
#ifdef SDPA_RECIPE_KV_ROWS
        auto k_row = [&](uint32_t t) { return kv_row(t) * SDPA_RECIPE_DHT; };
        auto v_row = [&](uint32_t t) { return kv_row(t) * SDPA_RECIPE_V_SRC_DHT; };
#endif
        const uint32_t kvbase = kv_head * k_chunks * kv_tiles;
#ifdef SDPA_RECIPE_Q_SLAB_JOBS
        // The job's chunk of the whole sequence Q holds (q_primary_rows rows per head).
        const uint32_t q_chunk = slabs.chunk(job % queries_per_head);
        const uint32_t qbase = (job_head * (q_primary_rows / (q_tiles * 32)) + q_chunk) * q_tiles * SDPA_RECIPE_DHT;
#else
        const uint32_t qbase = job * q_tiles * SDPA_RECIPE_DHT;
#endif
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
#ifdef SDPA_RECIPE_SINK_CB
        // The head's sink logit: the first value of its [1, H, 1, 1] tile (compute reads word 0).
        scb.reserve_back(1);
        noc.async_read(
            sink, CoreLocalMem<uint32_t>(scb.get_write_ptr()), 64, {.page_id = job_head % SDPA_RECIPE_SINK_HEADS}, {});
        noc.async_read_barrier();
        scb.push_back(1);
#endif

#ifdef SDPA_RECIPE_KRANGE
        // K chunks of this Q chunk (recipe_key_range.hpp; the writer sends compute the range and the edge masks):
        // the edge chunks, read here, then the full ones. With SDPA_RECIPE_KV_SHARE the full ones come from the
        // core one Q chunk later in this head (upstream) unless this core leads its round, and go on to the core one
        // Q chunk earlier (downstream) for as many as it needs (RecipeKvShare).
        // Out of line: K/V sharing also asks for the neighbouring Q chunks' ranges.
        auto chunks_of = [&](uint32_t q_job) __attribute__((noinline)) {
#ifdef SDPA_RECIPE_Q_SLAB_JOBS
            const uint32_t row0 = slabs.chunk(q_job % queries_per_head) * q_tiles * 32;
#else
            const uint32_t row0 = (q_job % queries_per_head) * q_tiles * 32;
#endif
            const uint32_t row_end = row0 + q_tiles * 32 < q_primary_rows ? row0 + q_tiles * 32 : q_primary_rows;
            return keys.chunks(row0, row_end, SDPA_K_CHUNK_TILES * 32, k_chunks);
        };
        const RecipeChunkRange range = chunks_of(job);
#ifdef SDPA_RECIPE_KV_SHARE
        const RecipeKvShare share(first_job, qi, SDPA_RECIPE_BATCH_HEADS * queries_per_head, job, chunks_of);
#endif
        for (uint32_t i = 0; i < range.count(); ++i) {
            const uint32_t ki = range.at(i);
            [[maybe_unused]] const uint32_t first_kv_page = kvbase + ki * kv_tiles;
#ifdef SDPA_RECIPE_KV_SHARE
            const bool shared = i >= range.edges();
            const bool receive = shared && share.receives(ki);
            const bool forward = shared && share.forwards(ki);
            const uint32_t barrier = shared ? shared_barrier_tiles : reader_barrier_tiles;
#else
            constexpr bool receive = false, forward = false;
            constexpr uint32_t barrier = reader_barrier_tiles;
#endif
            kcb.reserve_back(kv_tiles);
            const uint32_t kptr = kcb.get_write_ptr();
            if (receive) {
#ifdef SDPA_RECIPE_KV_SHARE
                share.receive(noc, kptr);
#endif
            } else {
#ifdef SDPA_RECIPE_KV_ROWS
                read_kv_rows<kbytes, true, SDPA_RECIPE_DHT>(
                    noc, k_tensor, ki * SDPA_K_CHUNK_TILES, k_row, kptr, barrier);
#else
                read_kv_from_dram<kbytes, true>(noc, k, first_kv_page, kptr, barrier);
#endif
            }
            // Compute may start on the chunk while it is forwarded: this reader overwrites the slot only later.
            kcb.push_back(kv_tiles);
            if (forward) {
#ifdef SDPA_RECIPE_KV_SHARE
                share.forward(noc, kptr, kv_tiles * kbytes);
#endif
            }
            vcb.reserve_back(v_tiles);
            const uint32_t vptr = vcb.get_write_ptr();
            if (receive) {
#ifdef SDPA_RECIPE_KV_SHARE
                share.receive(noc, vptr);
#endif
            } else {
#ifdef SDPA_RECIPE_KV_ROWS
                read_kv_rows<vbytes, false, SDPA_RECIPE_V_DHT>(
                    noc, v_tensor, ki * SDPA_K_CHUNK_TILES, v_row, vptr, barrier);
#else
                read_kv_from_dram<vbytes, false>(noc, v, first_kv_page, vptr, barrier);
#endif
            }
            // Compute may start on the chunk while it is forwarded: this reader overwrites the slot only later.
            vcb.push_back(v_tiles);
            if (forward) {
#ifdef SDPA_RECIPE_KV_SHARE
                share.forward(noc, vptr, v_tiles * vbytes);
#endif
            }
        }
#else
        const bool receive = link.should_receive(head);
        const bool forward = link.should_forward(head, qi);
        for (uint32_t ki = 0; ki < k_chunks; ++ki) {
            [[maybe_unused]] const uint32_t first_kv_page = kvbase + ki * kv_tiles;

            kcb.reserve_back(kv_tiles);
            const uint32_t kptr = kcb.get_write_ptr();
            if (receive) {
                link.receive(noc);
            } else {
#ifdef SDPA_RECIPE_KV_ROWS
                read_kv_rows<kbytes, true, SDPA_RECIPE_DHT>(noc, k_tensor, ki * SDPA_K_CHUNK_TILES, k_row, kptr);
#else
                read_kv_from_dram<kbytes, true>(noc, k, first_kv_page, kptr);
#endif
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

            vcb.reserve_back(v_tiles);
            const uint32_t vptr = vcb.get_write_ptr();
            if (receive) {
                link.receive(noc);
            } else {
#ifdef SDPA_RECIPE_KV_ROWS
                read_kv_rows<vbytes, false, SDPA_RECIPE_V_DHT>(noc, v_tensor, ki * SDPA_K_CHUNK_TILES, v_row, vptr);
#else
                read_kv_from_dram<vbytes, false>(noc, v, first_kv_page, vptr);
#endif
            }
            if (forward) {
                link.forward(noc, vptr, v_tiles, vbytes);
            }
            vcb.push_back(v_tiles);
        }
#endif
    }
    noc.async_write_barrier();
}
