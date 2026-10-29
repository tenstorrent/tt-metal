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
// Key ranges (krange: causal, sliding window, chunked, windowed; recipe_key_range.hpp): no chain link; arg 3 is the
// core's index in the snake deal of all heads' Q chunks, 4 its Q chunk count, 7-10 its K/V sharing partners (kv_share:
// RecipeKvShare, semaphore 3 too), 12 the scalar Q offset, 13-15 the Q offset, cu_window_seqlens and page table
// addresses (0 when absent); with Q slabs (ring-distributed SDPA) 16-17 the slabs' first Q chunks. Each Q chunk reads
// only its K range; the writer generates the edge chunks' masks.
// Attention sink (sink_cb): runtime arg sink_arg is the sink tensor's address; each Q chunk gets a page holding its
// head's sink logit.
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

#include <type_traits>

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
#include "recipe_key_range.hpp"

namespace recipe_reader {
// Joint SDPA: three joint accessors follow the primary Q/K/V ones, and the joint addresses runtime args 12-14.
constexpr bool has_joint = get_named_compile_time_arg_val("joint") != 0;
constexpr uint32_t k_chunk_tiles = get_named_compile_time_arg_val("k_chunk_tiles");
constexpr uint32_t d_tiles = get_named_compile_time_arg_val("d_tiles");
// Zero Q rows padding a paired recipe's odd chunk (host: recipe_compute_q_tiles).
constexpr uint32_t q_pad_tiles = get_named_compile_time_arg_val("q_pad_tiles");
// Grouped-query attention: query heads per KV head (1: one KV head per query head).
constexpr uint32_t q_per_kv_head = get_named_compile_time_arg_val("q_per_kv_head");
// K/V tile rows located one at a time (read_kv_rows; 0: whole chunks): the logical K length. Paged K/V, and MLA's V
// (v_d_tiles tile columns of rows v_src_d_tiles wide; v_is_k: K's first columns, no V accessor).
constexpr uint32_t kv_rows = get_named_compile_time_arg_val("kv_rows");
constexpr uint32_t v_d_tiles = get_named_compile_time_arg_val("v_d_tiles");
constexpr uint32_t v_src_d_tiles = get_named_compile_time_arg_val("v_src_d_tiles");
constexpr bool v_is_k = get_named_compile_time_arg_val("v_is_k") != 0;
// Additive attn_mask (dense only): its accessor follows the primary ones, its address is runtime arg 12.
constexpr bool has_mask = get_named_compile_time_arg_val("mask") != 0;
constexpr uint32_t mask_cb = get_named_compile_time_arg_val("mask_cb");
constexpr uint32_t mask_group_rows = get_named_compile_time_arg_val("mask_group_rows");
constexpr uint32_t mask_q_tiles = get_named_compile_time_arg_val("mask_q_tiles");
constexpr uint32_t mask_k_tiles = get_named_compile_time_arg_val("mask_k_tiles");
constexpr uint32_t mask_heads = get_named_compile_time_arg_val("mask_heads");
constexpr bool mask_bcast_batch = get_named_compile_time_arg_val("mask_bcast_batch") != 0;
constexpr bool mask_bcast_heads = get_named_compile_time_arg_val("mask_bcast_heads") != 0;
// Key ranges (recipe_key_range.hpp; dense only): the K range model, the snake deal over cores x batch_heads, K/V
// sharing, Q slabs (q_slab_jobs chunks each; 0: none) and the optional device tensors' page sizes (0: absent).
constexpr bool krange = get_named_compile_time_arg_val("krange") != 0;
constexpr bool causal = get_named_compile_time_arg_val("causal") != 0;
constexpr uint32_t window = get_named_compile_time_arg_val("window");
constexpr uint32_t segments = get_named_compile_time_arg_val("segments");
constexpr uint32_t scratch_cb = get_named_compile_time_arg_val("scratch_cb");
constexpr uint32_t cores = get_named_compile_time_arg_val("cores");
constexpr uint32_t batch_heads = get_named_compile_time_arg_val("batch_heads");
constexpr bool kv_share = get_named_compile_time_arg_val("kv_share") != 0;
constexpr uint32_t q_slab_jobs = get_named_compile_time_arg_val("q_slab_jobs");
constexpr bool q_slabs = q_slab_jobs != 0;
constexpr uint32_t q_offset_page = get_named_compile_time_arg_val("q_offset_page");
constexpr uint32_t segments_page = get_named_compile_time_arg_val("segments_page");
constexpr uint32_t page_table_page = get_named_compile_time_arg_val("page_table_page");
constexpr bool has_page_table = krange && page_table_page != 0;
// Chunked prefill's paged cache: heads, block size and the page-table row's offset in the scratch CB.
constexpr uint32_t q_heads = get_named_compile_time_arg_val("q_heads");
constexpr uint32_t kv_heads = get_named_compile_time_arg_val("kv_heads");
constexpr uint32_t page_block_tiles = get_named_compile_time_arg_val("page_block_tiles");
constexpr uint32_t page_table_offset = get_named_compile_time_arg_val("page_table_offset");
// DRAM read barrier interval: every 16 K/V tiles keeps enough reads in flight for the chain head (every 2 tiles
// serialized the fetch). Key ranges (no chain: every core reads) pass a smaller interval for many cores; with K/V
// sharing one core per head and round reads the shared chunks, so it keeps more reads in flight (measured 10 heads x
// 8192^2 causal: STANDARD 1.56 -> 1.49 ms, FAST BFP8 0.96 -> 0.88 ms; 16 tiles on every read slows sliding windows and
// GQA instead).
constexpr uint32_t read_barrier_tiles = get_named_compile_time_arg_val("read_barrier_tiles");
constexpr uint32_t shared_read_barrier_tiles = get_named_compile_time_arg_val("shared_read_barrier_tiles");
// Attention sink (0: none): its page CB, the sink tensor's heads, accessor args offset and address runtime arg.
constexpr uint32_t sink_cb = get_named_compile_time_arg_val("sink_cb");
constexpr uint32_t sink_heads = get_named_compile_time_arg_val("sink_heads");
constexpr uint32_t sink_cta = get_named_compile_time_arg_val("sink_cta");
constexpr uint32_t sink_arg = get_named_compile_time_arg_val("sink_arg");
constexpr bool has_sink = sink_cb != 0;
static_assert(!(has_mask && has_joint), "SDPA recipe masks are dense-only");
static_assert(!(krange && has_joint), "SDPA recipe key ranges are dense-only");
}  // namespace recipe_reader
using namespace recipe_reader;

constexpr uint32_t reader_barrier_tiles = read_barrier_tiles;
constexpr uint32_t shared_barrier_tiles = shared_read_barrier_tiles;
constexpr uint32_t kv_tiles = k_chunk_tiles * d_tiles;
constexpr uint32_t v_tiles = k_chunk_tiles * v_d_tiles;

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
        const uint32_t dst_tile = transpose ? (p % d_tiles) * k_chunk_tiles + p / d_tiles : p;
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
                const uint32_t dst_tile = transpose ? (p % d_tiles) * k_chunk_tiles + p / d_tiles : p;
                zero_tile_padding<tile_bytes>(write_ptr + dst_tile * tile_bytes, rows);
            }
        }
    }
}

// One K chunk of K or V, `width` tiles of K tile row t starting at page row_page(t) (paged K/V: through the page table;
// MLA: V narrower than its source rows, or read from K). Tile rows past the kv_rows keys are zero, as is the padding of
// a partial last row.
template <uint32_t tile_bytes, bool transpose, uint32_t width, typename Accessor, typename RowPage>
FORCE_INLINE void read_kv_rows(
    const Noc& noc,
    const Accessor& tensor,
    uint32_t row0,
    const RowPage& row_page,
    uint32_t write_ptr,
    uint32_t barrier_tiles = reader_barrier_tiles) {
    constexpr uint32_t row_tiles = (kv_rows + 31) / 32;
    auto tile = [](uint32_t r, uint32_t c) { return transpose ? c * k_chunk_tiles + r : r * width + c; };
    uint32_t issued = 0;
    bool zeroed = false;
    for (uint32_t r = 0; r < k_chunk_tiles; ++r) {
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
    if constexpr (kv_rows % 32 != 0) {
        if (row0 < row_tiles && row_tiles <= row0 + k_chunk_tiles) {
            for (uint32_t c = 0; c < width; ++c) {
                zero_tile_padding<tile_bytes>(write_ptr + tile(row_tiles - 1 - row0, c) * tile_bytes, kv_rows % 32);
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
        constexpr uint32_t heads = batch_heads;
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

// The compile-time args offset after an optional accessor's (args at args_offset when present).
template <bool present, uint32_t args_offset>
constexpr uint32_t next_args_offset() {
    if constexpr (present) {
        return TensorAccessorArgs<args_offset>().next_compile_time_args_offset();
    } else {
        return args_offset;
    }
}

// An optional tensor's accessor (args at args_offset, address in runtime arg address_arg); 0 when absent.
template <bool present, uint32_t args_offset>
FORCE_INLINE auto recipe_accessor(uint32_t address_arg) {
    if constexpr (present) {
        return TensorAccessor(TensorAccessorArgs<args_offset>(), get_arg_val<uint32_t>(address_arg));
    } else {
        return 0;
    }
}

// Without K/V sharing: every chunk is read here.
struct RecipeNoKvShare {
    template <typename ChunksOf>
    RecipeNoKvShare(uint32_t, uint32_t, uint32_t, uint32_t, const ChunksOf&) {}
    bool receives(uint32_t) const { return false; }
    bool forwards(uint32_t) const { return false; }
    void receive(const Noc&, uint32_t) const {}
    void forward(const Noc&, uint32_t, uint32_t) const {}
};

// The head's sink logit: the first value of its [1, H, 1, 1] tile (compute reads word 0).
template <typename Accessor>
FORCE_INLINE void read_sink_page(const Noc& noc, const Accessor& sink, CircularBuffer& cb, uint32_t head) {
    cb.reserve_back(1);
    noc.async_read(sink, CoreLocalMem<uint32_t>(cb.get_write_ptr()), 64, {.page_id = head % sink_heads}, {});
    noc.async_read_barrier();
    cb.push_back(1);
}

// The kernel, a template so that the build's unused paths (if constexpr on the named args above) are not instantiated.
template <uint32_t q_tiles, uint32_t k_chunks, uint32_t queries_per_head>
FORCE_INLINE void read_recipe() {
    static_assert(q_tiles > 0);
    static_assert(k_chunks > 0 && queries_per_head > 0);
    constexpr uint32_t q_primary_rows = get_compile_time_arg_val(3);
    constexpr uint32_t q_joint_rows = get_compile_time_arg_val(4);
    constexpr uint32_t kv_primary_rows = get_compile_time_arg_val(5);
    constexpr uint32_t kv_joint_rows = get_compile_time_arg_val(6);
    constexpr auto qa = TensorAccessorArgs<7>();
    constexpr auto ka = TensorAccessorArgs<qa.next_compile_time_args_offset()>();
    // MLA without a V tensor (v_is_k): V is K's first v_d_tiles tile columns (runtime arg 2 is K's address).
    constexpr uint32_t v_args = v_is_k ? qa.next_compile_time_args_offset() : ka.next_compile_time_args_offset();
    constexpr auto va = TensorAccessorArgs<v_args>();
    constexpr uint32_t after_primary = va.next_compile_time_args_offset();
    const auto q = recipe_sequence<q_primary_rows, q_joint_rows, q_tiles * 32, 7, after_primary, 0>(0);
    const auto k = recipe_sequence<
        kv_primary_rows,
        kv_joint_rows,
        k_chunk_tiles * 32,
        qa.next_compile_time_args_offset(),
        after_primary,
        1>(1);
    const auto v = recipe_sequence<kv_primary_rows, kv_joint_rows, k_chunk_tiles * 32, v_args, after_primary, 2>(2);
    const auto k_tensor = TensorAccessor(ka, get_arg_val<uint32_t>(1));
    const auto v_tensor = TensorAccessor(va, get_arg_val<uint32_t>(2));
    const auto sink = recipe_accessor<has_sink, sink_cta>(sink_arg);
    CircularBuffer scb(sink_cb);
    const auto mask = recipe_accessor<has_mask, after_primary>(12);
    CircularBuffer mcb(mask_cb);
    // Key ranges' optional device tensors, in this order: the Q offset, cu_window_seqlens, the page table.
    constexpr uint32_t kr_cta0 = after_primary;
    constexpr uint32_t kr_cta1 = next_args_offset<q_offset_page != 0, kr_cta0>();
    constexpr uint32_t kr_cta2 = next_args_offset<segments_page != 0, kr_cta1>();
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

    // Key ranges: runtime args 12-15: the scalar Q offset, then the Q offset, cu_window_seqlens and page table
    // addresses. The tensors land in this kernel's half of the scratch CB (recipe_key_range.hpp: RecipeScratch).
    RecipeKeyRange<causal, window, segments> keys{.q_offset = recipe_arg<krange>(12), .k_rows = kv_primary_rows};
    [[maybe_unused]] const uint32_t scratch = CircularBuffer(scratch_cb).get_write_ptr();
    if constexpr (krange && q_offset_page != 0) {
        keys.q_offset = recipe_read_index_page(
            noc, TensorAccessor(TensorAccessorArgs<kr_cta0>(), get_arg_val<uint32_t>(13)), 0, q_offset_page, scratch);
    }
    if constexpr (krange && segments_page != 0) {
        recipe_read_index_page(
            noc,
            TensorAccessor(TensorAccessorArgs<kr_cta1>(), get_arg_val<uint32_t>(14)),
            0,
            segments_page,
            scratch + RecipeScratch::Segments);
        keys.set_segments(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scratch + RecipeScratch::Segments));
    }
    // The current batch's page-table row: its cache blocks in sequence order, after both scratch halves.
    const auto page_table = recipe_accessor<has_page_table, kr_cta2>(15);
    uint32_t page_table_batch = UINT32_MAX;
    const auto* blocks = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scratch + page_table_offset);
    const RecipeQSlabs<q_slab_jobs> slabs{{recipe_arg<q_slabs>(16), recipe_arg<q_slabs>(17)}};
    for (uint32_t qi = 0; qi < jobs; ++qi) {
        uint32_t job;
        if constexpr (krange) {
            // Runtime arg 3 is this core's index in the snake deal (recipe_key_range.hpp: recipe_snake_job).
            job = recipe_snake_job(first_job, qi, cores, batch_heads, queries_per_head);
        } else {
            job = first_job + qi;
        }
        const uint32_t job_head = job / queries_per_head;
        uint32_t kv_head = job_head / q_per_kv_head;
        if constexpr (page_table_page != 0) {
            // Chunked prefill: K/V tile row t of the sequence is row t % block of cache block blocks[t / block], and
            // kv_head the head within the batch.
            const uint32_t batch = job_head / q_heads;
            if (batch != page_table_batch) {
                recipe_read_index_page(noc, page_table, batch, page_table_page, scratch + page_table_offset);
                page_table_batch = batch;
            }
            kv_head %= kv_heads;
        }
        auto kv_row = [&](uint32_t t) -> uint32_t {
            if constexpr (page_table_page != 0) {
                return (blocks[t / page_block_tiles] * kv_heads + kv_head) * page_block_tiles + t % page_block_tiles;
            } else {
                return kv_head * ((kv_rows + 31) / 32) + t;
            }
        };
        auto k_row = [&](uint32_t t) { return kv_row(t) * d_tiles; };
        auto v_row = [&](uint32_t t) { return kv_row(t) * v_src_d_tiles; };
        const uint32_t kvbase = kv_head * k_chunks * kv_tiles;
        uint32_t qbase;
        if constexpr (q_slab_jobs != 0) {
            // The job's chunk of the whole sequence Q holds (q_primary_rows rows per head).
            const uint32_t q_chunk = slabs.chunk(job % queries_per_head);
            qbase = (job_head * (q_primary_rows / (q_tiles * 32)) + q_chunk) * q_tiles * d_tiles;
        } else {
            qbase = job * q_tiles * d_tiles;
        }
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
        if constexpr (sink_cb != 0) {
            read_sink_page(noc, sink, scb, job_head);
        }

        if constexpr (krange) {
            // K chunks of this Q chunk (recipe_key_range.hpp; the writer sends compute the range and the edge masks):
            // the edge chunks, read here, then the full ones. With kv_share the full ones come from the core one Q
            // chunk later in this head (upstream) unless this core leads its round, and go on to the core one Q chunk
            // earlier (downstream) for as many as it needs (RecipeKvShare).
            // Out of line: K/V sharing also asks for the neighbouring Q chunks' ranges.
            auto chunks_of = [&](uint32_t q_job) __attribute__((noinline)) {
                uint32_t row0;
                if constexpr (q_slab_jobs != 0) {
                    row0 = slabs.chunk(q_job % queries_per_head) * q_tiles * 32;
                } else {
                    row0 = (q_job % queries_per_head) * q_tiles * 32;
                }
                const uint32_t row_end = row0 + q_tiles * 32 < q_primary_rows ? row0 + q_tiles * 32 : q_primary_rows;
                return keys.chunks(row0, row_end, k_chunk_tiles * 32, k_chunks);
            };
            const RecipeChunkRange range = chunks_of(job);
            using KvShare = std::conditional_t<kv_share, RecipeKvShare, RecipeNoKvShare>;
            [[maybe_unused]] const KvShare share(first_job, qi, batch_heads * queries_per_head, job, chunks_of);
            for (uint32_t i = 0; i < range.count(); ++i) {
                const uint32_t ki = range.at(i);
                [[maybe_unused]] const uint32_t first_kv_page = kvbase + ki * kv_tiles;
                bool receive = false, forward = false;
                uint32_t barrier = reader_barrier_tiles;
                if constexpr (kv_share) {
                    const bool shared = i >= range.edges();
                    receive = shared && share.receives(ki);
                    forward = shared && share.forwards(ki);
                    barrier = shared ? shared_barrier_tiles : reader_barrier_tiles;
                }
                kcb.reserve_back(kv_tiles);
                const uint32_t kptr = kcb.get_write_ptr();
                if (receive) {
                    if constexpr (kv_share) {
                        share.receive(noc, kptr);
                    }
                } else {
                    if constexpr (kv_rows != 0) {
                        read_kv_rows<kbytes, true, d_tiles>(noc, k_tensor, ki * k_chunk_tiles, k_row, kptr, barrier);
                    } else {
                        read_kv_from_dram<kbytes, true>(noc, k, first_kv_page, kptr, barrier);
                    }
                }
                // Compute may start on the chunk while it is forwarded: this reader overwrites the slot only later.
                kcb.push_back(kv_tiles);
                if (forward) {
                    if constexpr (kv_share) {
                        share.forward(noc, kptr, kv_tiles * kbytes);
                    }
                }
                vcb.reserve_back(v_tiles);
                const uint32_t vptr = vcb.get_write_ptr();
                if (receive) {
                    if constexpr (kv_share) {
                        share.receive(noc, vptr);
                    }
                } else {
                    if constexpr (kv_rows != 0) {
                        read_kv_rows<vbytes, false, v_d_tiles>(noc, v_tensor, ki * k_chunk_tiles, v_row, vptr, barrier);
                    } else {
                        read_kv_from_dram<vbytes, false>(noc, v, first_kv_page, vptr, barrier);
                    }
                }
                // Compute may start on the chunk while it is forwarded: this reader overwrites the slot only later.
                vcb.push_back(v_tiles);
                if (forward) {
                    if constexpr (kv_share) {
                        share.forward(noc, vptr, v_tiles * vbytes);
                    }
                }
            }
        } else {
            const bool receive = link.should_receive(head);
            const bool forward = link.should_forward(head, qi);
            for (uint32_t ki = 0; ki < k_chunks; ++ki) {
                [[maybe_unused]] const uint32_t first_kv_page = kvbase + ki * kv_tiles;

                kcb.reserve_back(kv_tiles);
                const uint32_t kptr = kcb.get_write_ptr();
                if (receive) {
                    link.receive(noc);
                } else {
                    if constexpr (kv_rows != 0) {
                        read_kv_rows<kbytes, true, d_tiles>(noc, k_tensor, ki * k_chunk_tiles, k_row, kptr);
                    } else {
                        read_kv_from_dram<kbytes, true>(noc, k, first_kv_page, kptr);
                    }
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
                        noc, mask, mcb, job_head, (job % queries_per_head) * q_tiles, ki * k_chunk_tiles);
                }

                vcb.reserve_back(v_tiles);
                const uint32_t vptr = vcb.get_write_ptr();
                if (receive) {
                    link.receive(noc);
                } else {
                    if constexpr (kv_rows != 0) {
                        read_kv_rows<vbytes, false, v_d_tiles>(noc, v_tensor, ki * k_chunk_tiles, v_row, vptr);
                    } else {
                        read_kv_from_dram<vbytes, false>(noc, v, first_kv_page, vptr);
                    }
                }
                if (forward) {
                    link.forward(noc, vptr, v_tiles, vbytes);
                }
                vcb.push_back(v_tiles);
            }
        }
    }
    noc.async_write_barrier();
}

void kernel_main() {
    read_recipe<get_compile_time_arg_val(0), get_compile_time_arg_val(1), get_compile_time_arg_val(2)>();
}
