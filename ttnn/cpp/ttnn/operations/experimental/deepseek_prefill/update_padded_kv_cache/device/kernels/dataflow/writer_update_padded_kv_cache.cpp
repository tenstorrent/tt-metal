// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/core_local_mem.h"
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/update_padded_kv_cache/device/kernels/dataflow/cache_dataflow_helpers.hpp"
#include "ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/update_padded_kv_cache/device/kernels/paged_kv_table.hpp"
#include "api/tensor/noc_traits.h"
#include "ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/pack_scaled_fp8_kv_cache/shared_with_host/packed_kv_layout.hpp"

namespace packed = ttnn::operations::experimental::deepseek_prefill::pack_scaled_fp8_kv_cache;
namespace paged = ttnn::operations::experimental::deepseek_prefill::paged_kv_table;

namespace {

// A global prefix maps to a staircase of local offsets in block-cyclic cache storage.
inline uint32_t local_cache_offset(uint32_t prefix, uint32_t coordinate, uint32_t factor, uint32_t chunk) {
    const uint32_t slab = prefix / (factor * chunk);
    const uint32_t boundary = (prefix / chunk) % factor;
    const uint32_t offset = prefix % chunk;
    return slab * chunk + (coordinate < boundary ? chunk : (coordinate == boundary ? offset : 0));
}

// Inverse of the block-cyclic placement: the global page-row a chip-local cache row holds.
inline uint32_t global_row_of_local(uint32_t local_row, uint32_t coordinate, uint32_t factor, uint32_t chunk) {
    return ((local_row / chunk) * factor + coordinate) * chunk + local_row % chunk;
}

// The two runs of the request's page-table row one write can touch, loaded into L1. A chip writes one contiguous
// run of local rows, which crosses at most one slab boundary; each slab is one contiguous run of logical pages.
struct PageWindows {
    uint32_t base[2] = {0, 0};
    uint32_t count[2] = {0, 0};
    uint32_t l1[2] = {0, 0};

    uint32_t lookup(uint32_t page) const {
        for (uint32_t w = 0; w < 2; ++w) {
            if (page - base[w] < count[w]) {
                return CoreLocalMem<volatile uint32_t>(l1[w])[page - base[w]];
            }
        }
        return paged::kUnmappedPage;
    }
};

// Loads the logical pages [first, last) of `row` into window `w`. Reads start on a DRAM-aligned entry and cover
// whole aligned groups, which stay inside the row because the host requires its width to be a whole number of groups.
template <typename TableAccessor>
inline void load_page_window(
    Noc& noc,
    const TableAccessor& table,
    PageWindows& windows,
    uint32_t w,
    uint32_t row,
    uint32_t first,
    uint32_t last) {
    constexpr uint32_t align = paged::kTableEntriesPerRead;
    const uint32_t aligned_first = first - first % align;
    const uint32_t entries = (last - aligned_first + align - 1) / align * align;
    noc.async_read(
        table,
        CoreLocalMem<uint32_t>(windows.l1[w]),
        entries * sizeof(uint32_t),
        {.page_id = row, .offset_bytes = aligned_first * static_cast<uint32_t>(sizeof(uint32_t))},
        {});
    windows.base[w] = aligned_first;
    windows.count[w] = entries;
}

}  // namespace

// CB -> cache writer for the per-chip-offset kv-cache update op.
//
// The per-request `slot_idx` and `kv_actual_global` reach the kernel one of two ways, selected by the
// `has_metadata` compile-time flag (the op sets it from whether the metadata tensors were supplied):
//   - metadata path: read on-device from two 1-element uint32 DRAM tensors (raw addresses in common
//     args 8 and 9) -> slot_idx = slot_idx tensor's element [0], kv_actual_global (tokens) =
//     kv_actual_global tensor's element [0]. The values stay off the host dispatch path, so the op is
//     traceable and one cached program per layer is reused across users/chunks.
//   - scalar path: read from common runtime args 8/9 (patched on cache hits by the op's
//     override_runtime_arguments). Kept out of the program hash the same way.
// `layer_idx`, `num_layers` and `cluster_axis` stay in the hash (structural) in both paths.
//
// Optional `valid_global` (end of the chunk's REAL tokens) arrives the same two ways in common arg 10.
// Set, the chip writes only the page-rows holding real tokens; unset, the whole padded slab.
//
// Compile args: [0]=cb_id_out, [1]=has_metadata, [2]=cb_id_meta, [3]=tile_height, [4]=has_valid,
// [5]=untilize, [6]=block_pages, [7]=scaled-FP8 packing, [8]=tiled RoPE CB, [9]=paged, [10]=page-table CB,
// [11..]=cache accessor, then (metadata path only) ONE metadata accessor (the 1-element tensors share an identical
// layout, so the same accessor serves every read), then (paged only) the page-table accessor. tile_height divides kv
// tokens into the page-row unit (TILE_HEIGHT for TILE, 1 for ROW_MAJOR), so one kernel handles both layouts.
//
// Paged: `cache` is a pool of page-sized entries (paged_kv_table.hpp) and `slot_idx` selects the request's page-table
// row. Common args [12]=page-table binding, [13]=page-rows per page, [14]=row width in pages, [15]=pool pages. A row
// whose page is unmapped (or outside the pool) is not written.
//
// The body lives in a template on `HasMeta` so the `if constexpr` below actually DISCARDS (does not
// instantiate) the unused branch — `kernel_main` is not a template, so an `if constexpr` there would
// still instantiate the metadata branch's TensorAccessor and fail to compile the scalar program.
// `HasValid` is a template param so the no-clamp program keeps the original loop.
template <bool HasMeta, bool HasValid, bool HasRope, bool HasScaled, bool TiledRope, bool Paged>
static void run_writer() {
    // Index 11 follows the eleven common args in update_padded_kv_cache_device_operation.cpp,
    // where writer_kernel.emplace_common_runtime_args({dst_buffer}) appends the destination binding.
    const uint32_t dst_addr = get_common_arg_val<uint32_t>(11);
    const uint32_t num_pages = get_arg_val<uint32_t>(0);
    const uint32_t core_blocks_written = get_arg_val<uint32_t>(1);

    // Common runtime args (same for all cores on this chip). Indices 0-7 are structural; index 8 (and
    // 9, scalar path) carry the per-request values resolved below.
    const uint32_t my_sp_coord = get_common_arg_val<uint32_t>(0);
    const uint32_t sp_factor = get_common_arg_val<uint32_t>(1);
    const uint32_t chunk_local_t = get_common_arg_val<uint32_t>(2);
    const uint32_t layer_idx = get_common_arg_val<uint32_t>(3);
    const uint32_t num_layers = get_common_arg_val<uint32_t>(4);
    const uint32_t Wt = get_common_arg_val<uint32_t>(5);
    const uint32_t cache_HtWt = get_common_arg_val<uint32_t>(6);
    const uint32_t cache_CHtWt = get_common_arg_val<uint32_t>(7);

    constexpr uint32_t cb_id_out = get_compile_time_arg_val(0);
    constexpr uint32_t tile_height = get_compile_time_arg_val(3);
    // [4] is has_valid, consumed as the HasValid template param.
    constexpr auto cache_args = TensorAccessorArgs<11>();

    Noc noc;

    // Resolve the per-request values (in page-row units) from whichever source this program was
    // compiled for.
    uint32_t slot_idx;
    uint32_t kv_actual_global_t;
    // Real tokens as page-rows, rounded up to 32 (zero_padded_kv_cache clears the partial block).
    uint32_t valid_tokens = 0;
    constexpr uint32_t kClampGranularityTokens = 32;
    if constexpr (HasMeta) {
        // Metadata path: NoC-read element [0] (page 0, 4 bytes) of each 1-element uint32 tensor into
        // the L1-scratch CB. Each read targets dst offset 0 (DRAM-read dst-alignment: a 4-byte read into
        // a non-16B-aligned dst offset lands wrong), so we read slot, barrier+extract, then overwrite the
        // same slot with kv_actual_global. Single reserve_back/push_back (a second reserve_back on a
        // single-page CB with no intervening pop would deadlock).
        constexpr uint32_t cb_id_meta = get_compile_time_arg_val(2);
        // ONE metadata accessor follows the cache accessor in the compile args; it serves both 1-element
        // tensors (identical layout). Gate the offset on HasMeta so this TensorAccessorArgs<> is a
        // *dependent* template-id: `if constexpr` only skips instantiation of the discarded branch's
        // template-parameter-dependent constructs, so the scalar program (no metadata accessor) must
        // not name a fixed out-of-range offset here.
        constexpr uint32_t kMetaArgsOffset = HasMeta ? cache_args.next_compile_time_args_offset() : 0;
        CircularBuffer cb_meta(cb_id_meta);
        cb_meta.reserve_back(1);

        slot_idx = kv_cache_dataflow::read_metadata<kMetaArgsOffset>(noc, cb_meta, get_common_arg_val<uint32_t>(8));
        kv_actual_global_t =
            kv_cache_dataflow::read_metadata<kMetaArgsOffset>(noc, cb_meta, get_common_arg_val<uint32_t>(9)) /
            tile_height;
        if constexpr (HasValid) {
            valid_tokens =
                kv_cache_dataflow::read_metadata<kMetaArgsOffset>(noc, cb_meta, get_common_arg_val<uint32_t>(10));
        }
        cb_meta.push_back(1);
    } else {
        // Scalar path: per-call values arrive as common runtime args (patched on cache hits).
        slot_idx = get_common_arg_val<uint32_t>(8);
        kv_actual_global_t = get_common_arg_val<uint32_t>(9) / tile_height;
        if constexpr (HasValid) {
            valid_tokens = get_common_arg_val<uint32_t>(10);
        }
    }

    const uint32_t valid_global_t = kClampGranularityTokens *
                                    ((valid_tokens + kClampGranularityTokens - 1) / kClampGranularityTokens) /
                                    tile_height;

    // Cache linearization: users outer, layers inner.
    const uint32_t batch_idx = slot_idx * num_layers + layer_idx;

    const uint32_t update_idxt = local_cache_offset(kv_actual_global_t, my_sp_coord, sp_factor, chunk_local_t);

    const uint32_t input_Ht = chunk_local_t;

    // Real rows are a prefix on every chip, so their end is the staircase above at valid_global_t.
    uint32_t rows_to_write = input_Ht;
    if constexpr (HasValid) {
        const uint32_t end_idxt = local_cache_offset(valid_global_t, my_sp_coord, sp_factor, chunk_local_t);
        rows_to_write = (end_idxt > update_idxt) ? (end_idxt - update_idxt) : 0;
        if (rows_to_write > input_Ht) {
            rows_to_write = input_Ht;
        }
    }

    const uint32_t start_idx = batch_idx * cache_CHtWt + update_idxt * Wt;

    static_assert(!(Paged && (HasRope || HasScaled || TiledRope)), "a paged cache takes the plain copy path only");
    PageWindows windows;
    uint32_t page_rows = 1;
    uint32_t pool_pages = 0;
    if constexpr (Paged) {
        // The page-table accessor follows the cache accessor and, on the metadata path, the metadata accessor
        // (whose args are named only when present, as for kMetaArgsOffset above).
        constexpr uint32_t after_cache = cache_args.next_compile_time_args_offset();
        constexpr uint32_t table_args_offset =
            HasMeta ? TensorAccessorArgs<HasMeta ? after_cache : 0>().next_compile_time_args_offset() : after_cache;
        constexpr auto table_args = TensorAccessorArgs<table_args_offset>();
        const auto table = TensorAccessor(table_args, get_common_arg_val<uint32_t>(12));
        page_rows = get_common_arg_val<uint32_t>(13);
        const uint32_t table_width = get_common_arg_val<uint32_t>(14);
        pool_pages = get_common_arg_val<uint32_t>(15);

        // DRAM reads need a DRAM-aligned L1 destination; the host pads the CB by one alignment unit.
        CircularBuffer cb_table(get_compile_time_arg_val(10));
        cb_table.reserve_back(1);
        constexpr uint32_t align_bytes = paged::kTableEntriesPerRead * sizeof(uint32_t);
        windows.l1[0] = (cb_table.get_write_ptr() + align_bytes - 1) / align_bytes * align_bytes;
        windows.l1[1] = windows.l1[0] + paged::window_capacity(chunk_local_t, page_rows) * sizeof(uint32_t);

        if (rows_to_write > 0) {
            const auto page_of = [&](uint32_t local_row) {
                return global_row_of_local(local_row, my_sp_coord, sp_factor, chunk_local_t) / page_rows;
            };
            // Local rows [lo, hi] of one slab -> one window; pages past the row's width stay unmapped.
            const auto load = [&](uint32_t w, uint32_t lo, uint32_t hi) {
                const uint32_t first_page = page_of(lo);
                const uint32_t end_page = page_of(hi) + 1 < table_width ? page_of(hi) + 1 : table_width;
                if (first_page < end_page) {
                    load_page_window(noc, table, windows, w, slot_idx, first_page, end_page);
                }
            };
            const uint32_t first = update_idxt;
            const uint32_t last = update_idxt + rows_to_write - 1;
            const uint32_t slab_last = (first / chunk_local_t + 1) * chunk_local_t - 1;
            load(0, first, last < slab_last ? last : slab_last);
            if (last > slab_last) {
                load(1, slab_last + 1, last);
            }
            // Completed once the first input page is in, so the table read hides under the reader's.
        }
        cb_table.push_back(1);
    }
    bool table_pending = Paged && rows_to_write > 0;
    // Paged rows are mapped incrementally: a core walks consecutive local rows, so the divisions run only when a
    // run starts (or wraps to the next head), and a slab boundary jumps over the other chips' pages.
    uint32_t next_row = 0;
    uint32_t row_in_chunk = 0;
    uint32_t row_in_page = 0;
    uint32_t logical_page = 0;
    uint32_t physical = paged::kUnmappedPage;
    const uint32_t slab_page_jump = Paged ? 1 + (sp_factor - 1) * (chunk_local_t / page_rows) : 0;

    const uint32_t page_bytes = get_local_cb_interface(cb_id_out).fifo_page_size;
    CircularBuffer cb(cb_id_out);

    constexpr uint32_t onepage = 1;
    // `cache_args` and `noc` are declared above (shared with the optional metadata read).
    const auto s = TensorAccessor(cache_args, dst_addr);

    // One block is one page-row (Wt pages) of one head, head-major / row-minor as the reader streams
    // them. The cache's head stride (cache_HtWt) exceeds the input's (input_Ht * Wt) once the cache is
    // deeper than one chunk, so each page id is recomputed from (head, row) rather than walked with ++,
    // which would cross into the previous head's rows. Only reachable at C > 1.
    // Exact: the program factory passes num_pages as num_blocks_per_core * Wt, so there is never a
    // partial block to drop here.
    const uint32_t num_blocks = num_pages / Wt;
    for (uint32_t blk = 0; blk < num_blocks; ++blk) {
        const uint32_t block = core_blocks_written + blk;
        const uint32_t row = block % input_Ht;
        uint32_t page0;
        bool mapped = true;
        if constexpr (Paged) {
            if (table_pending) {
                cb.wait_front(1);
                noc.async_read_barrier();
                invalidate_l1_cache();
                table_pending = false;
            }
            if (blk == 0 || row != next_row) {
                const uint32_t local_row = update_idxt + row;
                const uint32_t global_row = global_row_of_local(local_row, my_sp_coord, sp_factor, chunk_local_t);
                row_in_chunk = local_row % chunk_local_t;
                row_in_page = global_row % page_rows;
                logical_page = global_row / page_rows;
                physical = windows.lookup(logical_page);
            } else if (++row_in_chunk == chunk_local_t) {
                row_in_chunk = 0;
                row_in_page = 0;
                logical_page += slab_page_jump;
                physical = windows.lookup(logical_page);
            } else if (++row_in_page == page_rows) {
                row_in_page = 0;
                physical = windows.lookup(++logical_page);
            }
            next_row = row + 1;
            mapped = physical < pool_pages;
            page0 =
                (physical * num_layers + layer_idx) * cache_CHtWt + (block / input_Ht) * cache_HtWt + row_in_page * Wt;
        } else {
            page0 = start_idx + (block / input_Ht) * cache_HtWt + row * Wt;
        }
        const bool keep = (!HasValid || row < rows_to_write) && mapped;
        // The copy path consumes one page at a time; untilize publishes a complete tile-row block.
        constexpr uint32_t pages_per_batch = (HasRope || HasScaled) ? get_compile_time_arg_val(6) : onepage;
        constexpr uint32_t rows_per_batch = (HasRope || TiledRope) ? tile_height : 1;
        const uint32_t row_bytes = pages_per_batch * page_bytes / rows_per_batch;
        const uint32_t first_page = (HasRope || TiledRope) ? page0 * tile_height : page0;
        const uint32_t num_batches = (HasRope || TiledRope) ? 1 : Wt;
        for (uint32_t batch = 0; batch < num_batches; ++batch) {
            cb.wait_front(pages_per_batch);
            constexpr uint32_t rope_cb_id = get_compile_time_arg_val(8);
            constexpr uint32_t rope_tiles = packed::ROPE_WIDTH / tile_height;
            CircularBuffer rope_cb(rope_cb_id);
            if constexpr (TiledRope) {
                rope_cb.wait_front(rope_tiles);
            }
            if (keep) {
                if constexpr (HasScaled) {
                    constexpr uint32_t scale_bytes = packed::SCALE_WIDTH * sizeof(float);
                    constexpr uint32_t rope_bytes = packed::ROPE_WIDTH * sizeof(uint16_t);
                    for (uint32_t row = 0; row < rows_per_batch; ++row) {
                        const uint32_t destination_row = first_page + batch * rows_per_batch + row;
                        const uint32_t latent_offset = TiledRope ? 2 * row * page_bytes : 0;
                        const uint32_t scale_offset = TiledRope ? (2 * row + 1) * page_bytes : page_bytes;
                        noc.async_write(
                            cb, s, packed::LATENT_WIDTH, {.offset_bytes = latent_offset}, {.page_id = destination_row});
                        noc.async_write(
                            cb,
                            s,
                            scale_bytes,
                            {.offset_bytes = scale_offset},
                            {.page_id = destination_row, .offset_bytes = packed::LATENT_WIDTH});
                        if constexpr (TiledRope) {
                            noc.async_write(
                                rope_cb,
                                s,
                                rope_bytes,
                                {.offset_bytes = row * rope_bytes},
                                {.page_id = destination_row, .offset_bytes = packed::LATENT_WIDTH + scale_bytes});
                        } else {
                            noc.async_write(
                                cb,
                                s,
                                rope_bytes,
                                {.offset_bytes = 2 * page_bytes},
                                {.page_id = destination_row, .offset_bytes = packed::LATENT_WIDTH + scale_bytes});
                        }
                    }
                } else {
                    for (uint32_t r = 0; r < rows_per_batch; ++r) {
                        noc.async_write(
                            cb,
                            s,
                            row_bytes,
                            {.offset_bytes = r * row_bytes},
                            {.page_id = first_page + batch * rows_per_batch + r});
                    }
                }
            }
            if constexpr (HasRope || HasScaled) {
                noc.async_write_barrier();
            } else if (keep) {
                noc.async_writes_flushed();
            }
            // Consume skipped batches too: the reader streams the complete padded slab.
            if constexpr (TiledRope) {
                rope_cb.pop_front(rope_tiles);
            }
            cb.pop_front(pages_per_batch);
        }
    }
    noc.async_write_barrier();
}

void kernel_main() {
    constexpr bool has_metadata = get_compile_time_arg_val(1);
    constexpr bool has_valid = get_compile_time_arg_val(4);
    constexpr bool has_rope = get_compile_time_arg_val(5);
    constexpr bool has_scaled = get_compile_time_arg_val(7) != 0;
    constexpr bool tiled_rope = get_compile_time_arg_val(8) != 0;
    constexpr bool paged = get_compile_time_arg_val(9) != 0;
    run_writer<has_metadata, has_valid, has_rope, has_scaled, tiled_rope, paged>();
}
