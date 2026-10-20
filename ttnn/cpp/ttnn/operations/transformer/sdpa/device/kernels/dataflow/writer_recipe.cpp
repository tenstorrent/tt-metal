// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
#include "api/core_local_mem.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"
#include "sequence_accessor.hpp"
#include "recipe_key_range.hpp"

// Named CTAs (host: run_recipe_segments, sdpa_recipe.cpp).
namespace recipe_writer {
// Joint SDPA: the joint output's accessor follows the primary one, its address is runtime arg 3.
constexpr bool has_joint = get_named_compile_time_arg_val("joint") != 0;
constexpr uint32_t d_tiles = get_named_compile_time_arg_val("d_tiles");
// Zero Q rows padding a paired recipe's odd chunk (host: recipe_compute_q_tiles).
constexpr uint32_t q_pad_tiles = get_named_compile_time_arg_val("q_pad_tiles");
// Output [B, 1, Sq, H x Dv] (concat_heads heads; 0: [B, H, Sq, Dv]).
constexpr uint32_t concat_heads = get_named_compile_time_arg_val("concat_heads");
// Q chunks per head (key ranges and concatenated heads).
constexpr uint32_t q_jobs = get_named_compile_time_arg_val("q_jobs");
// Key ranges (recipe_key_range.hpp): the K range model, the snake deal over cores x batch_heads, Q slabs (q_slab_jobs
// chunks each; 0: none), the optional device tensors' page sizes (0: absent), K's length and chunking, and the CBs
// (control pages, edge masks in mask_group_rows-row groups, the all-masked template and mask_cache_tiles cached tiles,
// this kernel's half of the scratch CB at scratch_writer bytes).
constexpr bool krange = get_named_compile_time_arg_val("krange") != 0;
constexpr bool causal = get_named_compile_time_arg_val("causal") != 0;
constexpr uint32_t window = get_named_compile_time_arg_val("window");
constexpr uint32_t segments = get_named_compile_time_arg_val("segments");
constexpr uint32_t cores = get_named_compile_time_arg_val("cores");
constexpr uint32_t batch_heads = get_named_compile_time_arg_val("batch_heads");
constexpr uint32_t q_slab_jobs = get_named_compile_time_arg_val("q_slab_jobs");
constexpr bool q_slabs = q_slab_jobs != 0;
constexpr uint32_t q_offset_page = get_named_compile_time_arg_val("q_offset_page");
constexpr uint32_t segments_page = get_named_compile_time_arg_val("segments_page");
constexpr uint32_t k_rows = get_named_compile_time_arg_val("k_rows");
constexpr uint32_t k_chunks = get_named_compile_time_arg_val("k_chunks");
constexpr uint32_t k_chunk_tiles = get_named_compile_time_arg_val("k_chunk_tiles");
constexpr uint32_t key_range_cb = get_named_compile_time_arg_val("key_range_cb");
constexpr uint32_t mask_cb = get_named_compile_time_arg_val("mask_cb");
constexpr uint32_t mask_group_rows = get_named_compile_time_arg_val("mask_group_rows");
constexpr uint32_t masked_tile_cb = get_named_compile_time_arg_val("masked_tile_cb");
constexpr uint32_t mask_cache_tiles = get_named_compile_time_arg_val("mask_cache_tiles");
constexpr uint32_t scratch_cb = get_named_compile_time_arg_val("scratch_cb");
constexpr uint32_t scratch_writer = get_named_compile_time_arg_val("scratch_writer");
static_assert(!(krange && has_joint), "SDPA recipe key ranges are dense-only");
}  // namespace recipe_writer
using namespace recipe_writer;

// The output as one sequence: the primary output and, in joint builds, the joint output (args at joint_args).
template <uint32_t q_tiles, uint32_t primary_rows, uint32_t joint_rows, uint32_t args_offset>
FORCE_INLINE auto recipe_output_sequence() {
    constexpr auto oa = TensorAccessorArgs<args_offset>();
    if constexpr (has_joint) {
        constexpr auto joa = TensorAccessorArgs<oa.next_compile_time_args_offset()>();
        return sequence_accessor<primary_rows, joint_rows, q_tiles * 32, d_tiles>(
            TensorAccessor(oa, get_arg_val<uint32_t>(0)), TensorAccessor(joa, get_arg_val<uint32_t>(3)));
    } else {
        return sequence_accessor<primary_rows, q_tiles * 32, d_tiles>(TensorAccessor(oa, get_arg_val<uint32_t>(0)));
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

// One Q chunk's output rows, then drop the padding rows of a paired recipe's odd chunk (host: recipe_compute_q_tiles).
template <uint32_t q_tiles, uint32_t primary_rows, typename Output>
FORCE_INLINE void write_output_chunk(const Noc& noc, DataflowBuffer& cb, const Output& out, uint32_t job) {
    for (uint32_t row = 0; row < q_tiles; ++row) {
        cb.wait_front(d_tiles);
        if constexpr (concat_heads != 0) {
            // Output [B, 1, Sq, H x Dv]: this head's Dv columns of the sequence's tile row (padding rows dropped).
            constexpr uint32_t heads = concat_heads != 0 ? concat_heads : 1;
            constexpr uint32_t row_tiles = (primary_rows + 31) / 32;
            const uint32_t head = job / q_jobs;
            const uint32_t tile_row = (job % q_jobs) * q_tiles + row;
            if (tile_row < row_tiles) {
                const uint32_t page = ((head / heads * row_tiles + tile_row) * heads + head % heads) * d_tiles;
                for (uint32_t col = 0; col < d_tiles; ++col) {
                    noc.async_write(cb, out.primary, 2048, {.offset_bytes = col * 2048}, {.page_id = page + col});
                }
            }
        } else {
            for (uint32_t col = 0; col < d_tiles; ++col) {
                out.visit(job * q_tiles * d_tiles + row * d_tiles + col, [&](const auto& destination, uint32_t page) {
                    noc.async_write(cb, destination, 2048, {.offset_bytes = col * 2048}, {.page_id = page});
                });
            }
        }
        noc.async_write_barrier();
        cb.pop_front(d_tiles);
    }
    for (uint32_t row = 0; row < q_pad_tiles; ++row) {
        cb.wait_front(d_tiles);
        cb.pop_front(d_tiles);
    }
}

// The kernel, a template so that the build's unused paths (if constexpr on the named args above) are not instantiated.
template <uint32_t q_tiles, uint32_t primary_rows, uint32_t joint_rows>
FORCE_INLINE void write_recipe() {
    const auto out = recipe_output_sequence<q_tiles, primary_rows, joint_rows, 3>();
    const uint32_t first_job = get_arg_val<uint32_t>(1);
    const uint32_t jobs = get_arg_val<uint32_t>(2);
    Noc noc;
    DataflowBuffer cb(16);
    if constexpr (krange) {
        // Key ranges (recipe_key_range.hpp): per Q chunk, the control page compute reads and the edge chunks' masks,
        // before its output. Runtime args 3-5: the scalar Q offset, then the Q offset and cu_window_seqlens addresses
        // (0 when absent), read into this kernel's half of the scratch CB; with Q slabs 6-7 the slabs' first Q chunks.
        constexpr uint32_t kr_cta0 = TensorAccessorArgs<3>().next_compile_time_args_offset();
        constexpr uint32_t kr_cta1 = next_args_offset<q_offset_page != 0, kr_cta0>();
        RecipeKeyRange<causal, window, segments> keys{.q_offset = get_arg_val<uint32_t>(3), .k_rows = k_rows};
        [[maybe_unused]] const uint32_t scratch = CircularBuffer(scratch_cb).get_write_ptr() + scratch_writer;
        if constexpr (q_offset_page != 0) {
            keys.q_offset = recipe_read_index_page(
                noc,
                TensorAccessor(TensorAccessorArgs<kr_cta0>(), get_arg_val<uint32_t>(4)),
                0,
                q_offset_page,
                scratch);
        }
        if constexpr (segments_page != 0) {
            recipe_read_index_page(
                noc,
                TensorAccessor(TensorAccessorArgs<kr_cta1>(), get_arg_val<uint32_t>(5)),
                0,
                segments_page,
                scratch + RecipeScratch::Segments);
            keys.set_segments(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scratch + RecipeScratch::Segments));
        }
        // The all-masked template tile, copied into every all-masked mask tile.
        RecipeMaskCache<mask_cache_tiles> mask_cache{.base = CircularBuffer(masked_tile_cb).get_write_ptr()};
        for (uint32_t i = 0; i < kRecipeMaskTileBytes / 4; ++i) {
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(mask_cache.base)[i] =
                i < 16 ? kRecipeMaskExponents : kRecipeMaskedNibbles;
        }
        CircularBuffer rcb(key_range_cb), mcb(mask_cb);
        const RecipeQSlabs<q_slab_jobs> slabs{{recipe_arg<q_slabs>(6), recipe_arg<q_slabs>(7)}};
        // This core's Q chunks in the snake deal (runtime arg 1: the core's index; reader_recipe.cpp).
        for (uint32_t qi = 0; qi < jobs; ++qi) {
            const uint32_t job = recipe_snake_job(first_job, qi, cores, batch_heads, q_jobs);
            uint32_t q_row0, q_row_end;
            if constexpr (q_slab_jobs != 0) {
                // Rows of the whole sequence; slab chunks are whole.
                q_row0 = slabs.chunk(job % q_jobs) * q_tiles * 32;
                q_row_end = q_row0 + q_tiles * 32;
            } else {
                q_row0 = (job % q_jobs) * q_tiles * 32;
                q_row_end = q_row0 + q_tiles * 32 < primary_rows ? q_row0 + q_tiles * 32 : primary_rows;
            }
            const RecipeChunkRange range = keys.chunks(q_row0, q_row_end, k_chunk_tiles * 32, k_chunks);
            recipe_push_chunk_range(rcb, range);
            // The edge chunks come first in processing order (RecipeChunkRange::at).
            for (uint32_t i = 0; i < range.edges(); ++i) {
                generate_mask_chunk<q_tiles, k_chunk_tiles, mask_cb, mask_group_rows>(
                    noc, keys, mcb, mask_cache, q_row0, range.at(i) * k_chunk_tiles * 32);
            }
            write_output_chunk<q_tiles, primary_rows>(noc, cb, out, job);
        }
    } else {
        for (uint32_t job = first_job; job < first_job + jobs; ++job) {
            write_output_chunk<q_tiles, primary_rows>(noc, cb, out, job);
        }
    }
}

void kernel_main() {
    write_recipe<get_compile_time_arg_val(0), get_compile_time_arg_val(1), get_compile_time_arg_val(2)>();
}
