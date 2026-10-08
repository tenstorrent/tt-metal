// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"
#include "sequence_accessor.hpp"
#ifdef SDPA_RECIPE_KRANGE
#include "api/core_local_mem.h"
#include "api/dataflow/circular_buffer.h"
#include "recipe_key_range.hpp"
#endif
void kernel_main() {
    constexpr uint32_t q_tiles = get_compile_time_arg_val(0);
    constexpr uint32_t primary_rows = get_compile_time_arg_val(1);
    constexpr uint32_t joint_rows = get_compile_time_arg_val(2);
    constexpr auto oa = TensorAccessorArgs<3>();
#ifdef SDPA_JOINT
    constexpr auto joa = TensorAccessorArgs<oa.next_compile_time_args_offset()>();
    const auto out = sequence_accessor<primary_rows, joint_rows, q_tiles * 32, SDPA_RECIPE_DHT>(
        TensorAccessor(oa, get_arg_val<uint32_t>(0)), TensorAccessor(joa, get_arg_val<uint32_t>(3)));
#else
    const auto out = sequence_accessor<primary_rows, q_tiles * 32, SDPA_RECIPE_DHT>(TensorAccessor(oa, get_arg_val<uint32_t>(0)));
#endif
    const uint32_t first_job = get_arg_val<uint32_t>(1);
    const uint32_t jobs = get_arg_val<uint32_t>(2);
    Noc noc;
    DataflowBuffer cb(16);
#ifdef SDPA_RECIPE_KRANGE
    // Key ranges (recipe_key_range.hpp): per Q chunk, the control page compute reads and the edge chunks' masks,
    // before its output. Runtime args 3-5: the scalar Q offset, then the Q offset and cu_window_seqlens addresses
    // (0 when absent), read into this kernel's half of the scratch CB; with Q slabs 6-7 the slabs' first Q chunks.
    constexpr uint32_t kr_cta0 = oa.next_compile_time_args_offset();
#ifdef SDPA_RECIPE_Q_OFFSET_PAGE
    constexpr auto offset_args = TensorAccessorArgs<kr_cta0>();
    constexpr uint32_t kr_cta1 = offset_args.next_compile_time_args_offset();
#else
    constexpr uint32_t kr_cta1 = kr_cta0;
#endif
    RecipeKeyRange keys{.q_offset = get_arg_val<uint32_t>(3), .k_rows = SDPA_RECIPE_K_ROWS};
    [[maybe_unused]] const uint32_t scratch =
        CircularBuffer(SDPA_RECIPE_SCRATCH_CB).get_write_ptr() + SDPA_RECIPE_SCRATCH_WRITER;
#ifdef SDPA_RECIPE_Q_OFFSET_PAGE
    keys.q_offset = recipe_read_index_page(
        noc, TensorAccessor(offset_args, get_arg_val<uint32_t>(4)), 0, SDPA_RECIPE_Q_OFFSET_PAGE, scratch);
#endif
#ifdef SDPA_RECIPE_SEGMENTS_PAGE
    constexpr auto segment_args = TensorAccessorArgs<kr_cta1>();
    recipe_read_index_page(
        noc,
        TensorAccessor(segment_args, get_arg_val<uint32_t>(5)),
        0,
        SDPA_RECIPE_SEGMENTS_PAGE,
        scratch + RecipeScratch::Segments);
    keys.segments = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scratch + RecipeScratch::Segments);
#endif
    // The all-masked template tile, copied into every all-masked mask tile.
    RecipeMaskCache mask_cache{.base = CircularBuffer(SDPA_RECIPE_MASKED_TILE_CB).get_write_ptr()};
    for (uint32_t i = 0; i < kRecipeMaskTileBytes / 4; ++i) {
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(mask_cache.base)[i] =
            i < 16 ? kRecipeMaskExponents : kRecipeMaskedNibbles;
    }
    CircularBuffer rcb(SDPA_RECIPE_KEY_RANGE_CB), mcb(SDPA_RECIPE_MASK_CB);
#ifdef SDPA_RECIPE_Q_SLAB_JOBS
    const RecipeQSlabs slabs{{get_arg_val<uint32_t>(6), get_arg_val<uint32_t>(7)}};
#endif
    // Walk positions [first_job, first_job + jobs) of the heads' zigzag orders (reader_recipe.cpp).
    for (uint32_t z = first_job; z < first_job + jobs; ++z) {
        const uint32_t job = z - z % SDPA_RECIPE_Q_JOBS + recipe_zigzag_job(z % SDPA_RECIPE_Q_JOBS, SDPA_RECIPE_Q_JOBS);
#ifdef SDPA_RECIPE_Q_SLAB_JOBS
        // Rows of the whole sequence; slab chunks are whole.
        const uint32_t q_row0 = slabs.chunk(job % SDPA_RECIPE_Q_JOBS) * q_tiles * 32;
        const uint32_t q_row_end = q_row0 + q_tiles * 32;
#else
        const uint32_t q_row0 = (job % SDPA_RECIPE_Q_JOBS) * q_tiles * 32;
        const uint32_t q_row_end = q_row0 + q_tiles * 32 < primary_rows ? q_row0 + q_tiles * 32 : primary_rows;
#endif
        const RecipeChunkRange range = keys.chunks(q_row0, q_row_end, SDPA_K_CHUNK_TILES * 32, SDPA_RECIPE_K_CHUNKS);
        recipe_push_chunk_range(rcb, range);
        for (uint32_t ki = range.first; ki < range.end; ++ki) {
            if (ki < range.full_begin || ki >= range.full_end) {
                generate_mask_chunk<q_tiles>(noc, keys, mcb, mask_cache, q_row0, ki * SDPA_K_CHUNK_TILES * 32);
            }
        }
#else
    for (uint32_t job = first_job; job < first_job + jobs; ++job) {
#endif
        for (uint32_t row = 0; row < q_tiles; ++row) {
            cb.wait_front(SDPA_RECIPE_DHT);
#ifdef SDPA_RECIPE_CONCAT_HEADS
            // Output [B, 1, Sq, H x Dv]: this head's Dv columns of the sequence's tile row (padding rows dropped).
            constexpr uint32_t heads = SDPA_RECIPE_CONCAT_HEADS;
            constexpr uint32_t row_tiles = (primary_rows + 31) / 32;
            const uint32_t head = job / SDPA_RECIPE_Q_JOBS;
            const uint32_t tile_row = (job % SDPA_RECIPE_Q_JOBS) * q_tiles + row;
            if (tile_row < row_tiles) {
                const uint32_t page = ((head / heads * row_tiles + tile_row) * heads + head % heads) * SDPA_RECIPE_DHT;
                for (uint32_t col = 0; col < SDPA_RECIPE_DHT; ++col) {
                    noc.async_write(cb, out.primary, 2048, {.offset_bytes = col * 2048}, {.page_id = page + col});
                }
            }
#else
            for (uint32_t col = 0; col < SDPA_RECIPE_DHT; ++col) {
                out.visit(job * q_tiles * SDPA_RECIPE_DHT + row * SDPA_RECIPE_DHT + col, [&](const auto& destination, uint32_t page) {
                    noc.async_write(cb, destination, 2048, {.offset_bytes = col * 2048}, {.page_id = page});
                });
            }
#endif
            noc.async_write_barrier();
            cb.pop_front(SDPA_RECIPE_DHT);
        }
        // Drop the padding rows of a paired recipe's odd chunk (host: recipe_compute_q_tiles).
        for (uint32_t row = 0; row < SDPA_RECIPE_Q_PAD_TILES; ++row) {
            cb.wait_front(SDPA_RECIPE_DHT);
            cb.pop_front(SDPA_RECIPE_DHT);
        }
    }
}
