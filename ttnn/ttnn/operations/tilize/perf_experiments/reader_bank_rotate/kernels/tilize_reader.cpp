// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// tilize reader (NoC0) — `load_block`.
//
// One `read_sticks_for_tilize` call per BLOCK. The helper owns the whole block:
// it reserves `block_width_tiles` pages, issues `tile_h` NoC reads behind ONE
// barrier, and pushes — once per tile-row, for all `block_row_extent` tile-rows
// of the block. Nothing here is per-tile-row.
//
// The three block extents map onto the helper's parameters exactly:
//   total_num_rows          = block_row_extent * tile_h   (the tile_row extent)
//   row_bytes               = block_row_bytes             (the tile_col extent)
//   byte_offset_within_page = w_chunk * block_row_bytes    (the column-chunk id)
//
// `byte_offset_within_page` is the helper's documented wide-W chunking
// parameter: it selects this block's column slice INSIDE each stick page, so
// the CB footprint scales with the chunk width and not with the tensor's W.
//
// NATIVE SHARDED INPUT (`input_is_native`). When the block IS this core's own
// resident shard, cb_input_rows is PLACED ON the shard buffer by the host, so
// the block's bytes are already the CB's contents and there is nothing to move:
// `load_block` degenerates to marking the block's pages available. That is what
// consuming a shard means — an accessor read of a core's own shard would go out
// over the NoC to fetch bytes that are already in this core's L1. The accessor
// stays declared unconditionally (it owns the interleaved leg and the non-local
// cross-spec leg) so the compile-time arg indices never move.
//
// PADDED INPUT (`pad_active`) — op_design.md's `grid2d_padded`, ADDITIVE on the
// block that already exists. The block grid, the core assignment, the CBs and
// the compute call are unchanged; only `load_block` gains a fill. Two pad
// regions exist and they are separate arithmetic:
//   * the W tail — bytes [valid_bytes, block_row_bytes) of a row that HAS data.
//     At most one tile's worth (32 elements), because C = ceil(W/32); filled in
//     place with `fill_l1_range`, which is alignment-aware and is exactly the
//     helper written for a row whose pad offset is not 4-byte aligned.
//   * a fully padded ROW — an H tail row, a row of an all-pad tile COLUMN
//     (`pad_mode="explicit"` past the tile round), or a row of an all-pad
//     leading-dim slice. Sourced by ONE local L1->L1 NoC read from `cb_pad_row`,
//     so the DM engine moves the bytes rather than the RISC storing them.
// The two phases are ordered read-then-fill with the barrier BETWEEN them, so a
// CPU store into the tail of a row can never race the NoC write into its head.
//
// RETILE INPUT (`is_retile`) — a TILE input re-laid at ANOTHER tile height.
// This is a genuinely distinct block operation, not a parameterization of the
// stick reader: the source's pages are whole TILES, so the reader walks FACES.
// `read_sticks_for_tilize` cannot express it at all — it is stick-indexed by
// construction (`accessor.get_noc_addr(start_page + block_row + row, ...)`,
// tilize_helpers_dataflow.inl:121) and there are no sticks to index.
//
// The whole algorithm is ONE derived quantity, `retile_copy_unit` (host side,
// mirrored in the constexprs below): the largest byte run that is contiguous in
// BOTH tile layouts. A tile of height `h` is `h / min(h,16)` face-ROWS of two
// `min(h,16) x 16` faces, face-row-major, elements row-major inside a face
// (`tt_metal/impl/data_format/tile.cpp:TILE_FACE_HW_CHOICES`). So:
//   * equal face heights (`min(h_in,16) == min(h_out,16)`) — a face PAIR is
//     adjacent in both layouts, so the run is `min(h_in,h_out)` rows x all 32
//     columns (a slab: left face then right face, NOT row-major).
//   * different face heights — the run is one face FRAGMENT,
//     `min(h_in,h_out)` rows x 16 columns.
// Each run is issued as one `noc_async_read` from the source tile page's byte
// offset into the destination tile's byte offset, so the output tile is
// ASSEMBLED IN PLACE in cb_output_tiles and there is no compute stage and no
// row-major intermediate. This is the opposite of the untilize/re-tilize round
// trip op_design.md ranks `rejected`: one DRAM crossing each way, the minimum.
//
// This branch is therefore the only one that pushes to cb_output_tiles instead
// of cb_input_rows (which stays a one-page stub). The writer is unchanged — it
// already stores whole output tile pages in exactly this order.
//
// The H tail is also why this branch cannot call `read_sticks_for_tilize`:
// that helper spans ONE contiguous stick run (`start_page + block_row + row`),
// which is only a valid tile-row index when `H % tile_h == 0` — with an H tail
// the source rows RESTART at every image boundary. So the block's read is
// SEGMENTED per image here, one reserve / read-burst / barrier / push per
// tile-row, which is the helper's own shape with the segmentation written out.
// RECORDED GAP: `rows_per_segment` + a `pad_value` pair (or a `fill` callback)
// alongside `byte_offset_within_page` would close both in the helper.

// STAGE INSTRUMENTATION (Perf 1) — PERMANENT, see perf_instrumentation.hpp's
// durability contract. `reader_read_block` is deliberately at BLOCK granularity:
// `read_sticks_for_tilize` owns the reserve / issue / barrier / push cycle
// internally, so a zone at the call site is that whole cycle's OCCUPANCY, not
// the read's cost. The issue-vs-barrier split for this stage comes from the
// ablation switches below, not from a finer zone (splitting it would mean
// zoning inside a shared kernel_lib helper).
//
// ABLATION SWITCHES (`TILIZE_ABLATE=reads,writes,compute`, plumbed as kernel
// defines by tilize_program_descriptor). Each removes a stage's PAYLOAD and
// keeps its synchronization — the loop, the CB reserve/push and the barrier —
// so the pipeline still runs and the output is wrong by design. They exist
// because zones report occupancy and only a cumulative peel reports cost.
#include "api/dataflow/dataflow_api.h"
#include "ttnn/cpp/ttnn/kernel_lib/l1_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/perf_instrumentation.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers_dataflow.hpp"

#ifdef TILIZE_ABLATE_READS
#define TILIZE_NOC_READ(...) ((void)0)
#else
#define TILIZE_NOC_READ(...) noc_async_read(__VA_ARGS__)
#endif

namespace {

constexpr uint32_t kTileWidth = 32;
constexpr uint32_t kFaceWidth = 16;
constexpr uint32_t kFacesPerRow = kTileWidth / kFaceWidth;  // 2

// Byte-offset-in-elements of element (r, c) inside a tile of face height `a`.
// `a == min(tile_h, 16)`; faces are laid out face-row-major, elements row-major
// within a face. This is the ONE address rule the retile block operation needs.
template <uint32_t a>
FORCE_INLINE uint32_t tile_elem_offset(uint32_t r, uint32_t c) {
    const uint32_t face_row = r / a;
    const uint32_t row_in_face = r - face_row * a;
    const uint32_t face_col = c / kFaceWidth;
    const uint32_t col_in_face = c - face_col * kFaceWidth;
    return ((face_row * kFacesPerRow + face_col) * a + row_in_face) * kFaceWidth + col_in_face;
}

}  // namespace

void kernel_main() {
    constexpr uint32_t cb_input_rows = get_compile_time_arg_val(0);
    constexpr uint32_t block_width_tiles = get_compile_time_arg_val(1);
    constexpr uint32_t tile_h = get_compile_time_arg_val(2);
    constexpr uint32_t tensor_row_blocks = get_compile_time_arg_val(3);  // R
    constexpr uint32_t num_row_groups = get_compile_time_arg_val(4);
    constexpr uint32_t num_w_chunks = get_compile_time_arg_val(5);
    constexpr uint32_t block_row_bytes = get_compile_time_arg_val(6);  // block_width_tiles*32*elem
    constexpr uint32_t input_is_native = get_compile_time_arg_val(7);
    constexpr uint32_t input_pages_per_row = get_compile_time_arg_val(8);
    constexpr uint32_t in_page_width_bytes = get_compile_time_arg_val(9);
    // --- padding. All eight are inert and the branch compiles out at pad_active == 0.
    constexpr uint32_t pad_active = get_compile_time_arg_val(10);
    constexpr uint32_t cb_pad_row = get_compile_time_arg_val(11);
    constexpr uint32_t elem_size = get_compile_time_arg_val(12);
    constexpr uint32_t pad_word = get_compile_time_arg_val(13);
    constexpr uint32_t in_num_images = get_compile_time_arg_val(14);
    constexpr uint32_t in_rows_per_image = get_compile_time_arg_val(15);   // the INPUT's logical H
    constexpr uint32_t in_row_bytes = get_compile_time_arg_val(16);        // the INPUT's logical W, in bytes
    constexpr uint32_t rows_per_image_out = get_compile_time_arg_val(17);  // tile-rows per image, padded
    // --- retile. All six are inert and the branch compiles out at is_retile == 0.
    constexpr uint32_t is_retile = get_compile_time_arg_val(18);
    constexpr uint32_t in_tile_h = get_compile_time_arg_val(19);               // the INPUT's tile height
    constexpr uint32_t in_tile_rows_per_image = get_compile_time_arg_val(20);  // ceil(in H / in_tile_h)
    constexpr uint32_t cb_output_tiles = get_compile_time_arg_val(21);
    constexpr uint32_t tensor_col_tiles = get_compile_time_arg_val(22);  // C
    constexpr uint32_t out_tile_bytes = get_compile_time_arg_val(23);
    // --- ragged column tail (Refinement 6). The FIRST tile column this core
    // range's blocks cover. 0 for the full-width range and for every plan with
    // no tail, which is what keeps this arg inert on the smooth-`C` geometries.
    // The tail range is a SECOND core range running this same kernel with its
    // own `block_width_tiles`, so within one core the CB push/pop quantum is
    // still a single constant and neither endpoint can wrap mid-transfer.
    constexpr uint32_t col_tile_offset = get_compile_time_arg_val(24);
    constexpr uint32_t col_byte_offset = col_tile_offset * kTileWidth * elem_size;
    // --- split reader (Refinement 6). Inert at 0, which is every plan whose
    // read transaction is large enough not to be RISC-V-issue-bound. When it is
    // on, this kernel reads the LEADING `block_row_extent - rows_writer`
    // tile-rows of each block and the WRITER kernel reads the rest into its own
    // input CB (see tilize_writer.cpp). The block operation is unchanged — one
    // `read_sticks_for_tilize` call per block, over a shorter row range.
    constexpr uint32_t split_reader_rows = get_compile_time_arg_val(25);
    constexpr uint32_t split_writer_share_pct = get_compile_time_arg_val(26);
    constexpr auto in_args = TensorAccessorArgs<27>();

    const uint32_t src_addr = get_arg_val<uint32_t>(0);
    const uint32_t start_block_id = get_arg_val<uint32_t>(1);
    const uint32_t num_blocks_this_core = get_arg_val<uint32_t>(2);
    // 1 on the solved plan (contiguous block ranges); the core count on the
    // shard-driven plan, where core i owns shards {i, i+N, i+2N, ...}.
    const uint32_t block_stride = get_arg_val<uint32_t>(3);

    // Stick-indexed accessor over the ROW_MAJOR input; page size comes from the
    // accessor's own compile-time args (the tensor's aligned stick size).
    [[maybe_unused]] const auto in_acc = TensorAccessor(in_args, src_addr);

    // Seed cb_pad_row: ONE block row of the fill, built once per kernel.
    // 32 elements go in by hand (<= 128 B, so the store loop is bounded by the
    // TILE WIDTH and not by the block width), then local L1->L1 reads DOUBLE
    // the filled span until the whole row is covered — log2(block_width_tiles)
    // transfers, the DM engine moving the bulk. Filling `block_row_bytes` with a
    // RISC store loop instead would be a per-word walk of up to 32 KB.
    [[maybe_unused]] uint32_t pad_row_addr = 0;
    if constexpr (pad_active) {
        MaybeDeviceZoneScope("reader_pad_seed");
        pad_row_addr = get_write_ptr(cb_pad_row);
        constexpr uint32_t seed_bytes = block_row_bytes / block_width_tiles;  // == TILE_WIDTH * elem_size
        dataflow_kernel_lib::fill_l1_range<elem_size>(pad_row_addr, seed_bytes, pad_word);
        for (uint32_t filled = seed_bytes; filled < block_row_bytes;) {
            const uint32_t chunk = (filled < block_row_bytes - filled) ? filled : block_row_bytes - filled;
            noc_async_read(get_noc_addr(pad_row_addr), pad_row_addr + filled, chunk);
            noc_async_read_barrier();
            filled += chunk;
        }
    }

    for (uint32_t b = 0; b < num_blocks_this_core; ++b) {
        // resolve_block — index arithmetic only, from the same CT plan the
        // compute and writer kernels see.
        const uint32_t block_id = start_block_id + b * block_stride;
        const uint32_t row_group = block_id / num_w_chunks;
        const uint32_t w_chunk = block_id - row_group * num_w_chunks;

        // Balanced monotone row split; needs no remainder table.
        const uint32_t row_start = (row_group * tensor_row_blocks) / num_row_groups;
        const uint32_t row_end = ((row_group + 1) * tensor_row_blocks) / num_row_groups;
        const uint32_t block_row_extent = row_end - row_start;

        if constexpr (is_retile) {
            MaybeDeviceZoneScope("reader_retile_block");
            // retile_block — the face-walking block operation. Assembles this
            // block's output tiles IN PLACE in cb_output_tiles out of the input
            // tiles' faces, one `retile_copy_unit` run per NoC read.
            // `in_tile_h` is 0 on the ROW_MAJOR path, and `kernel_main` is not a
            // template, so the DISCARDED branch of an `if constexpr` is still
            // type-checked: every divisor below has to stay a valid constant
            // expression there too. Degenerating to the output's own geometry
            // makes this whole (dead) instantiation the identity re-lay.
            constexpr uint32_t src_tile_h = (in_tile_h > 0) ? in_tile_h : tile_h;
            constexpr uint32_t in_face_h = (src_tile_h < kFaceWidth) ? src_tile_h : kFaceWidth;
            constexpr uint32_t out_face_h = (tile_h < kFaceWidth) ? tile_h : kFaceWidth;
            // retile_copy_unit(in_tile_h, tile_h), mirrored from the host.
            constexpr uint32_t unit_rows = (src_tile_h < tile_h) ? src_tile_h : tile_h;
            constexpr uint32_t unit_cols = (in_face_h == out_face_h) ? kTileWidth : kFaceWidth;
            constexpr uint32_t unit_bytes = unit_rows * unit_cols * elem_size;
            constexpr uint32_t unit_rows_per_tile = tile_h / unit_rows;
            constexpr uint32_t unit_cols_per_tile = kTileWidth / unit_cols;

            const uint32_t col_base = col_tile_offset + w_chunk * block_width_tiles;
            for (uint32_t tr = 0; tr < block_row_extent; ++tr) {
                const uint32_t out_tile_row = row_start + tr;
                // Per-IMAGE split, for the same reason the padded branch needs
                // one: the input's and the output's padded per-image heights do
                // not have to agree (H=20 is 3 input tile-rows at 8 and 5
                // output tile-rows at 4), so a folded output tile-row is only a
                // folded INPUT tile-row after the image is factored out.
                const uint32_t image = out_tile_row / rows_per_image_out;
                const uint32_t row_in_image = (out_tile_row - image * rows_per_image_out) * tile_h;

                cb_reserve_back(cb_output_tiles, block_width_tiles);
                uint32_t tile_l1 = get_write_ptr(cb_output_tiles);

                // Every transfer of the whole tile-row goes behind ONE barrier,
                // matching the block quantum of every other branch here: the
                // in-flight depth is `block_width_tiles * unit_rows_per_tile *
                // unit_cols_per_tile` reads, which is where the tiny-unit
                // geometries (1 -> 32 issues 64 x 32 B per output tile) get
                // their amortization.
                for (uint32_t i = 0; i < block_width_tiles; ++i) {
                    const uint32_t tile_col = col_base + i;
                    for (uint32_t ur = 0; ur < unit_rows_per_tile; ++ur) {
                        const uint32_t out_row_local = ur * unit_rows;
                        const uint32_t src_row = row_in_image + out_row_local;
                        const uint32_t in_tile_row_in_image = src_row / src_tile_h;
                        const uint32_t in_row_local = src_row - in_tile_row_in_image * src_tile_h;
                        const uint32_t in_page =
                            (image * in_tile_rows_per_image + in_tile_row_in_image) * tensor_col_tiles + tile_col;
                        for (uint32_t uc = 0; uc < unit_cols_per_tile; ++uc) {
                            const uint32_t col = uc * unit_cols;
                            const uint32_t src_off = tile_elem_offset<in_face_h>(in_row_local, col) * elem_size;
                            const uint32_t dst_off = tile_elem_offset<out_face_h>(out_row_local, col) * elem_size;
                            TILIZE_NOC_READ(in_acc.get_noc_addr(in_page, src_off), tile_l1 + dst_off, unit_bytes);
                        }
                    }
                    tile_l1 += out_tile_bytes;
                }
                noc_async_read_barrier();
                cb_push_back(cb_output_tiles, block_width_tiles);
            }
        } else if constexpr (pad_active) {
            MaybeDeviceZoneScope("reader_pad_block");
            // load_block, PADDED + SEGMENTED. `pad_active` implies the input is
            // read through the accessor (the host clears `input_native`, since a
            // resident shard holds no room for the fill).
            //
            // The column extent is the same for every tile-row of the block, so
            // the W-tail split is computed once here:
            //   valid_bytes = the part of this w_chunk that the input's logical
            //                 row actually reaches (0 for an all-pad column)
            //   pad_bytes   = the rest, < TILE_WIDTH * elem_size whenever
            //                 valid_bytes > 0 (C = ceil(W / TILE_WIDTH))
            const uint32_t col_bytes = col_byte_offset + w_chunk * block_row_bytes;
            const uint32_t valid_bytes =
                (in_row_bytes > col_bytes)
                    ? ((in_row_bytes - col_bytes < block_row_bytes) ? in_row_bytes - col_bytes : block_row_bytes)
                    : 0;
            const uint32_t pad_bytes = block_row_bytes - valid_bytes;
            // Only meaningful (and only used) when valid_bytes > 0, where the
            // host guarantees the whole segment sits inside ONE source page.
            const uint32_t page_col = col_bytes / in_page_width_bytes;
            const uint32_t byte_in_page = col_bytes - page_col * in_page_width_bytes;

            for (uint32_t tr = 0; tr < block_row_extent; ++tr) {
                // SEGMENTED per image: with an H tail the source sticks restart
                // at each image boundary, so the tile-row index has to be split
                // before it can become a stick index.
                const uint32_t global_tr = row_start + tr;
                const uint32_t image = global_tr / rows_per_image_out;
                const uint32_t first_src_row = (global_tr - image * rows_per_image_out) * tile_h;
                uint32_t valid_rows = 0;
                if (image < in_num_images && first_src_row < in_rows_per_image) {
                    valid_rows = in_rows_per_image - first_src_row;
                    if (valid_rows > tile_h) {
                        valid_rows = tile_h;
                    }
                }

                cb_reserve_back(cb_input_rows, block_width_tiles);
                const uint32_t block_addr = get_write_ptr(cb_input_rows);
                const uint32_t stick_base =
                    (image * in_rows_per_image + first_src_row) * input_pages_per_row + page_col;

                // Phase 1 — every NoC transfer of the tile-row, behind ONE barrier.
                uint32_t l1_write_addr = block_addr;
                for (uint32_t row = 0; row < tile_h; ++row) {
                    if (row < valid_rows && valid_bytes > 0) {
                        TILIZE_NOC_READ(
                            in_acc.get_noc_addr(stick_base + row * input_pages_per_row, byte_in_page),
                            l1_write_addr,
                            valid_bytes);
                    } else {
                        // A fully padded row: one L1 -> L1 transfer from the seeded row.
                        TILIZE_NOC_READ(get_noc_addr(pad_row_addr), l1_write_addr, block_row_bytes);
                    }
                    l1_write_addr += block_row_bytes;
                }
                noc_async_read_barrier();

                // Phase 2 — the W tail of the rows that carried data. AFTER the
                // barrier: these are RISC stores into the same L1 words the NoC
                // was just writing the head of, and a store issued while that
                // write is in flight is a race on the row's last aligned word.
                if (pad_bytes > 0 && valid_bytes > 0) {
                    l1_write_addr = block_addr + valid_bytes;
                    for (uint32_t row = 0; row < valid_rows; ++row) {
                        dataflow_kernel_lib::fill_l1_range<elem_size>(l1_write_addr, pad_bytes, pad_word);
                        l1_write_addr += block_row_bytes;
                    }
                }
                cb_push_back(cb_input_rows, block_width_tiles);
            }
        } else if constexpr (!input_is_native && input_pages_per_row > 1) {
            MaybeDeviceZoneScope("reader_strided_block");
            // load_block, STRIDED. The source's shard cuts the width, so one row
            // spans `input_pages_per_row` pages and consecutive sticks are that
            // far apart in page index — `read_sticks_for_tilize` cannot express
            // it (it is stick-indexed by construction: `start_page + block_row +
            // row`, stride 1). RECORDED GAP: the helper would close this with a
            // `page_stride_per_row` parameter alongside `byte_offset_within_page`;
            // this branch is that parameter, written out. Everything else is the
            // helper's own shape — one reserve/read-burst/push per TILE-ROW, one
            // barrier per tile-row, `block_row_bytes` per stick.
            //
            // The host guarantees `block_row_bytes` divides `in_page_width_bytes`
            // (block_width_tiles is a common divisor of C and the page width in
            // tiles), so a block's row segment always sits inside ONE page.
            const uint32_t col_bytes = col_byte_offset + w_chunk * block_row_bytes;
            const uint32_t page_col = col_bytes / in_page_width_bytes;
            const uint32_t byte_in_page = col_bytes - page_col * in_page_width_bytes;
            for (uint32_t tr = 0; tr < block_row_extent; ++tr) {
                cb_reserve_back(cb_input_rows, block_width_tiles);
                uint32_t l1_write_addr = get_write_ptr(cb_input_rows);
                const uint32_t first_stick = (row_start + tr) * tile_h;
                for (uint32_t row = 0; row < tile_h; ++row) {
                    TILIZE_NOC_READ(
                        in_acc.get_noc_addr((first_stick + row) * input_pages_per_row + page_col, byte_in_page),
                        l1_write_addr,
                        block_row_bytes);
                    l1_write_addr += block_row_bytes;
                }
                noc_async_read_barrier();
                cb_push_back(cb_input_rows, block_width_tiles);
            }
        } else if constexpr (input_is_native) {
            MaybeDeviceZoneScope("reader_native_block");
            // load_block, zero-copy: the block's `block_row_extent` tile-rows are
            // already resident in this core's L1 behind cb_input_rows. Marking
            // the whole block available in one push keeps the reader's quantum a
            // BLOCK, matching the accessor leg; the compute helper still waits
            // and pops one tile-row at a time.
            const uint32_t block_pages = block_row_extent * block_width_tiles;
            cb_reserve_back(cb_input_rows, block_pages);
            cb_push_back(cb_input_rows, block_pages);
        } else {
            MaybeDeviceZoneScope("reader_read_block");
            // load_block. Valid as one contiguous stick run because H % tile_h == 0
            // on the tile-aligned path, so tile-row r starts at stick r * tile_h
            // exactly, even where R comes from the leading-dim fold.
            uint32_t rows_reader = block_row_extent;
            if constexpr (split_reader_rows > 0) {
                uint32_t rows_writer = (block_row_extent * split_writer_share_pct) / 100;
                if (rows_writer >= block_row_extent) {
                    rows_writer = block_row_extent - 1;
                }
                rows_reader = block_row_extent - rows_writer;
            }
#ifdef TILIZE_ABLATE_READS
            // The helper's own reserve / barrier / push cycle with the PAYLOAD
            // removed: same trip count, same CB quantum, no NoC read. Written
            // out here because the reads live inside `read_sticks_for_tilize`
            // and a shared kernel_lib helper is not the place for an op's
            // ablation switch.
            for (uint32_t tr = 0; tr < rows_reader; ++tr) {
                cb_reserve_back(cb_input_rows, block_width_tiles);
                noc_async_read_barrier();
                cb_push_back(cb_input_rows, block_width_tiles);
            }
#elif defined(TILIZE_ROTATE_VARIANT)
            // ---- perf_experiments/reader_bank_rotate (isolated bake-off) ----
            // Same bytes, same transaction size (block_row_bytes), same trip
            // count (tile_h stick reads per block iteration, behind ONE
            // barrier) as `read_sticks_for_tilize`'s TILE mode -- only the
            // ISSUE ORDER of the tile_h reads changes. Every stick still lands
            // at its helper-identical L1 offset (`l1_base + row *
            // block_row_bytes`), addressed by ROW not by issue slot, so
            // reordering behind one shared barrier is free and the block is
            // bit-identical to the baseline regardless of which order wins.
            //
            // Mechanism this chases: on an interleaved DRAM tensor, source
            // page p maps to bank (p % NUM_BANKS) (round-robin assignment,
            // `tensor_accessor.h:get_bank_and_offset_from_page_id`). On a
            // geometry where every core shares `start_page` (R==1, one row
            // group -- true for the focus shape and both R==1 domain points
            // below) the whole 64-core grid issues row 0 first, row 1 second,
            // ... in lockstep, so at read-step r every core hammers bank
            // `r % NUM_BANKS` while the other NUM_BANKS-1 banks idle. Rotating
            // core k's start row de-synchronizes the grid so all NUM_BANKS
            // banks are hit every step instead of one at a time.
            //
            // TILIZE_ROTATE_VARIANT (compile define, set by the test's
            // monkeypatched `_ablation_defines`):
            //   1 = raw loop, ASCENDING (r0 = 0). Isolates "raw vs helper" --
            //       the control that separates a win from writing the loop out
            //       (no rotation) from a win from the rotation itself.
            //   2 = rotate start row by `(w_chunk * TILIZE_ROTATE_STRIDE) %
            //       tile_h` (stride via a second compile define, default 1).
            //       Stride 1 IS "rotate by the core's own linear index": on
            //       the smooth/no-tail plans exercised here `w_chunk` is the
            //       core's own linear (raster) position whenever
            //       num_blocks_this_core == 1 (`split_work_to_cores`'s
            //       row-wise, contiguous block-id assignment), so this is the
            //       cheapest possible derivation -- no new runtime arg, reuses
            //       a value the block-resolve arithmetic already computed.
            //   3 = rotate so THIS block iteration's FIRST read lands on DRAM
            //       bank `w_chunk % NUM_DRAM_BANKS` exactly, solved from the
            //       page->bank rule above rather than guessed.
            {
                // NUM_DRAM_BANKS is a box constant (12, stated in the shared
                // perf-tournament context for this n150 Wormhole B0 box), NOT
                // a general derivation -- a graduated version would need this
                // read from the device/accessor rather than hardcoded. Fine
                // for an isolated bench; flagged in the report as a gap for
                // whoever integrates variant 3.
                constexpr uint32_t kNumDramBanks = 12;
#ifdef TILIZE_ROTATE_STRIDE
                constexpr uint32_t kStride = TILIZE_ROTATE_STRIDE;
#else
                constexpr uint32_t kStride = 1;
#endif
                const uint32_t byte_off = col_byte_offset + w_chunk * block_row_bytes;
                for (uint32_t blk = 0; blk < rows_reader; ++blk) {
                    const uint32_t start_page = row_start * tile_h + blk * tile_h;
                    uint32_t r0 = 0;
#if TILIZE_ROTATE_VARIANT == 2
                    r0 = (w_chunk * kStride) % tile_h;
#elif TILIZE_ROTATE_VARIANT == 3
                    {
                        const uint32_t target_bank = w_chunk % kNumDramBanks;
                        const uint32_t base_bank = start_page % kNumDramBanks;
                        // kNumDramBanks(12) < tile_h(32), so r0 < tile_h always.
                        r0 = (target_bank + kNumDramBanks - base_bank) % kNumDramBanks;
                    }
#endif
                    cb_reserve_back(cb_input_rows, block_width_tiles);
                    const uint32_t l1_base = get_write_ptr(cb_input_rows);
                    for (uint32_t i = 0; i < tile_h; ++i) {
                        const uint32_t row = (r0 + i) % tile_h;  // issue ORDER only
                        TILIZE_NOC_READ(
                            in_acc.get_noc_addr(start_page + row, byte_off),
                            l1_base + row * block_row_bytes,
                            block_row_bytes);
                    }
                    noc_async_read_barrier();
                    cb_push_back(cb_input_rows, block_width_tiles);
                }
            }
#else
            dataflow_kernel_lib::read_sticks_for_tilize<cb_input_rows, dataflow_kernel_lib::TilizeGranularity::TILE>(
                in_acc,
                /* total_num_rows          */ rows_reader * tile_h,
                /* row_bytes               */ block_row_bytes,
                /* start_page              */ row_start * tile_h,
                /* byte_offset_within_page */ col_byte_offset + w_chunk * block_row_bytes);
#endif
        }
    }
}
