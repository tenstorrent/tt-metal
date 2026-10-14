// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// tilize reader (NCRISC / NoC0) — block operation `load_sticks_block`.
//
// Walks this core's rectangle (row_start_tile, core_rows_tiles, col_start_tile, all columns)
// as column blocks of block_width_tiles (ragged last block = last_block_width_tiles, own CB ring),
// each split into row blocks of block_height_tiles. Per tile-row (the streaming quantum):
// reserve `width` tile-sized pages, issue tile_h stick-segment reads (all in flight), one
// barrier, push `width`. Page layout in the CB: tile_h rows of width*32 elements, which is
// what the symmetric tilize helper consumes.
//
// Pad fill (has_padding): the output tile grid is the PADDED shape; the input is that shape's
// prefix (left-expanded with 1s). Each segment reads only its in-range data bytes; the W tail,
// H-tail sticks, sticks of pad images and whole pad tile-columns are filled by local L1->L1 NoC
// copies from a one-segment fill row (the pad value in the input format), prepared once.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "ttnn/cpp/ttnn/kernel_lib/local_copy_helpers_dataflow.hpp"

void kernel_main() {
    constexpr uint32_t cb_rm_sticks = get_compile_time_arg_val(0);
    constexpr uint32_t cb_rm_sticks_tail = get_compile_time_arg_val(1);
    constexpr uint32_t tile_h = get_compile_time_arg_val(2);
    constexpr uint32_t tile_col_bytes = get_compile_time_arg_val(3);  // 32 * elem bytes
    constexpr uint32_t image_height_sticks = get_compile_time_arg_val(4);
    constexpr uint32_t tile_rows_per_image = get_compile_time_arg_val(5);
    constexpr uint32_t stick_page_bytes = get_compile_time_arg_val(6);
    constexpr uint32_t block_width_tiles = get_compile_time_arg_val(7);
    constexpr uint32_t last_block_width_tiles = get_compile_time_arg_val(8);
    constexpr uint32_t num_col_blocks = get_compile_time_arg_val(9);
    constexpr bool has_tail = get_compile_time_arg_val(10) != 0;
    // Bank-coalesced stick reads (only when one column block spans the whole stick and images are
    // tile-row aligned): interleaved page p lives in bank p % num_banks at offset
    // (p / num_banks) * page, so the sticks of one bank inside a run of consecutive sticks are
    // contiguous there and come in ONE wide read into a reader-private staging buffer
    // (staging_tile_rows tile-rows per chunk; the first chunk may be shorter). The staging ring has
    // staging_slots slots, each chunk tagged with its own transaction id, so a chunk is retired by
    // a per-trid barrier while the next chunks' DRAM reads stay on the wire. Local L1->L1 NoC
    // writes then restore row order for the tilize.
    constexpr bool coalesce_bank_reads = get_compile_time_arg_val(11) != 0;
    constexpr uint32_t num_banks = get_compile_time_arg_val(12);
    constexpr uint32_t cb_stick_staging = get_compile_time_arg_val(13);
    constexpr uint32_t staging_tile_rows = get_compile_time_arg_val(14);
    constexpr bool has_padding = get_compile_time_arg_val(15) != 0;
    constexpr uint32_t cb_pad_fill = get_compile_time_arg_val(16);
    constexpr uint32_t in_row_bytes = get_compile_time_arg_val(17);    // logical bytes of one input stick
    constexpr uint32_t fill_row_bytes = get_compile_time_arg_val(18);  // >= max segment + fill_phase_bytes
    constexpr uint32_t fill_phase_bytes = get_compile_time_arg_val(19);
    constexpr bool leading_pad = get_compile_time_arg_val(20) != 0;
    // Leading (batch) dims, outermost first: output (padded) and input extents.
    constexpr uint32_t out_l0 = get_compile_time_arg_val(21), out_l1 = get_compile_time_arg_val(22),
                       out_l2 = get_compile_time_arg_val(23), out_l3 = get_compile_time_arg_val(24);
    constexpr uint32_t in_l0 = get_compile_time_arg_val(25), in_l1 = get_compile_time_arg_val(26),
                       in_l2 = get_compile_time_arg_val(27), in_l3 = get_compile_time_arg_val(28);
    // Resident input (sharded regimes): cb_rm_sticks is backed on this core's own input shard, which
    // already holds the block in the layout the tilize consumes — no NoC traffic at all.
    constexpr bool in_resident = get_compile_time_arg_val(29) != 0;
    // Input pages per stick (> 1 for a width/block-sharded RM input: shard-width pieces) and the
    // logical bytes of one page; page id of (stick, piece) = stick * in_pages_per_stick + piece.
    constexpr uint32_t in_pages_per_stick = get_compile_time_arg_val(30);
    constexpr uint32_t in_page_data_bytes = get_compile_time_arg_val(31);
    constexpr uint32_t staging_slots = get_compile_time_arg_val(32);  // coalesced path: chunks in flight
    // Coalesced path, split reorder: the writer RISC copies the last writer_copy_sticks sticks of
    // every tile-row (on its own NoC, in parallel). Per tile-row the reader posts a copy job
    // {dst, staging slot base, first stick} on cb_copy_job and waits for the ack on cb_copy_done.
    constexpr uint32_t writer_copy_sticks_arg = get_compile_time_arg_val(33);  // 0 = reader copies all
    constexpr uint32_t cb_copy_job = get_compile_time_arg_val(34);
    constexpr uint32_t cb_copy_done = get_compile_time_arg_val(35);
    constexpr uint32_t first_chunk_tile_rows = get_compile_time_arg_val(36);  // <= staging_tile_rows
    // retile_tile_input: the input is TILE at in_tile_h (faces in_face_h x 16). Per unit
    // (lcm(in_tile_h, tile_h) rows of one column block) the reader gets the unit's input tiles —
    // read into a staging slot (retile_slots ring, one trid per slot), or straight from this core's
    // resident input shard (in_resident) — and de-faces them with local 16-element copies into the
    // same row layout the ROW_MAJOR path pushes to cb_rm_sticks.
    constexpr bool retile = get_compile_time_arg_val(37) != 0;
    constexpr uint32_t in_tile_h = get_compile_time_arg_val(38);
    constexpr uint32_t in_face_h = get_compile_time_arg_val(39);
    constexpr uint32_t in_tile_page_bytes = get_compile_time_arg_val(40);
    constexpr uint32_t in_tile_rows_per_image = get_compile_time_arg_val(41);
    constexpr uint32_t in_cols_tiles = get_compile_time_arg_val(42);
    constexpr uint32_t retile_slots = get_compile_time_arg_val(43);
    constexpr uint32_t shard_cols_tiles = get_compile_time_arg_val(44);  // resident retile source
    // Segment path: a stick segment whose byte width is not a multiple of the source read alignment
    // (1-byte dtypes on 64 B-aligned DRAM) would land at a non-congruent L1 row. Such a block reads
    // each stick into an aligned slot of cb_stick_staging (bounce) and local-copies it (L1 needs only
    // 16 B) into its row. 0 = never bounce.
    constexpr uint32_t seg_read_alignment = get_compile_time_arg_val(45);
    constexpr auto in_args = TensorAccessorArgs<46>();

    const uint32_t src_addr = get_arg_val<uint32_t>(0);
    const uint32_t row_start_tile = get_arg_val<uint32_t>(1);
    const uint32_t core_rows_tiles = get_arg_val<uint32_t>(2);
    const uint32_t col_start_tile = get_arg_val<uint32_t>(3);
    const uint32_t block_height_tiles = get_arg_val<uint32_t>(4);
    // coalesced path: < min(num_banks, chunk sticks); segment path: < tile_h (start stick)
    const uint32_t bank_stagger_offset = get_arg_val<uint32_t>(5);
    const uint32_t fill_word = get_arg_val<uint32_t>(6);            // pad value, replicated to 32 bits

    if constexpr (retile) {
        constexpr uint32_t half_bytes = tile_col_bytes / 2;                   // one face row: 16 elements
        constexpr uint32_t face_bytes = in_face_h * half_bytes;               // one face
        constexpr uint32_t in_padded_h = in_tile_rows_per_image * in_tile_h;  // input sticks per image
        // Output tile-rows sharing one input tile-row (in_tile_h > tile_h), else 1.
        constexpr uint32_t rows_per_in_row = in_tile_h > tile_h ? in_tile_h / tile_h : 1;
        constexpr uint32_t max_unit_in_rows = tile_h > in_tile_h ? tile_h / in_tile_h : 1;
        constexpr uint32_t unit_h = max_unit_in_rows * in_tile_h;  // lcm(in_tile_h, tile_h) rows
        static_assert(half_bytes <= NOC_MAX_BURST_SIZE, "face row copy must be one packet");
        static_assert(!in_resident || num_col_blocks == 1, "resident retile source: one column block");
        static_assert(retile_slots >= 2 && retile_slots <= NOC_MAX_TRANSACTION_ID, "trid range");

        const auto tile_acc = TensorAccessor(in_args, src_addr, in_tile_page_bytes);
        // Resident source: this core's shard starts at input tile-row shard_k0 (flattened rows).
        const uint32_t shard_k0 = row_start_tile * tile_h / in_tile_h;
        const uint32_t staging_base = in_resident ? 0 : get_write_ptr(cb_stick_staging);
        Noc noc;
        noc_async_write_one_packet_set_state(get_noc_addr(src_addr), half_bytes);

        for (uint32_t col_block = 0; col_block < num_col_blocks; ++col_block) {
            const bool is_tail = has_tail && (col_block == num_col_blocks - 1);
            const uint32_t cb_id = is_tail ? cb_rm_sticks_tail : cb_rm_sticks;
            const uint32_t width_tiles = is_tail ? last_block_width_tiles : block_width_tiles;
            const uint32_t seg_bytes = width_tiles * tile_col_bytes;
            const uint32_t col0 = col_start_tile + col_block * block_width_tiles;
            const uint32_t slot_bytes = max_unit_in_rows * width_tiles * in_tile_page_bytes;

            // Unit starting at core-relative output row r: its first input tile-row k0, its input
            // tile-row count nk, and the number of output rows it serves.
            struct Unit {
                uint32_t k0, nk, rows;
            };
            auto unit_at = [&](uint32_t r) -> Unit {
                const uint32_t tile_row = row_start_tile + r;
                const uint32_t image = tile_row / tile_rows_per_image;
                const uint32_t row_in_image = tile_row - image * tile_rows_per_image;
                const uint32_t first_h = row_in_image * tile_h;
                const uint32_t valid = in_padded_h - first_h < tile_h ? in_padded_h - first_h : tile_h;
                uint32_t run = rows_per_in_row - row_in_image % rows_per_in_row;
                if (tile_rows_per_image - row_in_image < run) {
                    run = tile_rows_per_image - row_in_image;
                }
                if (core_rows_tiles - r < run) {
                    run = core_rows_tiles - r;
                }
                return {image * in_tile_rows_per_image + first_h / in_tile_h, (valid + in_tile_h - 1) / in_tile_h, run};
            };
            auto issue_unit = [&](const Unit& u, uint32_t slot) {
                dataflow_kernel_lib::set_read_trid(noc, slot + 1);
                uint32_t dst = staging_base + slot * slot_bytes;
                for (uint32_t kk = 0; kk < u.nk; ++kk) {
                    const uint32_t tile0 = (u.k0 + kk) * in_cols_tiles + col0;
                    for (uint32_t c = 0; c < width_tiles; ++c) {
                        noc_async_read(tile_acc.get_noc_addr(tile0 + c), dst, in_tile_page_bytes);
                        dst += in_tile_page_bytes;
                    }
                }
            };

            Unit cur = unit_at(0);
            uint32_t u_idx = 0;
            if constexpr (!in_resident) {
                issue_unit(cur, 0);
            }
            for (uint32_t r = 0; r < core_rows_tiles; ++u_idx) {
                const uint32_t slot = u_idx % retile_slots;
                const bool has_next = r + cur.rows < core_rows_tiles;
                const Unit next = has_next ? unit_at(r + cur.rows) : cur;
                // Source tile (kk, c) of the current unit: staging slot, or the resident shard.
                uint32_t src_base;
                uint32_t src_row_stride;
                if constexpr (in_resident) {
                    src_base = src_addr + (cur.k0 - shard_k0) * shard_cols_tiles * in_tile_page_bytes;
                    src_row_stride = shard_cols_tiles * in_tile_page_bytes;
                } else {
                    if (has_next) {
                        issue_unit(next, (u_idx + 1) % retile_slots);  // lands while this unit is de-faced
                    }
                    dataflow_kernel_lib::async_read_barrier_with_trid(noc, slot + 1);
                    src_base = staging_base + slot * slot_bytes;
                    src_row_stride = width_tiles * in_tile_page_bytes;
                }

                for (uint32_t j = 0; j < cur.rows; ++j, ++r) {
                    const uint32_t tile_row = row_start_tile + r;
                    const uint32_t row_in_image = tile_row - (tile_row / tile_rows_per_image) * tile_rows_per_image;
                    const uint32_t first_h = row_in_image * tile_h;
                    const uint32_t valid = in_padded_h - first_h < tile_h ? in_padded_h - first_h : tile_h;
                    // Input row of output stick 0, relative to the unit's first input tile-row.
                    const uint32_t h0 = first_h % unit_h;
                    cb_reserve_back(cb_id, width_tiles);
                    const uint32_t l1_base = get_write_ptr(cb_id);
                    for (uint32_t i = 0; i < valid; ++i) {
                        const uint32_t h = h0 + i;
                        const uint32_t kk = h / in_tile_h;
                        const uint32_t rr = h - kk * in_tile_h;
                        const uint32_t face_row = rr / in_face_h;
                        const uint32_t src_off = face_row * 2 * face_bytes + (rr - face_row * in_face_h) * half_bytes;
                        uint32_t src = src_base + kk * src_row_stride + src_off;
                        uint32_t dst = l1_base + i * seg_bytes;
                        for (uint32_t c = 0; c < width_tiles; ++c) {
                            noc_async_write_one_packet_with_state(src, dst);                            // left face row
                            noc_async_write_one_packet_with_state(src + face_bytes, dst + half_bytes);  // right
                            src += in_tile_page_bytes;
                            dst += tile_col_bytes;
                        }
                    }
                    noc_async_write_barrier();
                    cb_push_back(cb_id, width_tiles);
                }
                cur = next;
            }
        }
        dataflow_kernel_lib::set_read_trid(noc, 0);  // restore untagged
        return;
    }

    if constexpr (in_resident) {
        // One block = the whole resident shard (single column block): publish it in one push.
        const uint32_t n_pages = core_rows_tiles * block_width_tiles;
        cb_reserve_back(cb_rm_sticks, n_pages);
        cb_push_back(cb_rm_sticks, n_pages);
        return;
    }

    const auto in_acc = TensorAccessor(in_args, src_addr, stick_page_bytes);
    // Read bytes [offset, offset + n_bytes) of `stick` to `dst_l1`, split at page boundaries.
    auto read_stick_bytes = [&](uint32_t stick, uint32_t offset, uint32_t dst_l1, uint32_t n_bytes) {
        if constexpr (in_pages_per_stick == 1) {
            noc_async_read(in_acc.get_noc_addr(stick, offset), dst_l1, n_bytes);
        } else {
            while (n_bytes > 0) {
                const uint32_t piece = offset / in_page_data_bytes;
                const uint32_t in_page = offset - piece * in_page_data_bytes;
                const uint32_t n = (in_page_data_bytes - in_page < n_bytes) ? in_page_data_bytes - in_page : n_bytes;
                noc_async_read(in_acc.get_noc_addr(stick * in_pages_per_stick + piece, in_page), dst_l1, n);
                offset += n;
                dst_l1 += n;
                n_bytes -= n;
            }
        }
    };

    if constexpr (coalesce_bank_reads) {
        // Whole-stick single column block; tile-rows of this core are consecutive sticks.
        constexpr uint32_t chunk_sticks_max = staging_tile_rows * tile_h;
        constexpr uint32_t max_per_group = (chunk_sticks_max + num_banks - 1) / num_banks;
        constexpr uint32_t slot_bytes = num_banks * max_per_group * stick_page_bytes;
        constexpr uint32_t seg_bytes = block_width_tiles * tile_col_bytes;
        const uint32_t staging_base = get_write_ptr(cb_stick_staging);
        const uint32_t first_core_stick = row_start_tile * tile_h;  // images are tile-row aligned here

        // Chunk geometry: chunk 0 has first_chunk_tile_rows (<= staging_tile_rows) so the first data
        // lands sooner and the pipeline fills early; every later chunk has staging_tile_rows.
        auto chunk_row0 = [&](uint32_t chunk) -> uint32_t {
            return chunk == 0 ? 0 : first_chunk_tile_rows + (chunk - 1) * staging_tile_rows;
        };
        auto chunk_rows = [&](uint32_t chunk) -> uint32_t {
            const uint32_t row0 = chunk_row0(chunk);
            const uint32_t nominal = chunk == 0 ? first_chunk_tile_rows : staging_tile_rows;
            return (core_rows_tiles - row0 < nominal) ? core_rows_tiles - row0 : nominal;
        };
        const uint32_t num_chunks =
            core_rows_tiles <= first_chunk_tile_rows
                ? 1
                : 1 + (core_rows_tiles - first_chunk_tile_rows + staging_tile_rows - 1) / staging_tile_rows;

        auto issue_chunk = [&](uint32_t chunk, uint32_t slot) {
            const uint32_t n_sticks = chunk_rows(chunk) * tile_h;
            const uint32_t stick0 = first_core_stick + chunk_row0(chunk) * tile_h;
            const uint32_t base = staging_base + slot * slot_bytes;
            const uint32_t groups = n_sticks < num_banks ? n_sticks : num_banks;
            for (uint32_t k = 0; k < groups; ++k) {
                // Bank stagger: each core starts its bank walk at a different bank.
                uint32_t g = k + bank_stagger_offset;
                g = g >= groups ? g - groups : g;
                const uint32_t count = (n_sticks - g + num_banks - 1) / num_banks;
                noc_async_read(
                    in_acc.get_noc_addr(stick0 + g),
                    base + g * max_per_group * stick_page_bytes,
                    count * stick_page_bytes);
            }
        };

        // Local copies all target this core with the same size: program the write cmd buffer once
        // (the reader issues no other writes), then each copy only sets src/dst addresses.
        constexpr bool one_packet_copies = seg_bytes <= NOC_MAX_BURST_SIZE;
        // The writer's share applies to single-packet stick copies only (the writer gates identically).
        constexpr uint32_t writer_copy_sticks = one_packet_copies ? writer_copy_sticks_arg : 0;
        static_assert(writer_copy_sticks <= tile_h, "writer share exceeds the tile-row");
        constexpr uint32_t reader_copy_sticks = tile_h - writer_copy_sticks;
        if constexpr (one_packet_copies) {
            noc_async_write_one_packet_set_state(get_noc_addr(staging_base), seg_bytes);
        }

        // Chunk c lands in slot c % staging_slots with trid slot + 1 (trid 0 = untagged). At most
        // staging_slots - 1 chunks are in flight ahead of the one being reordered: chunk
        // c + staging_slots - 1 is issued once chunk c has landed, into the slot chunk c - 1 just
        // drained — the earliest chunk always has the DRAM to itself first.
        static_assert(staging_slots >= 2 && staging_slots <= NOC_MAX_TRANSACTION_ID, "trid range");
        Noc noc;
        auto issue_tagged = [&](uint32_t chunk) {
            const uint32_t slot = chunk % staging_slots;
            dataflow_kernel_lib::set_read_trid(noc, slot + 1);
            issue_chunk(chunk, slot);
        };
        for (uint32_t c = 0; c + 1 < staging_slots && c < num_chunks; ++c) {
            issue_tagged(c);
        }
        for (uint32_t c = 0; c < num_chunks; ++c) {
            const uint32_t slot = c % staging_slots;
            dataflow_kernel_lib::async_read_barrier_with_trid(noc, slot + 1);  // only chunk c
            if (c + staging_slots - 1 < num_chunks) {
                issue_tagged(c + staging_slots - 1);
            }
            const uint32_t base = staging_base + slot * slot_bytes;
            const uint32_t rows = chunk_rows(c);
            for (uint32_t t = 0; t < rows; ++t) {
                cb_reserve_back(cb_rm_sticks, block_width_tiles);
                uint32_t dst_l1 = get_write_ptr(cb_rm_sticks);
                if constexpr (writer_copy_sticks > 0) {
                    cb_reserve_back(cb_copy_job, 1);
                    volatile tt_l1_ptr uint32_t* job =
                        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_copy_job));
                    job[0] = dst_l1 + reader_copy_sticks * seg_bytes;
                    job[1] = base;
                    job[2] = t * tile_h + reader_copy_sticks;
                    cb_push_back(cb_copy_job, 1);
                }
                for (uint32_t i = 0; i < reader_copy_sticks; ++i) {
                    const uint32_t s = t * tile_h + i;  // stick index inside the chunk
                    const uint32_t src_l1 = base + ((s % num_banks) * max_per_group + s / num_banks) * stick_page_bytes;
                    if constexpr (one_packet_copies) {
                        noc_async_write_one_packet_with_state(src_l1, dst_l1);
                    } else {
                        noc_async_write(src_l1, get_noc_addr(dst_l1), seg_bytes);
                    }
                    dst_l1 += seg_bytes;
                }
                noc_async_write_barrier();
                if constexpr (writer_copy_sticks > 0) {
                    cb_wait_front(cb_copy_done, 1);
                    cb_pop_front(cb_copy_done, 1);
                }
                cb_push_back(cb_rm_sticks, block_width_tiles);
            }
            // Slot drained (all copies of this chunk acked, both RISCs): chunk c + staging_slots
            // may land in it (issued after chunk c + 1 lands).
        }
        dataflow_kernel_lib::set_read_trid(noc, 0);  // restore untagged
        return;
    }

    // Fill row: seed FILL_PHASE bytes with RISC stores, then grow it by doubling local NoC copies.
    uint32_t fill_base = 0;
    if constexpr (has_padding) {
        fill_base = (get_write_ptr(cb_pad_fill) + fill_phase_bytes - 1) & ~(fill_phase_bytes - 1);
        volatile tt_l1_ptr uint32_t* seed = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(fill_base);
        for (uint32_t k = 0; k < fill_phase_bytes / sizeof(uint32_t); ++k) {
            seed[k] = fill_word;
        }
        for (uint32_t have = fill_phase_bytes; have < fill_row_bytes;) {
            const uint32_t n = (fill_row_bytes - have < have) ? fill_row_bytes - have : have;
            noc_async_write(fill_base, get_noc_addr(fill_base + have), n);
            noc_async_write_barrier();
            have += n;
        }
    }
    // Local fill copy: the source offset matches dst modulo fill_phase_bytes, so the copy is
    // NoC-alignment congruent and in element phase (segments start at multiples of 32 elements).
    auto fill = [&](uint32_t dst_l1, uint32_t n_bytes) {
        noc_async_write(fill_base + (dst_l1 & (fill_phase_bytes - 1)), get_noc_addr(dst_l1), n_bytes);
    };

    uint32_t bounce_base = 0;
    if constexpr (seg_read_alignment != 0) {
        bounce_base = (get_write_ptr(cb_stick_staging) + seg_read_alignment - 1) & ~(seg_read_alignment - 1);
    }

    for (uint32_t col_block = 0; col_block < num_col_blocks; ++col_block) {
        const bool is_tail = has_tail && (col_block == num_col_blocks - 1);
        const uint32_t cb_id = is_tail ? cb_rm_sticks_tail : cb_rm_sticks;
        const uint32_t width_tiles = is_tail ? last_block_width_tiles : block_width_tiles;
        const uint32_t seg_bytes = width_tiles * tile_col_bytes;
        const uint32_t seg_offset_bytes = (col_start_tile + col_block * block_width_tiles) * tile_col_bytes;
        // In-range data bytes of this segment (W tail / whole pad columns => fewer, possibly 0).
        uint32_t data_bytes = seg_bytes;
        if constexpr (has_padding) {
            data_bytes =
                seg_offset_bytes >= in_row_bytes
                    ? 0
                    : (in_row_bytes - seg_offset_bytes < seg_bytes ? in_row_bytes - seg_offset_bytes : seg_bytes);
        }

        for (uint32_t rb_start = 0; rb_start < core_rows_tiles; rb_start += block_height_tiles) {
            const uint32_t rb_end =
                (rb_start + block_height_tiles < core_rows_tiles) ? rb_start + block_height_tiles : core_rows_tiles;
            for (uint32_t r = rb_start; r < rb_end; ++r) {
                // Per-image ceil fold: global tile-row -> (image, tile-row within image).
                const uint32_t tile_row = row_start_tile + r;
                const uint32_t image = tile_row / tile_rows_per_image;
                const uint32_t row_in_image = tile_row - image * tile_rows_per_image;
                const uint32_t first_h = row_in_image * tile_h;
                // Output image -> input image (leading-dim padding: decode in output radices).
                uint32_t in_image = image;
                bool image_valid = true;
                if constexpr (leading_pad) {
                    uint32_t rem = image;
                    const uint32_t d3 = rem % out_l3;
                    rem /= out_l3;
                    const uint32_t d2 = rem % out_l2;
                    rem /= out_l2;
                    const uint32_t d1 = rem % out_l1;
                    const uint32_t d0 = rem / out_l1;
                    image_valid = d0 < in_l0 && d1 < in_l1 && d2 < in_l2 && d3 < in_l3;
                    in_image = ((d0 * in_l1 + d1) * in_l2 + d2) * in_l3 + d3;
                }
                // Sticks [first_h, first_h + valid_sticks) of this tile-row exist in the input.
                uint32_t valid_sticks = tile_h;
                if constexpr (has_padding) {
                    valid_sticks =
                        (!image_valid || data_bytes == 0 || first_h >= image_height_sticks)
                            ? 0
                            : (image_height_sticks - first_h < tile_h ? image_height_sticks - first_h : tile_h);
                }
                const uint32_t first_stick = in_image * image_height_sticks + first_h;

                cb_reserve_back(cb_id, width_tiles);
                const uint32_t l1_base = get_write_ptr(cb_id);
                const bool bounce = seg_read_alignment != 0 && (seg_bytes % seg_read_alignment) != 0;
                const uint32_t read_base = bounce ? bounce_base : l1_base;
                const uint32_t read_stride =
                    bounce ? (seg_bytes + seg_read_alignment - 1) & ~(seg_read_alignment - 1) : seg_bytes;
                // Bank stagger: start each core's stick walk at a different stick (consecutive
                // sticks live in consecutive banks), so column-split cores reading the same sticks
                // do not all queue on one bank at once.
                uint32_t i = bank_stagger_offset < valid_sticks ? bank_stagger_offset : 0;
                for (uint32_t k = 0; k < valid_sticks; ++k) {
                    read_stick_bytes(first_stick + i, seg_offset_bytes, read_base + i * read_stride, data_bytes);
                    i = (i + 1 == valid_sticks) ? 0 : i + 1;
                }
                noc_async_read_barrier();
                if (bounce && data_bytes > 0) {
                    for (uint32_t k = 0; k < valid_sticks; ++k) {
                        noc_async_write(read_base + k * read_stride, get_noc_addr(l1_base + k * seg_bytes), data_bytes);
                    }
                    noc_async_write_barrier();
                }
                if constexpr (has_padding) {
                    // After the read barrier: a partial tail fill shares a NoC word with the data.
                    if (data_bytes < seg_bytes) {
                        for (uint32_t i = 0; i < valid_sticks; ++i) {
                            fill(l1_base + i * seg_bytes + data_bytes, seg_bytes - data_bytes);
                        }
                    }
                    for (uint32_t i = valid_sticks; i < tile_h; ++i) {
                        fill(l1_base + i * seg_bytes, seg_bytes);
                    }
                    noc_async_write_barrier();
                }
                cb_push_back(cb_id, width_tiles);
            }
        }
    }
}
