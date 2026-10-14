// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// distributed_rms_norm — worker reader (NCRISC, NoC0).
//   load_block           : x block (block_rows x core_w_tiles) -> cb_x_resident, one tile row per barrier;
//                          W_local % 32 != 0: the padding columns of the tensor's last W tile are zeroed in the CB
//                          page before the push (sharded: in the resident shard), so sum(x^2) never sees them;
//                          gamma (row 0 of each tile, no tilize) and the reduce scaler once per call.
//                          Sharded x (R9/R10): the block IS the resident shard; cb_x_resident is backed on it, so
//                          load_block only publishes the shard's tiles (no NoC read).
//   local_gather          : (R5 pre-combine, combiners) wait until every W chunk of my row group has written its
//                          records into my inbox, then place the S_w records of each of my tile rows into cb_lc.
//   R2 (several exchange rounds): round b+lookahead's load_block is issued before round b's gather, so DRAM reads run
//   under round b's exchange (host ROUND_LOOKAHEAD; cb_x_resident holds 1 + lookahead blocks).
//   gather_sources_block : wait stats-ready (one counter per port of my link, >= round + 1), pull the G*rec_w fp32
//   records of every tile row of the block from the
//                          port cores into cb_stats_combine (source s -> row s%32 of combine tile s/32), one
//                          stat_rows batch at a time.
//
// Raw-API justification (gamma placement, record gather): the data lands in tile rows, not whole tiles; there is
// no dataflow helper for a sub-tile row placement (see op_design.md "Helpers considered and rejected").

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/noc.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"

void kernel_main() {
    constexpr uint32_t cb_x_resident = get_compile_time_arg_val(0);
    constexpr uint32_t cb_gamma_tiles = get_compile_time_arg_val(1);
    constexpr uint32_t cb_scaler = get_compile_time_arg_val(2);
    constexpr uint32_t cb_stats_combine = get_compile_time_arg_val(3);
    constexpr bool has_gamma = get_compile_time_arg_val(4) != 0;
    constexpr uint32_t sem_stats_ready_base = get_compile_time_arg_val(5);  // port (l, d) -> base + 2l + d
    constexpr uint32_t gamma_elem_bytes = get_compile_time_arg_val(6);
    constexpr uint32_t gamma_page_bytes = get_compile_time_arg_val(7);  // one stick = the whole W_local slice
    constexpr uint32_t record_bytes = get_compile_time_arg_val(8);      // host RECORD_BYTES: 32 fp32 partial sums
    constexpr bool gamma_full_rows = get_compile_time_arg_val(9) != 0;  // replicate row 0 (fused, no-prescale path)
    constexpr bool x_sharded = get_compile_time_arg_val(10) != 0;       // cb_x_resident is backed on the x shard
    constexpr uint32_t cb_lc = get_compile_time_arg_val(11);
    constexpr uint32_t sem_local_combine = get_compile_time_arg_val(12);
    constexpr uint32_t w_tail = get_compile_time_arg_val(13);  // valid columns of the last W tile (0: aligned)
    constexpr uint32_t round_lookahead = get_compile_time_arg_val(14);  // rounds whose loads run ahead (host knob)
    // x tile layout: Bfp8_b = 64 B of shared exponents (one per face row), then 1 B (sign + mantissa) per element
    constexpr uint32_t x_exp_bytes = get_compile_time_arg_val(15);  // 0: plain float (Float32 / Float16_b)
    constexpr bool gamma_early_knob = get_compile_time_arg_val(16) != 0;  // host GAMMA_EARLY
    constexpr auto x_args = TensorAccessorArgs<17>();
    [[maybe_unused]] constexpr auto gamma_args = TensorAccessorArgs<x_args.next_compile_time_args_offset()>();

    constexpr uint32_t half_record = record_bytes / 2;  // 16 fp32 = one face row
    constexpr uint32_t f32_tile_bytes = 4096;
    constexpr uint32_t f32_face_bytes = 1024;
    constexpr uint32_t f32_face_row_bytes = 64;

    // common args (one copy per device, multicast by dispatch): what every worker of the device shares, plus one
    // table entry per link. Per-core args: only what differs between cores (every per-core word is dispatched to
    // every core of every device, one device after the other: the dispatch time is cross-device launch skew).
    const uint32_t x_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t gamma_addr = get_common_arg_val<uint32_t>(1);
    const uint32_t block_rows = get_common_arg_val<uint32_t>(2);  // rows per block (knob)
    const uint32_t wt_total = get_common_arg_val<uint32_t>(3);    // tensor_w_tiles
    const uint32_t scratch_addr = get_common_arg_val<uint32_t>(4);
    const uint32_t slot_bytes = get_common_arg_val<uint32_t>(5);
    const uint32_t s_w = get_common_arg_val<uint32_t>(6);
    const uint32_t n_src_tiles = get_common_arg_val<uint32_t>(7);
    const uint32_t group_size = get_common_arg_val<uint32_t>(8);
    constexpr uint32_t link_table = 9;  // per link l: [round_recs, G x (noc x, noc y) of the port holding slot q,
                                        //   bwd_mask]
    size_t a = 0;
    const uint32_t row_start = get_arg_val<uint32_t>(a++);  // first global tile row of this core
    const uint32_t core_rows = get_arg_val<uint32_t>(a++);  // tile rows assigned to this core
    const uint32_t w_start = get_arg_val<uint32_t>(a++);    // first W tile of this core's chunk
    const uint32_t cw = get_arg_val<uint32_t>(a++);         // W tiles of this core's chunk
    const uint32_t rec_base = get_arg_val<uint32_t>(a++);   // link_rec_base of my row group
    const uint32_t my_link = get_arg_val<uint32_t>(a++);
    const uint32_t link_idx = link_table + my_link * (2 + 2 * group_size);
    const uint32_t round_recs = get_common_arg_val<uint32_t>(link_idx);  // records per round in my link's region
    const uint32_t ports_idx = link_idx + 1;                             // common: port holding slot q
    const uint32_t bwd_mask = get_common_arg_val<uint32_t>(ports_idx + 2 * group_size);  // bit q: slot q on port d=1
    const uint32_t all_slots = group_size >= 32 ? 0xFFFFFFFFu : ((1u << group_size) - 1u);
    // sharded-path args (R9/R10); the interleaved path keeps its Phase 0 arg list and these defaults
    uint32_t cw_phys = cw, stat_rows = block_rows, gamma_chunk = cw, lc_rows = 0, n_lc_tiles = 0, lc_sources = 0,
             inbox_addr = 0, my_noc_x = 0, my_noc_y = 0;
    if constexpr (x_sharded) {
        cw_phys = get_arg_val<uint32_t>(a++);      // tiles per tile row of the resident x block
        stat_rows = get_arg_val<uint32_t>(a++);    // tile rows per stats batch (knob)
        gamma_chunk = get_arg_val<uint32_t>(a++);  // gamma tiles per chunk (knob; == cw: one chunk)
        lc_rows = get_arg_val<uint32_t>(a++);      // tile rows I pre-combine (0: not a combiner)
        n_lc_tiles = get_arg_val<uint32_t>(a++);   // ceil(S_w / 32)
        lc_sources = get_arg_val<uint32_t>(a++);   // S_w: W chunks of my row group
        inbox_addr = get_arg_val<uint32_t>(a++);
        my_noc_x = get_arg_val<uint32_t>(a++);
        my_noc_y = get_arg_val<uint32_t>(a++);
    }

    if (core_rows == 0) {
        return;  // idle core of the worker rectangle
    }

    // scaler (once per kernel): SUM -> 1.0, full tile (the same pattern serves REDUCE_ROW and REDUCE_COL)
    dataflow_kernel_lib::
        calculate_and_prepare_reduce_scaler<cb_scaler, ckernel::PoolType::SUM, ckernel::ReduceDim::REDUCE_ROW>();

    const uint32_t x_tile_bytes = get_tile_size(cb_x_resident);
    const auto x = TensorAccessor(x_args, x_addr, x_tile_bytes);
    // one stats-ready counter per port of my link (direction 0 / 1); each counts that port's completed rounds
    volatile tt_l1_ptr uint32_t* stats_ready_fwd =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_stats_ready_base + 2 * my_link));
    volatile tt_l1_ptr uint32_t* stats_ready_bwd =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_stats_ready_base + 2 * my_link + 1));
    Noc noc;
    CircularBuffer x_cb(cb_x_resident);

    // ---- w_tail mask: zero columns w_tail..31 of the last W tile of each tile row ----
    // Whole padding faces (1 and 3 when w_tail <= 16) are zeroed by the DM engine; the partial face rows (16 rows
    // x (16 - c0) elements in each of two faces) are the only RISC stores. Only the core holding the tensor's last
    // W tile does this, once per tile row.
    const bool owns_w_tail = (w_tail != 0) && (w_start + cw == wt_total);
    auto mask_w_tail = [&](uint32_t base_offset, uint32_t rows_n, uint32_t row_stride_tiles) {
        if constexpr (w_tail != 0) {
            // a zero mantissa byte is 0.0 whatever the shared exponent, so block-float padding zeroes in place
            const uint32_t face_bytes = (x_tile_bytes - x_exp_bytes) / 4;
            const uint32_t elem_bytes = face_bytes / 256;
            constexpr uint32_t c0 = w_tail & 15;                    // first padding column inside the partial face
            constexpr uint32_t partial_face = w_tail < 16 ? 0 : 1;  // faces {0,2} or {1,3} hold the partial rows
            for (uint32_t j = 0; j < rows_n; ++j) {
                const uint32_t off = base_offset + (j * row_stride_tiles + cw - 1) * x_tile_bytes + x_exp_bytes;
                if constexpr (w_tail <= 16) {
                    noc.async_write_zeros(x_cb, face_bytes, {.offset_bytes = off + face_bytes});
                    noc.async_write_zeros(x_cb, face_bytes, {.offset_bytes = off + 3 * face_bytes});
                }
                if constexpr (c0 != 0) {
                    const uint32_t tile_addr = get_write_ptr(cb_x_resident) + off;
                    for (uint32_t f = partial_face; f < 4; f += 2) {
                        const uint32_t face_addr = tile_addr + f * face_bytes;
                        if (elem_bytes == 4) {
                            volatile tt_l1_ptr uint32_t* p = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(face_addr);
                            for (uint32_t r = 0; r < 16; ++r) {
                                for (uint32_t c = c0; c < 16; ++c) {
                                    p[r * 16 + c] = 0;
                                }
                            }
                        } else if (elem_bytes == 1) {
                            volatile tt_l1_ptr uint8_t* p = reinterpret_cast<volatile tt_l1_ptr uint8_t*>(face_addr);
                            for (uint32_t r = 0; r < 16; ++r) {
                                for (uint32_t c = c0; c < 16; ++c) {
                                    p[r * 16 + c] = 0;
                                }
                            }
                        } else {
                            volatile tt_l1_ptr uint16_t* p = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(face_addr);
                            for (uint32_t r = 0; r < 16; ++r) {
                                for (uint32_t c = c0; c < 16; ++c) {
                                    p[r * 16 + c] = 0;
                                }
                            }
                        }
                    }
                }
            }
            if constexpr (w_tail <= 16) {
                noc.write_zeros_l1_barrier();
            }
        }
    };

    // gamma (once per call; interleaved: issued with round 0's first x tile row, see front()):
    // one contiguous read of the chunk's cw*32 elements into the front of each group of <= 32 tiles, then
    // tile t's segment is placed in its row 0 (elements 0..15 -> face 0 row 0, 16..31 -> face 1 row 0).
    // Placement runs from the last tile down: tile t >= 1 lands beyond the staging bytes, tile 0 last.
    // Split in two so the read can be issued ahead and its (hot single-bank) latency hidden under the x reads:
    // gamma_issue reserves the chunk and issues one read per group of <= 32 tiles (no barrier); gamma_place runs
    // after a barrier that covered them, places row 0 of every tile and pushes.
    constexpr uint32_t gamma_seg_bytes = 32 * gamma_elem_bytes;
    constexpr uint32_t gamma_group_tiles = 32;  // staging (n * seg_bytes) stays inside the group's first tile
    auto gamma_issue = [&](uint32_t c0, uint32_t chunk, uint32_t reserve_tiles) {
        if constexpr (has_gamma) {
            const uint32_t gamma_tile_bytes = get_tile_size(cb_gamma_tiles);
            const auto gamma = TensorAccessor(gamma_args, gamma_addr, gamma_page_bytes);
            cb_reserve_back(cb_gamma_tiles, reserve_tiles);
            const uint32_t base = get_write_ptr(cb_gamma_tiles);
            for (uint32_t g0 = 0; g0 < chunk; g0 += gamma_group_tiles) {
                const uint32_t n = (chunk - g0) < gamma_group_tiles ? (chunk - g0) : gamma_group_tiles;
                // read no further than the stick's aligned page: with W_local % 32 != 0 the last segment is short
                // (its padding columns only ever scale padding columns of x)
                const uint32_t src_off = (w_start + c0 + g0) * gamma_seg_bytes;
                const uint32_t want = n * gamma_seg_bytes;
                const uint32_t avail = gamma_page_bytes - src_off;
                noc_async_read(
                    gamma.get_noc_addr(0, src_off), base + g0 * gamma_tile_bytes, want < avail ? want : avail);
            }
        }
    };
    auto gamma_place = [&](uint32_t chunk, uint32_t reserve_tiles) {
        if constexpr (has_gamma) {
            const uint32_t gamma_tile_bytes = get_tile_size(cb_gamma_tiles);
            constexpr uint32_t seg_bytes = gamma_seg_bytes;
            constexpr uint32_t half_words = 16 * gamma_elem_bytes / 4;
            constexpr uint32_t face_bytes = 256 * gamma_elem_bytes;
            constexpr uint32_t group_tiles = gamma_group_tiles;
            const uint32_t base = get_write_ptr(cb_gamma_tiles);
            invalidate_l1_cache();
            for (uint32_t g0 = 0; g0 < chunk; g0 += group_tiles) {
                const uint32_t n = (chunk - g0) < group_tiles ? (chunk - g0) : group_tiles;
                const uint32_t gbase = base + g0 * gamma_tile_bytes;
                for (uint32_t t = n; t-- > 0;) {
                    volatile tt_l1_ptr uint32_t* src =
                        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(gbase + t * seg_bytes);
                    volatile tt_l1_ptr uint32_t* row0 =
                        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(gbase + t * gamma_tile_bytes);
                    volatile tt_l1_ptr uint32_t* row0_face1 =
                        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(gbase + t * gamma_tile_bytes + face_bytes);
                    for (uint32_t i = 0; i < half_words; ++i) {
                        row0_face1[i] = src[half_words + i];
                    }
                    if (t > 0) {
                        for (uint32_t i = 0; i < half_words; ++i) {
                            row0[i] = src[i];
                        }
                    }
                }
                if constexpr (gamma_full_rows) {
                    // DEST-reuse multiply has no broadcast: copy row 0 of faces 0/1 into every row of all 4 faces
                    for (uint32_t t = 0; t < n; ++t) {
                        const uint32_t tb = gbase + t * gamma_tile_bytes;
                        volatile tt_l1_ptr uint32_t* f0 = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(tb);
                        volatile tt_l1_ptr uint32_t* f1 =
                            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(tb + face_bytes);
                        for (uint32_t face = 0; face < 4; ++face) {
                            volatile tt_l1_ptr uint32_t* src_row = (face & 1) ? f1 : f0;
                            volatile tt_l1_ptr uint32_t* dst =
                                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(tb + face * face_bytes);
                            for (uint32_t r = (face < 2) ? 1 : 0; r < 16; ++r) {
                                for (uint32_t i = 0; i < half_words; ++i) {
                                    dst[r * half_words + i] = src_row[i];
                                }
                            }
                        }
                    }
                }
            }
            cb_push_back(cb_gamma_tiles, reserve_tiles);
        }
    };
    auto load_gamma = [&](uint32_t c0, uint32_t chunk, uint32_t reserve_tiles) {
        gamma_issue(c0, chunk, reserve_tiles);
        noc_async_read_barrier();
        gamma_place(chunk, reserve_tiles);
    };
    // one chunk (== cw): pushed once, held for the call. Several: double-buffered gamma_chunk-page chunks, every
    // chunk occupying gamma_chunk pages (the ragged last one too) so the ring always wraps on a chunk boundary.
    const bool gamma_chunked = gamma_chunk < cw;
    auto stream_gamma_chunks = [&]() {
        for (uint32_t c0 = 0; c0 < cw; c0 += gamma_chunk) {
            load_gamma(c0, (cw - c0) < gamma_chunk ? (cw - c0) : gamma_chunk, gamma_chunk);
        }
    };

    // Pull records into a CB laid out as n_tiles_per_row source tiles per tile row (source tile k of row j at
    // k * rows + j; source s -> row s % 32). Records of (source q, tile row j) sit in one block of rw * 128 B:
    // [half 0 of w = 0..rw-1][half 1 of w = 0..rw-1], so a run of sources inside one 16-row face pair moves as
    // two reads (one per half). block_of(q, j) gives that block's L1 address on the core at (x_of(q), y_of(q)).
    // gather_open reserves the batch's tiles and zeroes the unused source rows; gather_issue issues the reads of the
    // sources q with in_phase(q) (no barrier); gather_close waits for every read issued and pushes. One phase per
    // port lets the slots of an early port (my own device's records) be pulled while a later port's are in flight.
    auto gather_open = [&](uint32_t cb, uint32_t n_tiles_per_row, uint32_t rows, uint32_t n_src) {
        CircularBuffer gather_cb(cb);
        cb_reserve_back(cb, n_tiles_per_row * rows);
        if ((n_src & 31) != 0) {
            // rows beyond the last source of the last source tile must not enter the sum
            noc.async_write_zeros(
                gather_cb, rows * f32_tile_bytes, {.offset_bytes = (n_tiles_per_row - 1) * rows * f32_tile_bytes});
            noc.write_zeros_l1_barrier();
        }
    };
    auto gather_issue =
        [&](uint32_t cb, uint32_t rows, uint32_t n_q, uint32_t rw, auto in_phase, auto x_of, auto y_of, auto block_of) {
            const uint32_t wbase = get_write_ptr(cb);
            const uint32_t half_block = rw * half_record;
            for (uint32_t q = 0; q < n_q; ++q) {
                if (!in_phase(q)) {
                    continue;
                }
                const uint32_t px = x_of(q);
                const uint32_t py = y_of(q);
                const uint32_t s_first = q * rw;
                for (uint32_t j = 0; j < rows; ++j) {
                    const uint32_t block = block_of(q, j);
                    for (uint32_t sa = s_first; sa < s_first + rw;) {
                        const uint32_t seg_end = (sa | 15) + 1;  // next 16-row face boundary
                        const uint32_t sb = seg_end < s_first + rw ? seg_end : s_first + rw;
                        const uint32_t r = sa & 31;
                        const uint32_t dst = wbase + ((sa >> 5) * rows + j) * f32_tile_bytes +
                                             (r >> 4) * 2 * f32_face_bytes + (r & 15) * f32_face_row_bytes;
                        const uint32_t src = block + (sa - s_first) * half_record;
                        const uint32_t bytes = (sb - sa) * half_record;
                        noc_async_read(get_noc_addr(px, py, src), dst, bytes);
                        noc_async_read(get_noc_addr(px, py, src + half_block), dst + f32_face_bytes, bytes);
                        sa = sb;
                    }
                }
            }
        };
    auto gather_close = [&](uint32_t cb, uint32_t n_tiles) {
        noc_async_read_barrier();
        cb_push_back(cb, n_tiles);
    };
    auto gather = [&](uint32_t cb,
                      uint32_t n_tiles_per_row,
                      uint32_t rows,
                      uint32_t n_q,
                      uint32_t rw,
                      auto x_of,
                      auto y_of,
                      auto block_of) {
        gather_open(cb, n_tiles_per_row, rows, n_q * rw);
        gather_issue(cb, rows, n_q, rw, [](uint32_t) { return true; }, x_of, y_of, block_of);
        gather_close(cb, n_tiles_per_row * rows);
    };

    const uint32_t num_blocks = (core_rows + block_rows - 1) / block_rows;
    auto rows_of = [&](uint32_t b) {
        const uint32_t r0 = b * block_rows;
        return (core_rows - r0) < block_rows ? (core_rows - r0) : block_rows;
    };

    const uint32_t gamma_round = (round_lookahead < num_blocks - 1) ? round_lookahead : num_blocks - 1;
    const bool gamma_early =
        gamma_early_knob && !x_sharded && !gamma_chunked;  // interleaved: gamma read rides round 0's first barrier

    // ---- front half of round b: load_block (+ gamma, R5 local gather and prescale gamma in round 0) ----
    auto front = [&](uint32_t round) {
        const uint32_t r0 = round * block_rows;
        const uint32_t rows = rows_of(round);
        if constexpr (x_sharded) {
            // the shard is already resident in cb_x_resident's L1: publish its valid tile rows, no NoC traffic
            cb_reserve_back(cb_x_resident, rows * cw_phys);
            if (owns_w_tail) {
                mask_w_tail(0, rows, cw_phys);  // padding of the resident shard's last tile column
            }
            cb_push_back(cb_x_resident, rows * cw_phys);
        } else {
            // one tile row (cw tiles) per barrier. gamma (once per call) is issued ahead of round 0's first tile row
            // and placed after its barrier: its single-bank read overlaps the x reads instead of following them
            for (uint32_t j = 0; j < rows; ++j) {
                const bool gamma_here = gamma_early && round == 0 && j == 0;
                if (gamma_here) {
                    gamma_issue(0, cw, cw);
                }
                cb_reserve_back(cb_x_resident, cw);
                uint32_t wp = get_write_ptr(cb_x_resident);
                const uint32_t tile0 = (row_start + r0 + j) * wt_total + w_start;
                for (uint32_t t = 0; t < cw; ++t) {
                    noc_async_read_page(tile0 + t, x, wp);
                    wp += x_tile_bytes;
                }
                noc_async_read_barrier();
                if (owns_w_tail) {
                    mask_w_tail(0, 1, cw);
                }
                cb_push_back(cb_x_resident, cw);
                if (gamma_here) {
                    gamma_place(cw, cw);
                }
            }
        }
        // sharded: gamma once per call after the shard publish (one round)
        if (!gamma_early && round == gamma_round && !gamma_chunked) {
            load_gamma(0, cw, cw);
        }

        // ---- local_gather (R5 pre-combine; combiners only, one round) ----
        if (round == 0 && lc_rows > 0) {
            volatile tt_l1_ptr uint32_t* lc_ready =
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_local_combine));
            noc_semaphore_wait_min(lc_ready, lc_sources);
            gather(
                cb_lc,
                n_lc_tiles,
                lc_rows,
                1,
                lc_sources,
                [&](uint32_t) { return my_noc_x; },
                [&](uint32_t) { return my_noc_y; },
                [&](uint32_t, uint32_t j) { return inbox_addr + j * lc_sources * record_bytes; });
        }
        if (round == 0 && gamma_chunked && !gamma_full_rows) {
            stream_gamma_chunks();  // prescale consumes them before the exchange
        }
    };

    // ---- back half of round b: gather_sources_block, one stats batch at a time ----
    auto back = [&](uint32_t round) {
        const uint32_t rows = rows_of(round);
        const uint32_t round_base = rec_base + round * round_recs;
        for (uint32_t b0 = 0; b0 < rows; b0 += stat_rows) {
            const uint32_t nb = (rows - b0) < stat_rows ? (rows - b0) : stat_rows;
            auto x_of = [&](uint32_t q) { return get_common_arg_val<uint32_t>(ports_idx + 2 * q); };
            auto y_of = [&](uint32_t q) { return get_common_arg_val<uint32_t>(ports_idx + 2 * q + 1); };
            auto block_of = [&](uint32_t q, uint32_t j) {
                return scratch_addr + q * slot_bytes + (round_base + (b0 + j) * s_w) * record_bytes;
            };
            gather_open(cb_stats_combine, n_src_tiles, nb, group_size * s_w);
            // gather_sources_block, one phase per port of my link, in the order the ports complete the round (their
            // stats-ready counters): a direction holding none of the slots is not waited for
            bool fwd_done = (bwd_mask & all_slots) == all_slots;
            bool bwd_done = (bwd_mask & all_slots) == 0;
            while (!(fwd_done && bwd_done)) {
                invalidate_l1_cache();
                if (!fwd_done && *stats_ready_fwd >= round + 1) {
                    gather_issue(
                        cb_stats_combine,
                        nb,
                        group_size,
                        s_w,
                        [&](uint32_t q) { return ((bwd_mask >> q) & 1) == 0; },
                        x_of,
                        y_of,
                        block_of);
                    fwd_done = true;
                }
                if (!bwd_done && *stats_ready_bwd >= round + 1) {
                    gather_issue(
                        cb_stats_combine,
                        nb,
                        group_size,
                        s_w,
                        [&](uint32_t q) { return ((bwd_mask >> q) & 1) != 0; },
                        x_of,
                        y_of,
                        block_of);
                    bwd_done = true;
                }
            }
            gather_close(cb_stats_combine, n_src_tiles * nb);
            if (gamma_chunked && gamma_full_rows) {
                stream_gamma_chunks();  // the fused normalize of this batch walks every chunk
            }
        }
    };

    // R2 schedule: round b+lookahead's loads run before round b's gather, so they overlap round b's exchange
    for (uint32_t round = 0; round < round_lookahead && round < num_blocks; ++round) {
        front(round);
    }
    for (uint32_t round = 0; round < num_blocks; ++round) {
        if (round + round_lookahead < num_blocks) {
            front(round + round_lookahead);
        }
        back(round);
    }
}
