// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

// groupnorm_sc_N_1_HW_C_b reader (NCRISC; NoC0, NoC1 on the DRAM odd-column cores: program descriptor "NoC pairing").
//
// Once per kernel: group map (NoC zeros + one-hot halfword stores, after block 0's last pass-1 push), ones tiles on
// unit roots (one 64 B row by RISC-V, then NoC self-copies doubling to a tile).
// Per block (one unit = image n, channel slice s, rows [row_start, row_start + rows_core)):
//   pilot : image row 0 of the slice -> staging -> cb_pilot_rows (row 0: -k, k the shift of each lane's group;
//           pass 1 accumulates x + (-k)) and, on the unit root, cb_group_shift (lane g = +k of group g, rows
//           replicated at step B); read together with chunk 0, built after a pass-1 credit push
//   pass 1: chunks 0..B-1 -> the input ring (TILE: both views, lockstep) or cb_input_rm sticks (RM);
//           compute keeps the last K chunks on the keep view
//   affine: gamma / beta slices -> staging -> the expand matmul's gamma map (gamma[c] at row g(c), column c's
//           lane) + beta map (row 0 = -beta[c]), cb_gamma_rows (stall shadow of the combine)
//   pass 2: re-read chunks 0..B-K-1 -> the input ring (cb_input_rm for RM)
//   output share (DRAM interleaved): compute packs each pass-2 chunk for whichever DM RISC has a free output slot
//           (the writer's cb_output_tiles preferred, else this RISC's cb_output_b) and names the chunk's pass-2
//           index q on that consumer's sequence CB (cb_out_seq_b here; END closes the block). This RISC writes its
//           share to DRAM on its own NoC: it drains cb_output_b while a pass-2 re-read waits for ring room, then
//           to END after the block's last re-read. Both NoCs carry pass-2 writes, in proportion to how fast each
//           drains (the pass-2 writer on NOC0 was ~6x slower per chunk than on NOC1 in DRAM-congested units).
// Every chunk pushes the nominal T pages (ragged tail: only rows_valid tile-rows transferred).
// SHARDED: the input CB (TILE: the ring, RM: cb_input_rm) is backed on this core's own shard, so a chunk is
// a credit push with no data movement; the pilot comes from the unit root's shard (row 0 of the image).
//
// A slice lane v (0 .. W*32) holds channel s*P + (v mod P), P = MAP_PERIOD: P = W*32 except a sharded RM
// slice narrower than a tile multiple, which compute tilizes as k = W*32/P sticks per tile row.
//
// raw_api rows (op_design.md API Mapping): map / affine-map one-hot stores and pilot row-0 copies have no helper
// (generate_index_tile writes a ramp, fill_tile_column writes all 32 rows); DRAM reads are TensorAccessor.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "ttnn/cpp/ttnn/kernel_lib/l1_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/perf_instrumentation.hpp"

namespace {
constexpr uint32_t cb_input_rm = 0;
constexpr uint32_t cb_input_tiles = 2;  // input ring, pass-1 view (alias group A)
constexpr uint32_t cb_input_keep = 3;   // input ring, pass-2 view (same bytes, pushed in lockstep)
constexpr uint32_t cb_group_map = 12;
constexpr uint32_t cb_gamma_rows = 13;  // the expand matmul's in1: gamma map (+ beta map)
constexpr uint32_t cb_ones_tiles = 17;
constexpr uint32_t cb_affine_staging = 21;
constexpr uint32_t cb_pilot_rows = 22;
constexpr uint32_t cb_group_shift = 24;
constexpr uint32_t cb_output_b = 27;   // DRAM interleaved: this RISC's share of the pass-2 output
constexpr uint32_t cb_out_seq_b = 29;  // pass-2 index of each chunk compute packed into cb_output_b (16 B pages)
constexpr uint32_t OUT_SEQ_END = 0xFFFFFFFFu;

constexpr uint32_t BF16_TILE_BYTES = 2048;
constexpr uint32_t BF16_FACE_BYTES = 512;
constexpr uint32_t F32_TILE_BYTES = 4096;
constexpr uint32_t BF16_ONE = 0x3F80;
constexpr uint32_t BF16_ONE_PAIR = 0x3F803F80;
constexpr uint32_t TILE_SEGMENT_BYTES = 64;  // 32 bf16 channels of one stick
constexpr uint32_t ALIGN = 64;               // DRAM read alignment (BH)

// Element offset of tile-row-0 lane l (0..31) inside a tile: faces 0 / 1 hold lanes 0..15 / 16..31.
FORCE_INLINE uint32_t row0_lane(uint32_t l) { return (l >> 4) * 256 + (l & 15); }

// Element offset of slice lane v (0 .. W*32) in row 0 of its tile (1024 elements per tile). The lanes of one half
// row (16-aligned run) are contiguous elements.
FORCE_INLINE uint32_t lane_elem(uint32_t v) { return (v >> 5) * 1024 + (v & 16) * 16 + (v & 15); }

// Row-0 lanes [v, v1) <- val, one tight store loop per contiguous half-row run (the shift-rows hot loop:
// NCRISC is in-order, so per-lane index math - not the stores - set its cost). E = 1: one element per lane;
// E = 2 (bf16 lane pairs in one word): v and v1 even.
template <uint32_t E, typename T>
FORCE_INLINE void fill_lanes(volatile tt_l1_ptr T* base, uint32_t v, uint32_t v1, T val) {
    while (v < v1) {
        const uint32_t run_end = ((v | 15) + 1) < v1 ? ((v | 15) + 1) : v1;
        volatile tt_l1_ptr T* p = base + lane_elem(v) / E;
        for (uint32_t i = 0, n = (run_end - v) / E; i < n; ++i) {
            p[i] = val;
        }
        v = run_end;
    }
}

}  // namespace

void kernel_main() {
    constexpr uint32_t W = get_compile_time_arg_val(0);         // Wt_core: channel tiles per block
    constexpr uint32_t CHUNK_HT = get_compile_time_arg_val(1);  // h: tile rows per chunk
    constexpr bool IS_RM = get_compile_time_arg_val(2) != 0;
    constexpr bool HAS_GAMMA = get_compile_time_arg_val(3) != 0;
    constexpr bool HAS_BETA = get_compile_time_arg_val(4) != 0;
    constexpr uint32_t R = get_compile_time_arg_val(5);  // gather tile rows (ones tiles)
    constexpr uint32_t CG = get_compile_time_arg_val(6);
    constexpr uint32_t WT_TOTAL = get_compile_time_arg_val(7);
    constexpr uint32_t HT = get_compile_time_arg_val(8);
    constexpr uint32_t GX = get_compile_time_arg_val(9);
    constexpr uint32_t IN_PAGE_BYTES = get_compile_time_arg_val(10);
    constexpr uint32_t P = get_compile_time_arg_val(11);  // MAP_PERIOD: channels per slice
    constexpr bool SHARDED = get_compile_time_arg_val(12) != 0;
    // Per-chunk zones only when the host found every chunk's zone executions inside the marker budget.
    constexpr bool CHUNK_ZONES = get_compile_time_arg_val(13) != 0;
    constexpr auto in_args = TensorAccessorArgs<14>();
    constexpr auto gamma_args = TensorAccessorArgs<in_args.next_compile_time_args_offset()>();
    constexpr auto beta_args = TensorAccessorArgs<gamma_args.next_compile_time_args_offset()>();
    // The output (written for this RISC's pass-2 share; DRAM interleaved only).
    constexpr uint32_t OUT_PAGE_BYTES = get_compile_time_arg_val(beta_args.next_compile_time_args_offset());
    constexpr auto out_args = TensorAccessorArgs<beta_args.next_compile_time_args_offset() + 1>();

    constexpr uint32_t T = CHUNK_HT * W;
    constexpr uint32_t HW = HT * 32;
    constexpr uint32_t C = WT_TOTAL * 32;
    constexpr uint32_t SLICE_BYTES = W * TILE_SEGMENT_BYTES;
    constexpr uint32_t G_CORE = P / CG;  // group lanes of a slice
    // Per-tensor affine staging: the slice's P channels from a 64 B-aligned start.
    constexpr uint32_t AFFINE_STAGE_BYTES = ((P * 2 + ALIGN - 1) / ALIGN) * ALIGN + ALIGN;

    uint32_t arg = 0;
    const uint32_t in_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t gamma_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t beta_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t unit_start = get_arg_val<uint32_t>(arg++);
    const uint32_t num_blocks = get_arg_val<uint32_t>(arg++);
    const uint32_t row_start = get_arg_val<uint32_t>(arg++);
    const uint32_t rows_core = get_arg_val<uint32_t>(arg++);
    const uint32_t resident_chunks = get_arg_val<uint32_t>(arg++);
    const uint32_t is_root = get_arg_val<uint32_t>(arg++);
    const uint32_t root_x = get_arg_val<uint32_t>(arg++);  // SHARDED: unit root (virtual), holds image row 0
    const uint32_t root_y = get_arg_val<uint32_t>(arg++);
    const uint32_t out_addr = get_arg_val<uint32_t>(arg++);

    const auto in_acc = TensorAccessor(in_args, in_addr, IN_PAGE_BYTES);
    [[maybe_unused]] const auto out_acc = TensorAccessor(out_args, out_addr, OUT_PAGE_BYTES);
    [[maybe_unused]] const auto gamma_acc = TensorAccessor(gamma_args, gamma_addr, C * 2);
    [[maybe_unused]] const auto beta_acc = TensorAccessor(beta_args, beta_addr, C * 2);

    Noc noc;

    const uint32_t num_chunks = (rows_core + CHUNK_HT - 1) / CHUNK_HT;
    const uint32_t K = resident_chunks < num_chunks ? resident_chunks : num_chunks;
    const uint32_t first_resident = num_chunks - K;
    const uint32_t staging = get_write_ptr(cb_affine_staging);
    // The prologue (shift rows, group shift) runs right after a chunk's credit push:
    // SHARDED after the last pass-1 push (the chunks are resident, so compute gets every credit before any prologue
    // work), DRAM after chunk 0's (whose read the pilot rides with). RM compute tilizes meanwhile; TILE compute
    // waits only for the pilot rows, which are emitted first.
    const uint32_t pro_step = SHARDED ? num_chunks - 1 : 0;

    // One output chunk at the front of cb_output_b -> DRAM: rows_valid tile rows from image tile row row0g
    // (n * HT + row0), the writer's page walk.
    auto write_b_chunk = [&](uint32_t row0g, uint32_t rows_valid, uint32_t wt0) {
        cb_wait_front(cb_output_b, T);
        uint32_t src = get_read_ptr(cb_output_b);
        for (uint32_t r = 0; r < rows_valid; ++r) {
            const uint32_t tile0 = (row0g + r) * WT_TOTAL + wt0;
            for (uint32_t wt = 0; wt < W; ++wt) {
                noc_async_write_page(tile0 + wt, out_acc, src);
                src += OUT_PAGE_BYTES;
            }
        }
        noc_async_writes_flushed();
        cb_pop_front(cb_output_b, T);
    };

    // Group map M (W tiles): element (slice lane r, group lane l) = 1.0 iff slice lane r holds a channel of group
    // lane l. Slices start on group boundaries, so the map is block-invariant: built once per kernel. Compute first
    // reads it at the collapse, so it is built off the critical path, after the last pass-1 chunk push of block 0.
    auto build_map = [&]() {
        MaybeDeviceZoneScope("r_map_setup");
        DataflowBuffer map_dfb(cb_group_map);
        cb_reserve_back(cb_group_map, W);
        noc.async_write_zeros(map_dfb, W * BF16_TILE_BYTES);
        noc.write_zeros_l1_barrier();
        volatile tt_l1_ptr uint16_t* map = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(get_write_ptr(cb_group_map));
        // Row r = v & 31 of tile v >> 5 has its one at column lane(v) = (v mod P) / CG: per group, each half-row run
        // of rows is one strided store loop (16 elements per face row).
        for (uint32_t v0 = 0; v0 < W * 32; v0 += CG) {
            const uint32_t lane = (v0 % P) / CG;
            for (uint32_t v = v0; v < v0 + CG;) {
                const uint32_t run_end = ((v | 15) + 1) < v0 + CG ? ((v | 15) + 1) : v0 + CG;
                const uint32_t r = v & 31;
                volatile tt_l1_ptr uint16_t* p =
                    map + (v >> 5) * 1024 + ((r >> 4) * 2 + (lane >> 4)) * 256 + (r & 15) * 16 + (lane & 15);
                for (uint32_t i = 0, n = run_end - v; i < n; ++i) {
                    p[i * 16] = BF16_ONE;
                }
                v = run_end;
            }
        }
        cb_push_back(cb_group_map, W);
    };

    for (uint32_t blk = 0; blk < num_blocks; ++blk) {
        MaybeDeviceZoneScope("r_block");
        const uint32_t unit = unit_start + blk;
        const uint32_t n = unit / GX;
        const uint32_t c_start = (unit % GX) * P;  // first channel of the slice
        const uint32_t wt0 = c_start / 32;         // its tile column (P = W*32 whenever the input is tiled in DRAM)
        // Output share (DRAM interleaved; SHARDED packs into the output shard and has none).
        [[maybe_unused]] bool b_done = false;
        // One cb_output_b chunk if its index and data are both there (block: wait for them); END ends the share.
        [[maybe_unused]] auto try_b = [&](bool block) {
            if (!block && !cb_pages_available_at_front(cb_out_seq_b, 1)) {
                return;
            }
            cb_wait_front(cb_out_seq_b, 1);
            invalidate_l1_cache();
            const uint32_t q = *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_read_ptr(cb_out_seq_b));
            if (q == OUT_SEQ_END) {
                cb_pop_front(cb_out_seq_b, 1);
                b_done = true;
                return;
            }
            if (!block && !cb_pages_available_at_front(cb_output_b, T)) {
                return;
            }
            cb_pop_front(cb_out_seq_b, 1);
            const uint32_t j = q < K ? first_resident + q : q - K;  // pass-2 order: resident tail, then the head
            const uint32_t rows_left = rows_core - j * CHUNK_HT;
            write_b_chunk(n * HT + row_start + j * CHUNK_HT, rows_left < CHUNK_HT ? rows_left : CHUNK_HT, wt0);
        };

        // One loop, one site per NoC pattern (kernel text must fit the kernel-config ring under --dev):
        // steps 0..B-1 are pass 1, step B emits the affine rows (stall shadow of the combine), and steps
        // B+1..B+first_resident re-read the non-resident head for pass 2.
        for (uint32_t step = 0; step <= num_chunks + first_resident; ++step) {
            if (step == num_chunks) {
                MaybeDeviceZoneScope("r_ones_shift_affine");
                if (is_root) {
                    // Ones tiles (R): matmul in0 of the root's sender sum. RISC-V writes one 64 B row, then NoC
                    // self-copies keep doubling the filled prefix until all R tiles are ones (one NoC site). The
                    // bytes persist: later blocks only re-credit them.
                    cb_reserve_back(cb_ones_tiles, R);
                    if (blk == 0) {
                        const uint32_t ones_base = get_write_ptr(cb_ones_tiles);
                        volatile tt_l1_ptr uint32_t* words = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ones_base);
                        for (uint32_t i = 0; i < 16; ++i) {
                            words[i] = BF16_ONE_PAIR;
                        }
                        constexpr uint32_t ONES_BYTES = R * BF16_TILE_BYTES;
                        for (uint32_t filled = 64; filled < ONES_BYTES;) {
                            const uint32_t len = filled < ONES_BYTES - filled ? filled : ONES_BYTES - filled;
                            noc_async_read(get_noc_addr(ones_base), ones_base + filled, len);
                            noc_async_read_barrier();
                            filled += len;
                        }
                    }
                    cb_push_back(cb_ones_tiles, R);

                    // Group shift tile: row 0 (written at step 0) -> all 32 rows, as the finalize's stats are
                    // row-replicated. NoC self-copies double rows inside faces 0 / 1, then faces 0-1 -> 2-3.
                    const uint32_t gs = get_write_ptr(cb_group_shift);
                    for (uint32_t len = 64; len < 1024; len <<= 1) {
                        for (uint32_t f = 0; f < 2; ++f) {
                            noc_async_read(get_noc_addr(gs + f * 1024), gs + f * 1024 + len, len);
                        }
                        noc_async_read_barrier();
                    }
                    noc_async_read(get_noc_addr(gs), gs + F32_TILE_BYTES / 2, F32_TILE_BYTES / 2);
                    noc_async_read_barrier();
                    cb_push_back(cb_group_shift, 1);
                }

                // ---------------- Affine expand maps ----------------
                // cb_gamma_rows holds the expand matmul's in1, [group lane x channel] per tile (not transposed):
                // W "gamma map" tiles, element (g(c), lane of c) = gamma[c], then (beta) W "beta map" tiles with
                // row 0 = -beta[c]. compute: a = rstd x gmap, b' = [mean*rstd, e] x [gmap; bmap] (e = 1), so
                // b' = mean*rstd*gamma - beta. Adjacent channels of one group are adjacent halfwords of one row,
                // so both maps are word copies of the staged slice.
                if constexpr (HAS_GAMMA) {
                    constexpr uint32_t AMAP_TILES = (HAS_BETA ? 2 : 1) * W;
                    const uint32_t lo = (c_start * 2) & ~(ALIGN - 1);
                    const uint32_t skew_hw = (c_start * 2 - lo) / 2;
                    const uint32_t len = ((c_start + P) * 2 - lo + ALIGN - 1) & ~(ALIGN - 1);
                    noc_async_read(gamma_acc.get_noc_addr(0, lo), staging, len);
                    if constexpr (HAS_BETA) {
                        noc_async_read(beta_acc.get_noc_addr(0, lo), staging + AFFINE_STAGE_BYTES, len);
                    }
                    cb_reserve_back(cb_gamma_rows, AMAP_TILES);
                    if (blk == 0) {
                        // Off-position elements stay zero across blocks: the one-hot positions are block-invariant
                        // (every slice starts on a group boundary) and the ring returns to its base.
                        DataflowBuffer amap_dfb(cb_gamma_rows);
                        noc.async_write_zeros(amap_dfb, AMAP_TILES * BF16_TILE_BYTES);
                        noc.write_zeros_l1_barrier();
                    }
                    noc_async_read_barrier();
                    const uint32_t am = get_write_ptr(cb_gamma_rows);
                    // Slice lane v: tile v >> 5, column row0_lane(v & 31); its group lane picks the gamma map row.
                    uint32_t lane = 0, in_group = 0, in_period = 0;  // in_period == v % P
                    if constexpr (CG % 2 == 0) {
                        // Two lanes per word (P, the skew and CG are even: a pair never straddles a group).
                        volatile tt_l1_ptr uint32_t* gs = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(staging);
                        volatile tt_l1_ptr uint32_t* bs =
                            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(staging + AFFINE_STAGE_BYTES);
                        volatile tt_l1_ptr uint32_t* gm = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(am);
                        volatile tt_l1_ptr uint32_t* bm =
                            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(am + W * BF16_TILE_BYTES);
                        for (uint32_t v = 0; v < W * 32; v += 2) {
                            const uint32_t col = (v >> 5) * 1024 + row0_lane(v & 31);
                            const uint32_t w = (skew_hw + in_period) >> 1;
                            gm[(col + (lane >> 4) * 512 + (lane & 15) * 16) >> 1] = gs[w];
                            if constexpr (HAS_BETA) {
                                bm[col >> 1] = bs[w] ^ 0x80008000u;  // -beta
                            }
                            in_period += 2;
                            in_group += 2;
                            if (in_period == P) {
                                in_period = in_group = lane = 0;
                            } else if (in_group == CG) {
                                in_group = 0;
                                ++lane;
                            }
                        }
                    } else {
                        volatile tt_l1_ptr uint16_t* gs = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(staging);
                        volatile tt_l1_ptr uint16_t* bs =
                            reinterpret_cast<volatile tt_l1_ptr uint16_t*>(staging + AFFINE_STAGE_BYTES);
                        volatile tt_l1_ptr uint16_t* gm = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(am);
                        volatile tt_l1_ptr uint16_t* bm =
                            reinterpret_cast<volatile tt_l1_ptr uint16_t*>(am + W * BF16_TILE_BYTES);
                        for (uint32_t v = 0; v < W * 32; ++v) {
                            const uint32_t col = (v >> 5) * 1024 + row0_lane(v & 31);
                            const uint32_t ch = skew_hw + in_period;
                            gm[col + (lane >> 4) * 512 + (lane & 15) * 16] = gs[ch];
                            if constexpr (HAS_BETA) {
                                bm[col] = bs[ch] ^ 0x8000;  // -beta
                            }
                            if (++in_period == P) {
                                in_period = in_group = lane = 0;
                            } else if (++in_group == CG) {
                                in_group = 0;
                                ++lane;
                            }
                        }
                    }
                    cb_push_back(cb_gamma_rows, AMAP_TILES);
                }
                continue;
            }

            // ---------------- One chunk (nominal T pages) ----------------
            const uint32_t j = step < num_chunks ? step : step - num_chunks - 1;
            const uint32_t row0 = row_start + j * CHUNK_HT;
            const uint32_t rows_left = rows_core - j * CHUNK_HT;
            const uint32_t rows_valid = rows_left < CHUNK_HT ? rows_left : CHUNK_HT;

            if (step == 0) {
                // Pilot source (image row 0 of the slice) -> staging; its barrier is the chunk's.
                // TILE: row 0 of faces 0 / 1 of each tile (64 B reads: rows 0-1), staged 128 B per tile.
                // RM: the slice of stick 0, staged contiguously.
                if constexpr (IS_RM) {
                    const uint64_t src = SHARDED ? get_noc_addr(root_x, root_y, in_addr)
                                                 : in_acc.get_noc_addr(n * HW, wt0 * TILE_SEGMENT_BYTES);
                    noc_async_read(src, staging, P * 2);
                } else {
                    for (uint32_t wt = 0; wt < W; ++wt) {
                        for (uint32_t f = 0; f < 2; ++f) {
                            const uint64_t src =
                                SHARDED
                                    ? get_noc_addr(root_x, root_y, in_addr + wt * BF16_TILE_BYTES + f * BF16_FACE_BYTES)
                                    : in_acc.get_noc_addr(n * HT * WT_TOTAL + wt0 + wt, f * BF16_FACE_BYTES);
                            noc_async_read(src, staging + wt * 128 + f * 64, 64);
                        }
                    }
                }
            }

            {
                // Back-pressure: compute has not freed the ring slot yet.
                MaybeDeviceZoneScopeIf(CHUNK_ZONES, "r_reserve");
                if constexpr (!SHARDED) {
                    // Pass-2 re-read: compute may be blocked on cb_output_b (this RISC's share), so write the share
                    // out while the ring has no room for the next chunk.
                    if (step > num_chunks) {
                        for (;;) {
                            const bool room = IS_RM ? cb_pages_reservable_at_back(cb_input_rm, T)
                                                    : cb_pages_reservable_at_back(cb_input_tiles, T) &&
                                                          cb_pages_reservable_at_back(cb_input_keep, T);
                            if (room) {
                                break;
                            }
                            if (!b_done) {
                                try_b(false);
                            }
                        }
                    }
                }
                if constexpr (IS_RM) {
                    cb_reserve_back(cb_input_rm, T);
                } else {
                    // Both views must have room: the keep view still holds the resident tail until pass 2.
                    cb_reserve_back(cb_input_tiles, T);
                    cb_reserve_back(cb_input_keep, T);
                }
            }
            if constexpr (SHARDED) {
                // The chunk is already resident in this core's shard: the reserve above hands compute its credits.
            } else if constexpr (IS_RM) {
                MaybeDeviceZoneScopeIf(CHUNK_ZONES, "r_issue");
                uint32_t page = n * HW + row0 * 32;
                uint32_t dst = get_write_ptr(cb_input_rm);
                for (uint32_t s = 0; s < rows_valid * 32; ++s) {
                    noc_async_read(in_acc.get_noc_addr(page, wt0 * TILE_SEGMENT_BYTES), dst, SLICE_BYTES);
                    ++page;
                    dst += SLICE_BYTES;
                }
            } else {
                MaybeDeviceZoneScopeIf(CHUNK_ZONES, "r_issue");
                uint32_t dst = get_write_ptr(cb_input_tiles);
                for (uint32_t r = 0; r < rows_valid; ++r) {
                    const uint32_t tile0 = (n * HT + row0 + r) * WT_TOTAL + wt0;
                    for (uint32_t wt = 0; wt < W; ++wt) {
                        noc_async_read_page(tile0 + wt, in_acc, dst);
                        dst += BF16_TILE_BYTES;
                    }
                }
            }

            if (!SHARDED || step == pro_step) {
                MaybeDeviceZoneScopeIf(CHUNK_ZONES, "r_barrier");
                noc_async_read_barrier();  // SHARDED: only the pilot read is outstanding
            }

            if constexpr (IS_RM) {
                cb_push_back(cb_input_rm, T);
            } else {
                cb_push_back(cb_input_tiles, T);
                cb_push_back(cb_input_keep, T);
            }

            if (step == pro_step) {
                // ---------------- Shift rows: lane v -> k = x[row 0, first channel of v's group] ----------------
                // cb_pilot_rows (bf16): -k (sign bit flipped: exact), the srcB operand of pass 1's x + (-k), a row
                // broadcast: the unpacker reads only row 0 (faces 0/1), so rows 1..31 are never zeroed.
                // cb_group_shift (root, Float32): +k per group lane, 0 past G_CORE (rows replicated at step B).
                // Lanes come in whole groups: P is a multiple of CG and every period starts on a group boundary.
                volatile tt_l1_ptr uint16_t* sv = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(staging);
                // TILE staging holds rows 0-1 of faces 0/1 per tile: lane l < 16 at halfword l, else l + 16.
                auto stage = [&](uint32_t c) -> uint32_t {
                    return sv[IS_RM ? c : (c >> 5) * 64 + (c & 31) + (c & 16)];
                };
                {
                    MaybeDeviceZoneScope("r_shift_rows");
                    cb_reserve_back(cb_pilot_rows, W);
                    const uint32_t pbase = get_write_ptr(cb_pilot_rows);
                    for (uint32_t v0 = 0; v0 < W * 32; v0 += CG) {
                        const uint32_t nk16 = stage(v0 % P) ^ 0x8000u;  // -k
                        if constexpr (CG % 2 == 0) {
                            // Even groups: lanes (v, v + 1), v even, share a group and one word of a face row.
                            fill_lanes<2>(
                                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(pbase),
                                v0,
                                v0 + CG,
                                nk16 | (nk16 << 16));
                        } else {
                            fill_lanes<1>(
                                reinterpret_cast<volatile tt_l1_ptr uint16_t*>(pbase),
                                v0,
                                v0 + CG,
                                static_cast<uint16_t>(nk16));
                        }
                    }
                    cb_push_back(cb_pilot_rows, W);  // compute's pass-1 moments walk starts here
                }

                if (is_root) {
                    cb_reserve_back(cb_group_shift, 1);
                    volatile tt_l1_ptr uint32_t* gv =
                        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_group_shift));
                    for (uint32_t g = 0; g < 32; ++g) {
                        gv[row0_lane(g)] = g < G_CORE ? stage(g * CG) << 16 : 0;
                    }
                }
            }
            if (step == num_chunks - 1 && blk == 0) {
                // After the last pass-1 push: no read is outstanding (the zero-fill's L1 barrier would otherwise wait
                // for a DRAM chunk too), and compute first reads the map after reducing this chunk.
                build_map();
            }
        }
        // The rest of this block's output share: whatever compute packs for this RISC after the last re-read was
        // issued (every chunk of the share when nothing is re-read), up to END.
        if constexpr (!SHARDED) {
            while (!b_done) {
                try_b(true);
            }
        }
    }
    if constexpr (!SHARDED) {
        noc_async_write_barrier();
    }
}
