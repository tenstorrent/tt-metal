// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

// groupnorm_sc_N_1_HW_C_b reader (NCRISC, NoC0).
//
// Once per kernel: reduce scaler (helper), group map (NoC zeros + 32 one-hot halfword stores per tile),
// ones tiles on unit roots (one 64 B row by RISC-V, then NoC self-copies doubling to a tile).
// Per block (one unit = image n, channel slice s, rows [row_start, row_start + rows_core)):
//   pilot : image row 0 of the slice -> staging -> cb_pilot_rows (row 0: the shift k of each lane's group),
//           cb_sum_seed (row 0: -n*k, exact, n = rows_core*32: the Sigma x accumulator starts there, so it ends at
//           Sigma (x - k)) and, on the unit root, cb_group_shift (lane g = k of group g, rows replicated at step B);
//           read together with chunk 0, so its latency hides under the first chunk's
//   pass 1: chunks 0..B-1 -> the input ring (TILE: both views, lockstep) or cb_input_rm sticks (RM);
//           compute keeps the last K chunks on the keep view
//   affine: gamma / beta slices -> staging -> row 0 of faces 0/1 of W tiles (stall shadow of the combine)
//   pass 2: re-read chunks 0..B-K-1 -> the input ring (cb_input_rm for RM)
// Every chunk pushes the nominal T pages (ragged tail: only rows_valid tile-rows transferred).
// SHARDED: the input CB (TILE: the ring, RM: cb_input_rm) is backed on this core's own shard, so a chunk is
// a credit push with no data movement; the pilot comes from the unit root's shard (row 0 of the image).
//
// A slice lane v (0 .. W*32) holds channel s*P + (v mod P), P = MAP_PERIOD: P = W*32 except a sharded RM
// slice narrower than a tile multiple, which compute tilizes as k = W*32/P sticks per tile row.
//
// raw_api rows (op_design.md API Mapping): map one-hot stores, pilot / affine row-0 copies have no helper
// (generate_index_tile writes a ramp, fill_tile_column writes all 32 rows); DRAM reads are TensorAccessor.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/l1_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/perf_instrumentation.hpp"

namespace {
constexpr uint32_t cb_input_rm = 0;
constexpr uint32_t cb_input_tiles = 2;  // input ring, pass-1 view (alias group A)
constexpr uint32_t cb_input_keep = 3;   // input ring, pass-2 view (same bytes, pushed in lockstep)
constexpr uint32_t cb_reduce_scaler = 6;
constexpr uint32_t cb_group_map = 12;
constexpr uint32_t cb_gamma_rows = 13;
constexpr uint32_t cb_beta_rows = 14;
constexpr uint32_t cb_ones_tiles = 17;
constexpr uint32_t cb_affine_staging = 21;
constexpr uint32_t cb_pilot_rows = 22;
constexpr uint32_t cb_sum_seed = 23;
constexpr uint32_t cb_group_shift = 24;

constexpr uint32_t BF16_TILE_BYTES = 2048;
constexpr uint32_t BF16_FACE_BYTES = 512;
constexpr uint32_t F32_TILE_BYTES = 4096;
constexpr uint32_t BF16_ONE = 0x3F80;
constexpr uint32_t BF16_ONE_PAIR = 0x3F803F80;
constexpr uint32_t TILE_SEGMENT_BYTES = 64;  // 32 bf16 channels of one stick
constexpr uint32_t ALIGN = 64;               // DRAM read alignment (BH)

// Element offset of tile-row-0 lane l (0..31) inside a tile: faces 0 / 1 hold lanes 0..15 / 16..31.
FORCE_INLINE uint32_t row0_lane(uint32_t l) { return (l >> 4) * 256 + (l & 15); }

// -(rows * 32 * k) as fp32 bits for a bf16 k, by integer arithmetic (no soft float): exact while the 8-bit
// significand times rows fits 24 bits (rows < 2^16), truncated beyond. Zero / denormal k -> 0.
FORCE_INLINE uint32_t neg_n_times_bf16(uint32_t k16, uint32_t rows) {
    const uint32_t e = (k16 >> 7) & 0xFF;
    if (e == 0) {
        return 0;
    }
    const uint32_t p = ((k16 & 0x7F) | 0x80) * rows;  // value = p * 2^(e - 127 - 7 + 5)
    uint32_t b = 7;                                   // msb of p (p >= 128)
    while (b < 31 && (p >> (b + 1)) != 0) {
        ++b;
    }
    const uint32_t m = (b <= 23 ? p << (23 - b) : p >> (b - 23)) & 0x7FFFFF;
    return ((~k16 & 0x8000) << 16) | ((b + e - 2) << 23) | m;
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
    // Pilot rows are emitted PILOT_REPEAT times (m copies of the W tiles): compute's pass-1 chain walks m tile
    // rows as one row of m*W tiles so its DEST window fills on narrow slices.
    constexpr uint32_t PILOT_TILES = get_compile_time_arg_val(13) * W;
    // Per-chunk zones only when the host found every chunk's zone executions inside the marker budget.
    constexpr bool CHUNK_ZONES = get_compile_time_arg_val(14) != 0;
    constexpr auto in_args = TensorAccessorArgs<15>();
    constexpr auto gamma_args = TensorAccessorArgs<in_args.next_compile_time_args_offset()>();
    constexpr auto beta_args = TensorAccessorArgs<gamma_args.next_compile_time_args_offset()>();

    constexpr uint32_t T = CHUNK_HT * W;
    constexpr uint32_t HW = HT * 32;
    constexpr uint32_t C = WT_TOTAL * 32;
    constexpr uint32_t SLICE_BYTES = W * TILE_SEGMENT_BYTES;
    constexpr uint32_t G_CORE = P / CG;  // group lanes of a slice
    constexpr uint32_t NUM_AFFINE = (HAS_GAMMA ? 1 : 0) + (HAS_BETA ? 1 : 0);
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

    const auto in_acc = TensorAccessor(in_args, in_addr, IN_PAGE_BYTES);
    [[maybe_unused]] const auto gamma_acc = TensorAccessor(gamma_args, gamma_addr, C * 2);
    [[maybe_unused]] const auto beta_acc = TensorAccessor(beta_args, beta_addr, C * 2);

    Noc noc;

    const uint32_t num_chunks = (rows_core + CHUNK_HT - 1) / CHUNK_HT;
    const uint32_t K = resident_chunks < num_chunks ? resident_chunks : num_chunks;
    const uint32_t first_resident = num_chunks - K;
    const uint32_t staging = get_write_ptr(cb_affine_staging);
    // The prologue (shift rows, seed, scaler, group shift, SHARDED: the map) runs right after a chunk's credit push:
    // SHARDED after the last pass-1 push (the chunks are resident, so compute gets every credit before any prologue
    // work), DRAM after chunk 0's (whose read the pilot rides with). RM compute tilizes meanwhile; TILE compute
    // waits only for the pilot rows, which are emitted first.
    const uint32_t pro_step = SHARDED ? num_chunks - 1 : 0;

    // Group map M (W tiles): element (slice lane r, group lane l) = 1.0 iff slice lane r holds a channel of group
    // lane l. Slices start on group boundaries, so the map is block-invariant: built once per kernel. Compute first
    // reads it at the collapse, so it is built off the critical path (DRAM: in chunk 0's read shadow; SHARDED: after
    // the pilot / seed rows, in pass 1's shadow).
    auto build_map = [&]() {
        MaybeDeviceZoneScope("r_map_setup");
        DataflowBuffer map_dfb(cb_group_map);
        cb_reserve_back(cb_group_map, W);
        noc.async_write_zeros(map_dfb, W * BF16_TILE_BYTES);
        noc.write_zeros_l1_barrier();
        volatile tt_l1_ptr uint16_t* map = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(get_write_ptr(cb_group_map));
        uint32_t lane = 0, in_group = 0, in_period = 0;
        for (uint32_t v = 0; v < W * 32; ++v) {
            const uint32_t r = v & 31;
            map[(v >> 5) * 1024 + ((r >> 4) * 2 + (lane >> 4)) * 256 + (r & 15) * 16 + (lane & 15)] = BF16_ONE;
            if (++in_period == P) {
                in_period = in_group = lane = 0;
            } else if (++in_group == CG) {
                in_group = 0;
                ++lane;
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

                // ---------------- Affine rows ----------------
                if constexpr (NUM_AFFINE > 0) {
                    // Slice bytes [c_start*2, (c_start+P)*2) from the 64 B-aligned start below them.
                    const uint32_t lo = (c_start * 2) & ~(ALIGN - 1);
                    const uint32_t skew_hw = (c_start * 2 - lo) / 2;
                    const uint32_t len = ((c_start + P) * 2 - lo + ALIGN - 1) & ~(ALIGN - 1);
                    uint64_t src_noc[NUM_AFFINE];
                    if constexpr (HAS_GAMMA) {
                        src_noc[0] = gamma_acc.get_noc_addr(0, lo);
                    }
                    if constexpr (HAS_BETA) {
                        src_noc[NUM_AFFINE - 1] = beta_acc.get_noc_addr(0, lo);
                    }
                    for (uint32_t k = 0; k < NUM_AFFINE; ++k) {
                        noc_async_read(src_noc[k], staging + k * AFFINE_STAGE_BYTES, len);
                    }
                    noc_async_read_barrier();

                    // Row 0 of faces 0/1 of each tile carries lane v = channel c_start + (v mod P), two lanes per
                    // word (P and the skew are even), consumed with BroadcastDim::Row.
                    for (uint32_t k = 0; k < NUM_AFFINE; ++k) {
                        const uint32_t cb = (HAS_GAMMA && k == 0) ? cb_gamma_rows : cb_beta_rows;
                        volatile tt_l1_ptr uint32_t* sv =
                            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(staging + k * AFFINE_STAGE_BYTES);
                        cb_reserve_back(cb, W);
                        const uint32_t dst = get_write_ptr(cb);
                        if (blk == 0) {
                            // Rows 1..31 stay zero across blocks (the W-page ring returns to its base).
                            DataflowBuffer rows_dfb(cb);
                            noc.async_write_zeros(rows_dfb, W * BF16_TILE_BYTES);
                            noc.write_zeros_l1_barrier();
                        }
                        volatile tt_l1_ptr uint32_t* dv = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(dst);
                        for (uint32_t v = 0; v < W * 32; v += 2) {
                            dv[((v >> 5) * 1024 + row0_lane(v & 31)) >> 1] = sv[(skew_hw + v % P) >> 1];
                        }
                        cb_push_back(cb, W);
                    }
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

            if constexpr (!SHARDED) {
                if (step == 0 && blk == 0) {
                    build_map();  // in the shadow of chunk 0's reads (and the pilot's)
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
                // cb_pilot_rows (bf16): k, the srcB operand of pass 1's x - k, consumed with BroadcastDim::Row: math
                // reads only srcB row 0 (faces 0/1), so rows 1..31 are never zeroed. cb_sum_seed (Float32, unpacked
                // to DEST): -n*k in row 0; its rows 1..31 only reach rows 1..31 of the Sigma x accumulator, the chan
                // row and the collapse output, whose row 0 alone is gathered - never zeroed either.
                // cb_group_shift (root, Float32): k per group lane, 0 past G_CORE (rows replicated at step B).
                // Lanes come in whole groups: P is a multiple of CG and every period starts on a group boundary.
                volatile tt_l1_ptr uint16_t* sv = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(staging);
                // TILE staging holds rows 0-1 of faces 0/1 per tile: lane l < 16 at halfword l, else l + 16.
                auto stage = [&](uint32_t c) -> uint32_t {
                    return sv[IS_RM ? c : (c >> 5) * 64 + (c & 31) + (c & 16)];
                };
                {
                    MaybeDeviceZoneScope("r_shift_rows");
                    cb_reserve_back(cb_pilot_rows, PILOT_TILES);
                    const uint32_t pbase = get_write_ptr(cb_pilot_rows);
                    // POISON (correctness probe only): every never-written pilot / seed element is a quiet NaN.
                    if (blk == 0) {
                        volatile tt_l1_ptr uint32_t* pp = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(pbase);
                        for (uint32_t i = 0; i < PILOT_TILES * BF16_TILE_BYTES / 4; ++i) {
                            pp[i] = 0x7FC07FC0;
                        }
                        cb_reserve_back(cb_sum_seed, W);
                        volatile tt_l1_ptr uint32_t* sp =
                            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_sum_seed));
                        for (uint32_t i = 0; i < W * F32_TILE_BYTES / 4; ++i) {
                            sp[i] = 0x7FC00000;
                        }
                    }
                    for (uint32_t v0 = 0; v0 < W * 32; v0 += CG) {
                        const uint32_t k16 = stage(v0 % P);
                        if constexpr (CG % 2 == 0) {
                            // Even groups: lanes (v, v + 1), v even, share a group and one word of a face row.
                            volatile tt_l1_ptr uint32_t* pw = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(pbase);
                            const uint32_t k2 = k16 | (k16 << 16);
                            for (uint32_t v = v0; v < v0 + CG; v += 2) {
                                const uint32_t off = ((v >> 5) * 1024 + row0_lane(v & 31)) >> 1;
                                for (uint32_t c = 0; c < PILOT_TILES; c += W) {
                                    pw[off + c * 512] = k2;
                                }
                            }
                        } else {
                            volatile tt_l1_ptr uint16_t* pv = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(pbase);
                            for (uint32_t v = v0; v < v0 + CG; ++v) {
                                const uint32_t off = (v >> 5) * 1024 + row0_lane(v & 31);
                                for (uint32_t c = 0; c < PILOT_TILES; c += W) {
                                    pv[off + c * 1024] = k16;
                                }
                            }
                        }
                    }
                    cb_push_back(cb_pilot_rows, PILOT_TILES);  // TILE compute's pass-1 chain starts here

                    cb_reserve_back(cb_sum_seed, W);
                    volatile tt_l1_ptr uint32_t* dv =
                        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_sum_seed));
                    for (uint32_t v0 = 0; v0 < W * 32; v0 += CG) {
                        const uint32_t seed = neg_n_times_bf16(stage(v0 % P), rows_core);
                        for (uint32_t v = v0; v < v0 + CG; ++v) {
                            dv[(v >> 5) * 1024 + row0_lane(v & 31)] = seed;
                        }
                    }
                    cb_push_back(cb_sum_seed, W);
                }

                if (blk == 0) {
                    MaybeDeviceZoneScope("r_scaler");
                    dataflow_kernel_lib::
                        prepare_reduce_scaler<cb_reduce_scaler, ckernel::PoolType::SUM, ckernel::ReduceDim::REDUCE_COL>(
                            1.0f);
                }

                if (is_root) {
                    cb_reserve_back(cb_group_shift, 1);
                    volatile tt_l1_ptr uint32_t* gv =
                        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_group_shift));
                    for (uint32_t g = 0; g < 32; ++g) {
                        gv[row0_lane(g)] = g < G_CORE ? stage(g * CG) << 16 : 0;
                    }
                }

                if constexpr (SHARDED) {
                    if (blk == 0) {
                        build_map();
                    }
                }
            }
        }
    }
}
