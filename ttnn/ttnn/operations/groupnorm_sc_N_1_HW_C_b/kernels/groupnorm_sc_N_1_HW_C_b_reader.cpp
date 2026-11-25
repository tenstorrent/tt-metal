// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

// groupnorm_sc_N_1_HW_C_b reader (NCRISC, NoC0).
//
// Once per kernel: reduce scaler (helper), group map (NoC zeros + 32 one-hot halfword stores per tile),
// ones tiles on unit roots (one 64 B row by RISC-V, then NoC self-copies doubling to a tile).
// Per block (one unit = image n, channel slice [wt0, wt0 + W), rows [row_start, row_start + rows_core)):
//   pass 1: chunks 0..B-K-1 -> cb_input_stream (or cb_input_rm sticks for RM),
//           chunks B-K..B-1 -> cb_input_resident + cb_input_resident_keep (both views, lockstep; TILE)
//   affine: gamma / beta slices -> staging -> row 0 of faces 0/1 of W tiles (stall shadow of the combine)
//   pass 2: re-read chunks 0..B-K-1 -> cb_input_stream (cb_input_rm for RM)
// Every chunk pushes the nominal T pages (ragged tail: only rows_valid tile-rows transferred).
//
// raw_api rows (op_design.md API Mapping): map one-hot stores and affine row-0 copies have no helper
// (generate_index_tile writes a ramp, fill_tile_column writes all 32 rows); DRAM reads are TensorAccessor.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/l1_helpers.hpp"

namespace {
constexpr uint32_t cb_input_rm = 0;
constexpr uint32_t cb_input_stream = 1;
constexpr uint32_t cb_input_resident = 2;
constexpr uint32_t cb_input_resident_keep = 3;
constexpr uint32_t cb_reduce_scaler = 6;
constexpr uint32_t cb_group_map = 12;
constexpr uint32_t cb_gamma_rows = 13;
constexpr uint32_t cb_beta_rows = 14;
constexpr uint32_t cb_ones_tiles = 17;
constexpr uint32_t cb_affine_staging = 21;

constexpr uint32_t BF16_TILE_BYTES = 2048;
constexpr uint32_t BF16_ONE = 0x3F80;
constexpr uint32_t BF16_ONE_PAIR = 0x3F803F80;
constexpr uint32_t TILE_SEGMENT_BYTES = 64;  // 32 bf16 channels of one stick
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
    constexpr auto in_args = TensorAccessorArgs<11>();
    constexpr auto gamma_args = TensorAccessorArgs<in_args.next_compile_time_args_offset()>();
    constexpr auto beta_args = TensorAccessorArgs<gamma_args.next_compile_time_args_offset()>();

    constexpr uint32_t T = CHUNK_HT * W;
    constexpr uint32_t HW = HT * 32;
    constexpr uint32_t AFFINE_PAGE_BYTES = WT_TOTAL * TILE_SEGMENT_BYTES;
    constexpr uint32_t SLICE_BYTES = W * TILE_SEGMENT_BYTES;

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

    const auto in_acc = TensorAccessor(in_args, in_addr, IN_PAGE_BYTES);
    [[maybe_unused]] const auto gamma_acc = TensorAccessor(gamma_args, gamma_addr, AFFINE_PAGE_BYTES);
    [[maybe_unused]] const auto beta_acc = TensorAccessor(beta_args, beta_addr, AFFINE_PAGE_BYTES);

    Noc noc;

    // ---------------- Constants (once per kernel) ----------------
    dataflow_kernel_lib::
        prepare_reduce_scaler<cb_reduce_scaler, ckernel::PoolType::SUM, ckernel::ReduceDim::REDUCE_COL>(1.0f);

    {
        // Group map M (W tiles): element (channel row r, group lane l) = 1.0 iff slice channel wt*32+r is in
        // group lane l. Slices start on span boundaries, so the map is block-invariant.
        DataflowBuffer map_dfb(cb_group_map);
        cb_reserve_back(cb_group_map, W);
        noc.async_write_zeros(map_dfb, W * BF16_TILE_BYTES);
        noc.write_zeros_l1_barrier();
        volatile tt_l1_ptr uint16_t* map = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(get_write_ptr(cb_group_map));
        for (uint32_t wt = 0; wt < W; ++wt) {
            for (uint32_t r = 0; r < 32; ++r) {
                const uint32_t lane = (wt * 32 + r) / CG;
                const uint32_t face = (r >> 4) * 2 + (lane >> 4);
                map[wt * 1024 + face * 256 + (r & 15) * 16 + (lane & 15)] = BF16_ONE;
            }
        }
        cb_push_back(cb_group_map, W);
    }

    if (is_root) {
        // Ones tiles (R): matmul in0 of the root's sender sum. RISC-V writes one 64 B row, then NoC
        // self-copies double it to a full tile and replicate the tile.
        cb_reserve_back(cb_ones_tiles, R);
        const uint32_t ones_base = get_write_ptr(cb_ones_tiles);
        volatile tt_l1_ptr uint32_t* words = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ones_base);
        for (uint32_t i = 0; i < 16; ++i) {
            words[i] = BF16_ONE_PAIR;
        }
        for (uint32_t filled = 64; filled < BF16_TILE_BYTES; filled *= 2) {
            noc_async_read(get_noc_addr(ones_base), ones_base + filled, filled);
            noc_async_read_barrier();
        }
        for (uint32_t t = 1; t < R; ++t) {
            noc_async_read(get_noc_addr(ones_base), ones_base + t * BF16_TILE_BYTES, BF16_TILE_BYTES);
        }
        noc_async_read_barrier();
        cb_push_back(cb_ones_tiles, R);
    }

    const uint32_t num_chunks = (rows_core + CHUNK_HT - 1) / CHUNK_HT;
    const uint32_t K = resident_chunks < num_chunks ? resident_chunks : num_chunks;
    const uint32_t first_resident = num_chunks - K;

    for (uint32_t blk = 0; blk < num_blocks; ++blk) {
        const uint32_t unit = unit_start + blk;
        const uint32_t n = unit / GX;
        const uint32_t wt0 = (unit % GX) * W;

        if (is_root && blk > 0) {
            // The ones bytes persist; re-credit them for this block's sender sum.
            cb_reserve_back(cb_ones_tiles, R);
            cb_push_back(cb_ones_tiles, R);
        }

        // Reads one chunk (nominal T pages) into its destination CB(s).
        auto read_chunk = [&](uint32_t j, bool resident) {
            const uint32_t row0 = row_start + j * CHUNK_HT;
            const uint32_t rows_left = rows_core - j * CHUNK_HT;
            const uint32_t rows_valid = rows_left < CHUNK_HT ? rows_left : CHUNK_HT;
            if constexpr (IS_RM) {
                cb_reserve_back(cb_input_rm, T);
                const uint32_t l1 = get_write_ptr(cb_input_rm);
                uint32_t page = n * HW + row0 * 32;
                uint32_t dst = l1;
                for (uint32_t s = 0; s < rows_valid * 32; ++s) {
                    noc_async_read(in_acc.get_noc_addr(page, wt0 * TILE_SEGMENT_BYTES), dst, SLICE_BYTES);
                    ++page;
                    dst += SLICE_BYTES;
                }
                noc_async_read_barrier();
                cb_push_back(cb_input_rm, T);
            } else {
                const uint32_t cb = resident ? cb_input_resident : cb_input_stream;
                cb_reserve_back(cb, T);
                if (resident) {
                    cb_reserve_back(cb_input_resident_keep, T);
                }
                uint32_t dst = get_write_ptr(cb);
                for (uint32_t r = 0; r < rows_valid; ++r) {
                    const uint32_t tile0 = (n * HT + row0 + r) * WT_TOTAL + wt0;
                    for (uint32_t wt = 0; wt < W; ++wt) {
                        noc_async_read_page(tile0 + wt, in_acc, dst);
                        dst += BF16_TILE_BYTES;
                    }
                }
                noc_async_read_barrier();
                cb_push_back(cb, T);
                if (resident) {
                    cb_push_back(cb_input_resident_keep, T);
                }
            }
        };

        // ---------------- Pass 1 ----------------
        for (uint32_t j = 0; j < num_chunks; ++j) {
            read_chunk(j, j >= first_resident);
        }

        // ---------------- Affine rows (stall shadow of the combine) ----------------
        if constexpr (HAS_GAMMA || HAS_BETA) {
            const uint32_t staging = get_write_ptr(cb_affine_staging);
            constexpr uint32_t beta_offset = HAS_GAMMA ? SLICE_BYTES : 0;
            if constexpr (HAS_GAMMA) {
                noc_async_read(gamma_acc.get_noc_addr(0, wt0 * TILE_SEGMENT_BYTES), staging, SLICE_BYTES);
            }
            if constexpr (HAS_BETA) {
                noc_async_read(beta_acc.get_noc_addr(0, wt0 * TILE_SEGMENT_BYTES), staging + beta_offset, SLICE_BYTES);
            }
            noc_async_read_barrier();

            // Row 0 of faces 0/1 of each tile carries the 32 channels (consumed with BroadcastDim::Row).
            auto emit_rows = [&](uint32_t cb, uint32_t src) {
                cb_reserve_back(cb, W);
                const uint32_t dst = get_write_ptr(cb);
                if (blk == 0) {
                    DataflowBuffer rows_dfb(cb);
                    noc.async_write_zeros(rows_dfb, W * BF16_TILE_BYTES);
                    noc.write_zeros_l1_barrier();
                }
                for (uint32_t wt = 0; wt < W; ++wt) {
                    volatile tt_l1_ptr uint32_t* s =
                        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(src + wt * TILE_SEGMENT_BYTES);
                    volatile tt_l1_ptr uint32_t* d0 =
                        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(dst + wt * BF16_TILE_BYTES);
                    volatile tt_l1_ptr uint32_t* d1 =
                        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(dst + wt * BF16_TILE_BYTES + 512);
                    for (uint32_t i = 0; i < 8; ++i) {
                        d0[i] = s[i];
                        d1[i] = s[8 + i];
                    }
                }
                cb_push_back(cb, W);
            };
            if constexpr (HAS_GAMMA) {
                emit_rows(cb_gamma_rows, staging);
            }
            if constexpr (HAS_BETA) {
                emit_rows(cb_beta_rows, staging + beta_offset);
            }
        }

        // ---------------- Pass 2 re-reads (non-resident head) ----------------
        for (uint32_t j = 0; j < first_resident; ++j) {
            read_chunk(j, false);
        }
    }
}
