// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

// groupnorm_sc_N_1_HW_C_b reader (NCRISC, NoC0).
//
// Once per kernel: reduce scaler (helper), group map (NoC zeros + 32 one-hot halfword stores per tile),
// ones tiles on unit roots (one 64 B row by RISC-V, then NoC self-copies doubling to a tile).
// Per block (one unit = image n, channel slice [wt0, wt0 + W), rows [row_start, row_start + rows_core)):
//   pass 1: chunks 0..B-1 -> the input ring (TILE: both views, lockstep) or cb_input_rm sticks (RM);
//           compute keeps the last K chunks on the keep view
//   affine: gamma / beta slices -> staging -> row 0 of faces 0/1 of W tiles (stall shadow of the combine)
//   pass 2: re-read chunks 0..B-K-1 -> the input ring (cb_input_rm for RM)
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
constexpr uint32_t cb_input_tiles = 2;  // input ring, pass-1 view (alias group A)
constexpr uint32_t cb_input_keep = 3;   // input ring, pass-2 view (same bytes, pushed in lockstep)
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
    constexpr uint32_t NUM_AFFINE = (HAS_GAMMA ? 1 : 0) + (HAS_BETA ? 1 : 0);

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
        // self-copies keep doubling the filled prefix until all R tiles are ones (one NoC site).
        cb_reserve_back(cb_ones_tiles, R);
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

        // One loop, one site per NoC pattern (kernel text must fit the kernel-config ring under --dev):
        // steps 0..B-1 are pass 1, step B emits the affine rows (stall shadow of the combine), and steps
        // B+1..B+first_resident re-read the non-resident head for pass 2.
        for (uint32_t step = 0; step <= num_chunks + first_resident; ++step) {
            if (step == num_chunks) {
                // ---------------- Affine rows ----------------
                if constexpr (NUM_AFFINE > 0) {
                    const uint32_t staging = get_write_ptr(cb_affine_staging);
                    uint64_t src_noc[NUM_AFFINE];
                    if constexpr (HAS_GAMMA) {
                        src_noc[0] = gamma_acc.get_noc_addr(0, wt0 * TILE_SEGMENT_BYTES);
                    }
                    if constexpr (HAS_BETA) {
                        src_noc[NUM_AFFINE - 1] = beta_acc.get_noc_addr(0, wt0 * TILE_SEGMENT_BYTES);
                    }
                    for (uint32_t k = 0; k < NUM_AFFINE; ++k) {
                        noc_async_read(src_noc[k], staging + k * SLICE_BYTES, SLICE_BYTES);
                    }
                    noc_async_read_barrier();

                    // Row 0 of faces 0/1 of each tile carries the 32 channels (consumed with BroadcastDim::Row).
                    for (uint32_t k = 0; k < NUM_AFFINE; ++k) {
                        const uint32_t cb = (HAS_GAMMA && k == 0) ? cb_gamma_rows : cb_beta_rows;
                        const uint32_t src = staging + k * SLICE_BYTES;
                        cb_reserve_back(cb, W);
                        const uint32_t dst = get_write_ptr(cb);
                        if (blk == 0) {
                            // Rows 1..31 stay zero across blocks (the W-page ring returns to its base).
                            DataflowBuffer rows_dfb(cb);
                            noc.async_write_zeros(rows_dfb, W * BF16_TILE_BYTES);
                            noc.write_zeros_l1_barrier();
                        }
                        for (uint32_t wt = 0; wt < W; ++wt) {
                            volatile tt_l1_ptr uint32_t* sv =
                                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(src + wt * TILE_SEGMENT_BYTES);
                            volatile tt_l1_ptr uint32_t* d0 =
                                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(dst + wt * BF16_TILE_BYTES);
                            volatile tt_l1_ptr uint32_t* d1 =
                                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(dst + wt * BF16_TILE_BYTES + 512);
                            for (uint32_t i = 0; i < 8; ++i) {
                                d0[i] = sv[i];
                                d1[i] = sv[8 + i];
                            }
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
            if constexpr (IS_RM) {
                cb_reserve_back(cb_input_rm, T);
                uint32_t page = n * HW + row0 * 32;
                uint32_t dst = get_write_ptr(cb_input_rm);
                for (uint32_t s = 0; s < rows_valid * 32; ++s) {
                    noc_async_read(in_acc.get_noc_addr(page, wt0 * TILE_SEGMENT_BYTES), dst, SLICE_BYTES);
                    ++page;
                    dst += SLICE_BYTES;
                }
                noc_async_read_barrier();
                cb_push_back(cb_input_rm, T);
            } else {
                // Both views must have room: the keep view still holds the resident tail until pass 2.
                cb_reserve_back(cb_input_tiles, T);
                cb_reserve_back(cb_input_keep, T);
                uint32_t dst = get_write_ptr(cb_input_tiles);
                for (uint32_t r = 0; r < rows_valid; ++r) {
                    const uint32_t tile0 = (n * HT + row0 + r) * WT_TOTAL + wt0;
                    for (uint32_t wt = 0; wt < W; ++wt) {
                        noc_async_read_page(tile0 + wt, in_acc, dst);
                        dst += BF16_TILE_BYTES;
                    }
                }
                noc_async_read_barrier();
                cb_push_back(cb_input_tiles, T);
                cb_push_back(cb_input_keep, T);
            }
        }
    }
}
