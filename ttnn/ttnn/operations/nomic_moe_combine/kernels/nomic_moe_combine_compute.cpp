// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// nomic_moe_combine — compute (op_design.md § Block schedule; realizations R1 / R2 of § Key Risks).
//
// Per core one block = the whole output shard, streamed per tile row; per DEST group of n <= DEST_AUTO_LIMIT tiles:
//     DEST[d]  = bias row-broadcast                         (unary_bcast<ROW>, one init a group)
//     DEST[d] += rows_k[j0 + d] * weights_k[tb] (col bcast)  k = 0 then 1 (fixed fp order), HiFi3 default, fp32 DEST
//     pack n tiles bf16 straight into the output shard       (the one rounding)
//
// The CT knob `fused_tilize` picks how rows_k reaches srcA in tile form:
//   R1 (fused_tilize = 1): the unpacker's tilize-A (llk_unpack_tilizeA_B) reads operand A straight from the row-major
//      rows block and pairs it with an ELWMUL NONE against weights pre-expanded once per core (expand_weights_block,
//      kernel-lib unary_bcast<Col>). No tilize round trip, but measured slow: the tilize-A unpack moves one 16-datum
//      row per instruction (~380 cycles a tile product; 8x512 compute 32.5 us vs reads 14.5 us).
//   R2 (fused_tilize = 0, default): kernel-lib fast tilize of each choice's W tiles into cb_rows_tilized, then
//      mul_tiles_bcast_cols on the raw weight tiles (column 0 of faces 0 / 2). No expansion needed.
//
// Raw-LLK / compute-API use (helper-first rule): the DEST group is n seeded slots each accumulated twice with the
// inits hoisted over the group. eltwise_chain's DestAccumulation reduces across the walk into one tile with
// per-element inits per tile (eltwise/api/chain.hpp:302-312), so the group is written with the compute API's
// unary_bcast<ROW> / mul_tiles_bcast_cols (the calls the kernel-lib elements wrap). R1's tilize-A x ELWMUL pairing
// has no helper at all (only reduce pairs with tilize-A: tilizeA_B_reduce_init); the HiFi ELWMUL NONE MOP consumes
// one srcA / srcB face per run and clears both data-valids per face, matching the tilize-A unpacker.
// bias_seed_srcb (lamp L4, CT knob, default off): unary_bcast<ROW> routes an fp32 CB through unpack-to-dest; the
// knob seeds through srcB instead (llk_unpack_A / B2D datacopy, the LLK calls unary_bcast wraps for 16-bit CBs),
// exact here because the bias is bf16-valued and srcB holds TF32. Measured neutral, so parked.

#include <cstdint>

#include "api/compute/bcast.h"
#include "api/compute/cb_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/tilize.h"
#include "api/compute/reg_api.h"
#include "ttnn/cpp/ttnn/kernel_lib/dest_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/broadcast/bcast.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp"
#if defined(ABLATE_ZONES) && !defined(ABLATE_NOCZONES)
#include "api/debug/kernel_profiler.hpp"
#define NMC_ZONE(name) DeviceZoneScopedN(name)
#else
#define NMC_ZONE(name)
#endif

void kernel_main() {
    constexpr uint32_t shard_col_tiles = get_compile_time_arg_val(0);                   // W
    [[maybe_unused]] constexpr uint32_t shard_row_tiles = get_compile_time_arg_val(1);  // r
    constexpr uint32_t cb_rows_choice0 = get_compile_time_arg_val(2);
    constexpr uint32_t cb_rows_choice1 = get_compile_time_arg_val(3);
    constexpr uint32_t cb_weights_raw_choice0 = get_compile_time_arg_val(4);
    constexpr uint32_t cb_weights_raw_choice1 = get_compile_time_arg_val(5);
    constexpr uint32_t cb_weights_bcast = get_compile_time_arg_val(6);  // R1 only
    constexpr uint32_t cb_bias_tiles = get_compile_time_arg_val(7);
    constexpr uint32_t cb_output_shard = get_compile_time_arg_val(8);
    constexpr uint32_t cb_rows_tilized = get_compile_time_arg_val(9);  // R2 only
    constexpr bool fused_tilize = get_compile_time_arg_val(10) != 0;
    constexpr uint32_t dest_group_knob = get_compile_time_arg_val(11);  // 0 = DEST_AUTO_LIMIT
    constexpr uint32_t top_k = get_compile_time_arg_val(12);            // host TOP_K (validated == 2)
    static_assert(top_k == 2, "one rows / weights CB pair per choice: the main loop is written for top 2");
    // 1: seed the bias through srcB (row-broadcast datacopy B2D, TF32 srcB: exact, the bias is bf16-valued) in the
    // srcB mode the multiply uses; 0: unary_bcast<ROW>, which takes an fp32 CB through unpack-to-dest (lamp L4).
    constexpr bool bias_seed_srcb = get_compile_time_arg_val(13) != 0;
    constexpr uint32_t dest_limit = dest_group_knob != 0 && dest_group_knob < compute_kernel_lib::DEST_AUTO_LIMIT
                                        ? dest_group_knob
                                        : compute_kernel_lib::DEST_AUTO_LIMIT;
    constexpr uint32_t dest_group_tiles = dest_limit < shard_col_tiles ? dest_limit : shard_col_tiles;

    // Core i = y * grid_x + x of the shard order; its record (common runtime arg i, the readers' layout): field 1 is
    // core_row_tiles. No per-core runtime args (Refinement 4).
    constexpr uint32_t grid_x = get_compile_time_arg_val(14);
    constexpr uint32_t record_field_bits = get_compile_time_arg_val(15);
    const uint32_t core_index = get_absolute_logical_y() * grid_x + get_absolute_logical_x();
    const uint32_t core_row_tiles =
        (get_common_arg_val<uint32_t>(core_index) >> record_field_bits) & ((1u << record_field_bits) - 1);

    DataflowBuffer rows_choice0(cb_rows_choice0);
    DataflowBuffer rows_choice1(cb_rows_choice1);

    if constexpr (fused_tilize) {
        compute_kernel_hw_startup(cb_weights_raw_choice0, cb_weights_bcast);
        // ---- expand_weights_block: once per core, both choices (tile k * core_row_tiles + tb) ----
        DataflowBuffer weights_raw_choice0(cb_weights_raw_choice0);
        DataflowBuffer weights_raw_choice1(cb_weights_raw_choice1);
        DataflowBuffer weights_bcast(cb_weights_bcast);
        using compute_kernel_lib::BroadcastDim;
        using compute_kernel_lib::input;
        using compute_kernel_lib::IterationShape;
        using compute_kernel_lib::output;
        compute_kernel_lib::unary_bcast<BroadcastDim::Col, input(cb_weights_raw_choice0), output(cb_weights_bcast)>(
            IterationShape::tiles(core_row_tiles), weights_raw_choice0, weights_bcast);
        compute_kernel_lib::unary_bcast<BroadcastDim::Col, input(cb_weights_raw_choice1), output(cb_weights_bcast)>(
            IterationShape::tiles(core_row_tiles), weights_raw_choice1, weights_bcast);
        // main loop formats, once: srcA = bf16 rows, srcB = fp32 (TF32) weights and bias, pack = bf16 output
        reconfig_data_format(cb_rows_choice0, cb_weights_bcast);
        pack_reconfig_data_format(cb_output_shard);
        cb_wait_front(cb_weights_bcast, top_k * core_row_tiles);
    } else {
        // srcA = bf16 rows / tilized rows, srcB = fp32 (TF32) weights and bias, pack = bf16 (tilized rows and output):
        // no data-format reconfig anywhere in the main loop.
        compute_kernel_hw_startup(cb_rows_choice0, cb_weights_raw_choice0, cb_output_shard);
    }
    DataflowBuffer rows_tilized(cb_rows_tilized);

    for (uint32_t row_idx = 0; row_idx < core_row_tiles; ++row_idx) {
#ifdef ABLATE_ZONES
        if constexpr (!fused_tilize) {
            {
                NMC_ZONE("NMC-C-WAITR0");
                cb_wait_front(cb_rows_choice0, shard_col_tiles);
            }
            {
                NMC_ZONE("NMC-C-WAITR1");
                cb_wait_front(cb_rows_choice1, shard_col_tiles);
            }
            if (row_idx == 0) {
                NMC_ZONE("NMC-C-WAITB");
                cb_wait_front(cb_bias_tiles, shard_col_tiles);
            }
            {
                NMC_ZONE("NMC-C-WAITW");
                cb_wait_front(cb_weights_raw_choice0, row_idx + 1);
                cb_wait_front(cb_weights_raw_choice1, row_idx + 1);
            }
        }
#endif
        NMC_ZONE("NMC-C-ROW");
        if constexpr (fused_tilize) {
            cb_wait_front(cb_rows_choice0, shard_col_tiles);
            cb_wait_front(cb_rows_choice1, shard_col_tiles);
        } else {
            // Each choice's 32 x W*32 row-major block -> W tiles (choice k at tiles k*W .. k*W + W - 1).
#ifndef ABLATE_TILIZE
            NMC_ZONE("NMC-C-TILIZE");
            using namespace compute_kernel_lib::tilize_config;
            compute_kernel_lib::tilize<
                shard_col_tiles,
                cb_rows_choice0,
                cb_rows_tilized,
                InitUninitMode::InitOnly,
                WaitMode::WaitBlock,
                ReconfigureRegisterDatatypeMode::NoReconfigure>(rows_choice0, rows_tilized, 1);
            compute_kernel_lib::tilize<
                shard_col_tiles,
                cb_rows_choice1,
                cb_rows_tilized,
                InitUninitMode::UninitOnly,
                WaitMode::WaitBlock,
                ReconfigureRegisterDatatypeMode::NoReconfigure>(rows_choice1, rows_tilized, 1);
#else
            cb_wait_front(cb_rows_choice0, shard_col_tiles);
            cb_wait_front(cb_rows_choice1, shard_col_tiles);
            cb_pop_front(cb_rows_choice0, shard_col_tiles);
            cb_pop_front(cb_rows_choice1, shard_col_tiles);
            cb_reserve_back(cb_rows_tilized, top_k * shard_col_tiles);
            cb_push_back(cb_rows_tilized, top_k * shard_col_tiles);
#endif
            cb_wait_front(cb_rows_tilized, top_k * shard_col_tiles);
        }
        // The DRAM-side inputs are waited only now, so the tile row's tilize overlaps their arrival; each tile row's
        // weight tile is pushed on its own (read pointer stays at tile 0: tile row_idx is the one just waited for).
        if (row_idx == 0) {
            cb_wait_front(cb_bias_tiles, shard_col_tiles);
        }
        if constexpr (!fused_tilize) {
            cb_wait_front(cb_weights_raw_choice0, row_idx + 1);
            cb_wait_front(cb_weights_raw_choice1, row_idx + 1);
        }
        for (uint32_t first = 0; first < shard_col_tiles; first += dest_group_tiles) {
            const uint32_t count =
                first + dest_group_tiles <= shard_col_tiles ? dest_group_tiles : shard_col_tiles - first;
            cb_reserve_back(cb_output_shard, count);
            tile_regs_acquire();
            // DEST starts as the bias, its row 0 broadcast down the 32 rows.
#ifndef ABLATE_BIAS
            {
                NMC_ZONE("NMC-C-BIAS");
                if constexpr (bias_seed_srcb) {
                    UNPACK((llk_unpack_A_init<BroadcastType::ROW, false, EltwiseBinaryReuseDestType::NONE, false>(
                        false, false, cb_bias_tiles)));
                    MATH((llk_math_eltwise_unary_datacopy_init<DataCopyType::B2D, DST_ACCUM_MODE, BroadcastType::ROW>(
                        cb_bias_tiles)));
                    for (uint32_t d = 0; d < count; ++d) {
                        UNPACK((llk_unpack_A<BroadcastType::ROW, false, EltwiseBinaryReuseDestType::NONE, false>(
                            cb_bias_tiles, first + d)));
                        MATH((llk_math_eltwise_unary_datacopy<
                              DataCopyType::B2D,
                              DST_ACCUM_MODE,
                              BroadcastType::ROW,
                              false>(d, cb_bias_tiles)));
                    }
                    UNPACK((llk_unpack_A_uninit<BroadcastType::ROW>()));
                    MATH((llk_math_eltwise_unary_datacopy_uninit<BroadcastType::ROW, false>()));
                } else {
                    unary_bcast_init<BroadcastType::ROW>(cb_bias_tiles);
                    for (uint32_t d = 0; d < count; ++d) {
                        unary_bcast<BroadcastType::ROW>(cb_bias_tiles, first + d, d);
                    }
                    unary_bcast_uninit<BroadcastType::ROW>(cb_bias_tiles);
                }
            }
#endif
#ifndef ABLATE_MULTIPLY
            {
                NMC_ZONE("NMC-C-MUL");
                if constexpr (fused_tilize) {
                    // rows_k (tilized in the unpacker) x the expanded weights accumulate on it, k = 0 then 1.
                    UNPACK((llk_unpack_tilizeA_B_init<false, false, false, false>(
                        cb_rows_choice0, cb_weights_bcast, shard_col_tiles)));
                    MATH((llk_math_eltwise_binary_init<EltwiseBinaryType::ELWMUL, BroadcastType::NONE, MATH_FIDELITY>(
                        cb_rows_choice0, cb_weights_bcast, 1 /*acc_to_dest*/)));
                    for (uint32_t k = 0; k < top_k; ++k) {
                        const uint32_t cb_rows = k == 0 ? cb_rows_choice0 : cb_rows_choice1;
                        const uint32_t weight_tile = k * core_row_tiles + row_idx;
                        for (uint32_t d = 0; d < count; ++d) {
                            UNPACK((llk_unpack_tilizeA_B<false, false, false, false>(
                                cb_rows, cb_weights_bcast, first + d, weight_tile, shard_col_tiles)));
                            MATH((llk_math_eltwise_binary<
                                  EltwiseBinaryType::ELWMUL,
                                  BroadcastType::NONE,
                                  DST_ACCUM_MODE,
                                  MATH_FIDELITY,
                                  EltwiseBinaryReuseDestType::NONE>(cb_rows, cb_weights_bcast, d, true)));
                        }
                    }
                    UNPACK((llk_unpack_tilizeA_B_uninit(cb_rows_choice0)));
                } else {
                    // tile(rows_k) x weights_k column 0 broadcast along the columns accumulate on it (ELWMUL
                    // accumulates into DEST), k = 0 then 1; one init for both choices (same formats and face geometry).
                    bcast_init<EltwiseBinaryType::ELWMUL, BroadcastType::COL>(cb_rows_tilized, cb_weights_raw_choice0);
                    for (uint32_t d = 0; d < count; ++d) {
                        mul_tiles_bcast_cols(cb_rows_tilized, cb_weights_raw_choice0, first + d, row_idx, d);
                    }
                    for (uint32_t d = 0; d < count; ++d) {
                        mul_tiles_bcast_cols(
                            cb_rows_tilized, cb_weights_raw_choice1, shard_col_tiles + first + d, row_idx, d);
                    }
                }
            }
#endif
            tile_regs_commit();
            NMC_ZONE("NMC-C-PACK");
            tile_regs_wait();
            for (uint32_t d = 0; d < count; ++d) {
                pack_tile(d, cb_output_shard);
            }
            tile_regs_release();
            cb_push_back(cb_output_shard, count);
        }
        if constexpr (fused_tilize) {
            cb_pop_front(cb_rows_choice0, shard_col_tiles);
            cb_pop_front(cb_rows_choice1, shard_col_tiles);
        } else {
            cb_pop_front(cb_rows_tilized, top_k * shard_col_tiles);
        }
    }
    cb_pop_front(cb_bias_tiles, shard_col_tiles);
    if constexpr (fused_tilize) {
        cb_pop_front(cb_weights_bcast, top_k * core_row_tiles);
    } else {
        cb_pop_front(cb_weights_raw_choice0, core_row_tiles);
        cb_pop_front(cb_weights_raw_choice1, core_row_tiles);
    }
}
