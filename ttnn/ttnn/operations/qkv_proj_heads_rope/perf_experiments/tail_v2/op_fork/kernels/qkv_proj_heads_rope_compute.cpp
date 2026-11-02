// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// qkv_proj_heads_rope compute: per block, seed the Float32 partials with the row-broadcast bias, project_block
// (matmul_block over K-blocks 0..n-2 of K' = hidden, L1-accumulating onto the seed from K-block 0), then the
// last K-block one row at a time, each row's
// epilogue following it: rotate_block on q/k cores (two DEST-accumulated FPU muls per output tile, raw
// whole-tile phase-major ELWMUL from tile_src_mul.hpp: 1.95x over mul_tiles, bit-identical),
// convert_block (copy + pack to bf16) on v cores. Output rows alternate between cb_out_tiles (writer, NoC1)
// and cb_out_tiles_noc0 (reader, NoC0) when out_write_nocs == 2.
//
// Perf 1 (measured, 8x512: 106.3 -> 100.5 us): running the last K-block per row lets each row's epilogue and
// store overlap the rest of the K-loop instead of bursting after it. The last K-block is raw LLK because
// compute_kernel_lib::matmul_block sets llk_pack_reconfig_l1_acc(block == 0 ? 0 : 1), so a per-row call
// carrying only the last K-block would overwrite the partials of K-blocks 0..n-2, and it has no per-row hook
// between subblock packs. The matmul of row r + LAST_K_LEAD runs before the epilogue of row r, so the
// epilogue's unpack does not wait on that row's final pack (lead 0: 102.1, 1: 100.5, 2: 100.3 us).
//
// Perf 1 (measured, 8x512 with K' = 24 / k_block 4 vs the 25th bias K tile: -2.3%): the bias seed is
// unary_bcast<ROW> + pack_tile, each DEST tile packed to all block_m rows - one unpack per bias tile,
// block_m packs; no eltwise_chain element packs one DEST tile to several CB tiles. It runs while math waits
// for the first x / W K-block.
//
// Raw calls: pack_reconfig_l1_acc(0/1) around the per-row last K-block - no helper exposes the L1-acc
// toggle; the epilogue setup (reconfig + mul_init + tile_src_mul_init / copy_init) once per row (the last-K
// matmul init between rows reprograms unpack and math), which the convert chain's InitReconfigOwner::Caller
// hands to the caller.

#include <cstdint>

#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/pack.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/bcast.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/reconfig_data_format.h"
#include "api/dataflow/circular_buffer.h"
#include "ttnn/cpp/ttnn/kernel_lib/matmul_block_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/perf_instrumentation.hpp"
#include "tile_src_mul.hpp"

void kernel_main() {
    constexpr uint32_t cb_activation_block = get_compile_time_arg_val(0);
    constexpr uint32_t cb_weight_block = get_compile_time_arg_val(1);
    constexpr uint32_t cb_matmul_partials = get_compile_time_arg_val(2);
    constexpr uint32_t cb_rotary_table = get_compile_time_arg_val(3);
    constexpr uint32_t cb_out_tiles = get_compile_time_arg_val(4);
    constexpr uint32_t block_m = get_compile_time_arg_val(5);
    constexpr uint32_t heads_per_core = get_compile_time_arg_val(6);
    constexpr uint32_t tiles_per_head = get_compile_time_arg_val(7);
    constexpr uint32_t k_block = get_compile_time_arg_val(8);
    constexpr uint32_t num_k_blocks = get_compile_time_arg_val(9);
    constexpr uint32_t sb_h = get_compile_time_arg_val(10);
    constexpr uint32_t sb_w = get_compile_time_arg_val(11);
    constexpr uint32_t out_write_nocs = get_compile_time_arg_val(12);
    constexpr uint32_t cb_out_tiles_noc0 = get_compile_time_arg_val(13);
    constexpr uint32_t seq_tiles = get_compile_time_arg_val(14);
    // DEST tiles per acquire (fp32 / sync mode), as the host sized the matmul subblock.
    constexpr uint32_t dest_tiles = get_compile_time_arg_val(15);
    constexpr uint32_t cb_bias = get_compile_time_arg_val(16);

    const uint32_t core_rows = get_arg_val<uint32_t>(0);
    const bool is_rotary = get_arg_val<uint32_t>(1) != 0;

    using namespace compute_kernel_lib;

    constexpr uint32_t n_tiles = heads_per_core * tiles_per_head;
    constexpr uint32_t partials_tiles = block_m * n_tiles;
    constexpr uint32_t table_tiles_per_row = 2 * tiles_per_head;
    static_assert(block_m % sb_h == 0 && n_tiles % sb_w == 0);
    static_assert(tiles_per_head == 2, "rotate_half as a tile swap needs D/2 == one tile");
    static_assert(out_write_nocs == 1 || out_write_nocs == 2);

    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_activation_block, cb_weight_block, cb_matmul_partials);

    CircularBuffer activation(cb_activation_block);
    CircularBuffer weight(cb_weight_block);
    CircularBuffer partials(cb_matmul_partials);
    DataflowBuffer partials_dfb(cb_matmul_partials);
    DataflowBuffer out_dfb(cb_out_tiles);

    static_assert(num_k_blocks >= 2, "the last K-block is split off the helper's K-loop");
    constexpr auto shape = MatmulBlockShape::of(block_m / sb_h, n_tiles / sb_w, sb_h, sb_w, k_block, num_k_blocks - 1);
    // Last K-block runs one row at a time: widest divisor of n_tiles that fits the host's DEST subblock.
    constexpr uint32_t row_sb_w = [] {
        uint32_t w = sb_h * sb_w < n_tiles ? sb_h * sb_w : n_tiles;
        while (n_tiles % w) {
            --w;
        }
        return w;
    }();
    // Bias seed: widest divisor of n_tiles that fits one DEST acquire.
    constexpr uint32_t seed_w = [] {
        uint32_t w = dest_tiles < n_tiles ? dest_tiles : n_tiles;
        while (n_tiles % w) {
            --w;
        }
        return w;
    }();
#ifdef TAIL_LEAD2
    constexpr uint32_t LAST_K_LEAD = 2;
#else
    constexpr uint32_t LAST_K_LEAD = 1;
#endif
    // tail_v2 grouped schedule: groups of TAIL_GROUP rows share one matmul / epilogue init; with
    // TAIL_FINAL1 the block's last row is a group of its own (shortest exposed tail).
#if defined(TAIL_G2)
    constexpr uint32_t TAIL_GROUP = 2;
#elif defined(TAIL_G3)
    constexpr uint32_t TAIL_GROUP = 3;
#elif defined(TAIL_G4)
    constexpr uint32_t TAIL_GROUP = 4;
#elif defined(TAIL_G11)
    constexpr uint32_t TAIL_GROUP = 11;
#else
    constexpr uint32_t TAIL_GROUP = 0;
#endif
#ifdef TAIL_FINAL1
    constexpr bool TAIL_FINAL_SINGLE = true;
#else
    constexpr bool TAIL_FINAL_SINGLE = false;
#endif
    constexpr uint32_t act_block_tiles = block_m * k_block;
    constexpr uint32_t weight_block_tiles = k_block * n_tiles;

    // convert_block: Float32 partials row -> bf16 output row.
    using ConvertCopy = CopyTile<input(
        cb_matmul_partials,
        WaitPolicy::None,
        PopPolicy::None,
        InputTileMapping::Block,
        DataFormatReconfig::Disabled,
        TileAddressing::Offset)>;
    using ConvertPack =
        PackTile<output(cb_out_tiles, ReservePolicy::Upfront, PushPolicy::AtEnd, DataFormatReconfig::Disabled)>;
    using ConvertPackNoc0 =
        PackTile<output(cb_out_tiles_noc0, ReservePolicy::Upfront, PushPolicy::AtEnd, DataFormatReconfig::Disabled)>;

    // Epilogue of one valid row into cb_out (convert_pack is bound to the same CB).
    // rotate_block: out_lo = P0 * cos_lo + P1 * (-sin_lo), out_hi = P0 * sin_hi + P1 * cos_hi against the row's
    // table slot [cos_lo, -sin_lo, sin_hi, cos_hi], window_heads heads per DEST window accumulated in place.
    auto epilogue_row = [&](uint32_t row, uint32_t cb_out, auto convert_pack) {
        const uint32_t row_base = row * n_tiles;
#if defined(TAIL_ABL_EPI_NONE) || defined(TAIL_ABL_EPI_ROT_NONE) || defined(TAIL_ABL_EPI_V_NONE)
#if defined(TAIL_ABL_EPI_NONE)
        if (true) {
#elif defined(TAIL_ABL_EPI_ROT_NONE)
        if (is_rotary) {
#else
        if (!is_rotary) {
#endif
            cb_reserve_back(cb_out, n_tiles);
            cb_push_back(cb_out, n_tiles);
            return;
        }
#endif
        if (is_rotary) {
            constexpr uint32_t window_heads = dest_tiles / tiles_per_head;
            static_assert(window_heads >= 1);
            const uint32_t table_base = (row % seq_tiles) * table_tiles_per_row;
            cb_reserve_back(cb_out, n_tiles);
            for (uint32_t head0 = 0; head0 < heads_per_core; head0 += window_heads) {
                const uint32_t heads = heads_per_core - head0 < window_heads ? heads_per_core - head0 : window_heads;
                tile_regs_acquire();
                for (uint32_t head = 0; head < heads; ++head) {
                    const uint32_t p0 = row_base + (head0 + head) * tiles_per_head;
                    const uint32_t dst = head * tiles_per_head;
#ifdef TAIL_PREUSE
                    // Same per-DEST-tile accumulation order (P0 term, then P1 term) as the 4-call form.
                    tile_src::tile_src_mul_pair(
                        cb_matmul_partials, cb_rotary_table, p0, table_base + 0, table_base + 2, dst, dst + 1);
                    tile_src::tile_src_mul_pair(
                        cb_matmul_partials, cb_rotary_table, p0 + 1, table_base + 1, table_base + 3, dst, dst + 1);
#else
                    tile_src::tile_src_mul(cb_matmul_partials, cb_rotary_table, p0, table_base + 0, dst);
                    tile_src::tile_src_mul(cb_matmul_partials, cb_rotary_table, p0 + 1, table_base + 1, dst);
                    tile_src::tile_src_mul(cb_matmul_partials, cb_rotary_table, p0, table_base + 2, dst + 1);
                    tile_src::tile_src_mul(cb_matmul_partials, cb_rotary_table, p0 + 1, table_base + 3, dst + 1);
#endif
                }
                tile_regs_commit();
                tile_regs_wait();
                for (uint32_t tile = 0; tile < heads * tiles_per_head; ++tile) {
                    pack_tile(tile, cb_out);
                }
                tile_regs_release();
            }
            cb_push_back(cb_out, n_tiles);
        } else {
#ifdef TAIL_VBLK
            // One DEST window per dest_tiles tiles instead of per tile.
            eltwise_chain<InitReconfigOwner::Caller>(
                IterationShape::tiles(n_tiles).block_size(dest_tiles),
                ConvertCopy{partials_dfb, row_base},
                convert_pack);
#else
            eltwise_chain<InitReconfigOwner::Caller>(
                IterationShape::tiles(n_tiles), ConvertCopy{partials_dfb, row_base}, convert_pack);
#endif
        }
    };

    // Every tile of the reserved partials block = the row-broadcast bias of its column (bf16 -> fp32 exact).
    auto seed_partials = [&]() {
        reconfig_data_format_srca(cb_bias);
        unary_bcast_init<BroadcastType::ROW>(cb_bias);
        pack_reconfig_data_format(cb_matmul_partials);
        pack_reconfig_l1_acc(0);
        for (uint32_t col = 0; col < n_tiles; col += seed_w) {
            tile_regs_acquire();
            for (uint32_t c = 0; c < seed_w; ++c) {
                unary_bcast<BroadcastType::ROW>(cb_bias, col + c, c);
            }
            tile_regs_commit();
            tile_regs_wait();
            for (uint32_t row = 0; row < block_m; ++row) {
                for (uint32_t c = 0; c < seed_w; ++c) {
                    pack_tile<true>(c, cb_matmul_partials, row * n_tiles + col + c);
                }
            }
            tile_regs_release();
        }
    };

    // Last K-block of one row: accumulate onto the partials K-blocks 0..n-2 left in L1 (the row's partials
    // sit at the CB write pointer, which advances one row per push).
    auto last_k_rows = [&](uint32_t first_row, uint32_t count) {
        reconfig_data_format(cb_weight_block, cb_activation_block);
        ckernel::matmul_block_init(cb_activation_block, cb_weight_block, false, row_sb_w, 1, k_block);
        pack_reconfig_data_format(cb_matmul_partials);
        pack_reconfig_l1_acc(1);
        for (uint32_t row = first_row; row < first_row + count; ++row) {
            for (uint32_t col = 0; col < n_tiles; col += row_sb_w) {
                tile_regs_acquire();
#ifdef TAIL_ABL_LASTK_NOMM
                for (uint32_t inner = 0; inner < 0; ++inner) {
#else
                for (uint32_t inner = 0; inner < k_block; ++inner) {
#endif
                    ckernel::matmul_block(
                        cb_activation_block,
                        cb_weight_block,
                        row * k_block + inner,
                        inner * n_tiles + col,
                        0,
                        false,
                        row_sb_w,
                        1,
                        k_block);
                }
                tile_regs_commit();
                tile_regs_wait();
#ifndef TAIL_ABL_LASTK_NOPACK
                for (uint32_t c = 0; c < row_sb_w; ++c) {
                    pack_tile<true>(c, cb_matmul_partials, col + c);
                }
#endif
                tile_regs_release();
            }
            partials.push_back(n_tiles);
        }
        pack_reconfig_l1_acc(0);
    };
    auto last_k_row = [&](uint32_t row) { last_k_rows(row, 1); };

    auto epilogue_setup = [&]() {
        if (is_rotary) {
            reconfig_data_format(cb_matmul_partials, cb_rotary_table);
            mul_init(cb_matmul_partials, cb_rotary_table, true);
            tile_src::tile_src_mul_init();
        } else {
            reconfig_data_format_srca(cb_matmul_partials);
            copy_init(cb_matmul_partials);
        }
        pack_reconfig_data_format(cb_out_tiles);
    };

    // tail_v2 halfsplit: each DEST window of a row is pushed on its own, even windows to cb_out_tiles
    // (writer, NoC1), odd to cb_out_tiles_noc0 (reader, NoC0): every row's store uses both NoCs and starts
    // one window earlier.
    constexpr uint32_t split_win = dest_tiles < n_tiles ? dest_tiles : n_tiles;
    constexpr uint32_t split_windows = (n_tiles + split_win - 1) / split_win;
#ifdef TAIL_HALFSPLIT
    constexpr bool halfsplit = out_write_nocs == 2 && split_windows >= 2 && split_win % tiles_per_head == 0;
#else
    constexpr bool halfsplit = false;
#endif
    auto epilogue_row_split = [&](uint32_t row) {
        const uint32_t row_base = row * n_tiles;
        const uint32_t table_base = (row % seq_tiles) * table_tiles_per_row;
        DataflowBuffer out_noc0_dfb(cb_out_tiles_noc0);
        for (uint32_t w = 0; w < split_windows; ++w) {
            const uint32_t t0 = w * split_win;
            const uint32_t count = n_tiles - t0 < split_win ? n_tiles - t0 : split_win;
            const uint32_t cb_out = (w & 1) ? cb_out_tiles_noc0 : cb_out_tiles;
            if (is_rotary) {
                cb_reserve_back(cb_out, count);
                tile_regs_acquire();
                for (uint32_t dst = 0; dst < count; dst += tiles_per_head) {
                    const uint32_t p0 = row_base + t0 + dst;
#ifdef TAIL_PREUSE
                    tile_src::tile_src_mul_pair(
                        cb_matmul_partials, cb_rotary_table, p0, table_base + 0, table_base + 2, dst, dst + 1);
                    tile_src::tile_src_mul_pair(
                        cb_matmul_partials, cb_rotary_table, p0 + 1, table_base + 1, table_base + 3, dst, dst + 1);
#else
                    tile_src::tile_src_mul(cb_matmul_partials, cb_rotary_table, p0, table_base + 0, dst);
                    tile_src::tile_src_mul(cb_matmul_partials, cb_rotary_table, p0 + 1, table_base + 1, dst);
                    tile_src::tile_src_mul(cb_matmul_partials, cb_rotary_table, p0, table_base + 2, dst + 1);
                    tile_src::tile_src_mul(cb_matmul_partials, cb_rotary_table, p0 + 1, table_base + 3, dst + 1);
#endif
                }
                tile_regs_commit();
                tile_regs_wait();
                for (uint32_t tile = 0; tile < count; ++tile) {
                    pack_tile(tile, cb_out);
                }
                tile_regs_release();
                cb_push_back(cb_out, count);
            } else if (w & 1) {
                eltwise_chain<InitReconfigOwner::Caller>(
                    IterationShape::tiles(count).block_size(dest_tiles),
                    ConvertCopy{partials_dfb, row_base + t0},
                    ConvertPackNoc0{out_noc0_dfb});
            } else {
                eltwise_chain<InitReconfigOwner::Caller>(
                    IterationShape::tiles(count).block_size(dest_tiles),
                    ConvertCopy{partials_dfb, row_base + t0},
                    ConvertPack{out_dfb});
            }
        }
    };

    auto epilogue_row_routed = [&](uint32_t row) {
        if constexpr (halfsplit) {
            if (is_rotary) {
                cb_wait_front(cb_rotary_table, (row + 1 < seq_tiles ? row + 1 : seq_tiles) * table_tiles_per_row);
            }
            partials.wait_front((row + 1) * n_tiles);
            epilogue_row_split(row);
            return;
        }
        if (is_rotary) {
            cb_wait_front(cb_rotary_table, (row + 1 < seq_tiles ? row + 1 : seq_tiles) * table_tiles_per_row);
        }
        partials.wait_front((row + 1) * n_tiles);
        if constexpr (out_write_nocs == 2) {
            if (row & 1) {
                DataflowBuffer out_noc0_dfb(cb_out_tiles_noc0);
                epilogue_row(row, cb_out_tiles_noc0, ConvertPackNoc0{out_noc0_dfb});
                return;
            }
        }
        epilogue_row(row, cb_out_tiles, ConvertPack{out_dfb});
    };

    {
        MaybeDeviceZoneScope("compute_bias_wait");
        cb_wait_front(cb_bias, n_tiles);
    }
    for (uint32_t block_row = 0; block_row < core_rows; block_row += block_m) {
        const uint32_t rows = (core_rows - block_row) < block_m ? (core_rows - block_row) : block_m;

        {
            MaybeDeviceZoneScope("compute_partials_reserve");
            partials.reserve_back(partials_tiles);
        }
        {
            MaybeDeviceZoneScope("compute_bias_seed");
            seed_partials();
        }
        {
            // Occupancy: matmul_block waits per K-block on x / W inside the helper.
            MaybeDeviceZoneScope("compute_matmul");
            matmul_block<
                false,
                true,
                LastBlockTarget::Interm,
                OutputCBLayout::TileRowMajor,
                matmul_config::InitMode::Short,
                InputPolicy::WaitAndPopPerKBlock,
                InputPolicy::WaitAndPopPerKBlock,
                NoPostCompute,
                NoPreKBlock,
                NoPostKBlock,
                0,
                NoKBlockInnerDimFn,
                NoIn0Source,
                NoIn1BaseOffset,
                true,
                true>(activation, weight, partials, partials, shape, {}, {}, n_tiles, n_tiles);
        }

        // Last K-block interleaved with the epilogue: matmul row r + LAST_K_LEAD, then the epilogue of row r.
        {
            MaybeDeviceZoneScope("compute_last_k_wait");
            activation.wait_front(act_block_tiles);
            weight.wait_front(weight_block_tiles);
        }
        // Occupancy: per-row table / partials waits and the output-CB reserves sit inside.
        MaybeDeviceZoneScope("compute_last_k_epilogue");
        if constexpr (TAIL_GROUP > 0) {
            auto group_size = [&](uint32_t start) -> uint32_t {
                const uint32_t end = TAIL_FINAL_SINGLE && rows > 1 ? rows - 1 : rows;
                if (start >= end) {
                    return rows - start;
                }
                return end - start < TAIL_GROUP ? end - start : TAIL_GROUP;
            };
            uint32_t cur = 0;
            uint32_t cur_n = group_size(0);
            last_k_rows(0, cur_n);
            while (cur < rows) {
                const uint32_t nxt = cur + cur_n;
                const uint32_t nxt_n = nxt < rows ? group_size(nxt) : 0;
                if (nxt_n) {
                    last_k_rows(nxt, nxt_n);
                }
                epilogue_setup();
                for (uint32_t row = cur; row < nxt; ++row) {
                    epilogue_row_routed(row);
                }
                cur = nxt;
                cur_n = nxt_n;
            }
        } else {
            const uint32_t lead = rows < LAST_K_LEAD ? rows : LAST_K_LEAD;
            for (uint32_t row = 0; row < lead; ++row) {
                last_k_row(row);
            }
            for (uint32_t row = 0; row < rows; ++row) {
                if (row + lead < rows) {
                    last_k_row(row + lead);
                }
                epilogue_setup();
                epilogue_row_routed(row);
            }
        }
        // Nominal rows past `rows` are never consumed; push them so the block's pop balances.
        if (rows < block_m) {
            partials.push_back((block_m - rows) * n_tiles);
        }
        activation.pop_front(act_block_tiles);
        weight.pop_front(weight_block_tiles);

        partials.wait_front(partials_tiles);
        partials.pop_front(partials_tiles);
        if (is_rotary) {
            const uint32_t table_tiles = (rows < seq_tiles ? rows : seq_tiles) * table_tiles_per_row;
            cb_pop_front(cb_rotary_table, table_tiles);
        }
    }
    cb_pop_front(cb_bias, n_tiles);
}
