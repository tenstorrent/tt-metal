// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// qkv_proj_heads_rope compute: per block, seed the Float32 partials with the row-broadcast bias, project_block
// (K-blocks 0..n-2 of K' = hidden, L1-accumulating onto the seed from K-block 0), then the
// last K-block one row at a time, each row's
// epilogue following it: rotate_block on q/k cores (two DEST-accumulated FPU muls per output tile, raw
// whole-tile phase-major ELWMUL from tile_src_mul.hpp, each partial tile unpacked once for two products),
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
// Perf 2 (measured, 8x512: 94.9 -> 92.1 us, bit-identical): every matmul call goes through mm_fused_mop.hpp
// (one math MOP per call instead of one per tile product, 52.4 -> 49.3 cycles per product at HiFi3), so
// K-blocks 0..n-2 are a raw K-loop instead of compute_kernel_lib::matmul_block: the helper's math is
// ckernel::matmul_block, which has no fused-MOP mode (a raw K-loop with the stock LLK measured flat vs the
// helper: 95.3 vs 94.9 us).
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
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/perf_instrumentation.hpp"
#include "tile_src_mul.hpp"
#include "mm_fused_mop.hpp"
#include "f3_sched.hpp"

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

    static_assert(num_k_blocks >= 2, "the last K-block is split off the block-wide K-loop");
    static_assert(f3::valid<k_block, num_k_blocks, 24>());
    constexpr uint32_t last_k_size = f3::ksize<k_block>(num_k_blocks - 1);
#ifndef F3_SEED_AFTER
#define F3_SEED_AFTER 0
#endif
    static_assert(F3_SEED_AFTER + 1 < num_k_blocks);
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
    constexpr uint32_t LAST_K_LEAD = 1;
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
                    // Per DEST tile the P0 term lands before the P1 term, as in the one-product-per-call order.
                    tile_src::tile_src_mul_pair(
                        cb_matmul_partials, cb_rotary_table, p0, table_base + 0, table_base + 2, dst, dst + 1);
                    tile_src::tile_src_mul_pair(
                        cb_matmul_partials, cb_rotary_table, p0 + 1, table_base + 1, table_base + 3, dst, dst + 1);
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
            // One DEST window per dest_tiles tiles (Perf 2: per-tile windows cost 8 handshakes per row).
            eltwise_chain<InitReconfigOwner::Caller>(
                IterationShape::tiles(n_tiles).block_size(dest_tiles),
                ConvertCopy{partials_dfb, row_base},
                convert_pack);
        }
    };

    // Every tile of the reserved partials block = the row-broadcast bias of its column (bf16 -> fp32 exact).
    auto seed_partials = [&](bool accumulate) {
        reconfig_data_format_srca(cb_bias);
        unary_bcast_init<BroadcastType::ROW>(cb_bias);
        pack_reconfig_data_format(cb_matmul_partials);
        pack_reconfig_l1_acc(accumulate ? 1 : 0);
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
    auto last_k_row = [&](uint32_t row) {
        reconfig_data_format(cb_weight_block, cb_activation_block);
        ckernel::matmul_block_init(cb_activation_block, cb_weight_block, false, row_sb_w, 1, k_block);
        mmk::fused_init(row_sb_w, 1);
        pack_reconfig_data_format(cb_matmul_partials);
        pack_reconfig_l1_acc(1);
        for (uint32_t col = 0; col < n_tiles; col += row_sb_w) {
            tile_regs_acquire();
            for (uint32_t inner = 0; inner < last_k_size; ++inner) {
                mmk::fused_matmul_block(
                    cb_activation_block,
                    cb_weight_block,
                    row * last_k_size + inner,
                    inner * n_tiles + col,
                    0,
                    row_sb_w,
                    1,
                    last_k_size);
            }
            tile_regs_commit();
            tile_regs_wait();
            for (uint32_t c = 0; c < row_sb_w; ++c) {
                pack_tile<true>(c, cb_matmul_partials, col + c);
            }
            tile_regs_release();
        }
        partials.push_back(n_tiles);
        pack_reconfig_l1_acc(0);
    };

    // K-blocks 0..n-2 of the block, accumulating onto the seed in L1 (init once per block, not per subblock).
    auto mm_init = [&]() {
        reconfig_data_format(cb_weight_block, cb_activation_block);
        pack_reconfig_data_format(cb_matmul_partials);
        ckernel::matmul_block_init(cb_activation_block, cb_weight_block, false, sb_w, sb_h, k_block);
        mmk::fused_init(sb_w, sb_h);
    };
    auto block_k_loop = [&](bool first_block) {
        mm_init();
#if defined(F3_LATESEED) || defined(F3_DSEED_KB)
        // K-block 0 overwrites the partials; the bias seed is L1-accumulated after K-block F3_SEED_AFTER, so
        // the bias read is off the fill (F3_SEED_AFTER 0: bias + K0 == K0 + bias, bit-identical).
        pack_reconfig_l1_acc(0);
#else
        pack_reconfig_l1_acc(1);
#endif
        for (uint32_t kb = 0; kb + 1 < num_k_blocks; ++kb) {
            const uint32_t k_size = f3::ksize<k_block>(kb);
#ifdef F3_DSEED_KB
            if (kb == F3_DSEED_KB && first_block) {
                MaybeDeviceZoneScope("compute_bias_wait");
                cb_wait_front(cb_bias, n_tiles);
            }
#endif
            {
                MaybeDeviceZoneScope("compute_kb_wait");
                activation.wait_front(act_block_tiles);
                weight.wait_front(weight_block_tiles);
            }
            for (uint32_t m = 0; m < block_m / sb_h; ++m) {
                for (uint32_t n = 0; n < n_tiles / sb_w; ++n) {
                    tile_regs_acquire();
#ifdef F3_DSEED_KB
                    // Bias row broadcast into DEST under this K-block's products: no extra pack.
                    if (kb == F3_DSEED_KB) {
                        unary_bcast_init<BroadcastType::ROW>(cb_bias);
                        for (uint32_t r = 0; r < sb_h; ++r) {
                            for (uint32_t c = 0; c < sb_w; ++c) {
                                unary_bcast<BroadcastType::ROW>(cb_bias, n * sb_w + c, r * sb_w + c);
                            }
                        }
                        ckernel::matmul_block_init(cb_activation_block, cb_weight_block, false, sb_w, sb_h, k_block);
                        mmk::fused_init(sb_w, sb_h);
                    }
#endif
                    for (uint32_t inner = 0; inner < k_size; ++inner) {
                        mmk::fused_matmul_block(
                            cb_activation_block,
                            cb_weight_block,
                            m * sb_h * k_size + inner,
                            inner * n_tiles + n * sb_w,
                            0,
                            sb_w,
                            sb_h,
                            k_size);
                    }
                    tile_regs_commit();
                    tile_regs_wait();
                    for (uint32_t r = 0; r < sb_h; ++r) {
                        for (uint32_t c = 0; c < sb_w; ++c) {
                            pack_tile<true>(r * sb_w + c, cb_matmul_partials, (m * sb_h + r) * n_tiles + n * sb_w + c);
                        }
                    }
                    tile_regs_release();
                }
            }
            activation.pop_front(act_block_tiles);
            weight.pop_front(weight_block_tiles);
#ifdef F3_DSEED_KB
            if (kb == 0) {
                pack_reconfig_l1_acc(1);
            }
#endif
#ifdef F3_LATESEED
            if (kb == 0) {
                pack_reconfig_l1_acc(1);
            }
            if (kb == F3_SEED_AFTER) {
                if (first_block) {
                    MaybeDeviceZoneScope("compute_bias_wait");
                    cb_wait_front(cb_bias, n_tiles);
                }
                {
                    MaybeDeviceZoneScope("compute_bias_seed");
                    seed_partials(true);
                }
                mm_init();
            }
#endif
        }
        pack_reconfig_l1_acc(0);
    };

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

    auto epilogue_row_routed = [&](uint32_t row) {
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

#if !defined(F3_LATESEED) && !defined(F3_DSEED_KB)
    {
        MaybeDeviceZoneScope("compute_bias_wait");
        cb_wait_front(cb_bias, n_tiles);
    }
#endif
    for (uint32_t block_row = 0; block_row < core_rows; block_row += block_m) {
        const uint32_t rows = (core_rows - block_row) < block_m ? (core_rows - block_row) : block_m;

        {
            MaybeDeviceZoneScope("compute_partials_reserve");
            partials.reserve_back(partials_tiles);
        }
#if !defined(F3_LATESEED) && !defined(F3_DSEED_KB)
        {
            MaybeDeviceZoneScope("compute_bias_seed");
            seed_partials(false);
        }
#endif
        {
            // Occupancy: waits per K-block on x / W inside.
            MaybeDeviceZoneScope("compute_matmul");
            block_k_loop(block_row == 0);
        }

        // Last K-block interleaved with the epilogue: matmul row r + LAST_K_LEAD, then the epilogue of row r.
        {
            MaybeDeviceZoneScope("compute_last_k_wait");
            activation.wait_front(act_block_tiles);
            weight.wait_front(weight_block_tiles);
        }
        // Occupancy: per-row table / partials waits and the output-CB reserves sit inside.
        MaybeDeviceZoneScope("compute_last_k_epilogue");
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
    if (core_rows > 0) {
        cb_pop_front(cb_bias, n_tiles);
    }
}
